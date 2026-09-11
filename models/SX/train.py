from __future__ import annotations

import argparse
import random
from contextlib import nullcontext
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from models.SX.model import HandSignLSTM


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred is not None and preferred != "":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def extract_body_pose_from_rgb(
    rgb: np.ndarray,
    pose=None,
) -> np.ndarray:
    """Convert an RGB image into a flattened body pose landmark vector."""
    owns_pose = pose is None
    detector = pose or mp.solutions.pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        results = detector.process(rgb)
    finally:
        if owns_pose:
            detector.close()

    pose_dim = 99
    body_vector = np.zeros(pose_dim, dtype=np.float32)

    if results.pose_landmarks:
        coords = []
        for landmark in results.pose_landmarks.landmark:
            coords.extend([landmark.x, landmark.y, landmark.z])
        body_vector = np.asarray(coords[:pose_dim], dtype=np.float32)
        if body_vector.shape[0] < pose_dim:
            pad = np.zeros(pose_dim - body_vector.shape[0], dtype=np.float32)
            body_vector = np.concatenate([body_vector, pad])

    return body_vector


def extract_hand_landmarks_from_image(
    image_path: Path,
    two_hands: bool = True,
    sequence_length: int = 1,
    hands=None,
    use_body_pose: bool = False,
    pose=None,
) -> np.ndarray:
    """Convert one image into a flattened landmark vector sequence."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    owns_hands = hands is None
    detector = hands or mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        results = detector.process(rgb)
    finally:
        if owns_hands:
            detector.close()

    hand_output_dim = 126 if two_hands else 63
    feature_dim = hand_output_dim + (99 if use_body_pose else 0)
    sequence = np.zeros((sequence_length, feature_dim), dtype=np.float32)
    hand_vector = np.zeros(hand_output_dim, dtype=np.float32)

    if results.multi_hand_landmarks:
        hand_vectors: list[np.ndarray] = []
        for hand in results.multi_hand_landmarks[:2]:
            coords = []
            for landmark in hand.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])
            hand_vectors.append(np.asarray(coords, dtype=np.float32))

        while len(hand_vectors) < 2 and two_hands:
            hand_vectors.append(np.zeros(63, dtype=np.float32))

        if two_hands:
            combined = np.concatenate(hand_vectors[:2], axis=0)
        else:
            combined = hand_vectors[0] if hand_vectors else np.zeros(63, dtype=np.float32)
        combined = combined.astype(np.float32)
        combined = combined[:hand_output_dim]
        if combined.shape[0] < hand_output_dim:
            pad = np.zeros(hand_output_dim - combined.shape[0], dtype=np.float32)
            combined = np.concatenate([combined, pad])
        hand_vector = combined

    if use_body_pose:
        body_vector = extract_body_pose_from_rgb(rgb, pose=pose)
        combined = np.concatenate([hand_vector, body_vector], axis=0)
    else:
        combined = hand_vector

    combined = combined[:feature_dim]
    if combined.shape[0] < feature_dim:
        pad = np.zeros(feature_dim - combined.shape[0], dtype=np.float32)
        combined = np.concatenate([combined, pad])

    sequence[0, :] = combined
    return sequence


def cache_landmarks(
    samples: list[tuple[Path, str]],
    cache_dir: Path,
    two_hands: bool = True,
    sequence_length: int = 1,
    use_body_pose: bool = False,
) -> tuple[int, int]:
    """Extract each image once and save features for fast repeated training."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    detected = 0
    missing = 0
    hand_output_dim = 126 if two_hands else 63
    feature_dim = hand_output_dim + (99 if use_body_pose else 0)

    pose_context = (
        mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if use_body_pose
        else nullcontext(None)
    )

    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2 if two_hands else 1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands, pose_context as pose:
        for image_path, _ in samples:
            class_cache_dir = cache_dir / image_path.parent.name.lower()
            class_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = class_cache_dir / f"{image_path.stem}.npz"
            if cache_path.is_file():
                cached = np.load(cache_path)
                if cached["features"].shape == (sequence_length, feature_dim):
                    if np.any(cached["features"]):
                        detected += 1
                    else:
                        missing += 1
                    continue
            features = extract_hand_landmarks_from_image(
                image_path,
                two_hands=two_hands,
                sequence_length=sequence_length,
                hands=hands,
                use_body_pose=use_body_pose,
                pose=pose,
            )
            np.savez_compressed(cache_path, features=features)
            if np.any(features):
                detected += 1
            else:
                missing += 1
    return detected, missing


class SignLanguageDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[Path, str]],
        class_to_idx: dict[str, int],
        two_hands: bool = True,
        sequence_length: int = 1,
        cache_dir: Path | None = None,
        use_body_pose: bool = False,
    ) -> None:
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.two_hands = two_hands
        self.sequence_length = sequence_length
        self.use_body_pose = use_body_pose
        self.feature_dim = (126 if two_hands else 63) + (99 if use_body_pose else 0)
        self.cache_dir = cache_dir

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, class_name = self.samples[index]
        if self.cache_dir is not None:
            cache_path = self.cache_dir / image_path.parent.name.lower() / f"{image_path.stem}.npz"
            features = np.load(cache_path)["features"]
        else:
            features = extract_hand_landmarks_from_image(
                image_path,
                two_hands=self.two_hands,
                sequence_length=self.sequence_length,
                use_body_pose=self.use_body_pose,
            )

        if features.shape[0] < self.sequence_length:
            pad = np.zeros((self.sequence_length - features.shape[0], self.feature_dim), dtype=np.float32)
            features = np.concatenate([features, pad], axis=0)
        elif features.shape[0] > self.sequence_length:
            features = features[: self.sequence_length]

        label = self.class_to_idx[class_name]
        return torch.from_numpy(features).float(), torch.tensor(label, dtype=torch.long)


def load_image_samples(data_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for class_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        class_name = class_dir.name.lower()
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                samples.append((image_path, class_name))
    return samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a hand-sign word recognition model from image folders.")
    parser.add_argument("--data-dir", type=Path, default=Path("SuXin_TEMP/data"), help="Root folder with train/valid/test splits.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sequence-length", type=int, default=1, help="Number of frames in the landmark sequence. For single-image data, this is repeated.")
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Use cuda for GPU training when available.",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--two-hands", action="store_true", default=True, help="Use both hands with flattened 126-dimensional input.")
    parser.add_argument("--one-hand", dest="two_hands", action="store_false", help="Use one hand with 63-dimensional input.")
    parser.add_argument(
        "--use-body-pose",
        action="store_true",
        default=False,
        help="Append MediaPipe body pose landmarks to the hand landmark vector.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_dir = args.data_dir / "train"
    valid_dir = args.data_dir / "valid"
    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset splits in {args.data_dir}, but train/valid folders were not found."
        )

    train_samples = load_image_samples(train_dir)
    valid_samples = load_image_samples(valid_dir)
    if not train_samples or not valid_samples:
        raise ValueError("No valid image samples were found in train/valid folders.")

    labels = sorted({label for _, label in train_samples + valid_samples})
    class_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_dataset = SignLanguageDataset(
        train_samples,
        class_to_idx,
        two_hands=args.two_hands,
        sequence_length=args.sequence_length,
        use_body_pose=args.use_body_pose,
    )
    valid_dataset = SignLanguageDataset(
        valid_samples,
        class_to_idx,
        two_hands=args.two_hands,
        sequence_length=args.sequence_length,
        use_body_pose=args.use_body_pose,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)

    input_size = (126 if args.two_hands else 63) + (99 if args.use_body_pose else 0)
    model = HandSignLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(labels),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "sx_hand_sign_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for sequences, labels_batch in train_loader:
            sequences = sequences.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels_batch.size(0)
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels_batch).sum().item()
            total += labels_batch.size(0)

        train_acc = correct / max(total, 1)
        train_loss = train_loss / max(total, 1)

        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for sequences, labels_batch in valid_loader:
                sequences = sequences.to(device)
                labels_batch = labels_batch.to(device)
                logits = model(sequences)
                loss = criterion(logits, labels_batch)
                valid_loss += loss.item() * labels_batch.size(0)
                valid_correct += (logits.argmax(dim=1) == labels_batch).sum().item()
                valid_total += labels_batch.size(0)

        valid_acc = valid_correct / max(valid_total, 1)
        valid_loss = valid_loss / max(valid_total, 1)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train loss: {train_loss:.4f} | train acc: {train_acc:.4f} | "
            f"valid loss: {valid_loss:.4f} | valid acc: {valid_acc:.4f}"
        )

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_to_idx,
                "idx_to_class": {v: k for k, v in class_to_idx.items()},
                "input_size": input_size,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "num_classes": len(labels),
                "sequence_length": args.sequence_length,
                "two_hands": args.two_hands,
                "best_val_acc": best_val_acc,
                "epoch": epoch,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved best checkpoint: {checkpoint_path}")

    print(f"Finished training. Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
