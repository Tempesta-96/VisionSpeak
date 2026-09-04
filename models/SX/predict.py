from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from models.SX.model import HandSignLSTM


def extract_landmarks_from_image(
    image_path: str | Path,
    two_hands: bool = True,
    sequence_length: int = 1,
) -> np.ndarray:
    """Extract MediaPipe hand landmarks from an image as a sequence vector."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    try:
        results = hands.process(rgb)
    finally:
        hands.close()

    feature_dim = 126 if two_hands else 63
    sequence = np.zeros((sequence_length, feature_dim), dtype=np.float32)

    if not results.multi_hand_landmarks:
        return sequence

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
    sequence[0, : combined.shape[0]] = combined[:feature_dim]
    return sequence


def load_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple[HandSignLSTM, dict[str, int], int, bool]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = HandSignLSTM(
        input_size=checkpoint.get("input_size", 126),
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2),
        num_classes=checkpoint.get("num_classes", len(checkpoint["class_to_idx"])),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint["class_to_idx"], checkpoint.get("sequence_length", 1), checkpoint.get("two_hands", True)


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device_name: str | None = None,
) -> str:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, class_to_idx, sequence_length, two_hands = load_checkpoint(Path(checkpoint_path), device)

    features = extract_landmarks_from_image(image_path, two_hands=two_hands, sequence_length=sequence_length)
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    logits = model(tensor)
    prediction_index = int(torch.argmax(logits, dim=1).item())
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    return idx_to_class[prediction_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a hand-sign word from an image using a trained MediaPipe LSTM model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to checkpoint .pt file.")
    parser.add_argument("--image", type=Path, required=True, help="Path to the hand-sign image.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Use cuda for GPU inference when available.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    word = predict_image(args.image, args.checkpoint, args.device)
    print(f"Predicted sign: {word}")
