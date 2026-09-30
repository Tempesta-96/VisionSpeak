from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch

from models.SX.inference import load_sx_checkpoint


def extract_landmarks_from_image(
    image_path: str | Path,
    two_hands: bool = True,
    sequence_length: int = 1,
    use_body_pose: bool = False,
) -> np.ndarray:
    """Extract MediaPipe hand and optional body landmarks as a sequence vector.

    ``sequence_length`` repeats the single observed pose so that windowed
    models receive the shape they expect. With ``sequence_length > 1`` every
    frame is identical, so this is only a shape adapter for still images; use a
    recorded clip to exercise the temporal window meaningfully.
    """
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
    pose = (
        mp.solutions.pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if use_body_pose
        else None
    )
    try:
        results = hands.process(rgb)
        pose_results = pose.process(rgb) if pose is not None else None
    finally:
        hands.close()
        if pose is not None:
            pose.close()

    hand_feature_dim = 126 if two_hands else 63
    body_feature_dim = 99 if use_body_pose else 0
    feature_dim = hand_feature_dim + body_feature_dim
    sequence = np.zeros((sequence_length, feature_dim), dtype=np.float32)

    hand_features = np.zeros(hand_feature_dim, dtype=np.float32)
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
            combined_hands = np.concatenate(hand_vectors[:2], axis=0)
        else:
            combined_hands = hand_vectors[0] if hand_vectors else np.zeros(63, dtype=np.float32)
        hand_features[:hand_feature_dim] = combined_hands[:hand_feature_dim]

    if use_body_pose:
        body_features = np.zeros(body_feature_dim, dtype=np.float32)
        if pose_results is not None and pose_results.pose_landmarks:
            coords = []
            for landmark in pose_results.pose_landmarks.landmark:
                coords.extend([landmark.x, landmark.y, landmark.z])
            body_values = np.asarray(coords[:body_feature_dim], dtype=np.float32)
            body_features[:body_values.shape[0]] = body_values
        features = np.concatenate([hand_features, body_features])
    else:
        features = hand_features

    # A still image has no temporal extent, so one pose fills the whole window.
    sequence[:] = features[:feature_dim]
    return sequence


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, int], int, bool, bool]:
    """Load the model a checkpoint declares, plus its input layout.

    Returns ``(model, class_to_idx, sequence_length, two_hands, use_body_pose)``.
    Delegates to :func:`models.SX.inference.load_sx_checkpoint` so this script
    and the live worker agree on what a checkpoint means.
    """
    model, labels, config = load_sx_checkpoint(checkpoint_path, device)
    return (
        model,
        labels,
        config.window_size,
        config.two_hands,
        config.use_body_pose,
    )


@torch.no_grad()
def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path,
    device_name: str | None = None,
) -> str:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, class_to_idx, sequence_length, two_hands, use_body_pose = load_checkpoint(
        Path(checkpoint_path), device
    )

    features = extract_landmarks_from_image(
        image_path,
        two_hands=two_hands,
        sequence_length=sequence_length,
        use_body_pose=use_body_pose,
    )
    tensor = torch.from_numpy(features).unsqueeze(0).to(device)
    logits = model(tensor)
    prediction_index = int(torch.argmax(logits, dim=1).item())
    idx_to_class = {value: key for key, value in class_to_idx.items()}
    return idx_to_class[prediction_index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a hand-sign word from an image using a trained MediaPipe landmark model.")
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
