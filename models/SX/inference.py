"""Load SX checkpoints without guessing an incompatible input layout."""
from dataclasses import dataclass
from pathlib import Path

import torch

from models.SX.model import HandSignLSTM


@dataclass(frozen=True)
class SXConfig:
    input_size: int
    two_hands: bool
    use_body_pose: bool
    sequence_length: int


def choose_device(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_sx_checkpoint(path: str | Path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    size = int(checkpoint.get("input_size", 126))
    two = checkpoint.get("two_hands", True)
    body = checkpoint.get("use_body_pose", size == (126 if two else 63) + 99)
    length = checkpoint.get("sequence_length", 1)
    if size != (126 if two else 63) + (99 if body else 0):
        raise ValueError("SX checkpoint input_size conflicts with hand/body metadata.")
    if length != 1:
        raise ValueError("SX inference currently supports sequence_length=1 checkpoints only.")
    labels = checkpoint["class_to_idx"]
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("SX class indices must be contiguous from zero.")
    if checkpoint.get("num_classes", len(labels)) != len(labels):
        raise ValueError("SX class count conflicts with class_to_idx.")
    model = HandSignLSTM(input_size=size,
        hidden_size=checkpoint.get("hidden_size", 64),
        num_layers=checkpoint.get("num_layers", 2), num_classes=len(labels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, labels, SXConfig(size, two, body, length)
