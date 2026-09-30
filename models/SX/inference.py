"""Load SX checkpoints without guessing an incompatible input layout.

A checkpoint declares its own architecture, window size, and landmark layout.
Legacy checkpoints written before the transformer experiment have no
``architecture`` key and load as the original single-frame LSTM.
"""
from dataclasses import dataclass
from pathlib import Path

import torch

from models.SX.transformer import build_gesture_model


@dataclass(frozen=True)
class SXConfig:
    """Everything the live worker needs to build model inputs.

    Attributes:
        input_size: Landmark features per frame.
        two_hands: Whether both hands contribute landmarks.
        use_body_pose: Whether body-pose landmarks are appended.
        window_size: Frames per prediction. ``1`` is the original
            single-frame behaviour; anything larger needs a frame buffer.
        architecture: ``"lstm"`` or ``"transformer"``.
    """

    input_size: int
    two_hands: bool
    use_body_pose: bool
    window_size: int
    architecture: str = "lstm"

    @property
    def sequence_length(self) -> int:
        """Deprecated alias for :attr:`window_size`, kept for older callers."""
        return self.window_size


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
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("SX checkpoint must be a mapping containing model_state_dict.")
    if "class_to_idx" not in checkpoint:
        raise ValueError("SX checkpoint must contain class_to_idx.")

    size = int(checkpoint.get("input_size", 126))
    two = checkpoint.get("two_hands", True)
    body = checkpoint.get("use_body_pose", size == (126 if two else 63) + 99)
    if size != (126 if two else 63) + (99 if body else 0):
        raise ValueError("SX checkpoint input_size conflicts with hand/body metadata.")

    architecture = str(checkpoint.get("architecture", "lstm")).lower()
    if architecture not in ("lstm", "transformer"):
        raise ValueError(
            f"SX checkpoint architecture must be 'lstm' or 'transformer', got {architecture!r}."
        )

    # Modern checkpoints write window_size and keep sequence_length in sync;
    # legacy ones only have sequence_length (usually absent, meaning 1).
    window_size = int(
        checkpoint.get("window_size", checkpoint.get("sequence_length", 1))
    )
    if window_size < 1:
        raise ValueError("SX checkpoint window_size must be at least 1.")

    declared_sequence_length = checkpoint.get("sequence_length")
    if declared_sequence_length is not None and int(declared_sequence_length) != window_size:
        raise ValueError(
            "SX checkpoint window_size and sequence_length disagree "
            f"({window_size} vs {declared_sequence_length})."
        )

    if architecture == "lstm" and window_size != 1:
        raise ValueError(
            "SX inference supports the LSTM checkpoint only with window_size=1."
        )

    labels = checkpoint["class_to_idx"]
    if sorted(labels.values()) != list(range(len(labels))):
        raise ValueError("SX class indices must be contiguous from zero.")
    if checkpoint.get("num_classes", len(labels)) != len(labels):
        raise ValueError("SX class count conflicts with class_to_idx.")

    settings = {
        "input_size": size,
        "architecture": architecture,
        "hidden_size": checkpoint.get("hidden_size", 64),
        "num_layers": checkpoint.get("num_layers", 2),
        "dropout": checkpoint.get("dropout", 0.2),
        "d_model": checkpoint.get("d_model", 128),
        "num_heads": checkpoint.get("num_heads", 4),
        "feedforward_dim": checkpoint.get("feedforward_dim", 256),
        "pooling": checkpoint.get("pooling", "cls"),
        # +1 for the prepended [CLS] token.
        "max_tokens": checkpoint.get("max_tokens", window_size + 1),
    }
    model = build_gesture_model(settings, num_classes=len(labels)).to(device)
    if settings["max_tokens"] < window_size + 1:
        raise ValueError(
            f"SX checkpoint max_tokens {settings['max_tokens']} cannot hold a "
            f"window of {window_size} frames plus the CLS token."
        )
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as error:
        raise ValueError(
            f"{Path(path).name} does not match its recorded architecture "
            f"{architecture!r}: {error}"
        ) from error
    model.eval()
    return model, labels, SXConfig(size, two, body, window_size, architecture)
