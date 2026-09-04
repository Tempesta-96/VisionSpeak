"""Predict one GRID vocabulary word from a single-word video clip.

The input video must contain one spoken word.  Its frames are cropped to the
mouth region with the same MediaPipe-based loader used by the data pipeline,
converted to grayscale, and passed to the isolated-word classifier.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from models.LBY.model import LipReadingClassifier
from processing.preprocessor import Preprocessor
from processing.video_loader import VideoLoader


ARCHIVE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT_PATH = ARCHIVE_ROOT / "checkpoints/best.pt"
FRAME_HEIGHT = 50
FRAME_WIDTH = 100


def get_device(preferred: str | None = None) -> torch.device:
    """Return an explicit device or the best available training-compatible one."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: torch.device | None = None,
) -> tuple[LipReadingClassifier, dict[int, str]]:
    """Restore the isolated-word model and its reverse vocabulary mapping."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Word-classifier checkpoint not found: {checkpoint_path}"
        )

    device = device or get_device()
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )
    except (OSError, EOFError, pickle.UnpicklingError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"Could not load word-classifier checkpoint: {checkpoint_path}"
        ) from error

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary of model data.")
    missing_keys = {"model_state_dict", "word_to_index"}.difference(checkpoint)
    if missing_keys:
        raise ValueError(
            "Checkpoint is missing required key(s): "
            f"{', '.join(sorted(missing_keys))}"
        )

    word_to_index = checkpoint["word_to_index"]
    if not isinstance(word_to_index, dict) or not word_to_index:
        raise ValueError("Checkpoint word_to_index must be a non-empty dictionary.")

    try:
        index_to_word = {
            int(index): str(word) for word, index in word_to_index.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError("Checkpoint word_to_index contains invalid class indices.") from error
    if len(index_to_word) != len(word_to_index):
        raise ValueError("Checkpoint word_to_index contains duplicate class indices.")

    model = LipReadingClassifier(num_classes=len(word_to_index)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError(
            "Checkpoint model_state_dict is incompatible with "
            "LipReadingClassifier."
        ) from error
    model.eval()
    return model, index_to_word


def prepare_frames(frames: np.ndarray) -> torch.Tensor:
    """Convert grayscale ``(T, 50, 100)`` uint8 frames to model input."""
    frame_array = np.asarray(frames)
    if frame_array.ndim != 3:
        raise ValueError(
            "Expected grayscale frames with shape (time, height, width); "
            f"received {frame_array.shape}."
        )
    if frame_array.shape[0] == 0:
        raise ValueError("Video contains no decoded frames.")
    if frame_array.shape[1:] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            "Expected mouth crops with shape (time, 50, 100); "
            f"received {frame_array.shape}."
        )

    tensor = torch.from_numpy(frame_array.astype(np.float32, copy=False)) / 255.0
    return tensor.unsqueeze(0).unsqueeze(2)


def predict_frames(
    frames: np.ndarray,
    model: LipReadingClassifier,
    index_to_word: dict[int, str],
    device: torch.device,
) -> str:
    """Return the predicted word for already-cropped grayscale mouth frames."""
    model_input = prepare_frames(frames).to(device)
    with torch.inference_mode():
        logits = model(model_input)
    predicted_index = int(logits.argmax(dim=1).item())
    try:
        return index_to_word[predicted_index]
    except KeyError as error:
        raise RuntimeError(
            f"Model predicted class index {predicted_index}, which is absent "
            "from the checkpoint vocabulary."
        ) from error


def load_video_frames(video_path: str | Path) -> np.ndarray:
    """Load a video and return its grayscale, resized mouth-crop frame stack."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    video_data = VideoLoader().load(str(video_path))
    if not video_data.get("loaded"):
        raise ValueError(f"Video contains no decoded frames: {video_path}")
    video_data = Preprocessor().convert_to_greyscale(video_data)
    return np.asarray(video_data["frames"])


def predict_word(
    video_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
    device: str | None = None,
) -> str:
    """Predict the one word spoken in ``video_path``.

    The video should contain only one spoken word, matching the isolated-word
    segments used when training ``LipReadingClassifier``.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    selected_device = get_device(device)
    model, index_to_word = load_model(checkpoint_path, selected_device)
    frames = load_video_frames(video_path)
    return predict_frames(frames, model, index_to_word, selected_device)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict one spoken GRID word from a video clip."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        required=True,
        help="Path to a video containing exactly one spoken word.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"Word-classifier checkpoint (default: {DEFAULT_CHECKPOINT_PATH}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="PyTorch device, such as cpu, mps, or cuda. Auto-detected if unset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        print(predict_word(args.video_path, args.checkpoint, args.device))
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(f"Prediction failed: {error}") from error


if __name__ == "__main__":
    main()
