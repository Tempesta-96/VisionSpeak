"""Run CTC sentence inference with a trained Valente LipNet checkpoint."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# Matches the existing notebook setup for the project's Windows environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from aligned_dataframe import load_aligned_dataframe, load_word_frames
from models.valente.model import LipNet, ctc_greedy_decode, tokens_to_text
from processing.preprocessor import Preprocessor
from processing.video_loader import VideoLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict text from a mouth video.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/valente_lipnet_ctc.pt"),
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--video-path", type=Path, default=None)
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Predict a processed aligned sentence identified by video_id.",
    )
    return parser.parse_args()


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[LipNet, dict[int, str], int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run models.valente.train first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    char_to_index = checkpoint["char_to_index"]
    index_to_char = {index: character for character, index in char_to_index.items()}
    blank_index = int(checkpoint["blank_index"])

    model = LipNet(num_classes=len(char_to_index) + 1).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, index_to_char, blank_index


def prepare_frames(frames: np.ndarray) -> torch.Tensor:
    """Normalise grayscale frames in the same way as sentence training."""
    tensor = torch.from_numpy(frames.astype(np.float32) / 255.0)
    tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
    return tensor.unsqueeze(0).unsqueeze(2)


@torch.no_grad()
def predict_frames(
    frames: np.ndarray,
    model: LipNet,
    index_to_char: dict[int, str],
    blank_index: int,
    device: torch.device,
) -> str:
    tensor = prepare_frames(frames).to(device)
    input_lengths = torch.tensor([tensor.size(1)], dtype=torch.long)
    log_probs, output_lengths = model.ctc_log_probs(tensor, input_lengths)
    token_ids = ctc_greedy_decode(
        log_probs.cpu(),
        output_lengths.cpu(),
        blank_index,
    )[0]
    return tokens_to_text(token_ids, index_to_char)


def predict_from_video_file(
    video_path: Path,
    model: LipNet,
    index_to_char: dict[int, str],
    blank_index: int,
    device: torch.device,
) -> str:
    loader = VideoLoader()
    preprocessor = Preprocessor()
    video_data = preprocessor.convert_to_greyscale(loader.load(str(video_path)))
    frames = np.asarray(video_data["frames"], dtype=np.uint8)
    return predict_frames(frames, model, index_to_char, blank_index, device)


def predict_from_aligned_video(
    video_id: str,
    model: LipNet,
    index_to_char: dict[int, str],
    blank_index: int,
    device: torch.device,
) -> str:
    """Predict an indexed sentence using the same joined word clips as training."""
    dataframe = load_aligned_dataframe()
    rows = dataframe[dataframe["video_id"] == video_id].sort_values("word_index")
    if rows.empty:
        raise ValueError(f"No aligned sentence found for video_id '{video_id}'.")
    frames = np.concatenate(
        [load_word_frames(row) for _, row in rows.iterrows()],
        axis=0,
    )
    return predict_frames(frames, model, index_to_char, blank_index, device)


def main() -> None:
    args = parse_args()
    if (args.video_path is None) == (args.video_id is None):
        raise SystemExit("Provide exactly one of --video-path or --video-id.")

    device = get_device(args.device)
    model, index_to_char, blank_index = load_model(args.checkpoint, device)
    if args.video_path is not None:
        text = predict_from_video_file(
            args.video_path,
            model,
            index_to_char,
            blank_index,
            device,
        )
    else:
        text = predict_from_aligned_video(
            args.video_id,
            model,
            index_to_char,
            blank_index,
            device,
        )
    print(f"Predicted text: {text}")


if __name__ == "__main__":
    main()
