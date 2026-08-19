"""Run a trained PyTorch LipNet checkpoint on a single video."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.valente.model import LipNetCTC, ctc_greedy_decode, tokens_to_text
from processing.preprocessor import Preprocessor
from processing.video_loader import VideoLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict text from one video file.")
    parser.add_argument("video_path", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/best_lipnet_ctc.pt"),
    )
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def prepare_frames(video_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    loader = VideoLoader()
    preprocessor = Preprocessor()

    video_data = loader.load(str(video_path))
    video_data = preprocessor.convert_to_greyscale(video_data)
    video_data = preprocessor.pad_frames(video_data)
    frames = video_data["frames"]

    tensor = torch.from_numpy(frames).float() / 255.0
    tensor = tensor.unsqueeze(1).unsqueeze(0)
    input_lengths = torch.tensor([tensor.shape[1]], dtype=torch.long)
    return tensor, input_lengths


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device)

    char_to_num = checkpoint["char_to_num"]
    num_to_char = checkpoint["num_to_char"]
    model = LipNetCTC(vocab_size=len(char_to_num)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    frames, input_lengths = prepare_frames(args.video_path)
    frames = frames.to(device)
    input_lengths = input_lengths.to(device)

    with torch.no_grad():
        log_probs = model(frames)
        decoded = ctc_greedy_decode(log_probs, input_lengths, model.blank_index)[0]

    print(tokens_to_text(decoded, num_to_char))


if __name__ == "__main__":
    main()
