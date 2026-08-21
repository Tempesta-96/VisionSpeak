"""Train the Valente LipNet model on complete GRID sentences with CTC."""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Matches the existing notebook setup for the project's Windows environment.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aligned_dataframe import load_aligned_dataframe, load_word_frames
from models.valente.model import LipNet, ctc_greedy_decode, tokens_to_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train sentence-level LipNet with CTC loss."
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_char_vocab(dataframe: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    """Build a character vocabulary with index zero reserved for CTC blank."""
    characters = sorted(set(" ".join(dataframe["word"].astype(str))))
    char_to_index = {character: index + 1 for index, character in enumerate(characters)}
    index_to_char = {index: character for character, index in char_to_index.items()}
    return char_to_index, index_to_char


class SentenceDataset(Dataset):
    """One lazily loaded sample per video, labelled by its complete sentence."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        video_ids: np.ndarray,
        char_to_index: dict[str, int],
    ) -> None:
        self.char_to_index = char_to_index
        self.items: list[tuple[str, pd.DataFrame, str]] = []

        for video_id in video_ids:
            rows = dataframe[dataframe["video_id"] == video_id].sort_values("word_index")
            text = " ".join(rows["word"].astype(str))
            self.items.append((str(video_id), rows, text))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str, str]:
        video_id, rows, text = self.items[index]
        frames = np.concatenate(
            [load_word_frames(row) for _, row in rows.iterrows()],
            axis=0,
        )
        frames_tensor = torch.from_numpy(frames.astype(np.float32) / 255.0)
        frames_tensor = (frames_tensor - frames_tensor.mean()) / (
            frames_tensor.std() + 1e-6
        )
        frames_tensor = frames_tensor.unsqueeze(1)
        targets = torch.tensor(
            [self.char_to_index[character] for character in text],
            dtype=torch.long,
        )
        return frames_tensor, targets, text, video_id


def sentence_collate(
    batch: list[tuple[torch.Tensor, torch.Tensor, str, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], list[str]]:
    """Pad video frames and concatenate variable-length CTC targets."""
    frames, targets, texts, video_ids = zip(*batch)
    input_lengths = torch.tensor([item.size(0) for item in frames], dtype=torch.long)
    target_lengths = torch.tensor([item.size(0) for item in targets], dtype=torch.long)
    padded_frames = torch.zeros(
        len(frames),
        int(input_lengths.max()),
        *frames[0].shape[1:],
        dtype=frames[0].dtype,
    )
    for index, item in enumerate(frames):
        padded_frames[index, : item.size(0)] = item
    return (
        padded_frames,
        torch.cat(targets),
        input_lengths,
        target_lengths,
        list(texts),
        list(video_ids),
    )


def create_dataloaders(
    dataframe: pd.DataFrame,
    char_to_index: dict[str, int],
    batch_size: int,
    test_size: float,
    seed: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("--test-size must be between zero and one.")

    video_ids = np.array(sorted(dataframe["video_id"].dropna().unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(video_ids)
    split_index = int(len(video_ids) * (1.0 - test_size))
    if split_index == 0 or split_index == len(video_ids):
        raise ValueError("The requested split leaves no train or test sentences.")

    train_dataset = SentenceDataset(dataframe, video_ids[:split_index], char_to_index)
    test_dataset = SentenceDataset(dataframe, video_ids[split_index:], char_to_index)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=sentence_collate,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=sentence_collate,
        num_workers=num_workers,
    )
    return train_loader, test_loader


def character_error_rate(prediction: str, target: str) -> float:
    """Compute Levenshtein distance normalised by the target length."""
    previous = list(range(len(target) + 1))
    for row, predicted_character in enumerate(prediction, start=1):
        current = [row]
        for column, target_character in enumerate(target, start=1):
            current.append(
                min(
                    previous[column] + 1,
                    current[column - 1] + 1,
                    previous[column - 1] + (predicted_character != target_character),
                )
            )
        previous = current
    return previous[-1] / max(len(target), 1)


def decode_log_probs(
    log_probs: torch.Tensor,
    input_lengths: torch.Tensor,
    blank_index: int,
    index_to_char: dict[int, str],
) -> list[str]:
    tokens = ctc_greedy_decode(log_probs.detach().cpu(), input_lengths.cpu(), blank_index)
    return [tokens_to_text(sequence, index_to_char) for sequence in tokens]


def run_epoch(
    model: LipNet,
    loader: DataLoader,
    ctc_loss: nn.CTCLoss,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    blank_index: int,
    index_to_char: dict[int, str],
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = total_cer = 0.0
    total_exact = total_sentences = 0

    for frames, targets, input_lengths, target_lengths, texts, _ in loader:
        frames = frames.to(device)
        targets = targets.to(device)
        with torch.set_grad_enabled(training):
            log_probs, output_lengths = model.ctc_log_probs(frames, input_lengths)
            loss = ctc_loss(log_probs, targets, output_lengths.cpu(), target_lengths.cpu())
            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        predictions = decode_log_probs(
            log_probs,
            output_lengths,
            blank_index,
            index_to_char,
        )
        total_loss += float(loss.item()) * len(texts)
        total_cer += sum(
            character_error_rate(prediction, target)
            for prediction, target in zip(predictions, texts)
        )
        total_exact += sum(
            prediction == target for prediction, target in zip(predictions, texts)
        )
        total_sentences += len(texts)

    return (
        total_loss / max(total_sentences, 1),
        total_cer / max(total_sentences, 1),
        total_exact / max(total_sentences, 1),
    )


@torch.no_grad()
def preview_predictions(
    model: LipNet,
    loader: DataLoader,
    device: torch.device,
    blank_index: int,
    index_to_char: dict[int, str],
    limit: int = 3,
) -> None:
    model.eval()
    frames, _, input_lengths, _, texts, _ = next(iter(loader))
    log_probs, output_lengths = model.ctc_log_probs(frames.to(device), input_lengths)
    predictions = decode_log_probs(log_probs, output_lengths, blank_index, index_to_char)
    for target, prediction in zip(texts[:limit], predictions[:limit]):
        print(f"Target:     {target}")
        print(f"Prediction: {prediction}")
        print("~" * 80)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    dataframe = load_aligned_dataframe()
    char_to_index, index_to_char = build_char_vocab(dataframe)
    blank_index = 0
    train_loader, test_loader = create_dataloaders(
        dataframe,
        char_to_index,
        args.batch_size,
        args.test_size,
        args.seed,
        args.num_workers,
    )

    print(f"Device: {device}")
    print(f"Train sentences: {len(train_loader.dataset)}")
    print(f"Test sentences: {len(test_loader.dataset)}")
    print(f"Character classes: {len(char_to_index)} + CTC blank")

    model = LipNet(num_classes=len(char_to_index) + 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 85],
        gamma=0.1,
    )
    ctc_loss = nn.CTCLoss(blank=blank_index, zero_infinity=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_dir / "valente_lipnet_ctc.pt"
    best_cer = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_cer, train_exact = run_epoch(
            model,
            train_loader,
            ctc_loss,
            optimizer,
            device,
            blank_index,
            index_to_char,
        )
        test_loss, test_cer, test_exact = run_epoch(
            model,
            test_loader,
            ctc_loss,
            None,
            device,
            blank_index,
            index_to_char,
        )
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} CER {train_cer:.3f} exact {train_exact:.3f} | "
            f"test loss {test_loss:.4f} CER {test_cer:.3f} exact {test_exact:.3f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
        )
        preview_predictions(model, test_loader, device, blank_index, index_to_char)

        if test_cer < best_cer:
            best_cer = test_cer
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "char_to_index": char_to_index,
                    "blank_index": blank_index,
                    "test_cer": test_cer,
                },
                checkpoint_path,
            )

    print(f"Best test CER: {best_cer:.4f}")
    print(f"Best checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
