"""Train a LipNet-style CTC model with the existing PyTorch data pipeline."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import get_dataloaders
from models.valente.model import LipNetCTC, ctc_greedy_decode, tokens_to_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the PyTorch LipNet model.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
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


def scheduler(epoch: int, current_lr: float) -> float:
    if epoch < 30:
        return current_lr
    return current_lr * float(np.exp(-0.1))


def flatten_targets(labels: torch.Tensor, target_lengths: torch.Tensor) -> torch.Tensor:
    chunks = [
        labels[index, : int(target_lengths[index].item())]
        for index in range(labels.size(0))
    ]
    return torch.cat(chunks, dim=0)


def move_batch_to_device(batch, device: torch.device):
    frames, labels, input_lengths, target_lengths = batch
    return (
        frames.to(device),
        labels.to(device),
        input_lengths.to(device),
        target_lengths.to(device),
    )


def run_epoch(
    model: LipNetCTC,
    loader,
    criterion,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    total_batches = 0

    for batch in loader:
        frames, labels, input_lengths, target_lengths = move_batch_to_device(
            batch, device
        )
        flattened_targets = flatten_targets(labels, target_lengths)

        if is_training:
            optimizer.zero_grad()

        log_probs = model(frames)
        loss = criterion(log_probs, flattened_targets, input_lengths, target_lengths)

        if is_training:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_batches += 1

    return total_loss / max(total_batches, 1)


@torch.no_grad()
def preview_predictions(
    model: LipNetCTC,
    loader,
    device: torch.device,
    num_to_char: dict[int, str],
    limit: int = 2,
) -> None:
    model.eval()
    frames, labels, input_lengths, target_lengths = move_batch_to_device(
        next(iter(loader)), device
    )
    log_probs = model(frames)
    decoded = ctc_greedy_decode(log_probs, input_lengths, model.blank_index)

    for index in range(min(limit, frames.size(0))):
        target_tokens = labels[index, : int(target_lengths[index].item())].tolist()
        print("Original:", tokens_to_text(target_tokens, num_to_char))
        print("Prediction:", tokens_to_text(decoded[index], num_to_char))
        print("~" * 80)


def save_checkpoint(
    out_dir: Path,
    epoch: int,
    model: LipNetCTC,
    optimizer: torch.optim.Optimizer,
    char_to_num: dict[str, int],
    num_to_char: dict[int, str],
    val_loss: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "char_to_num": char_to_num,
        "num_to_char": num_to_char,
        "val_loss": val_loss,
    }
    torch.save(checkpoint, out_dir / "best_lipnet_ctc.pt")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_loader, test_loader, char_to_num, num_to_char = get_dataloaders(
        test_size=args.test_size,
        seed=args.seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        stratify=False,
    )

    model = LipNetCTC(vocab_size=len(char_to_num)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CTCLoss(blank=model.blank_index, zero_infinity=True)

    best_val_loss = float("inf")
    current_lr = args.lr

    for epoch in range(args.epochs):
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr

        train_loss = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss = run_epoch(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | lr {current_lr:.6f}"
        )

        preview_predictions(model, test_loader, device, num_to_char)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                args.out_dir,
                epoch + 1,
                model,
                optimizer,
                char_to_num,
                num_to_char,
                val_loss,
            )

        current_lr = scheduler(epoch + 1, current_lr)

    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
