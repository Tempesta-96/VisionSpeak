"""Train the LipReadingClassifier on the processed GRID word data."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataset import get_word_classification_dataloaders
from model import LipReadingClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a word-level lip-reading classifier."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu, cuda or mps (auto-detected if unset).",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for frames, labels in loader:
        frames = frames.to(device)
        labels = labels.to(device)

        logits = model(frames)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    train_loader, test_loader, word_to_index, index_to_word = (
        get_word_classification_dataloaders(
            test_size=0.2,
            seed=args.seed,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            stratify=True,
        )
    )
    num_classes = len(word_to_index)
    print(
        f"Train samples: {len(train_loader.dataset)}, "
        f"Test samples: {len(test_loader.dataset)}, "
        f"Classes: {num_classes}"
    )

    model = LipReadingClassifier(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    best_accuracy = 0.0
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"test_loss {test_loss:.4f} test_acc {test_acc:.4f}"
        )

        if test_acc > best_accuracy:
            best_accuracy = test_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "word_to_index": word_to_index,
                "test_accuracy": test_acc,
            }
            torch.save(checkpoint, args.out_dir / "best.pt")

    print(f"Best test accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
