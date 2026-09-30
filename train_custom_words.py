"""Train LipReadingClassifier on the custom isolated-word dataset and evaluate.

The temporal core is selectable with ``--architecture``. Both options share the
same 3D-CNN frontend, so comparing them isolates attention versus recurrence:

* ``transformer`` (default): a pre-layer-norm transformer encoder with
  attention pooling.
* ``bigru``: the original single bidirectional GRU.

The model classes themselves default to ``bigru`` so older checkpoints keep
loading; this script defaults to training the transformer.

Every architecture setting is written into the checkpoint, so
``inference.lip_worker`` rebuilds the right model without being told which one
it is looking at.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from custom_words_dataset import get_custom_words_dataloaders
from models.LBY.model import build_lip_reading_classifier

EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 32
SEED = 42
CHECKPOINT = Path("checkpoints/custom_words.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--architecture",
        choices=("bigru", "transformer"),
        default="transformer",
        help="Temporal core to train (default: transformer).",
    )
    parser.add_argument(
        "--pooling",
        choices=("attention", "mean"),
        default="attention",
        help="Sequence reduction on the transformer path (default: attention).",
    )
    parser.add_argument("--layers", type=int, default=3, help="Transformer encoder layers.")
    parser.add_argument("--heads", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--feedforward", type=int, default=512, help="Encoder MLP width.")
    parser.add_argument("--hidden", type=int, default=256, help="GRU or transformer width.")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument(
        "--legacy-milestones",
        action="store_true",
        help="Use the original MultiStepLR schedule instead of the transformer-friendly cosine one.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, label_to_index, index_to_label = (
        get_custom_words_dataloaders(
            test_size=0.2, seed=args.seed, batch_size=args.batch_size
        )
    )
    num_classes = len(label_to_index)

    settings = {
        "architecture": args.architecture,
        "gru_hidden": args.hidden,
        "dropout": args.dropout,
        "pooling": args.pooling,
        "transformer_layers": args.layers,
        "transformer_heads": args.heads,
        "transformer_feedforward": args.feedforward,
    }

    print(f"Device: {device}")
    print(f"Architecture: {args.architecture}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Classes: {num_classes} {label_to_index}")

    model = build_lip_reading_classifier(settings, num_classes=num_classes).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.legacy_milestones:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 60], gamma=0.1
        )
    else:
        # Transformers are more sensitive to a late, large learning-rate drop
        # than the GRU was; anneal smoothly instead.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
    criterion = nn.CrossEntropyLoss()

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0

    def run_epoch(loader, training):
        model.train(training)
        total_loss = correct = total = 0
        for frames, labels in loader:
            frames, labels = frames.to(device), labels.to(device)
            with torch.set_grad_enabled(training):
                logits = model(frames)
                loss = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / max(total, 1), correct / max(total, 1)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, training=True)
        test_loss, test_acc = run_epoch(test_loader, training=False)
        scheduler.step()
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train_loss {train_loss:.4f} train_acc {train_acc:.4f} | "
            f"test_loss {test_loss:.4f} test_acc {test_acc:.4f} | "
            f"lr {optimizer.param_groups[0]['lr']:.1e}"
        )
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "label_to_index": label_to_index,
                    "test_accuracy": test_acc,
                    **settings,
                },
                args.checkpoint,
            )

    print(f"Best test accuracy: {best_accuracy:.4f}")

    # ---- Final evaluation: per-word accuracy + confusion matrix ----
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for frames, labels in test_loader:
            frames, labels = frames.to(device), labels.to(device)
            all_preds.append(model(frames).argmax(dim=1).cpu())
            all_targets.append(labels.cpu())
    preds = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    overall = float((preds == targets).mean())
    print(f"\nFinal overall test accuracy: {overall:.4f}")

    print("\nPer-word accuracy:")
    for word, idx in label_to_index.items():
        mask = targets == idx
        if mask.sum() == 0:
            continue
        acc = float((preds[mask] == idx).mean())
        print(f"  {word:>8}: {acc:.4f} ({int(mask.sum())} samples)")

    print("\nConfusion matrix (rows=truth, cols=prediction):")
    conf = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, preds):
        conf[t, p] += 1
    header = "        " + " ".join(f"{index_to_label[i][:4]:>4}" for i in range(num_classes))
    print(header)
    for i in range(num_classes):
        row = " ".join(f"{v:4d}" for v in conf[i])
        print(f"{index_to_label[i]:>8} {row}")

    print(
        "\nNote: this split is per-clip random, so all four speakers appear in "
        "both train and test. Treat the accuracy as a same-speaker upper bound."
    )


if __name__ == "__main__":
    main()
