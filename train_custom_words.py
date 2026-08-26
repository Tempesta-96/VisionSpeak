"""Train LipReadingClassifier on the custom isolated-word dataset and evaluate."""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from custom_words_dataset import get_custom_words_dataloaders
from models.LBY.model import LipReadingClassifier

EPOCHS = 80
LR = 1e-3
BATCH_SIZE = 32
SEED = 42
CHECKPOINT = Path("checkpoints/custom_words.pt")

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader, label_to_index, index_to_label = (
    get_custom_words_dataloaders(
        test_size=0.2, seed=SEED, batch_size=BATCH_SIZE
    )
)
num_classes = len(label_to_index)

print(f"Device: {device}")
print(f"Train samples: {len(train_loader.dataset)}")
print(f"Test samples: {len(test_loader.dataset)}")
print(f"Classes: {num_classes} {label_to_index}")

model = LipReadingClassifier(num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[40, 60], gamma=0.1
)
criterion = nn.CrossEntropyLoss()

CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
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


for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = run_epoch(train_loader, training=True)
    test_loss, test_acc = run_epoch(test_loader, training=False)
    scheduler.step()
    print(
        f"Epoch {epoch:2d}/{EPOCHS} | "
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
            },
            CHECKPOINT,
        )

print(f"Best test accuracy: {best_accuracy:.4f}")

# ---- Final evaluation: per-word accuracy + confusion matrix ----
checkpoint = torch.load(CHECKPOINT, map_location=device)
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
