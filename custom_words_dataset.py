"""Data loading for the custom isolated-word lip-reading dataset.

Loads ``data/custom_words/manifest.csv`` and lazily reads per-clip grayscale
mouth frames from the ``.npz`` archives produced by
``record_custom_word_dataset.ipynb``.

Each clip stores ``frames`` as ``(T, 50, 100)`` uint8. Samples are returned as
``(T, 1, 50, 100)`` float32 in ``[0, 1]`` to match a LipNet-style 3D-CNN input,
with a whole-word label index derived from the manifest's ``label`` column.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

DEFAULT_DATA_DIR = Path("data/custom_words")


def load_manifest(data_dir: str | Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Load the clip manifest, resolving each clip to an absolute ``clip_path``."""
    data_dir = Path(data_dir).expanduser().resolve()
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Custom-words manifest not found: {manifest_path}. "
            "Run record_custom_word_dataset.ipynb first."
        )

    dataframe = pd.read_csv(
        manifest_path,
        dtype={
            "sample_id": "string",
            "speaker_id": "string",
            "label": "string",
            "relative_path": "string",
        },
    )
    dataframe["clip_path"] = dataframe["relative_path"].map(
        lambda relative: data_dir / relative
    )
    return dataframe


def build_label_vocab(
    dataframe: pd.DataFrame,
) -> tuple[dict[str, int], dict[int, str]]:
    """Derive a stable whole-word vocabulary from the ``label`` column."""
    labels = sorted(dataframe["label"].astype(str).unique())
    label_to_index = {label: index for index, label in enumerate(labels)}
    index_to_label = {index: label for label, index in label_to_index.items()}
    return label_to_index, index_to_label


class CustomWordsDataset(Dataset):
    """Lazily loads one ``(mouth frames, word label)`` pair per clip row."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        label_to_index: dict[str, int],
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.label_to_index = label_to_index

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        with np.load(Path(row["clip_path"]), allow_pickle=False) as archive:
            frames = np.asarray(archive["frames"]).copy()
        tensor = torch.from_numpy(frames.astype(np.float32)) / 255.0
        # Per-clip standardisation: these mouth crops are low-contrast, and
        # centring/scaling is what lets the 3D-CNN learn a usable signal.
        tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
        tensor = tensor.unsqueeze(1)  # (T, 1, H, W)
        label = torch.tensor(
            self.label_to_index[str(row["label"])], dtype=torch.long
        )
        return tensor, label


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length clips to the batch maximum time."""
    batch = sorted(batch, key=lambda item: item[0].shape[0], reverse=True)
    frames = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    max_time = max(frame.shape[0] for frame in frames)
    padded_frames = torch.zeros(
        len(frames), max_time, *frames[0].shape[1:], dtype=frames[0].dtype
    )
    for index, frame in enumerate(frames):
        padded_frames[index, : frame.shape[0]] = frame

    return padded_frames, labels


def stratified_split(
    dataframe: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-label random split without scikit-learn."""
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for _, group in dataframe.groupby("label", sort=False):
        indices = np.asarray(group.index.to_numpy()).copy()
        rng.shuffle(indices)
        n_test = max(1, math.ceil(len(indices) * test_size))
        test_indices.append(indices[:n_test])
        train_indices.append(indices[n_test:])

    train_index = np.concatenate(train_indices) if train_indices else np.array([], dtype=int)
    test_index = np.concatenate(test_indices) if test_indices else np.array([], dtype=int)
    rng.shuffle(train_index)
    rng.shuffle(test_index)

    train_df = dataframe.loc[train_index].reset_index(drop=True)
    test_df = dataframe.loc[test_index].reset_index(drop=True)
    return train_df, test_df


def get_custom_words_dataloaders(
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 16,
    num_workers: int = 0,
    data_dir: str | Path = DEFAULT_DATA_DIR,
) -> tuple[DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    """Build train/test DataLoaders and return them with the vocab maps.

    Returns ``(train_loader, test_loader, label_to_index, index_to_label)``.
    """
    dataframe = load_manifest(data_dir)
    label_to_index, index_to_label = build_label_vocab(dataframe)

    train_df, test_df = stratified_split(dataframe, test_size, seed)
    train_dataset = CustomWordsDataset(train_df, label_to_index)
    test_dataset = CustomWordsDataset(test_df, label_to_index)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    return train_loader, test_loader, label_to_index, index_to_label
