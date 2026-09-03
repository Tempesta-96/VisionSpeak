"""Train/test split and PyTorch data loaders for the processed GRID data.

Builds on ``aligned_dataframe.py`` (one row per spoken word in
``data/lips-dataset/processedData/aligned_words.csv``) and lazily loads per-word frame
segments from the ``.npz`` archives.

Requires ``torch`` (installed separately, see ``lipnet.ipynb``).
"""

import math

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from archive.kaggle_lip_reading.aligned_dataframe import (
    load_aligned_dataframe,
    load_word_frames,
)


def split_aligned_dataframe(
    df: pd.DataFrame | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    stratify: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the word-level manifest into train and test DataFrames.

    The split is done per word row (not per video). Note: words that come
    from the same sentence/video may therefore appear in both sets. Pass
    ``stratify=True`` to preserve the class distribution of ``word``.

    Deterministic and free of extra dependencies (uses ``numpy`` only).
    """
    if df is None:
        df = load_aligned_dataframe()

    if stratify:
        return _stratified_split(df, test_size, seed)

    n_test = math.ceil(len(df) * test_size)
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df = shuffled.iloc[:n_test].reset_index(drop=True)
    train_df = shuffled.iloc[n_test:].reset_index(drop=True)
    return train_df, test_df


def _stratified_split(
    df: pd.DataFrame, test_size: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-class random split on ``word`` without scikit-learn."""
    rng = np.random.default_rng(seed)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for _, group in df.groupby("word", sort=False):
        indices = np.asarray(group.index.to_numpy()).copy()
        rng.shuffle(indices)
        n_test = max(1, math.ceil(len(indices) * test_size))
        test_indices.append(indices[:n_test])
        train_indices.append(indices[n_test:])

    train_index = np.concatenate(train_indices) if train_indices else np.array([], dtype=int)
    test_index = np.concatenate(test_indices) if test_indices else np.array([], dtype=int)
    rng.shuffle(train_index)
    rng.shuffle(test_index)

    train_df = df.loc[train_index].reset_index(drop=True)
    test_df = df.loc[test_index].reset_index(drop=True)
    return train_df, test_df


def build_vocab(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[int, str]]:
    """Derive a character-level vocabulary from the word labels."""
    chars = sorted({char for word in df["word"].astype(str) for char in word})
    char_to_num = {char: index for index, char in enumerate(chars)}
    num_to_char = {index: char for char, index in char_to_num.items()}
    return char_to_num, num_to_char


def build_word_vocab(
    df: pd.DataFrame,
) -> tuple[dict[str, int], dict[int, str]]:
    """Derive a whole-word vocabulary from the word labels."""
    words = sorted(df["word"].astype(str).unique())
    word_to_index = {word: index for index, word in enumerate(words)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    return word_to_index, index_to_word


class LipReadingDataset(Dataset):
    """Lazily loads one (mouth frames, label) pair per word row.

    Frames are stored as single-channel grayscale ``(T, H, W)`` uint8 and are
    returned as ``(T, 1, H, W)`` float32 in ``[0, 1]`` to match a LipNet-style
    3D-CNN input. ``transform``, if given, is applied after normalisation.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        char_to_num: dict[str, int] | None = None,
        transform=None,
        word_to_index: dict[str, int] | None = None,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True)
        self.char_to_num = char_to_num
        self.word_to_index = word_to_index
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataframe.iloc[index]
        frames = load_word_frames(row)  # (T, H, W) uint8
        tensor = torch.from_numpy(np.asarray(frames)).float() / 255.0
        tensor = tensor.unsqueeze(1)  # (T, 1, H, W)

        if self.transform is not None:
            tensor = self.transform(tensor)

        if self.word_to_index is not None:
            label = torch.tensor(
                self.word_to_index[str(row["word"])], dtype=torch.long
            )
        else:
            label = torch.tensor(
                [self.char_to_num[char] for char in str(row["word"])],
                dtype=torch.long,
            )
        return tensor, label


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length word segments (CTC convention).

    Returns ``(frames, labels, input_lengths, target_lengths)`` where frames
    are sorted by descending length and padded to the batch maximum.
    """
    batch = sorted(batch, key=lambda item: item[0].shape[0], reverse=True)
    frames = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    input_lengths = torch.tensor([f.shape[0] for f in frames], dtype=torch.long)
    target_lengths = torch.tensor([l.shape[0] for l in labels], dtype=torch.long)

    max_time = int(input_lengths.max())
    max_label = int(target_lengths.max())

    padded_frames = torch.zeros(
        len(frames), max_time, *frames[0].shape[1:], dtype=frames[0].dtype
    )
    padded_labels = torch.zeros(len(labels), max_label, dtype=torch.long)

    for i, frame in enumerate(frames):
        padded_frames[i, : frame.shape[0]] = frame
    for i, label in enumerate(labels):
        padded_labels[i, : label.shape[0]] = label

    return padded_frames, padded_labels, input_lengths, target_lengths


def get_dataloaders(
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
    stratify: bool = False,
) -> tuple[DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    """Build train/test DataLoaders and return them with the vocab maps.

    Returns ``(train_loader, test_loader, char_to_num, num_to_char)``.
    """
    train_df, test_df = split_aligned_dataframe(
        test_size=test_size, seed=seed, stratify=stratify
    )
    char_to_num, num_to_char = build_vocab(pd.concat([train_df, test_df]))

    train_dataset = LipReadingDataset(train_df, char_to_num)
    test_dataset = LipReadingDataset(test_df, char_to_num)

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
    return train_loader, test_loader, char_to_num, num_to_char


def collate_fn_classification(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a batch of variable-length word segments for classification.

    Returns ``(frames, labels)`` where frames are sorted by descending length
    and padded to ``(B, T_max, 1, H, W)`` and labels are ``(B,)`` word indices.
    """
    batch = sorted(batch, key=lambda item: item[0].shape[0], reverse=True)
    frames = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)

    max_time = max(frame.shape[0] for frame in frames)
    padded_frames = torch.zeros(
        len(frames), max_time, *frames[0].shape[1:], dtype=frames[0].dtype
    )
    for i, frame in enumerate(frames):
        padded_frames[i, : frame.shape[0]] = frame

    return padded_frames, labels


def get_word_classification_dataloaders(
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 32,
    num_workers: int = 0,
    stratify: bool = True,
) -> tuple[DataLoader, DataLoader, dict[str, int], dict[int, str]]:
    """Build word-classification train/test DataLoaders.

    Returns ``(train_loader, test_loader, word_to_index, index_to_word)``.
    """
    train_df, test_df = split_aligned_dataframe(
        test_size=test_size, seed=seed, stratify=stratify
    )
    word_to_index, index_to_word = build_word_vocab(pd.concat([train_df, test_df]))

    train_dataset = LipReadingDataset(train_df, word_to_index=word_to_index)
    test_dataset = LipReadingDataset(test_df, word_to_index=word_to_index)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_classification,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_classification,
        num_workers=num_workers,
    )
    return train_loader, test_loader, word_to_index, index_to_word
