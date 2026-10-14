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


def available_speakers(dataframe: pd.DataFrame) -> list[str]:
    """Return the speaker IDs present in the manifest, sorted."""
    return sorted(dataframe["speaker_id"].astype(str).unique())


def choose_holdout_speaker(dataframe: pd.DataFrame) -> str:
    """Pick a held-out speaker whose class coverage best matches the dataset.

    A speaker missing a word cannot be tested on it, so the most representative
    speaker is the one that best covers the full label set. Ties break
    alphabetically to keep the choice deterministic.
    """
    labels = set(dataframe["label"].astype(str))
    scored: list[tuple[float, str]] = []
    for speaker in available_speakers(dataframe):
        rows = dataframe[dataframe["speaker_id"].astype(str) == speaker]
        coverage = len(labels.intersection(set(rows["label"].astype(str)))) / max(len(labels), 1)
        scored.append((-coverage, speaker))
    scored.sort()
    return scored[0][1]


def speaker_disjoint_split(
    dataframe: pd.DataFrame,
    test_speaker: str | None = None,
    test_size: float = 0.2,
    seed: int = 42,
    return_validation: bool = False,
    carve_validation: bool = True,
):
    """Hold out one speaker entirely from the training set.

    ``stratified_split`` samples clips at random, so every speaker appears in
    both train and test and the resulting accuracy measures same-speaker
    recall. This split instead keeps the held-out speaker's clips only in the
    test set, which is what makes a score evidence of generalisation.

    Args:
        dataframe: Manifest rows.
        test_speaker: Speaker to hold out. Defaults to
            :func:`choose_holdout_speaker`.
        test_size: Fraction of the *remaining* speakers set aside as
            validation, so the test speaker is never used for model selection.
        seed: Seed for the validation split.
        return_validation: Return the validation frame as well.
        carve_validation: When ``False``, no validation frame is carved out and
            every non-test-speaker clip goes to training. Use this for a
            fixed-budget run whose deliverable is the final epoch; the returned
            validation frame is then empty. ``test_size`` is ignored.

    Returns:
        ``(train_df, test_df, test_speaker)``, or
        ``(train_df, validation_df, test_df, test_speaker)`` when
        ``return_validation`` is set.

    Raises:
        ValueError: If the requested speaker is absent, or if removing it would
            leave a class with no training examples.
    """
    dataframe = dataframe.reset_index(drop=True)
    speakers = available_speakers(dataframe)
    if len(speakers) < 2:
        raise ValueError(
            "A speaker-disjoint split needs at least two speakers; "
            f"the manifest has {len(speakers)}."
        )

    test_speaker = test_speaker or choose_holdout_speaker(dataframe)
    if test_speaker not in speakers:
        raise ValueError(
            f"Unknown speaker {test_speaker!r}. Available: {speakers}."
        )

    is_test = dataframe["speaker_id"].astype(str) == test_speaker
    test_df = dataframe[is_test].reset_index(drop=True)
    remaining = dataframe[~is_test].reset_index(drop=True)
    if test_df.empty:
        raise ValueError(f"Speaker {test_speaker!r} has no clips.")

    # Validation is drawn from the non-held-out speakers, never from the test
    # speaker, so the test set stays untouched by model selection. With
    # carve_validation=False every remaining clip trains and validation is empty.
    if carve_validation:
        train_df, validation_df = stratified_split(remaining, test_size, seed)
    else:
        train_df = remaining
        validation_df = remaining.iloc[0:0]

    missing = sorted(
        set(test_df["label"].astype(str)) - set(train_df["label"].astype(str))
    )
    if missing:
        raise ValueError(
            f"Holding out {test_speaker!r} leaves no training examples for "
            f"{missing}; these words could never be learned."
        )
    if train_df.empty:
        raise ValueError(f"Holding out {test_speaker!r} leaves no training clips.")

    if not validation_df.empty and not set(validation_df["speaker_id"]).isdisjoint({test_speaker}):
        raise ValueError("The held-out speaker must not appear in validation.")

    if return_validation:
        return train_df, validation_df, test_df, test_speaker
    return train_df, test_df, test_speaker


def _make_loader(
    dataframe: pd.DataFrame,
    label_to_index: dict[str, int],
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        CustomWordsDataset(dataframe, label_to_index),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )


def get_custom_words_dataloaders(
    test_size: float = 0.2,
    seed: int = 42,
    batch_size: int = 16,
    num_workers: int = 0,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    speaker_holdout: str | None = None,
    return_validation: bool = False,
    carve_validation: bool = True,
):
    """Build train/test DataLoaders and return them with the vocab maps.

    Returns ``(train_loader, test_loader, label_to_index, index_to_label)``, or
    with ``return_validation`` a 5-tuple whose second element is the validation
    loader; in that case ``test_loader`` moves to third position.

    ``speaker_holdout`` selects the speaker-disjoint protocol: the named
    speaker's clips appear only in the test loader. This is the honest
    evaluation; the default per-clip split leaks speakers between train and
    test and should be treated as an upper bound only.

    With ``speaker_holdout`` set, the validation loader holds out
    non-test speakers, so selecting a checkpoint on it never touches the
    held-out speaker. Without it, validation is ``None``: the per-clip split
    already spends its hold-out on the test set.

    ``carve_validation=False`` (requires ``speaker_holdout``) trains on every
    non-test-speaker clip and returns a ``None`` validation loader, so the
    caller must supply its own stopping rule; see
    ``models/LBY/train_custom_words.py --all-training-data``.
    """
    dataframe = load_manifest(data_dir)
    label_to_index, index_to_label = build_label_vocab(dataframe)
    validation_loader: DataLoader | None = None

    if speaker_holdout is None:
        if not carve_validation:
            raise ValueError("carve_validation=False requires a speaker_holdout.")
        train_df, test_df = stratified_split(dataframe, test_size, seed)
    else:
        if return_validation:
            train_df, validation_df, test_df, _ = speaker_disjoint_split(
                dataframe,
                test_speaker=speaker_holdout,
                test_size=test_size,
                seed=seed,
                return_validation=True,
                carve_validation=carve_validation,
            )
            if not validation_df.empty:
                validation_loader = _make_loader(
                    validation_df, label_to_index, batch_size, num_workers, shuffle=False
                )
        else:
            train_df, test_df, _ = speaker_disjoint_split(
                dataframe,
                test_speaker=speaker_holdout,
                test_size=test_size,
                seed=seed,
                carve_validation=carve_validation,
            )

    train_loader = _make_loader(
        train_df, label_to_index, batch_size, num_workers, shuffle=True
    )
    test_loader = _make_loader(
        test_df, label_to_index, batch_size, num_workers, shuffle=False
    )

    if return_validation:
        return train_loader, validation_loader, test_loader, label_to_index, index_to_label
    return train_loader, test_loader, label_to_index, index_to_label
