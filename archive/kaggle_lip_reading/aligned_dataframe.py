"""Load the processed GRID video alignment index as a pandas DataFrame."""

from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

ARCHIVE_ROOT = Path(__file__).resolve().parent


def load_aligned_dataframe(
    processed_dir: str | Path = ARCHIVE_ROOT / "data/lips-dataset/processedData",
) -> pd.DataFrame:
    """Load one row per spoken word without loading video arrays into memory.

    ``end_frame`` is an exclusive boundary, matching ordinary NumPy slicing.
    """
    processed_dir = Path(processed_dir).expanduser().resolve()
    manifest_path = processed_dir / "aligned_words.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Aligned-word manifest not found: {manifest_path}. "
            "Run align_videos.py first."
        )

    dataframe = pd.read_csv(
        manifest_path,
        dtype={
            "video_id": "string",
            "video_file": "string",
            "alignment_file": "string",
            "npz_file": "string",
            "word": "string",
        },
    )
    dataframe["npz_path"] = dataframe["npz_file"].map(
        lambda name: processed_dir / name
    )
    return dataframe


def load_word_frames(row: Mapping) -> np.ndarray:
    """Load only the unpadded frames for one DataFrame row."""
    npz_path = Path(row["npz_path"])
    word_index = int(row["word_index"])
    frame_count = int(row["frame_count"])

    with np.load(npz_path, allow_pickle=False) as archive:
        if "word_segments" in archive:
            return archive["word_segments"][word_index, :frame_count].copy()

        start_frame = int(row["start_frame"])
        end_frame = int(row["end_frame"])
        return archive["frames"][start_frame:end_frame].copy()
