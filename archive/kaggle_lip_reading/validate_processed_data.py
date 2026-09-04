"""Validate processed GRID archives and their aligned-word manifest."""

import argparse
from pathlib import Path
import zipfile

import numpy as np

from archive.kaggle_lip_reading.aligned_dataframe import load_aligned_dataframe


ARCHIVE_ROOT = Path(__file__).resolve().parent


REQUIRED_ARCHIVE_KEYS = {
    "frames",
    "words",
    "word_ranges",
    "word_time_ranges",
    "word_time_ranges_seconds",
    "word_segments",
    "segment_lengths",
    "align_lines",
    "sample_rate_hz",
    "video_fps",
    "time_units_per_frame",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate every processed video archive and timing row."
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=ARCHIVE_ROOT / "data/lips-dataset/processedData",
        help="Directory containing .npz files and aligned_words.csv.",
    )
    return parser.parse_args()


def validate_archive(archive_path: Path, rows) -> int:
    """Validate one archive and return its decoded frame count."""
    with np.load(archive_path, allow_pickle=False) as archive:
        missing_keys = REQUIRED_ARCHIVE_KEYS.difference(archive.files)
        if missing_keys:
            raise ValueError(f"missing keys: {sorted(missing_keys)}")

        frames = archive["frames"]
        words = archive["words"]
        word_ranges = archive["word_ranges"]
        time_ranges = archive["word_time_ranges"]
        time_ranges_seconds = archive["word_time_ranges_seconds"]
        segments = archive["word_segments"]
        lengths = archive["segment_lengths"]

        word_count = len(rows)
        if word_count != 6:
            raise ValueError(f"manifest has {word_count} words instead of 6")
        if words.shape != (word_count,):
            raise ValueError(f"unexpected words shape: {words.shape}")
        if word_ranges.shape != (word_count, 2):
            raise ValueError(f"unexpected word_ranges shape: {word_ranges.shape}")
        if time_ranges.shape != (word_count, 2):
            raise ValueError(
                f"unexpected word_time_ranges shape: {time_ranges.shape}"
            )
        if lengths.shape != (word_count,):
            raise ValueError(f"unexpected segment_lengths shape: {lengths.shape}")

        expected_lengths = word_ranges[:, 1] - word_ranges[:, 0]
        if not np.array_equal(lengths, expected_lengths):
            raise ValueError("segment lengths do not match frame ranges")
        if np.any(lengths <= 0):
            raise ValueError("one or more lexical words has no frames")
        if np.any(word_ranges[:, 0] < 0) or np.any(
            word_ranges[:, 1] > len(frames)
        ):
            raise ValueError("one or more frame ranges is out of bounds")
        if np.any(word_ranges[1:, 0] < word_ranges[:-1, 1]):
            raise ValueError("adjacent lexical-word frame ranges overlap")

        if words.tolist() != rows["word"].astype(str).tolist():
            raise ValueError("archive words do not match the manifest")
        if not np.array_equal(
            time_ranges,
            rows[["start_units", "end_units"]].to_numpy(dtype=np.int64),
        ):
            raise ValueError("exact time units do not match the manifest")
        if not np.allclose(time_ranges_seconds, time_ranges / 25000):
            raise ValueError("seconds do not match the 25 kHz time units")
        if not np.array_equal(
            word_ranges,
            rows[["start_frame", "end_frame"]].to_numpy(dtype=np.int32),
        ):
            raise ValueError("frame ranges do not match the manifest")
        if not np.array_equal(
            lengths, rows["frame_count"].to_numpy(dtype=np.int32)
        ):
            raise ValueError("frame counts do not match the manifest")

        for index, (start_frame, end_frame) in enumerate(word_ranges):
            length = int(lengths[index])
            if not np.array_equal(
                segments[index, :length], frames[start_frame:end_frame]
            ):
                raise ValueError(f"pixel segment mismatch at word {index}")
            if np.any(segments[index, length:]):
                raise ValueError(f"nonzero segment padding at word {index}")

        return len(frames)


def main() -> None:
    args = parse_args()
    processed_dir = args.processed_dir.expanduser().resolve()
    dataframe = load_aligned_dataframe(processed_dir)
    archive_paths = sorted(processed_dir.glob("*.npz"))

    if dataframe.empty:
        raise SystemExit("The aligned-word manifest is empty")
    if dataframe.duplicated(["video_id", "word_index"]).any():
        raise SystemExit("Duplicate (video_id, word_index) rows in manifest")

    manifest_stems = set(dataframe["video_id"].astype(str))
    archive_stems = {path.stem for path in archive_paths}
    missing_archives = manifest_stems.difference(archive_stems)
    stale_archives = archive_stems.difference(manifest_stems)
    if missing_archives or stale_archives:
        raise SystemExit(
            "Manifest/archive mismatch: "
            f"{len(missing_archives)} missing archive(s), "
            f"{len(stale_archives)} stale archive(s)"
        )

    grouped_rows = {
        video_id: rows.sort_values("word_index")
        for video_id, rows in dataframe.groupby("video_id", sort=False)
    }
    failures = []
    frame_counts = []
    for number, archive_path in enumerate(archive_paths, start=1):
        rows = grouped_rows.get(archive_path.stem)
        if rows is None:
            failures.append(f"{archive_path.name}: missing manifest rows")
            continue
        try:
            frame_counts.append(validate_archive(archive_path, rows))
        except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
            failures.append(f"{archive_path.name}: {error}")

        if number % 100 == 0:
            print(f"Validated {number}/{len(archive_paths)} archives")

    if failures:
        print("\n".join(failures))
        raise SystemExit(f"Validation failed for {len(failures)} archives")

    counts = {
        count: frame_counts.count(count)
        for count in sorted(set(frame_counts))
    }
    print(
        f"Validation passed: {len(archive_paths)} archives, "
        f"{len(dataframe)} aligned-word rows, frame counts {counts}."
    )


if __name__ == "__main__":
    main()
