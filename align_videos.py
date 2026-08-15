"""Align GRID-corpus videos with their word-level alignment files.

For each video in --video-dir (data/s1/*.mpg) a matching alignment file in
--align-dir (data/alignments/s1/*.align) provides word boundaries in units of
1/25000 s (25 kHz audio samples). Videos play at 25 fps, so a timestamp t maps
to frame index t / 1000.

Each video is saved as one .npz in --output-dir containing the full frame
stack, the per-word frame ranges and the per-word frame segments.
"""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

SAMPLE_RATE_HZ = 25000
VIDEO_FPS = 25
TIME_UNITS_PER_FRAME = SAMPLE_RATE_HZ / VIDEO_FPS
NON_SPEECH_TOKENS = {"sil", "sp"}

MANIFEST_FIELDS = (
    "video_id",
    "video_file",
    "alignment_file",
    "npz_file",
    "word_index",
    "word",
    "start_units",
    "end_units",
    "start_seconds",
    "end_seconds",
    "start_frame",
    "end_frame",
    "frame_count",
    "fps",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Slice videos into per-word frame segments using alignment files."
        )
    )
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("data/s1"),
        help="Directory containing the .mpg video files.",
    )
    parser.add_argument(
        "--align-dir",
        type=Path,
        default=Path("data/alignments/s1"),
        help="Directory containing the .align alignment files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processedData"),
        help="Directory where the .npz files are written.",
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Store frames as single-channel grayscale instead of BGR.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N videos (useful for testing).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-video details.",
    )
    return parser.parse_args()


def read_alignment(
    align_path: Path,
) -> tuple[list[str], list[tuple[int, int, str]]]:
    """Return source lines and spoken (start, end, word) alignment tokens."""
    align_lines = align_path.read_text().splitlines()
    tokens = []
    for line in align_lines:
        parts = line.split()
        if len(parts) < 3:
            continue
        start, end, word = int(parts[0]), int(parts[1]), parts[2]
        if start < 0 or end < start:
            raise ValueError(f"Invalid alignment range in {align_path}: {line}")
        if word in NON_SPEECH_TOKENS:
            continue
        tokens.append((start, end, word))
    return align_lines, tokens


def load_video(video_path: Path, grayscale: bool) -> tuple[np.ndarray, float]:
    """Read every frame and return the frame stack plus reported frame rate."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = float(VIDEO_FPS)

    frames = []
    while True:
        success, frame = capture.read()
        if not success:
            break
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
    capture.release()

    if not frames:
        raise ValueError(f"Video contains no frames: {video_path}")
    return np.stack(frames), fps


def time_to_frame(
    t: int,
    total_frames: int,
    time_units_per_frame: float = TIME_UNITS_PER_FRAME,
    *,
    end_bound: bool = False,
) -> int:
    """Convert a timestamp to a clipped frame boundary.

    End boundaries are exclusive and may equal ``total_frames``. Start
    boundaries must refer to an existing frame.
    """
    upper_bound = total_frames if end_bound else total_frames - 1
    # Alignment times are non-negative. Half-up rounding avoids Python's
    # tie-to-even behaviour, which can collapse a one-frame token at x.5.
    frame_boundary = math.floor((t / time_units_per_frame) + 0.5)
    return min(max(frame_boundary, 0), upper_bound)


def extract_word_segments(
    frames: np.ndarray, word_ranges: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Slice frames into a padded segment array plus per-segment lengths."""
    lengths = (word_ranges[:, 1] - word_ranges[:, 0]).astype(np.int32)
    max_len = int(lengths.max()) if len(lengths) else 0
    shape = (len(word_ranges), max_len, *frames.shape[1:])
    segments = np.zeros(shape, dtype=frames.dtype)

    for i, (start, end) in enumerate(word_ranges):
        segments[i, : end - start] = frames[start:end]
    return segments, lengths


def process_video(
    video_path: Path, align_path: Path, output_dir: Path, grayscale: bool
) -> dict:
    frames, fps = load_video(video_path, grayscale)
    total_frames = frames.shape[0]
    time_units_per_frame = SAMPLE_RATE_HZ / fps

    align_lines, tokens = read_alignment(align_path)
    words = [token[2] for token in tokens]
    word_time_ranges = np.array(
        [[start, end] for start, end, _ in tokens], dtype=np.int64
    ).reshape(-1, 2)
    frame_ranges = []
    for start, end, _ in tokens:
        start_frame = time_to_frame(
            start, total_frames, time_units_per_frame
        )
        end_frame = time_to_frame(
            end,
            total_frames,
            time_units_per_frame,
            end_bound=True,
        )
        if end > start and end_frame <= start_frame:
            end_frame = min(start_frame + 1, total_frames)
        frame_ranges.append([start_frame, end_frame])
    word_ranges = np.asarray(frame_ranges, dtype=np.int32).reshape(-1, 2)

    segments, segment_lengths = extract_word_segments(frames, word_ranges)

    output_path = output_dir / f"{video_path.stem}.npz"
    temporary_path = output_path.with_suffix(".npz.tmp")
    with temporary_path.open("wb") as output_file:
        np.savez_compressed(
            output_file,
            frames=frames,
            words=np.array(words),
            word_ranges=word_ranges,
            word_time_ranges=word_time_ranges,
            word_time_ranges_seconds=(word_time_ranges / SAMPLE_RATE_HZ),
            word_segments=segments,
            segment_lengths=segment_lengths,
            align_lines=np.array(align_lines),
            sample_rate_hz=np.int32(SAMPLE_RATE_HZ),
            video_fps=np.float64(fps),
            time_units_per_frame=np.float64(time_units_per_frame),
        )
    temporary_path.replace(output_path)

    manifest_rows = []
    for word_index, ((start, end, word), (start_frame, end_frame)) in enumerate(
        zip(tokens, word_ranges)
    ):
        manifest_rows.append(
            {
                "video_id": video_path.stem,
                "video_file": video_path.name,
                "alignment_file": align_path.name,
                "npz_file": output_path.name,
                "word_index": word_index,
                "word": word,
                "start_units": start,
                "end_units": end,
                "start_seconds": start / SAMPLE_RATE_HZ,
                "end_seconds": end / SAMPLE_RATE_HZ,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "frame_count": int(end_frame - start_frame),
                "fps": fps,
            }
        )

    return {
        "name": video_path.stem,
        "total_frames": total_frames,
        "n_words": len(words),
        "speech_frames": int(
            (word_ranges[-1, 1] - word_ranges[0, 0]) if len(word_ranges) else 0
        ),
        "manifest_rows": manifest_rows,
    }


def write_manifest(output_dir: Path, rows: list[dict]) -> Path:
    """Write a portable, word-level CSV index using an atomic replacement."""
    manifest_path = output_dir / "aligned_words.csv"
    temporary_path = manifest_path.with_suffix(".csv.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(manifest_path)
    return manifest_path


def main() -> None:
    args = parse_args()
    if not args.video_dir.is_dir():
        raise SystemExit(f"Video directory not found: {args.video_dir}")
    if not args.align_dir.is_dir():
        raise SystemExit(f"Alignment directory not found: {args.align_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = sorted(args.video_dir.glob("*.mpg"))
    if not video_paths:
        raise SystemExit(f"No .mpg files found in: {args.video_dir}")
    if args.limit is not None:
        video_paths = video_paths[: args.limit]

    selected_stems = {path.stem for path in video_paths}
    existing_stems = {
        path.stem for path in args.output_dir.glob("*.npz")
    }
    stale_stems = existing_stems.difference(selected_stems)
    if stale_stems:
        examples = ", ".join(sorted(stale_stems)[:5])
        raise SystemExit(
            f"Output directory contains {len(stale_stems)} archive(s) outside "
            f"this run, including: {examples}. Use an empty output directory "
            "or process the complete matching input set."
        )

    align_paths = {
        path.stem: path for path in args.align_dir.glob("*.align")
    }

    processed, skipped = 0, 0
    manifest_rows = []
    for video_path in video_paths:
        align_path = align_paths.get(video_path.stem)
        if align_path is None:
            skipped += 1
            print(f"SKIP {video_path.name}: no matching alignment file")
            continue

        try:
            info = process_video(
                video_path, align_path, args.output_dir, args.grayscale
            )
        except (ValueError, OSError) as error:
            skipped += 1
            print(f"ERROR {video_path.name}: {error}")
            continue

        processed += 1
        manifest_rows.extend(info["manifest_rows"])
        if args.verbose:
            print(
                f"{info['name']}: {info['total_frames']} frames, "
                f"{info['n_words']} words, "
                f"{info['speech_frames']} speech frames "
                f"-> {args.output_dir / (info['name'] + '.npz')}"
            )

    manifest_path = write_manifest(args.output_dir, manifest_rows)
    print(
        f"\nDone: {processed} videos aligned, {skipped} skipped. "
        f"Output: {args.output_dir}. Manifest: {manifest_path}"
    )
    if skipped:
        raise SystemExit(f"Processing was incomplete: {skipped} video(s) skipped")


if __name__ == "__main__":
    main()
