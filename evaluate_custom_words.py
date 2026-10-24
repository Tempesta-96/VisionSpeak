"""Offline evaluation of the deployed valente CTC lip reader on the recorded dataset.

Reads ``data/custom_words/manifest.csv`` and runs the exact inference path used
by the live webcam tools (``inference.lip_worker.LipPredictor``) over the
recorded clips, then reports overall / per-speaker / per-word accuracy and a
confusion matrix. No webcam is required.

Two clip sources:

* ``npz`` (default) — mouth-crop frames read straight from each clip's
  ``.npz`` archive. Works for every recorded clip.
* ``mp4`` — full-face raw videos decoded and re-cropped frame-by-frame
  through the live ``MouthCropper`` pipeline, including the live quality
  gates (minimum frames and fresh-detection rate). Only speakers recorded
  with the updated ``record_custom_word_dataset.ipynb`` have raw videos.

A single raw MP4 file can also be predicted directly with ``--video``,
bypassing the dataset and manifest entirely.

Examples::

    .venv/bin/python evaluate_custom_words.py
    .venv/bin/python evaluate_custom_words.py --speakers speaker_test_samples
    .venv/bin/python evaluate_custom_words.py --source mp4 --speakers speaker_test_samples
    .venv/bin/python evaluate_custom_words.py --video data/custom_words/raw/speaker_test_samples/hello/clip.mp4
    .venv/bin/python evaluate_custom_words.py --checkpoint checkpoints/valente_custom_words_lipnet_ctc_0.743_acc.pt
    .venv/bin/python evaluate_custom_words.py --output-csv predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

from prepare_custom_word_dataset import load_manifest as load_manifest_frame
from inference.lip_worker import (
    LipPredictor,
    TimestampClock,
    WORDS,
    clip_quality_message,
)
from processing.mouth_cropper import MouthCropper

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = Path("checkpoints/valente_custom_words_lipnet_ctc_0.852_acc.pt")
DEFAULT_DATA_DIR = Path("data/custom_words")
CSV_COLUMNS = [
    "sample_id",
    "speaker_id",
    "label",
    "repetition",
    "source",
    "predicted",
    "confidence",
    "correct",
    "status",
    "note",
]


@dataclass(frozen=True)
class ClipResult:
    """Outcome of running the model on one manifest row."""

    sample_id: str
    speaker_id: str
    label: str
    repetition: int
    predicted: str
    confidence: float
    correct: bool
    status: str  # 'predicted' | 'missing-file' | 'rejected' | 'error'
    note: str = ""


def select_rows(
    manifest: pd.DataFrame, speakers: list[str] | None
) -> tuple[pd.DataFrame, list[str]]:
    """Return the manifest rows for the selected speakers plus any unknown names."""
    if not speakers:
        return manifest, []
    known = set(manifest["speaker_id"].astype(str))
    unknown = [speaker for speaker in speakers if speaker not in known]
    rows = manifest[manifest["speaker_id"].astype(str).isin(speakers)]
    return rows, unknown


def predict_npz_result(row, predictor, *, max_frames: int) -> ClipResult:
    """Predict one clip from its mouth-crop ``.npz`` archive."""
    clip_path = Path(row.clip_path)
    if not clip_path.is_file():
        return ClipResult(
            sample_id=str(row.sample_id),
            speaker_id=str(row.speaker_id),
            label=str(row.label),
            repetition=int(row.repetition),
            predicted="",
            confidence=0.0,
            correct=False,
            status="missing-file",
            note=f"missing archive: {clip_path}",
        )

    with np.load(clip_path, allow_pickle=False) as archive:
        frames = archive["frames"]
    prediction = predictor.predict(frames, max_frames=max_frames)
    return ClipResult(
        sample_id=str(row.sample_id),
        speaker_id=str(row.speaker_id),
        label=str(row.label),
        repetition=int(row.repetition),
        predicted=prediction.label,
        confidence=float(prediction.confidence),
        correct=prediction.label == str(row.label),
        status="predicted",
    )


def extract_mp4_mouth_frames(
    video_path: Path,
    cropper,
    clock: TimestampClock,
    *,
    min_frames: int,
    min_detection_rate: float,
) -> tuple[np.ndarray | None, float, str]:
    """Decode one raw video through the live mouth-cropping pipeline.

    Mirrors ``inference.lip_worker.capture_clip``: every frame is cropped
    (with box fallback), converted to grayscale, and the clip must pass the
    same quality gates used live. ``clock`` must be shared across every
    video processed by ``cropper`` because MediaPipe requires monotonically
    increasing timestamps per landmarker instance. Returns
    ``(frames, detection_rate, "")`` on success or ``(None, detection_rate,
    reason)`` when the clip is unusable.
    """
    camera = cv2.VideoCapture(str(video_path))
    if not camera.isOpened():
        return None, 0.0, f"could not open video: {video_path}"

    frames: list[np.ndarray] = []
    fresh_detections = 0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break
            crop, _box, fresh = cropper.crop_with_metadata(frame, clock.next())
            fresh_detections += int(fresh)
            frames.append(
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.uint8, copy=False)
            )
    finally:
        camera.release()

    detection_rate = fresh_detections / max(len(frames), 1)
    retry_message = clip_quality_message(
        len(frames),
        detection_rate,
        min_frames=min_frames,
        min_detection_rate=min_detection_rate,
    )
    if retry_message:
        return None, detection_rate, retry_message
    return np.stack(frames).astype(np.uint8, copy=False), detection_rate, ""


def predict_mp4_result(
    row,
    cropper,
    clock: TimestampClock,
    predictor,
    *,
    data_dir: Path,
    max_frames: int,
    min_frames: int,
    min_detection_rate: float,
) -> ClipResult:
    """Predict one clip from its raw full-face MP4 through the live pipeline."""
    raw_relative = "" if pd.isna(row.raw_video_path) else str(row.raw_video_path)
    if not raw_relative:
        return ClipResult(
            sample_id=str(row.sample_id),
            speaker_id=str(row.speaker_id),
            label=str(row.label),
            repetition=int(row.repetition),
            predicted="",
            confidence=0.0,
            correct=False,
            status="missing-file",
            note="no raw video recorded for this clip",
        )
    video_path = Path(data_dir) / raw_relative
    if not video_path.is_file():
        return ClipResult(
            sample_id=str(row.sample_id),
            speaker_id=str(row.speaker_id),
            label=str(row.label),
            repetition=int(row.repetition),
            predicted="",
            confidence=0.0,
            correct=False,
            status="missing-file",
            note=f"missing raw video: {video_path}",
        )

    cropper.reset_tracking()
    frames, _detection_rate, message = extract_mp4_mouth_frames(
        video_path,
        cropper,
        clock,
        min_frames=min_frames,
        min_detection_rate=min_detection_rate,
    )
    if frames is None:
        return ClipResult(
            sample_id=str(row.sample_id),
            speaker_id=str(row.speaker_id),
            label=str(row.label),
            repetition=int(row.repetition),
            predicted="",
            confidence=0.0,
            correct=False,
            status="rejected",
            note=message,
        )

    prediction = predictor.predict(frames, max_frames=max_frames)
    return ClipResult(
        sample_id=str(row.sample_id),
        speaker_id=str(row.speaker_id),
        label=str(row.label),
        repetition=int(row.repetition),
        predicted=prediction.label,
        confidence=float(prediction.confidence),
        correct=prediction.label == str(row.label),
        status="predicted",
    )


def evaluate_rows(rows, predict_one) -> list[ClipResult]:
    """Run ``predict_one`` over every row, isolating per-clip failures."""
    results: list[ClipResult] = []
    total = len(rows)
    for index, row in enumerate(rows.itertuples(index=False), start=1):
        try:
            result = predict_one(row)
        except Exception as error:  # Keep one broken clip from stopping the run.
            result = ClipResult(
                sample_id=str(row.sample_id),
                speaker_id=str(row.speaker_id),
                label=str(row.label),
                repetition=int(row.repetition),
                predicted="",
                confidence=0.0,
                correct=False,
                status="error",
                note=str(error),
            )
        results.append(result)
        if index % 50 == 0 or index == total:
            print(f"  evaluated {index}/{total}", flush=True)
    return results


def summarize(results: list[ClipResult]) -> dict:
    """Aggregate counts, accuracy, and mean confidence over predicted clips."""
    predicted = [result for result in results if result.status == "predicted"]
    correct = sum(result.correct for result in predicted)
    return {
        "total": len(results),
        "predicted": len(predicted),
        "correct": correct,
        "accuracy": correct / len(predicted) if predicted else 0.0,
        "mean_confidence": (
            float(np.mean([result.confidence for result in predicted]))
            if predicted
            else 0.0
        ),
        "status_counts": dict(Counter(result.status for result in results)),
    }


def build_confusion_matrix(
    results: list[ClipResult], words: tuple[str, ...]
) -> np.ndarray:
    """Confusion matrix over predicted clips (rows=truth, cols=prediction)."""
    index = {word: position for position, word in enumerate(words)}
    confusion = np.zeros((len(words), len(words)), dtype=int)
    for result in results:
        if result.status != "predicted":
            continue
        if result.label not in index or result.predicted not in index:
            continue
        confusion[index[result.label], index[result.predicted]] += 1
    return confusion


def print_report(
    results: list[ClipResult],
    words: tuple[str, ...],
    *,
    source: str,
    checkpoint: Path,
) -> None:
    """Print the evaluation report in the style of ``train_custom_words.py``."""
    summary = summarize(results)
    status_line = ", ".join(
        f"{status}: {count}"
        for status, count in sorted(summary["status_counts"].items())
    )
    print(f"Source: {source} | Checkpoint: {checkpoint.name}")
    print(f"Clips: {summary['total']} ({status_line})")
    if summary["predicted"]:
        print(
            f"Overall accuracy: {summary['accuracy']:.4f} "
            f"({summary['correct']}/{summary['predicted']})"
        )
        print(f"Mean confidence: {summary['mean_confidence']:.4f}")
    else:
        print("Overall accuracy: n/a (no clip could be evaluated)")
        return

    predicted = [result for result in results if result.status == "predicted"]

    print("\nPer-speaker accuracy:")
    for speaker in sorted({result.speaker_id for result in predicted}):
        speaker_results = [
            result for result in predicted if result.speaker_id == speaker
        ]
        speaker_correct = sum(result.correct for result in speaker_results)
        print(
            f"  {speaker:>22}: {speaker_correct / len(speaker_results):.4f} "
            f"({speaker_correct}/{len(speaker_results)})"
        )

    print("\nPer-word accuracy:")
    for word in words:
        word_results = [result for result in predicted if result.label == word]
        if not word_results:
            continue
        word_correct = sum(result.correct for result in word_results)
        print(
            f"  {word:>8}: {word_correct / len(word_results):.4f} "
            f"({word_correct}/{len(word_results)})"
        )

    confusion = build_confusion_matrix(results, words)
    print("\nConfusion matrix (rows=truth, cols=prediction):")
    header = "        " + " ".join(f"{word[:4]:>4}" for word in words)
    print(header)
    for position, word in enumerate(words):
        row = " ".join(f"{value:4d}" for value in confusion[position])
        print(f"{word:>8} {row}")

    for status in ("missing-file", "rejected", "error"):
        flagged = [result for result in results if result.status == status]
        if flagged:
            print(f"\n{status} ({len(flagged)}):")
            for result in flagged[:3]:
                print(f"  {result.sample_id} [{result.label}]: {result.note}")


def write_results_csv(
    results: list[ClipResult], path: Path, *, source: str
) -> None:
    """Write one row per evaluated clip for later analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "sample_id": result.sample_id,
                    "speaker_id": result.speaker_id,
                    "label": result.label,
                    "repetition": result.repetition,
                    "source": source,
                    "predicted": result.predicted,
                    "confidence": f"{result.confidence:.6f}",
                    "correct": result.correct,
                    "status": result.status,
                    "note": result.note,
                }
            )


def run_single_video(
    video_path: Path,
    *,
    checkpoint: Path,
    device: str,
    max_frames: int,
    min_frames: int,
    min_detection_rate: float,
) -> int:
    """Predict the word in one raw MP4 file through the live pipeline."""
    if not video_path.is_file():
        print(f"Video file not found: {video_path}", file=sys.stderr)
        return 1

    print(f"Video: {video_path}")
    try:
        predictor = LipPredictor(checkpoint, device)
        with MouthCropper() as cropper:
            # One clock for the whole run, exactly like the live webcam
            # session: the shared landmarker needs ever-increasing timestamps.
            clock = TimestampClock()
            frames, detection_rate, message = extract_mp4_mouth_frames(
                video_path,
                cropper,
                clock,
                min_frames=min_frames,
                min_detection_rate=min_detection_rate,
            )
    except Exception as error:
        print(f"Could not process {video_path}: {error}", file=sys.stderr)
        return 1

    print(
        f"Checkpoint: {checkpoint.name} "
        f"(mode: {predictor.prediction_mode}, core: {predictor.architecture}) "
        f"| Device: {predictor.device}"
    )
    if frames is None:
        print(f"Rejected: {message}", file=sys.stderr)
        return 1

    prediction = predictor.predict(frames, max_frames=max_frames)
    print(f"Mouth frames: {len(frames)} | fresh-detection rate: {detection_rate:.0%}")
    print(f"Prediction: {prediction.label.upper()} ({prediction.confidence:.1%})")
    print("Word scores:")
    for word, score in sorted(prediction.scores.items(), key=lambda item: -item[1]):
        print(f"  {word:>8}: {score:.1%}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the deployed lip reader on the recorded custom-word "
            "dataset without a webcam, or predict a single raw MP4 file."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Lip checkpoint (default: {DEFAULT_CHECKPOINT}).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dataset directory containing manifest.csv (default: {DEFAULT_DATA_DIR}).",
    )
    parser.add_argument(
        "--source",
        choices=("npz", "mp4"),
        default="npz",
        help=(
            "Clip source: npz mouth-crop archives (default) or raw full-face "
            "MP4 videos through the live mouth-cropping pipeline."
        ),
    )
    parser.add_argument(
        "--speakers",
        default=None,
        help=(
            "Comma-separated speaker ids to evaluate (default: all). "
            "Example: --speakers speaker_007,speaker_test_samples"
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "Predict the word in a single raw MP4 file (e.g. a recording "
            "from record_custom_word_dataset.ipynb) instead of evaluating "
            "the dataset."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device for the lip model (default: auto).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=90,
        help="Maximum frames passed to the CTC model (default: 90).",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=30,
        help="mp4 source only: minimum mouth frames per clip (default: 30).",
    )
    parser.add_argument(
        "--min-detection-rate",
        type=float,
        default=0.75,
        help="mp4 source only: minimum fresh mouth-detection rate (default: 0.75).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path for per-clip predictions.",
    )
    return parser


def _resolve_against_project_root(path: Path) -> Path:
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def main(argv: list[str] | None = None) -> int:
    """Run the offline evaluation and return a process exit code."""
    args = build_parser().parse_args(argv)
    checkpoint = _resolve_against_project_root(args.checkpoint)
    data_dir = _resolve_against_project_root(args.data_dir)
    if not checkpoint.is_file():
        print(f"Lip checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    if args.video:
        dataset_only = [
            flag
            for flag, given in (
                ("--source", args.source != "npz"),
                ("--speakers", args.speakers is not None),
                ("--data-dir", args.data_dir != DEFAULT_DATA_DIR),
                ("--output-csv", args.output_csv is not None),
            )
            if given
        ]
        if dataset_only:
            print(
                f"--video cannot be combined with {' '.join(dataset_only)}; "
                "single-video mode predicts one file directly.",
                file=sys.stderr,
            )
            return 1
        return run_single_video(
            Path(args.video).expanduser(),
            checkpoint=checkpoint,
            device=args.device,
            max_frames=args.max_frames,
            min_frames=args.min_frames,
            min_detection_rate=args.min_detection_rate,
        )

    try:
        manifest = load_manifest_frame(data_dir)
    except FileNotFoundError as error:
        print(str(error), file=sys.stderr)
        return 1

    speakers = (
        [name.strip() for name in args.speakers.split(",") if name.strip()]
        if args.speakers
        else None
    )
    rows, unknown = select_rows(manifest, speakers)
    if unknown:
        print(f"Warning: unknown speakers ignored: {', '.join(sorted(unknown))}")
    if rows.empty:
        print("No clips match the selected speakers.", file=sys.stderr)
        return 1
    if args.source == "mp4":
        with_raw = rows["raw_video_path"].notna().sum()
        if not with_raw:
            print(
                "No selected clip has a raw video; record with the updated "
                "recording notebook or use --source npz.",
                file=sys.stderr,
            )
            return 1

    print(
        f"Evaluating {len(rows)} clip(s) from {data_dir} "
        f"({args.source} source, checkpoint {checkpoint.name})"
    )
    predictor = LipPredictor(checkpoint, args.device)
    print(f"Device: {predictor.device}")

    if args.source == "npz":
        results = evaluate_rows(
            rows,
            lambda row: predict_npz_result(row, predictor, max_frames=args.max_frames),
        )
    else:
        with MouthCropper() as cropper:
            # One clock for the whole run, exactly like the live webcam
            # session: the shared landmarker needs ever-increasing timestamps.
            clock = TimestampClock()
            results = evaluate_rows(
                rows,
                lambda row: predict_mp4_result(
                    row,
                    cropper,
                    clock,
                    predictor,
                    data_dir=data_dir,
                    max_frames=args.max_frames,
                    min_frames=args.min_frames,
                    min_detection_rate=args.min_detection_rate,
                ),
            )

    print()
    print_report(results, WORDS, source=args.source, checkpoint=checkpoint)

    if args.output_csv:
        output_csv = _resolve_against_project_root(args.output_csv)
        write_results_csv(results, output_csv, source=args.source)
        print(f"\nPer-clip predictions written to {output_csv}")

    if not any(result.status == "predicted" for result in results):
        print("No clip could be evaluated.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
