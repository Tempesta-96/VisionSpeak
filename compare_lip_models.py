"""Compare the CTC (CNN + BiGRU) and transformer lip readers on raw MP4 clips.

Both model families are scored on *exactly the same mouth frames*: every raw
video is decoded and cropped once through the live MediaPipe pipeline
(``processing.mouth_cropper``, the same code the webcam tools use), the frames
are cached, and each checkpoint is then run over the cached arrays. Cropping is
by far the slowest step, so sharing it keeps the comparison both fair and fast.

Outputs, written to ``reports/lip_comparison`` by default:

* ``comparison.md`` — headline table, per-word accuracy, confusion matrices
* ``comparison.json`` — the same numbers, machine-readable
* ``predictions_<tag>.csv`` — one row per clip, per model
* ``confusion_<tag>.png`` and ``confusion_side_by_side.png`` — heatmaps

Every clip in ``data/custom_words/raw/`` is used, with the ground-truth word
taken from its directory name via the manifest.

Examples::

    .venv/bin/python compare_lip_models.py
    .venv/bin/python compare_lip_models.py --speakers speaker_test_samples
    .venv/bin/python compare_lip_models.py --refresh-frames --limit 20
"""

from __future__ import annotations

import os
from pathlib import Path

# Keep matplotlib's cache inside the project (its default path is read-only here)
# and call this before matplotlib is imported anywhere.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".cache" / "matplotlib"))

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from prepare_custom_word_dataset import load_manifest
from evaluate_custom_words import (
    ClipResult,
    build_confusion_matrix,
    extract_mp4_mouth_frames,
    summarize,
    write_results_csv,
)
from inference.lip_worker import LipPredictor, TimestampClock
from processing.mouth_cropper import MouthCropper

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path("data/custom_words")
DEFAULT_CTC_CHECKPOINT = Path("checkpoints/valente_custom_words_lipnet_ctc_0.852_acc.pt")
DEFAULT_TRANSFORMER_CHECKPOINT = Path(
    "checkpoints/lip_transformer/transformer-attention-holdout-speaker_test_samples/"
    "epoch020_valid0.5234.pt"
)
DEFAULT_OUTPUT_DIR = Path("reports/lip_comparison")
DEFAULT_CACHE_DIR = Path(".cache/lip_mp4_frames")

WORDS = ("bad", "bye", "good", "hello", "hi", "no", "thanks", "yes")


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint to score, with the display tag used in the report."""

    tag: str
    path: Path
    note: str = ""


@dataclass
class ClipFrames:
    """Cached mouth frames for one raw clip, or the reason they are unusable."""

    sample_id: str
    frames: np.ndarray
    detection_rate: float
    reason: str
    from_cache: bool = False


def resolve(path: Path) -> Path:
    """Resolve a CLI path against the project root."""
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def parse_model_specs(
    ctc_checkpoint: Path, transformer_checkpoint: Path, extra: list[str]
) -> list[ModelSpec]:
    """Build the ordered list of checkpoints to evaluate."""
    specs = [
        ModelSpec("ctc_bigru", resolve(ctc_checkpoint), "CNN + BiGRU + CTC"),
        ModelSpec("transformer", resolve(transformer_checkpoint), "CNN + transformer"),
    ]
    for item in extra:
        tag, _, raw_path = item.partition("=")
        if not raw_path:
            tag, raw_path = Path(item).stem, item
        specs.append(ModelSpec(tag.strip(), resolve(Path(raw_path.strip())), "extra checkpoint"))
    for spec in specs:
        if not spec.path.is_file():
            raise SystemExit(f"Checkpoint not found for {spec.tag!r}: {spec.path}")
    return specs


def checkpoint_metadata(path: Path) -> dict:
    """Read the settings a checkpoint records about its own training run."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    settings = {
        key: checkpoint[key]
        for key in (
            "architecture",
            "pooling",
            "gru_hidden",
            "transformer_layers",
            "transformer_heads",
            "transformer_feedforward",
        )
        if key in checkpoint
    }
    provenance = path.parent / "PROVENANCE.md"
    return {
        "name": path.name,
        "path": str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path),
        "epoch": checkpoint.get("epoch"),
        "architecture": checkpoint.get("architecture", "bigru (default)"),
        "temporal_core": "ctc+birnn" if "char_to_index" in checkpoint else "transformer",
        "selection_split": checkpoint.get("selection_split"),
        "selection_accuracy": checkpoint.get("selection_accuracy", checkpoint.get("test_accuracy")),
        "validation_loss": checkpoint.get("validation_loss"),
        "speaker_holdout": checkpoint.get("speaker_holdout"),
        "written_at": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "provenance_file": (
            str(provenance.relative_to(PROJECT_ROOT)) if provenance.is_file() else None
        ),
        "settings": settings,
    }


def cache_path(cache_dir: Path, sample_id: str, *, min_frames: int, min_detection_rate: float) -> Path:
    """Cache file for one clip, keyed by the quality gates that produced it."""
    return cache_dir / f"{sample_id}__min{min_frames}_rate{min_detection_rate:g}.npz"


def load_or_extract_frames(
    row,
    *,
    cropper: MouthCropper,
    clock: TimestampClock,
    data_dir: Path,
    cache_dir: Path,
    max_frames: int,
    min_frames: int,
    min_detection_rate: float,
    refresh: bool,
) -> ClipFrames:
    """Return the mouth frames for one manifest row, using the disk cache."""
    sample_id = str(row.sample_id)
    cached = cache_path(
        cache_dir, sample_id, min_frames=min_frames, min_detection_rate=min_detection_rate
    )
    if cached.is_file() and not refresh:
        with np.load(cached, allow_pickle=False) as archive:
            return ClipFrames(
                sample_id=sample_id,
                frames=archive["frames"],
                detection_rate=float(archive["detection_rate"]),
                reason=str(archive["reason"]),
                from_cache=True,
            )

    raw_relative = "" if pd.isna(row.raw_video_path) else str(row.raw_video_path)
    video_path = data_dir / raw_relative if raw_relative else None
    if video_path is None or not video_path.is_file():
        return ClipFrames(sample_id, np.zeros((0, 50, 100), np.uint8), 0.0,
                          f"missing raw video: {video_path}")

    cropper.reset_tracking()
    frames, detection_rate, reason = extract_mp4_mouth_frames(
        video_path,
        cropper,
        clock,
        min_frames=min_frames,
        min_detection_rate=min_detection_rate,
    )
    stored = np.zeros((0, 50, 100), np.uint8) if frames is None else frames
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cached,
        frames=stored,
        detection_rate=np.float64(detection_rate),
        reason=np.array(reason),
    )
    return ClipFrames(sample_id, stored, float(detection_rate), reason)


def extract_frames(
    rows: pd.DataFrame,
    *,
    data_dir: Path,
    cache_dir: Path,
    max_frames: int,
    min_frames: int,
    min_detection_rate: float,
    refresh: bool,
) -> dict[str, ClipFrames]:
    """Decode every raw clip once through the live mouth-cropping pipeline."""
    collected: dict[str, ClipFrames] = {}
    total = len(rows)
    started = time.perf_counter()
    with MouthCropper() as cropper:
        # One clock for the whole run, exactly like a live webcam session: the
        # shared MediaPipe landmarker needs ever-increasing timestamps.
        clock = TimestampClock()
        for index, row in enumerate(rows.itertuples(index=False), start=1):
            collected[str(row.sample_id)] = load_or_extract_frames(
                row,
                cropper=cropper,
                clock=clock,
                data_dir=data_dir,
                cache_dir=cache_dir,
                max_frames=max_frames,
                min_frames=min_frames,
                min_detection_rate=min_detection_rate,
                refresh=refresh,
            )
            if index % 10 == 0 or index == total:
                used_cache = collected[str(row.sample_id)].from_cache
                elapsed = time.perf_counter() - started
                print(
                    f"  frames {index}/{total} ({elapsed:.0f}s, cache={used_cache})",
                    flush=True,
                )
    return collected


def predict_frames(
    spec: ModelSpec,
    rows: pd.DataFrame,
    frames_by_id: dict[str, ClipFrames],
    *,
    device: str,
    max_frames: int,
) -> tuple[list[ClipResult], dict]:
    """Score one checkpoint over the cached frames of every row."""
    predictor = LipPredictor(spec.path, device)
    results: list[ClipResult] = []
    started = time.perf_counter()
    for row in rows.itertuples(index=False):
        sample_id = str(row.sample_id)
        clip = frames_by_id[sample_id]
        if clip.frames.size == 0:
            results.append(
                ClipResult(
                    sample_id=sample_id,
                    speaker_id=str(row.speaker_id),
                    label=str(row.label),
                    repetition=int(row.repetition),
                    predicted="",
                    confidence=0.0,
                    correct=False,
                    status="rejected",
                    note=clip.reason or "no usable frames",
                )
            )
            continue
        try:
            prediction = predictor.predict(clip.frames, max_frames=max_frames)
        except Exception as error:  # One broken clip must not stop the run.
            results.append(
                ClipResult(
                    sample_id=sample_id,
                    speaker_id=str(row.speaker_id),
                    label=str(row.label),
                    repetition=int(row.repetition),
                    predicted="",
                    confidence=0.0,
                    correct=False,
                    status="error",
                    note=str(error),
                )
            )
            continue
        results.append(
            ClipResult(
                sample_id=sample_id,
                speaker_id=str(row.speaker_id),
                label=str(row.label),
                repetition=int(row.repetition),
                predicted=prediction.label,
                confidence=float(prediction.confidence),
                correct=prediction.label == str(row.label),
                status="predicted",
            )
        )
    elapsed = time.perf_counter() - started
    metadata = checkpoint_metadata(spec.path)
    metadata.update(
        {
            "tag": spec.tag,
            "note": spec.note,
            "device": str(predictor.device),
            "prediction_mode": predictor.prediction_mode,
            "architecture": predictor.architecture,
            "inference_seconds": round(elapsed, 2),
        }
    )
    return results, metadata


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    if total <= 0:
        return (0.0, 0.0)
    proportion = correct / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    half = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for paired predictions.

    ``b`` and ``c`` are the discordant counts: clips model A got right and B
    got wrong, and vice versa.
    """
    total = b + c
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(b, c) + 1)) / 2**total
    return min(1.0, 2 * tail)


def model_summary(results: list[ClipResult]) -> dict:
    """Accuracy, per-word breakdown and confusion matrix for one model."""
    summary = summarize(results)
    predicted = [result for result in results if result.status == "predicted"]
    correct = int(sum(result.correct for result in predicted))
    low, high = wilson_interval(correct, len(predicted))
    confusion = build_confusion_matrix(results, WORDS)
    per_word = {}
    for word in WORDS:
        word_results = [result for result in predicted if result.label == word]
        if not word_results:
            continue
        word_correct = sum(result.correct for result in word_results)
        per_word[word] = {
            "accuracy": word_correct / len(word_results),
            "correct": word_correct,
            "total": len(word_results),
        }
    summary.update(
        {
            "accuracy": correct / len(predicted) if predicted else 0.0,
            "correct": correct,
            "accuracy_ci95": [low, high],
            "accuracy_over_all_clips": correct / len(results) if results else 0.0,
            "confusion_matrix": confusion.tolist(),
            "labels": list(WORDS),
            "per_word": per_word,
        }
    )
    return summary


def render_confusion(axis, matrix: np.ndarray, title: str) -> None:
    """Draw one annotated confusion-matrix heatmap."""
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, row_totals, out=np.zeros_like(matrix, dtype=float),
                           where=row_totals > 0)
    axis.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(WORDS)), WORDS, rotation=45, ha="right")
    axis.set_yticks(range(len(WORDS)), WORDS)
    axis.set_xlabel("prediction")
    axis.set_ylabel("truth")
    axis.set_title(title, fontsize=11)
    for row in range(len(WORDS)):
        for column in range(len(WORDS)):
            count = int(matrix[row, column])
            if count == 0:
                continue
            share = normalised[row, column]
            axis.text(
                column,
                row,
                f"{count}\n{share:.0%}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if share > 0.55 else "black",
            )


def confusion_table_text(matrix: np.ndarray) -> str:
    """Plain-text confusion matrix with word labels, rows = truth."""
    header = "| truth \\ pred | " + " | ".join(WORDS) + " | total |"
    separator = "| --- | " + " | ".join("---" for _ in WORDS) + " | --- |"
    lines = [header, separator]
    for position, word in enumerate(WORDS):
        row = matrix[position]
        cells = " | ".join(str(int(value)) for value in row)
        lines.append(f"| **{word}** | {cells} | {int(row.sum())} |")
    totals = " | ".join(str(int(value)) for value in matrix.sum(axis=0))
    lines.append(f"| **total** | {totals} | {int(matrix.sum())} |")
    return "\n".join(lines)


def build_report(
    *,
    rows: pd.DataFrame,
    data_dir: str,
    speaker_label: str,
    source: str,
    results_by_tag: dict[str, list[ClipResult]],
    metadata_by_tag: dict[str, dict],
    summaries: dict[str, dict],
    frames_by_id: dict[str, ClipFrames],
    device: str,
    extraction_seconds: float,
    output_dir: Path,
) -> dict:
    """Assemble the comparison report payload and write the artefacts."""
    rejected = [clip for clip in frames_by_id.values() if clip.frames.size == 0]
    detection_rates = [clip.detection_rate for clip in frames_by_id.values() if clip.frames.size]
    frame_counts = [int(clip.frames.shape[0]) for clip in frames_by_id.values() if clip.frames.size]
    tags = list(results_by_tag)

    headline = []
    for tag in tags:
        summary = summaries[tag]
        meta = metadata_by_tag[tag]
        selection = meta.get("selection_accuracy")
        headline.append(
            {
                "model": tag,
                "architecture": meta["architecture"],
                "note": meta["note"],
                "clips": summary["predicted"],
                "correct": summary["correct"],
                "accuracy": summary["accuracy"],
                "accuracy_ci95": summary["accuracy_ci95"],
                "macro_per_word_accuracy": float(
                    np.mean([value["accuracy"] for value in summary["per_word"].values()])
                ),
                "mean_confidence": summary["mean_confidence"],
                "checkpoint": meta["path"],
                "checkpoint_epoch": meta["epoch"],
                "checkpoint_selection_split": meta["selection_split"],
                "checkpoint_selection_accuracy": selection,
                "checkpoint_speaker_holdout": meta["speaker_holdout"],
                "checkpoint_written_at": meta["written_at"],
                "checkpoint_provenance_file": meta["provenance_file"],
            }
        )

    # Paired comparison between the first two models (CTC vs transformer).
    paired = None
    if len(tags) >= 2:
        first, second = tags[0], tags[1]
        lookup_first = {result.sample_id: result for result in results_by_tag[first]}
        lookup_second = {result.sample_id: result for result in results_by_tag[second]}
        both_right = both_wrong = first_only = second_only = 0
        for sample_id, result in lookup_first.items():
            other = lookup_second.get(sample_id)
            if other is None or result.status != "predicted" or other.status != "predicted":
                continue
            if result.correct and other.correct:
                both_right += 1
            elif result.correct:
                first_only += 1
            elif other.correct:
                second_only += 1
            else:
                both_wrong += 1
        paired = {
            "first": first,
            "second": second,
            "both_correct": both_right,
            "both_wrong": both_wrong,
            f"{first}_only_correct": first_only,
            f"{second}_only_correct": second_only,
            "agreement": (both_right + both_wrong) / max(both_right + both_wrong + first_only + second_only, 1),
            "mcnemar_exact_p": mcnemar_exact(first_only, second_only),
        }

    # Recording window of the evaluated clips, used for the leakage check.
    recorded_from = recorded_to = None
    if "captured_at" in rows.columns:
        stamps = pd.to_datetime(rows["captured_at"], errors="coerce", utc=True).dropna()
        if not stamps.empty:
            recorded_from = stamps.min().isoformat(timespec="seconds")
            recorded_to = stamps.max().isoformat(timespec="seconds")
    for entry in headline:
        written = entry["checkpoint_written_at"]
        entry["predates_evaluated_clips"] = bool(
            recorded_from and written and written < recorded_from
        )

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "data_dir": data_dir,
            "speakers": speaker_label,
            "clips": len(rows),
            "labels": list(WORDS),
            "source": source,
            "device": device,
            "extraction_seconds": round(extraction_seconds, 1),
            "rejected_clips": len(rejected),
            "mean_detection_rate": float(np.mean(detection_rates)) if detection_rates else 0.0,
            "frame_counts": {
                "min": int(np.min(frame_counts)) if frame_counts else 0,
                "median": int(np.median(frame_counts)) if frame_counts else 0,
                "max": int(np.max(frame_counts)) if frame_counts else 0,
            },
            "recorded_from": recorded_from,
            "recorded_to": recorded_to,
        },
        "models": headline,
        "paired_comparison": paired,
        "per_word_accuracy": {
            tag: {word: value["accuracy"] for word, value in summaries[tag]["per_word"].items()}
            for tag in tags
        },
        "confusion_matrices": {tag: summaries[tag]["confusion_matrix"] for tag in tags},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for tag in tags:
        write_results_csv(results_by_tag[tag], output_dir / f"predictions_{tag}.csv", source=source)

    # ---- confusion matrix figures ----
    for tag in tags:
        figure, axis = plt.subplots(figsize=(6.4, 5.6))
        render_confusion(
            axis,
            np.asarray(summaries[tag]["confusion_matrix"]),
            f"{tag} — accuracy {summaries[tag]['accuracy']:.1%} "
            f"({summaries[tag]['correct']}/{summaries[tag]['predicted']})",
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_{tag}.png", dpi=150)
        plt.close(figure)

    if len(tags) >= 2:
        figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))
        for axis, tag in zip(axes, tags[:2]):
            render_confusion(
                axis,
                np.asarray(summaries[tag]["confusion_matrix"]),
                f"{tag} — {summaries[tag]['accuracy']:.1%}",
            )
        figure.suptitle(
            f"Lip reading on {len(rows)} raw MP4 clips ({speaker_label}) — live mouth-crop pipeline",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
        figure.savefig(output_dir / "confusion_side_by_side.png", dpi=150)
        plt.close(figure)

    # ---- markdown report ----
    lines = [
        "# Lip-reading model comparison on raw MP4 clips",
        "",
        f"- Generated: {payload['generated_at']}",
        f"- Clips: **{len(rows)}** raw MP4 files from `{data_dir}/raw/{speaker_label}/` "
        f"({len(WORDS)} words)",
        f"- Pipeline: full-face video decoded and mouth-cropped by the live "
        f"`processing.mouth_cropper` path, then normalised per model family",
        f"- Device: `{device}` | mean fresh-detection rate "
        f"{payload['dataset']['mean_detection_rate']:.1%} | "
        f"clips rejected by the live quality gates: {len(rejected)}",
        f"- Mouth frames per clip after cropping: "
        f"{payload['dataset']['frame_counts']['min']}–"
        f"{payload['dataset']['frame_counts']['max']} "
        f"(median {payload['dataset']['frame_counts']['median']}), the same "
        f"geometry the models were trained on",
        "",
        "## Headline scores",
        "",
        "| model | architecture | clips | correct | accuracy | 95% CI | macro per-word | mean confidence |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in headline:
        low, high = entry["accuracy_ci95"]
        lines.append(
            f"| `{entry['model']}` | {entry['architecture']} | {entry['clips']} | "
            f"{entry['correct']} | **{entry['accuracy']:.1%}** | "
            f"{low:.1%}–{high:.1%} | {entry['macro_per_word_accuracy']:.1%} | "
            f"{entry['mean_confidence']:.1%} |"
        )
    lines += ["", "### Checkpoints", "",
              "| model | checkpoint | epoch | split recorded | selection accuracy | held-out speaker |",
              "| --- | --- | --- | --- | --- | --- |"]
    for entry in headline:
        selection = entry["checkpoint_selection_accuracy"]
        selection_text = f"{selection:.4f}" if isinstance(selection, (int, float)) else "n/a"
        lines.append(
            f"| `{entry['model']}` | `{entry['checkpoint']}` | {entry['checkpoint_epoch']} | "
            f"{entry['checkpoint_selection_split'] or 'n/a'} | {selection_text} | "
            f"{entry['checkpoint_speaker_holdout'] or 'not recorded'} |"
        )

    lines += ["", "### Training-data leakage check", ""]
    if recorded_from and recorded_to:
        lines.append(f"The evaluated clips were recorded between {recorded_from} and {recorded_to}.")
        lines.append("")
    for entry in headline:
        written = entry["checkpoint_written_at"]
        if entry["predates_evaluated_clips"]:
            lines.append(
                f"- `{entry['model']}`: checkpoint written {written}, **before the earliest "
                f"evaluated clip** — it cannot have trained on any of them."
            )
        elif entry["checkpoint_provenance_file"]:
            lines.append(
                f"- `{entry['model']}`: checkpoint written {written}, after the clips were "
                f"recorded, so file timestamps prove nothing. Its split is verified separately "
                f"in `{entry['checkpoint_provenance_file']}`."
            )
        elif entry["checkpoint_speaker_holdout"]:
            lines.append(
                f"- `{entry['model']}`: trained with `{entry['checkpoint_speaker_holdout']}` held "
                f"out, which is the evaluated speaker — no leakage."
            )
        else:
            lines.append(
                f"- `{entry['model']}`: checkpoint written {written}; the file records no split, "
                f"so its score should be treated with caution."
            )

    if paired:
        first, second = paired["first"], paired["second"]
        first_only = paired.get(first + "_only_correct")
        second_only = paired.get(second + "_only_correct")
        lines += [
            "",
            "## Paired comparison",
            "",
            f"- Both models correct: **{paired['both_correct']}**",
            f"- Both models wrong: **{paired['both_wrong']}**",
            f"- Only `{first}` correct: **{first_only}**",
            f"- Only `{second}` correct: **{second_only}**",
            f"- Prediction agreement: **{paired['agreement']:.1%}**",
            f"- Exact McNemar p-value: **{paired['mcnemar_exact_p']:.4f}**",
        ]

    lines += ["", "## Per-word accuracy", "",
              "| word | " + " | ".join(f"`{tag}`" for tag in tags) + " |",
              "| --- | " + " | ".join("---" for _ in tags) + " |"]
    for word in WORDS:
        cells = []
        for tag in tags:
            value = summaries[tag]["per_word"].get(word)
            cells.append(
                f"{value['accuracy']:.0%} ({value['correct']}/{value['total']})" if value else "n/a"
            )
        lines.append(f"| **{word}** | " + " | ".join(cells) + " |")

    for tag in tags:
        lines += [
            "",
            f"## Confusion matrix — `{tag}` ({summaries[tag]['accuracy']:.1%})",
            "",
            confusion_table_text(np.asarray(summaries[tag]["confusion_matrix"])),
            "",
            f"![confusion {tag}](confusion_{tag}.png)",
        ]

    if rejected:
        lines += ["", "## Clips rejected by the live quality gates", ""]
        for clip in rejected[:10]:
            lines.append(f"- `{clip.sample_id}`: {clip.reason}")

    lines += [
        "",
        "## How to read this",
        "",
        "- Accuracy counts clips the live quality gates accepted; rejected clips are "
        "excluded from the percentage and listed above.",
        "- The 95% interval is a Wilson score interval, and the McNemar p-value tests "
        "the two models on the *same* clips (paired), which is the right test here.",
        "- Check the leakage section above before trusting a score: a checkpoint trained on "
        "a per-clip random split has seen this speaker, so its accuracy is a same-speaker "
        "upper bound rather than a generalisation number.",
        "",
    ]
    (output_dir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--speakers",
        default="speaker_test_samples",
        help="Comma-separated speaker ids whose raw MP4s are evaluated "
        "(default: speaker_test_samples).",
    )
    parser.add_argument("--ctc-checkpoint", type=Path, default=DEFAULT_CTC_CHECKPOINT)
    parser.add_argument("--transformer-checkpoint", type=Path, default=DEFAULT_TRANSFORMER_CHECKPOINT)
    parser.add_argument(
        "--extra-checkpoint",
        action="append",
        default=[],
        metavar="[TAG=]PATH",
        help="Additional checkpoint to score over the same frames (repeatable).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--refresh-frames",
        action="store_true",
        help="Re-decode every video instead of reusing the cached mouth frames.",
    )
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--min-detection-rate", type=float, default=0.75)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N clips.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = resolve(args.data_dir)
    output_dir = resolve(args.output_dir)
    cache_dir = resolve(args.cache_dir)
    speakers = [name.strip() for name in args.speakers.split(",") if name.strip()]

    specs = parse_model_specs(args.ctc_checkpoint, args.transformer_checkpoint, args.extra_checkpoint)

    manifest = load_manifest(data_dir)
    rows = manifest[manifest["speaker_id"].astype(str).isin(speakers)].reset_index(drop=True)
    rows = rows[rows["raw_video_path"].notna()].reset_index(drop=True)
    if args.limit:
        rows = rows.head(args.limit).reset_index(drop=True)
    if rows.empty:
        print(f"No raw MP4 clips found for {speakers} in {data_dir}", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Comparing {len(specs)} checkpoints on {len(rows)} raw MP4 clips from {speakers}")
    for spec in specs:
        print(f"  {spec.tag:>12}: {spec.path.relative_to(PROJECT_ROOT)}")
    print(f"Device: {device}")

    print("\nStep 1/2 — mouth-cropping every raw video once through the live pipeline")
    extraction_started = time.perf_counter()
    frames_by_id = extract_frames(
        rows,
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        min_detection_rate=args.min_detection_rate,
        refresh=args.refresh_frames,
    )
    extraction_seconds = time.perf_counter() - extraction_started
    rejected = sum(1 for clip in frames_by_id.values() if clip.frames.size == 0)
    print(f"  frames ready in {extraction_seconds:.0f}s ({rejected} rejected)")

    print("\nStep 2/2 — scoring each checkpoint on the identical frames")
    results_by_tag: dict[str, list[ClipResult]] = {}
    metadata_by_tag: dict[str, dict] = {}
    summaries: dict[str, dict] = {}
    for spec in specs:
        results, metadata = predict_frames(
            spec, rows, frames_by_id, device=args.device, max_frames=args.max_frames
        )
        summary = model_summary(results)
        results_by_tag[spec.tag] = results
        metadata_by_tag[spec.tag] = metadata
        summaries[spec.tag] = summary
        print(
            f"  {spec.tag:>12}: accuracy {summary['accuracy']:.4f} "
            f"({summary['correct']}/{summary['predicted']}) "
            f"[{metadata['architecture']}, epoch {metadata['epoch']}]",
            flush=True,
        )

    payload = build_report(
        rows=rows,
        data_dir=str(args.data_dir),
        speaker_label=",".join(speakers),
        source="raw mp4 (live mouth-crop pipeline)",
        results_by_tag=results_by_tag,
        metadata_by_tag=metadata_by_tag,
        summaries=summaries,
        frames_by_id=frames_by_id,
        device=str(device),
        extraction_seconds=extraction_seconds,
        output_dir=output_dir,
    )

    # ---- console summary ----
    print("\nConfusion matrices (rows=truth, cols=prediction):")
    for tag in results_by_tag:
        matrix = np.asarray(summaries[tag]["confusion_matrix"])
        print(f"\n{tag} — accuracy {summaries[tag]['accuracy']:.4f} "
              f"({summaries[tag]['correct']}/{summaries[tag]['predicted']})")
        print("        " + " ".join(f"{word[:4]:>4}" for word in WORDS))
        for position, word in enumerate(WORDS):
            print(f"{word:>8} " + " ".join(f"{value:4d}" for value in matrix[position]))

    if payload["paired_comparison"]:
        paired = payload["paired_comparison"]
        print(
            f"\nPaired: both correct {paired['both_correct']}, both wrong {paired['both_wrong']}, "
            f"agreement {paired['agreement']:.1%}, McNemar p={paired['mcnemar_exact_p']:.4f}"
        )

    print(f"\nReport: {output_dir.relative_to(PROJECT_ROOT)}/comparison.md")
    print(f"JSON:   {output_dir.relative_to(PROJECT_ROOT)}/comparison.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
