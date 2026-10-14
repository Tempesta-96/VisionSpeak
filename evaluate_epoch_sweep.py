"""Score the epoch-sweep checkpoints on the held-out speaker's raw MP4 clips.

This evaluates one grid: every architecture against every training budget, on
the *same* mouth frames, so the resulting table is a like-for-like comparison.
By default that is four architectures (CTC+BiGRU, CTC+Transformer, Word+BiGRU,
Word+Transformer) at 50, 100 and 200 epochs, which is the twelve checkpoints
``scripts/retrain_lip_models.sh`` writes as ``<arch>_ep<epoch>.pt``.

The headline output is a 3x4 table of overall test accuracy in percent: rows are
epoch budgets, columns are architectures.

Every raw video is decoded and mouth-cropped once through the live MediaPipe
pipeline (``processing.mouth_cropper``, the same code the webcam tools use) and
the frames are cached under ``.cache/lip_mp4_frames``, so a second run only pays
for the forward passes. Cropping and scoring are both delegated to
``compare_lip_models`` so this table and that report cannot drift apart.

The clips come from ``--speakers`` in ``--data-dir``, so pointing the sweep at a
different held-out speaker is a matter of recording that speaker, retraining
with ``TEST_SPEAKER`` set to them, and naming them here. Every clip needs its
raw video on disk: this path always decodes the recording.

Two guards make the table trustworthy rather than merely plausible:

* a checkpoint whose recorded ``epoch`` disagrees with its grid position is a
  hard error, because that is how a wrong number lands in a row;
* a checkpoint that did not hold out the evaluated speaker is a hard error,
  because its accuracy would not be a held-out number at all. Pass
  ``--allow-speaker-mismatch`` to override when you know why.

Examples::

    .venv/bin/python evaluate_epoch_sweep.py
    .venv/bin/python evaluate_epoch_sweep.py --device cpu
    .venv/bin/python evaluate_epoch_sweep.py --epochs 50,200 --refresh-frames

    # A different held-out speaker: record them, retrain with
    # TEST_SPEAKER=<id> and a separate OUT directory, then sweep those
    # checkpoints over that speaker's recordings.
    .venv/bin/python evaluate_epoch_sweep.py \
        --speakers speaker_008 \
        --checkpoint-dir checkpoints/retrain_speaker_008 \
        --output-dir reports/retrain_speaker_008 --stem epoch_sweep_speaker_008
"""

from __future__ import annotations
import os

# Same environment preparation compare_lip_models does, before it is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch

from compare_lip_models import (
    DEFAULT_CACHE_DIR,
    PROJECT_ROOT,
    ModelSpec,
    extract_frames,
    model_summary,
    predict_frames,
    resolve,
)
from custom_words_dataset import load_manifest

#: ``(key, display label)`` in the column order the table is printed in.
ARCHITECTURES: tuple[tuple[str, str], ...] = (
    ("ctc_bigru", "CTC + BiGRU"),
    ("ctc_transformer", "CTC + Transformer"),
    ("word_bigru", "Word + BiGRU"),
    ("word_transformer_attention", "Word + Transformer"),
)

#: Training budgets, in the row order the table is printed in.
EPOCH_BUDGETS: tuple[int, ...] = (50, 100, 200)

#: Checkpoint filename for one grid cell.
CHECKPOINT_TEMPLATE = "{arch}_ep{epoch}.pt"

#: Mirrors ``compare_lip_models.cache_path``: the quality gates are part of the
#: cache key, so the reported value has to be the gates actually used. Kept in
#: step by ``tests/test_evaluate_epoch_sweep.py``.
CACHE_KEY_TEMPLATE = "{{sample_id}}__min{min_frames}_rate{min_detection_rate:g}.npz"

DEFAULT_CHECKPOINT_DIR = Path("checkpoints/retrain")
DEFAULT_OUTPUT_DIR = Path("reports/retrain")
DEFAULT_SPEAKER = "speaker_test_samples"
DEFAULT_MAX_FRAMES = 90
DEFAULT_MIN_FRAMES = 30
DEFAULT_MIN_DETECTION_RATE = 0.75


class SweepError(RuntimeError):
    """The grid cannot be evaluated as asked, so any table would mislead."""


def build_grid(
    checkpoint_dir: Path,
    epochs: tuple[int, ...] = EPOCH_BUDGETS,
    architectures: tuple[tuple[str, str], ...] = ARCHITECTURES,
) -> list[list[ModelSpec]]:
    """Expand the epoch/architecture grid into concrete checkpoint specs.

    Returns one row per epoch, each holding one spec per architecture.

    Raises:
        SweepError: If any cell's checkpoint file is missing, naming every one
            so a partial run cannot masquerade as a full table.
    """
    grid: list[list[ModelSpec]] = []
    missing: list[str] = []

    for epoch in epochs:
        row: list[ModelSpec] = []
        for key, label in architectures:
            path = resolve(checkpoint_dir / CHECKPOINT_TEMPLATE.format(arch=key, epoch=epoch))
            if not path.is_file():
                missing.append(f"  {label} @ {epoch} epochs  ->  {path}")
            row.append(ModelSpec(tag=f"{key}_ep{epoch}", path=path, note=f"{label}, {epoch} epochs"))
        grid.append(row)

    if missing:
        raise SweepError(
            "Missing checkpoint(s) for the requested grid:\n"
            + "\n".join(missing)
            + "\n\nProduce them with, for example:\n"
            "  EPOCHS_WORD=<n> EPOCHS_CTC=<n> TAG=ep<n> scripts/retrain_lip_models.sh"
        )
    return grid


def validate_grid(
    grid: list[list[ModelSpec]],
    epochs: tuple[int, ...] = EPOCH_BUDGETS,
    architectures: tuple[tuple[str, str], ...] = ARCHITECTURES,
    *,
    expected_speaker: str = DEFAULT_SPEAKER,
    allow_speaker_mismatch: bool = False,
) -> list[str]:
    """Check each checkpoint agrees with the cell it is about to fill.

    Returns non-fatal warnings. A recorded ``epoch`` that contradicts the row,
    or a recorded ``test_speaker`` that is not the speaker being evaluated, is
    fatal: the first would mislabel a row and the second would report a number
    that is not held out.
    """
    problems: list[str] = []
    warnings: list[str] = []

    for row_index, row in enumerate(grid):
        epoch = epochs[row_index]
        for column_index, spec in enumerate(row):
            _, label = architectures[column_index]
            checkpoint = torch.load(spec.path, map_location="cpu", weights_only=True)

            recorded_epoch = checkpoint.get("epoch")
            if recorded_epoch is not None and int(recorded_epoch) != int(epoch):
                problems.append(
                    f"{spec.path.name} records epoch {recorded_epoch} but the table "
                    f"places it in the {epoch}-epoch row ({label})."
                )

            recorded_speaker = checkpoint.get("test_speaker")
            if recorded_speaker is None:
                warnings.append(
                    f"{spec.path.name} does not record test_speaker, so it cannot be "
                    f"confirmed that it held out {expected_speaker!r}."
                )
            elif str(recorded_speaker) != expected_speaker and not allow_speaker_mismatch:
                problems.append(
                    f"{spec.path.name} records test speaker {recorded_speaker!r}, but this "
                    f"run evaluates {expected_speaker!r}; its accuracy would not be held "
                    "out. Pass --allow-speaker-mismatch to evaluate it anyway."
                )

    if problems:
        raise SweepError("Refusing to build a misleading table:\n" + "\n".join(f"  {p}" for p in problems))
    return warnings


def format_percent(fraction: float) -> str:
    """Render a 0..1 fraction as a percentage with two decimals."""
    return f"{100.0 * fraction:.2f}%"


def render_grid(cell_for, epochs=EPOCH_BUDGETS, architectures=ARCHITECTURES) -> list[list[str]]:
    """Build a text grid, including its header row and left-hand epoch column."""
    header = ["epoch"] + [label for _, label in architectures]
    rows = [header]
    for epoch in epochs:
        rows.append([str(epoch)] + [cell_for(epoch, key) for key, _ in architectures])
    return rows


def format_table(rows: list[list[str]]) -> str:
    """Left-align the label column and right-align every numeric column."""
    widths = [max(len(row[index]) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for position, row in enumerate(rows):
        cells = [
            cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])
            for index, cell in enumerate(row)
        ]
        lines.append("  ".join(cells).rstrip())
        if position == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)


def format_markdown(rows: list[list[str]]) -> str:
    """Render a text grid as a markdown table.

    Takes the same input as :func:`format_table` — header first, no separator
    row — and generates the markdown separator itself.
    """
    header, *body = rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


def write_artifacts(
    summaries: dict[tuple[int, str], dict],
    output_dir: Path,
    stem: str,
    *,
    epochs: tuple[int, ...] = EPOCH_BUDGETS,
    architectures: tuple[tuple[str, str], ...] = ARCHITECTURES,
    per_clip: list[dict] | None = None,
    context: dict | None = None,
) -> tuple[Path, Path, Path]:
    """Write the accuracy table as markdown, the tidy scores as CSV, and the per-clip rows.

    Returns the three paths written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    accuracy_grid = render_grid(
        lambda epoch, key: format_percent(summaries[(epoch, key)]["accuracy"]), epochs, architectures
    )
    count_grid = render_grid(
        lambda epoch, key: f"{summaries[(epoch, key)]['correct']}/{summaries[(epoch, key)]['predicted']}",
        epochs,
        architectures,
    )

    markdown_path = output_dir / f"{stem}.md"
    lines = ["# Epoch sweep on raw MP4 clips", ""]
    if context:
        lines.extend(f"- {key}: {value}" for key, value in context.items())
        lines.append("")
    lines.extend(["## Overall test accuracy", "", format_markdown(accuracy_grid), ""])
    lines.extend(["## Clips correct", "", format_markdown(count_grid), ""])
    markdown_path.write_text("\n".join(lines))

    csv_path = output_dir / f"{stem}.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["architecture", "epoch", "correct", "predicted", "accuracy", "accuracy_ci95_low", "accuracy_ci95_high"]
        )
        for epoch in epochs:
            for key, label in architectures:
                summary = summaries[(epoch, key)]
                low, high = summary["accuracy_ci95"]
                writer.writerow(
                    [label, epoch, summary["correct"], summary["predicted"],
                     f"{summary['accuracy']:.6f}", f"{low:.6f}", f"{high:.6f}"]
                )

    per_clip_path = output_dir / f"{stem}_predictions.csv"
    with per_clip_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["architecture", "epoch", "sample_id", "label", "predicted", "confidence", "correct", "status"],
        )
        writer.writeheader()
        for row in per_clip or []:
            writer.writerow(row)

    return markdown_path, csv_path, per_clip_path


def parse_epochs(raw: str) -> tuple[int, ...]:
    """Parse a comma-separated epoch list into a non-empty tuple of ints."""
    epochs = tuple(int(part) for part in raw.replace(" ", "").split(",") if part)
    if not epochs:
        raise argparse.ArgumentTypeError("at least one epoch budget is required")
    return epochs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/custom_words"))
    parser.add_argument("--speakers", default=DEFAULT_SPEAKER,
                        help=f"Comma-separated speaker ids whose clips are evaluated (default: {DEFAULT_SPEAKER}).")
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR,
                        help="Directory holding <arch>_ep<epoch>.pt checkpoints.")
    parser.add_argument("--epochs", type=parse_epochs, default=EPOCH_BUDGETS,
                        help="Comma-separated epoch budgets to tabulate (default: 50,100,200).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="epoch_sweep_mp4", help="Base name for the written artefacts.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--refresh-frames", action="store_true",
                        help="Re-decode every video instead of reusing the cached mouth frames.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    parser.add_argument("--min-detection-rate", type=float, default=DEFAULT_MIN_DETECTION_RATE)
    parser.add_argument("--allow-speaker-mismatch", action="store_true",
                        help="Evaluate checkpoints even if they did not hold out --speakers.")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N clips.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data_dir = Path(args.data_dir).expanduser().resolve()
    cache_dir = resolve(args.cache_dir)
    speakers = [name.strip() for name in args.speakers.split(",") if name.strip()]
    expected_speaker = speakers[0] if len(speakers) == 1 else args.speakers

    try:
        grid = build_grid(args.checkpoint_dir, args.epochs, ARCHITECTURES)
        warnings = validate_grid(
            grid,
            args.epochs,
            ARCHITECTURES,
            expected_speaker=expected_speaker,
            allow_speaker_mismatch=args.allow_speaker_mismatch,
        )
    except SweepError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)

    manifest = load_manifest(data_dir)
    rows = manifest[manifest["speaker_id"].astype(str).isin(speakers)].reset_index(drop=True)
    rows = rows[rows["raw_video_path"].notna()].reset_index(drop=True)
    if args.limit:
        rows = rows.head(args.limit).reset_index(drop=True)
    if rows.empty:
        print(f"error: no raw MP4 clips for speaker(s) {speakers} in {data_dir}", file=sys.stderr)
        return 1

    print(f"Clips: {len(rows)} raw MP4 files for {', '.join(speakers)}")
    print(f"Grid:  {len(args.epochs)} epoch budgets x {len(ARCHITECTURES)} architectures "
          f"= {len(args.epochs) * len(ARCHITECTURES)} checkpoints")
    frames_by_id = extract_frames(
        rows,
        data_dir=data_dir,
        cache_dir=cache_dir,
        max_frames=args.max_frames,
        min_frames=args.min_frames,
        min_detection_rate=args.min_detection_rate,
        refresh=args.refresh_frames,
    )

    summaries: dict[tuple[int, str], dict] = {}
    per_clip: list[dict] = []
    for row_index, row_specs in enumerate(grid):
        epoch = args.epochs[row_index]
        for column_index, spec in enumerate(row_specs):
            key, label = ARCHITECTURES[column_index]
            results, metadata = predict_frames(
                spec, rows, frames_by_id, device=args.device, max_frames=args.max_frames
            )
            summary = model_summary(results)
            summaries[(epoch, key)] = summary
            per_clip.extend(
                {
                    "architecture": label,
                    "epoch": epoch,
                    "sample_id": result.sample_id,
                    "label": result.label,
                    "predicted": result.predicted,
                    "confidence": f"{result.confidence:.6f}",
                    "correct": result.correct,
                    "status": result.status,
                }
                for result in results
            )
            print(
                f"  [{row_index + 1}/{len(grid)}] {label:<18} @ {epoch:>3} epochs  "
                f"{format_percent(summary['accuracy']):>7}  "
                f"({summary['correct']}/{summary['predicted']}, {metadata['device']})",
                flush=True,
            )

    accuracy_grid = render_grid(
        lambda epoch, key: format_percent(summaries[(epoch, key)]["accuracy"]), args.epochs
    )
    count_grid = render_grid(
        lambda epoch, key: f"{summaries[(epoch, key)]['correct']}/{summaries[(epoch, key)]['predicted']}",
        args.epochs,
    )

    print("\nOverall test accuracy")
    print(format_table(accuracy_grid))
    print("\nClips correct")
    print(format_table(count_grid))

    rejected = {
        epoch: sum(
            summaries[(epoch, key)]["status_counts"].get("rejected", 0)
            + summaries[(epoch, key)]["status_counts"].get("error", 0)
            for key, _ in ARCHITECTURES
        )
        for epoch in args.epochs
    }
    total_rejected = sum(rejected.values())
    if total_rejected:
        print(
            f"\nNote: {total_rejected} model-clip pair(s) were rejected by the live quality "
            "gates or raised, and are excluded from the accuracy denominator."
        )
    else:
        print(f"\nAll {len(rows)} clips were usable for every checkpoint (no rejected clips).")

    markdown_path, csv_path, per_clip_path = write_artifacts(
        summaries,
        resolve(args.output_dir),
        args.stem,
        epochs=args.epochs,
        per_clip=per_clip,
        context={
            "Speaker held out": ", ".join(speakers),
            "Clips": len(rows),
            "Frames per clip": f"max {args.max_frames}, min {args.min_frames}, "
                               f"min detection rate {args.min_detection_rate:g}",
            "Checkpoints": str(args.checkpoint_dir),
            "Cache key": CACHE_KEY_TEMPLATE.format(
                min_frames=args.min_frames, min_detection_rate=args.min_detection_rate
            ),
        },
    )
    print(f"\nMarkdown: {markdown_path.relative_to(PROJECT_ROOT)}")
    print(f"CSV:      {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"Per-clip: {per_clip_path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
