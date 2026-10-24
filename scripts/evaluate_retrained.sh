#!/usr/bin/env bash
#
# Score every lip-reading checkpoint on the held-out test speaker's mouth-crop
# clips, through the deployed inference path (inference.lip_worker.LipPredictor),
# so the numbers are comparable across model families.
#
# The four checkpoints under checkpoints/retrain/ are this repository's retrain
# on all 640 non-test clips. The reference checkpoints are listed too: they were
# trained under weaker protocols (per-clip splits that leak speakers, or 512 of
# the 640 clips), and scoring them on the same 160 clips is what makes the
# "did retraining help?" question answerable.
#
# Usage:  scripts/evaluate_retrained.sh
#
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
export KMP_DUPLICATE_LIB_OK=TRUE

PY="${PY:-.venv/bin/python}"
SPEAKER="${SPEAKER:-speaker_test_samples}"
DEVICE="${DEVICE:-auto}"
OUT="${OUT:-checkpoints/retrain}"
LOGS="${LOGS:-reports/retrain}"
mkdir -p "$LOGS"

# tag:checkpoint pairs. Missing files are skipped with a note rather than failing
# the whole sweep, so a partial retrain still yields the numbers it earned.
CANDIDATES=(
  "retrain_word_bigru:$OUT/word_bigru.pt"
  "retrain_word_transformer_attention:$OUT/word_transformer_attention.pt"
  "retrain_ctc_bigru:$OUT/ctc_bigru.pt"
  "retrain_ctc_transformer:$OUT/ctc_transformer.pt"
  "ref_word_bigru:checkpoints/custom_words_speaker_holdout.pt"
  "ref_word_transformer:checkpoints/lip_transformer/transformer-attention-holdout-speaker_test_samples/epoch020_valid0.5234.pt"
  "ref_ctc_bigru:checkpoints/valente_custom_words_lipnet_ctc_0.852_acc.pt"
)

status=0
for entry in "${CANDIDATES[@]}"; do
  tag="${entry%%:*}"
  checkpoint="${entry#*:}"
  log="$LOGS/eval_$tag.log"

  if [[ ! -f "$checkpoint" ]]; then
    echo "### $tag: SKIPPED (no $checkpoint)"
    continue
  fi

  echo "############ $tag ############"
  if "$PY" evaluation/lip-reading/evaluate_custom_words.py \
      --source npz \
      --speakers "$SPEAKER" \
      --checkpoint "$checkpoint" \
      --device "$DEVICE" \
      --output-csv "$LOGS/preds_$tag.csv" > "$log" 2>&1; then
    grep -E "Overall accuracy|accuracy:|Confusion|Macro|samples" "$log" | head -20
    echo "### $tag: OK -> $log"
  else
    echo "### $tag: FAILED -> $log"
    tail -5 "$log"
    status=1
  fi
done

exit "$status"
