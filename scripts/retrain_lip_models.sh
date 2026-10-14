#!/usr/bin/env bash
#
# Retrain the four lip-reading configurations on every clip from the non-test
# speakers, holding one speaker out entirely as the test set.
#
# Protocol (see the README's "the split matters more than the architecture"):
#   * train : all clips from every speaker except TEST_SPEAKER
#   * test  : TEST_SPEAKER only, never read until training ends
#   * `--all-training-data` drops the validation carve-out, so no clip is spent
#     on model selection; the final epoch is the deliverable.
#
# Usage:
#   scripts/retrain_lip_models.sh                 # full budget
#   EPOCHS_WORD=2 EPOCHS_CTC=2 scripts/retrain_lip_models.sh   # smoke test
#
# Overridable environment:
#   EPOCHS_WORD (50)   EPOCHS_CTC (50)   BATCH_WORD (32)   BATCH_CTC (16)
#   TEST_SPEAKER (speaker_test_samples)   DEVICE (auto)     TAG (run label)
#
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mplcache}"
export KMP_DUPLICATE_LIB_OK=TRUE

EPOCHS_WORD="${EPOCHS_WORD:-50}"
EPOCHS_CTC="${EPOCHS_CTC:-50}"
BATCH_WORD="${BATCH_WORD:-32}"
BATCH_CTC="${BATCH_CTC:-16}"
TEST_SPEAKER="${TEST_SPEAKER:-speaker_test_samples}"
DEVICE="${DEVICE:-auto}"
TAG="${TAG:-}"
PY="${PY:-.venv/bin/python}"

SUFFIX="${TAG:+_$TAG}"
OUT="${OUT:-checkpoints/retrain}"
LOGS="${LOGS:-reports/retrain}"
mkdir -p "$OUT" "$LOGS"

status=0

# run <name> <command...>
run() {
  local name="$1"; shift
  local log="$LOGS/$name.log"
  echo "############ $name ############"
  echo "# $*" | tee "$log"
  local started finished
  started=$(date +%s)
  if "$@" 2>&1 | tee -a "$log"; then
    echo "### $name: OK"
  else
    echo "### $name: FAILED (see $log)"
    status=1
  fi
  finished=$(date +%s)
  echo "### $name: $(( (finished - started) / 60 ))m $(( (finished - started) % 60 ))s"
}

COMMON_WORD=(
  --all-training-data
  --speaker-holdout "$TEST_SPEAKER"
  --epochs "$EPOCHS_WORD"
  --batch-size "$BATCH_WORD"
)

COMMON_CTC=(
  --all-training-data
  --speaker-holdout "$TEST_SPEAKER"
  --epochs "$EPOCHS_CTC"
  --batch-size "$BATCH_CTC"
  --device "$DEVICE"
)

run "word_bigru$SUFFIX" "$PY" -m models.LBY.train_custom_words \
  --architecture bigru \
  "${COMMON_WORD[@]}" \
  --checkpoint "$OUT/word_bigru$SUFFIX.pt"

run "word_transformer_attention$SUFFIX" "$PY" -m models.LBY.train_custom_words \
  --architecture transformer --pooling attention \
  "${COMMON_WORD[@]}" \
  --checkpoint "$OUT/word_transformer_attention$SUFFIX.pt"

run "ctc_bigru$SUFFIX" "$PY" -m models.valente.train_custom_words \
  --architecture bigru \
  "${COMMON_CTC[@]}" \
  --checkpoint "$OUT/ctc_bigru$SUFFIX.pt"

run "ctc_transformer$SUFFIX" "$PY" -m models.valente.train_custom_words \
  --architecture transformer \
  "${COMMON_CTC[@]}" \
  --checkpoint "$OUT/ctc_transformer$SUFFIX.pt"

echo
echo "############ summary ############"
for name in "word_bigru$SUFFIX" "word_transformer_attention$SUFFIX" "ctc_bigru$SUFFIX" "ctc_transformer$SUFFIX"; do
  printf '%-34s %s\n' "$name" "$(ls -la "$OUT/$name.pt" 2>/dev/null | awk '{print $5" bytes"}' || echo MISSING)"
done
exit "$status"
