# VisionSpeak

VisionSpeak is a multimodal computer-vision prototype for converting visible
lip movements and hand gestures into text.

## Active lip-reading workflow

The active lip-reading experiments recognise these eight isolated words:

```text
hello, hi, yes, no, good, bad, thanks, bye
```

1. Record mouth-only training clips with `record_custom_word_dataset.ipynb`.
2. Load and split those clips with `prepare_custom_word_dataset.py`.
3. Train the isolated-word classifier with
   `python -m models.LBY.train_custom_words` (`--architecture transformer` or
   `--architecture bigru`), with the notebook version in
   `models/LBY/lip_transformer_training.ipynb` — which gives every run its own
   `checkpoints/lip_transformer/<run_name>/` directory so runs can be compared —
   or with the archived custom CTC experiment in
   `archive/kaggle_lip_reading/models/valente/custom_word_training.ipynb`,
   whose `ARCHITECTURE` setting selects the same two cores for the CTC model.
4. Test a trained checkpoint live with `evaluation/lip-reading/webcam_lip_model_comparison.ipynb`,
   or offline with `evaluation/lip-reading/evaluate_custom_words.py` — over recorded clips
   (`--source npz` / `--source mp4`) or a single raw MP4 recording
   (`--video path/to/clip.mp4`).
5. Compare two checkpoints head to head on the raw MP4 clips with
   `evaluation/lip-reading/compare_lip_models.py`, which mouth-crops every video once through the live
   pipeline and scores both models on the identical frames:

   ```bash
   .venv/bin/python evaluation/lip-reading/compare_lip_models.py
   ```

   It writes accuracy, a 95% Wilson interval, per-word accuracy, confusion
   matrices (text, JSON and PNG heatmaps) and an exact McNemar test of the two
   models to `reports/lip_comparison/`.

### Temporal cores: BiGRU and transformer

Both lip-reading models keep their convolutional frontend and expose a
selectable temporal core, so accuracy differences can be attributed to
attention versus recurrence rather than to a redesigned frontend:

| Model | Core | Where |
| --- | --- | --- |
| `models/valente/model.py` (`LipNet`, CTC) | `bigru` \| `transformer` | `architecture=` argument or `build_lipnet(checkpoint, ...)` |
| `models/LBY/model.py` (`LipReadingClassifier`) | `bigru` \| `transformer` | `architecture=` / `pooling=` argument |

`bigru` remains the default, so checkpoints trained before this experiment load
unchanged. A checkpoint records its own settings, and
`inference/lip_worker.py` rebuilds whatever architecture the file declares — you
do not tell the inference path which one it is loading.

The word classifier's head also records a `head_activation`. New runs use
`"gelu"`; a checkpoint that does not record the key rebuilds with the historical
ReLU, so older files keep predicting exactly what they used to. The distinction
matters: a ReLU whose input goes fully negative receives zero gradient and can
never recover, and because backprop through it is zero the GRU and the CNN
freeze with it. That is how the Word + BiGRU runs collapsed — every clip was
labelled with whichever class had the largest output bias. GELU has a non-zero
gradient for negative inputs, so that failure cannot be permanent.

```bash
# Run as a module from the repository root: the script imports the repository's
# top-level modules, so `python models/LBY/train_custom_words.py` fails.
# Transformer word-classifier experiment (the default core), served afterwards
# through the webcam CLI:
.venv/bin/python -m models.LBY.train_custom_words \
    --architecture transformer \
    --speaker-holdout speaker_test_samples \
    --epochs 100 \
    --checkpoint checkpoints/lip_transformer/transformer-attention-holdout-speaker_test_samples/best.pt

.venv/bin/python main.py --mode lip \
    --lip-checkpoint checkpoints/lip_transformer/transformer-attention-holdout-speaker_test_samples/best.pt

# The BiGRU core, same protocol, its own checkpoint directory:
.venv/bin/python -m models.LBY.train_custom_words \
    --architecture bigru \
    --speaker-holdout speaker_test_samples \
    --epochs 100 \
    --checkpoint checkpoints/lip_bigru/bigru-holdout-speaker_test_samples/best.pt
```

Always pass `--checkpoint`: without it the script writes to
`checkpoints/custom_words.pt` and overwrites that legacy checkpoint.

The transformer path strides time by 2 in the *first* convolutional block only.
That bound is not arbitrary: CTC requires `input_length >= 2 * target_length + 1`,
the longest word is `thanks` (6 characters), and the shortest recorded clip is
45 frames. The frontend must therefore leave at least 13 emissions, and it
leaves 23. `tests/test_lip_transformer.py` re-derives that inequality from the
manifest, so changing the stride fails the tests rather than silently breaking
the loss.

A note on reading the accuracy numbers: the default split in
`prepare_custom_word_dataset.stratified_split` is per-clip random, so **all four
speakers appear in both train and test**. Use `--speaker-holdout` for a number
that means something:

```bash
# Hold out one speaker entirely; auto-picks the most representative speaker
.venv/bin/python -m models.LBY.train_custom_words --architecture transformer --speaker-holdout

# Or name the speaker (this is the speaker whose raw MP4s are the video test set)
.venv/bin/python -m models.LBY.train_custom_words --architecture bigru \
    --speaker-holdout speaker_test_samples \
    --checkpoint checkpoints/lip_bigru/bigru-holdout-speaker_test_samples/best.pt
```

The `--legacy-milestones` flag restores the original `MultiStepLR([40, 60])`
schedule instead of the transformer-friendly cosine one; use it to reproduce the
older BiGRU runs. Omitting `--speaker-holdout` altogether gives the per-clip
random split, whose accuracy is a same-speaker upper bound only.

With `--speaker-holdout`, validation is drawn from the non-test speakers and the
checkpoint is selected on it, so the held-out speaker never influences model
choice.

### Spending every clip on training

`--all-training-data` (available on both trainers) drops that validation
carve-out: all 640 clips from the non-test speakers train and the final epoch is
the deliverable. The test speaker is not read until training ends, so nothing
selects on it — but there is also no early stopping, so the epoch count is a
commitment rather than a ceiling. Reach for it when you want the model to see
every available clip and can accept that trade:

```bash
# Word classifier: 640 training clips, scored once on the held-out speaker
.venv/bin/python -m models.LBY.train_custom_words \
    --architecture transformer --pooling attention \
    --all-training-data --speaker-holdout speaker_test_samples \
    --epochs 100 --checkpoint checkpoints/retrain/word_transformer_attention.pt

# CTC model: same protocol, through the command-line trainer
.venv/bin/python -m models.valente.train_custom_words \
    --architecture bigru \
    --all-training-data --speaker-holdout speaker_test_samples \
    --epochs 100 --checkpoint checkpoints/retrain/ctc_bigru.pt
```

`models/valente/train_custom_words.py` is the command-line counterpart of
the archived `archive/kaggle_lip_reading/models/valente/custom_word_training.ipynb`.
The notebook is fine for
exploration, but it splits clips at random — every speaker lands in train *and*
test — and then selects its checkpoint on that same set. The script uses the
speaker-disjoint protocol instead, and reports two test numbers that are not
interchangeable:

- **greedy** — CTC greedy decode counted correct only on an exact string match,
  so a near-miss like `helo` scores zero.
- **scored** — `inference.lip_worker.score_vocabulary`, the deployed path that
  ranks the eight known words by CTC sequence likelihood. This is the number to
  quote next to `evaluation/lip-reading/compare_lip_models.py` and the webcam tools.

### Batch training all four cores

`scripts/retrain_lip_models.sh` trains all four configurations in one run — the
word classifier and the CTC model, each with the BiGRU and the transformer
temporal core — using the all-training-data protocol above.

> **`speaker_test_samples` is the test data.** The script holds that speaker out
> entirely: the other four speakers' 640 clips train, and the held-out speaker's
> 160 clips are not read until training ends. No checkpoint selection touches
> them, so the accuracy a run reports is a generalisation number rather than a
> same-speaker one. `TEST_SPEAKER` changes which speaker is held out; see
> [Testing against a different speaker](#testing-against-a-different-speaker).

One invocation trains one epoch budget and writes four checkpoints, so a 50/100/200
sweep is three invocations:

```bash
EPOCHS_WORD=50  EPOCHS_CTC=50  TAG=ep50  scripts/retrain_lip_models.sh
EPOCHS_WORD=100 EPOCHS_CTC=100 TAG=ep100 scripts/retrain_lip_models.sh
EPOCHS_WORD=200 EPOCHS_CTC=200 TAG=ep200 scripts/retrain_lip_models.sh
```

`TAG` must be `ep<N>` matching the epoch count: that is the filename convention
`evaluation/lip-reading/evaluate_epoch_sweep.py` looks for. The three runs are independent, so they can
be issued back to back, or concurrently in separate terminals if the GPU has
room.

| Variable | Default | Meaning |
| --- | --- | --- |
| `EPOCHS_WORD` / `EPOCHS_CTC` | 50 / 50 | Training epochs for each family |
| `TEST_SPEAKER` | `speaker_test_samples1` | Speaker held out as test data |
| `TAG` | *(empty)* | Suffix appended to every checkpoint and log name |
| `OUT` | `checkpoints/retrain` | Directory the checkpoints are written to |
| `LOGS` | `reports/retrain` | Directory the per-run logs are written to |
| `BATCH_WORD` / `BATCH_CTC` | 32 / 16 | Batch sizes |
| `DEVICE` | `auto` | `auto` picks CUDA/ROCm when the device is visible |

It writes `$OUT/<arch>$SUFFIX.pt` for `word_bigru`, `word_transformer_attention`,
`ctc_bigru` and `ctc_transformer`, with one log per model under `$LOGS`. Each
model's log ends with its accuracy on the held-out speaker. `scripts/evaluate_retrained.sh`
scores the same checkpoints one at a time through the deployed inference path.

### Sweeping epoch budgets

`evaluation/lip-reading/evaluate_epoch_sweep.py` scores the whole architecture × epoch grid on the
held-out speaker's clips and prints it as a 3×4 table of overall test accuracy,
with epoch budgets down the rows and architectures across the columns:

```bash
.venv/bin/python evaluation/lip-reading/evaluate_epoch_sweep.py
```

```text
epoch  CTC + BiGRU  CTC + Transformer  Word + BiGRU  Word + Transformer
-----  -----------  -----------------  ------------  ------------------
50          23.75%             25.00%        41.25%              73.75%
100         40.62%             25.00%        35.62%              76.88%
200         69.38%             45.62%        53.12%              78.75%
```

By default it scores the 160 raw MP4 recordings, mouth-cropping each one **once**
through the live `processing.mouth_cropper` pipeline and caching the frames under
`.cache/lip_mp4_frames`. Those same frames are reused for all twelve
checkpoints, so the columns are directly comparable rather than twelve
independent runs. A cold cache decodes every video; after that a run only pays
for the forward passes.

Artefacts land in `reports/retrain/`:

| File | Contents |
| --- | --- |
| `epoch_sweep_mp4.md` | the accuracy table, the correct-clip counts, and the run context |
| `epoch_sweep_mp4.csv` | tidy rows: architecture, epoch, correct, accuracy, 95% CI |
| `epoch_sweep_mp4_predictions.csv` | one row per model/clip pair, for paired tests |

Useful options: `--epochs 50,200` tabulates a subset of the rows, `--limit 50`
scores only the first 50 clips as a smoke test, `--device cpu` forces the CPU,
`--refresh-frames` re-crops from the videos, and `--stem <name>` renames the
artefacts.

Two guards make the table safe to quote, and both stop the run rather than print
a number:

- a checkpoint whose recorded `epoch` contradicts the row it is about to fill
  (which is how a mislabelled checkpoint would land in the wrong row);
- a checkpoint that did not hold out the speaker being evaluated — such a score
  is not a held-out number, and on this dataset it can look implausibly good.

### Testing against a different speaker

`speaker_test_samples1` is the current test set, but it is a parameter rather
than a constant in both scripts, so a future recording session can become the
held-out set without changing any code.

The sweep always scores **raw MP4 recordings** — it decodes each video through
the live mouth-cropping pipeline. So the only thing a new test speaker needs is
their recordings on disk: record them with `record_custom_word_dataset.ipynb`,
which writes the MP4s under `data/custom_words/raw/<speaker_id>/` and the
matching manifest rows.

Then hold that speaker out of training, keeping their checkpoints in their own
directory so the `speaker_test_samples1` ones survive:

```bash
# Train with speaker_008 held out at each budget.
for budget in 50 100 200; do
  TEST_SPEAKER=speaker_008 \
  OUT=checkpoints/retrain_speaker_008 \
  LOGS=reports/retrain_speaker_008 \
  EPOCHS_WORD=$budget EPOCHS_CTC=$budget TAG=ep$budget \
  scripts/retrain_lip_models.sh
done

# Score those checkpoints on that speaker's recordings.
.venv/bin/python evaluation/lip-reading/evaluate_epoch_sweep.py \
    --speakers speaker_008 \
    --checkpoint-dir checkpoints/retrain_speaker_008 \
    --output-dir reports/retrain_speaker_008 \
    --stem epoch_sweep_speaker_008
```

Pair `TAG` with `OUT` as above rather than `TAG` alone, so each run keeps the
`<arch>_ep<N>.pt` names the sweep expects and neither overrides the other. If
the recordings live outside `data/custom_words`, point `--data-dir` at that
dataset root instead — the manifest there needs the same `speaker_id`, `label`
and `raw_video_path` columns.

Two things to expect. A speaker with no raw MP4s cannot be swept: `--speakers`
finds no clips and the run stops with `no raw MP4 clips for speaker(s) …`
rather than reporting an empty table. And evaluating the existing
`speaker_test_samples1` checkpoints against a different speaker is refused by the
guard above, because that speaker is inside their training set — such a score is
a same-speaker number, not a held-out one. `--allow-speaker-mismatch` overrides
that refusal when you knowingly want it.

## Live Web Cam Testing

For a single webcam interface that can switch between custom lip reading and
YOLO gesture recognition, use `combinedVisionSpeak.ipynb`. Its OpenCV window
uses these controls:

- `Space`: record and classify one three-second lip clip.
- `M`: switch between lip and gesture modes.
- `Q`: quit the combined session.

The recorded dataset is stored under `data/custom_words/`. Each accepted clip
contains grayscale mouth frames with shape `(time, 50, 100)` and is indexed by
`data/custom_words/manifest.csv`.

## Hand-gesture workflows

- `models/Lawrence/gesture_detector.py` and `models/Lawrence/main_yolo8n.ipynb` contain the
  YOLOv8 gesture detector.
- `models/SX/` contains the MediaPipe-landmark experiment. It now has two
  temporal cores:

| Core | Model | Window |
| --- | --- | --- |
| `lstm` | `models/SX/model.py` (`HandSignLSTM`) | 1 frame |
| `transformer` | `models/SX/transformer.py` (`GestureTransformer`) | any window |

### The gesture model has no time axis by default

Worth knowing before comparing the two cores. `models/SX/train.py` defaults to
`--sequence-length 1` and the gesture sources are still images, so the LSTM is
trained and served one frame at a time. It never observes motion, and attention
over a single token is degenerate.

`--window-size` (default 16) is what makes the transformer meaningful. Windows
are synthesized from each still image: the pose is copied across the window,
then per-frame jitter, landmark dropout, and a smooth scale drift are applied so
the frames genuinely differ. That is enough for the model to exercise its time
axis, but it does **not** invent a real gesture trajectory.

```bash
# Transformer over 16-frame windows
.venv/bin/python -m models.SX.train \
    --data-dir SuXin_TEMP/data --use-body-pose \
    --architecture transformer --window-size 16

# Original single-frame LSTM, unchanged
.venv/bin/python -m models.SX.train \
    --data-dir SuXin_TEMP/data --architecture lstm --window-size 1
```

`--clips-dir` accepts recorded per-frame landmark sequences (one `.npz` per
sample holding a `frames` array of shape `(T, feature_dim)`). When a clip exists
it is used instead of a synthesized window. Recording real gesture clips is the
follow-up that actually raises accuracy; the synthetic window is a scaffold that
makes the pipeline correct now.

A window inherits the center frame's class label. If a window straddles a class
boundary the label is approximate, which is acceptable at 16 frames but worth
remembering.

### Serving a windowed model

`inference/sx_worker.py` keeps a rolling buffer of recent per-frame landmark
vectors and classifies the newest frame in its temporal context. The buffer is
cleared whenever no hand is detected, because stale frames describe a hand that
is no longer present. Until the window fills, the newest real frame is repeated
so predictions are available immediately rather than after `window_size` frames.

A checkpoint declares its own architecture and window size, so the worker is
told nothing and infers everything:

```bash
.venv/bin/python -m inference.sx_worker --checkpoint checkpoints/sx_hand_sign_best.pt
```

The gesture image data is **not** in this repository. `models/SX/train.py`
expects `SuXin_TEMP/data/{train,valid}` split into per-class folders; the
workflow notebook referenced a Windows path
(`c:\Sean Su\NUS_Project\VisionSpeak\SuXin_TEMP\...`). Supply that folder before
training. The code, checkpoint handling, and live buffering are all exercised by
`tests/test_gesture_transformer.py`, but no trained accuracy is reported here
because the data is absent.

### Read the first gesture accuracy with suspicion

Two things will inflate whatever number this dataset produces, and both are
worth fixing before the result is trusted:

1. **Signer leakage.** `--data-dir` splits into `train`/`valid` folders, and
   nothing prevents the same person appearing in both. On the lip-reading side
   the equivalent per-clip split put all four speakers in train *and* test, and
   a speaker-disjoint re-evaluation dropped the score to roughly chance. The
   `transformer/lip-reading` branch has `speaker_disjoint_split()` in
   `prepare_custom_word_dataset.py`; the same protocol should be applied here by
   splitting on signer before any accuracy figure is quoted.
2. **Synthetic windows.** Windows are built from still images, so they carry
   jitter and scale drift rather than a real signing trajectory. A model can
   score well on those and still fail on live motion, which is exactly what the
   webcam loop provides.

Recording real per-frame landmark clips and re-running with `--clips-dir`, plus
a signer-disjoint split, are the two changes that would make the next reported
number mean something.

## Legacy GRID/Kaggle workflow

The original GRID dataset experiments, dataset files, scripts, checkpoints,
tests, and notebook cells have been moved to
`archive/kaggle_lip_reading/`. See its README for the archived layout and
commands.

The reusable model architectures in `models/LBY/model.py` and
`models/valente/model.py` remain active because the custom-word workflows use
them too.

## Environments

MediaPipe and Ultralytics depend on different OpenCV wheel distributions. Keep
them in separate environments to avoid loading two incompatible `cv2` packages
into one Python process.

Create the lip-reading/MediaPipe environment and use it as the Jupyter kernel:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name visionspeak-lip --display-name "VisionSpeak Lip"
```

Create the isolated YOLO worker environment:

```bash
python3.11 -m venv .venv-yolo
.venv-yolo/bin/python -m pip install -r requirements-yolo.txt
```

On Windows, use `.venv\Scripts\python.exe` and
`.venv-yolo\Scripts\python.exe` in place of the `bin/python` paths. The
combined notebook automatically discovers both layouts, and its configuration
cell also accepts explicit interpreter paths.

Do not install `requirements.txt` and `requirements-yolo.txt` into the same
environment.

No new dependency is needed for the transformer cores: they use
`torch.nn.TransformerEncoder`, which ships with the `torch` already pinned in
both profiles.

## Running the tests

The suite uses `unittest`, and `tests/` is not an importable package root, so
discover from inside it:

```bash
.venv/bin/python -m unittest discover -s tests -t tests
```

`tests/test_sx.py` currently fails to import because `inference/protocol.py`
does not define `EXIT_BACKEND_SWITCH`, which `inference/sx_worker.py` imports.
That predates the transformer work and is unrelated to it.
