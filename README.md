# VisionSpeak

VisionSpeak is a multimodal computer-vision prototype for converting visible
lip movements and hand gestures into text.

## Active lip-reading workflow

The active lip-reading experiments recognise these eight isolated words:

```text
hello, hi, yes, no, good, bad, thanks, bye
```

1. Record mouth-only training clips with `record_custom_word_dataset.ipynb`.
2. Load and split those clips with `custom_words_dataset.py`.
3. Train the isolated-word classifier with `train_custom_words.py`, or use the
   custom CTC experiment in `models/valente/custom_word_training.ipynb`.
4. Test a trained checkpoint live with `webcam_custom_word_detection.ipynb`,
   or offline with `evaluate_custom_words.py` — over recorded clips
   (`--source npz` / `--source mp4`) or a single raw MP4 recording
   (`--video path/to/clip.mp4`).

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

```bash
# Transformer CTC experiment (default), then serve it through the webcam CLI
.venv/bin/python train_custom_words.py --architecture transformer
.venv/bin/python main.py --mode lip \
    --lip-checkpoint checkpoints/custom_words.pt
```

The transformer path strides time by 2 in the *first* convolutional block only.
That bound is not arbitrary: CTC requires `input_length >= 2 * target_length + 1`,
the longest word is `thanks` (6 characters), and the shortest recorded clip is
45 frames. The frontend must therefore leave at least 13 emissions, and it
leaves 23. `tests/test_lip_transformer.py` re-derives that inequality from the
manifest, so changing the stride fails the tests rather than silently breaking
the loss.

A note on reading the accuracy numbers: the default split in
`custom_words_dataset.stratified_split` is per-clip random, so **all four
speakers appear in both train and test**. Use `--speaker-holdout` for a number
that means something:

```bash
# Hold out one speaker entirely; auto-picks the most representative speaker
.venv/bin/python train_custom_words.py --architecture transformer --speaker-holdout

# Or name the speaker
.venv/bin/python train_custom_words.py --architecture bigru --speaker-holdout TestID
```

With `--speaker-holdout`, validation is drawn from the non-test speakers and the
checkpoint is selected on it, so the held-out speaker never influences model
choice.

### Measured: the split matters more than the architecture

Both cores, ~14 epochs on this CPU, holding out one speaker (384 train / 160
test clips, 8 classes, so chance is 12.5%):

| Core | Training accuracy | Test accuracy |
| --- | --- | --- |
| transformer | 0.15 → **0.27** | **0.26 peak**, ~0.13 typical |
| bigru (cosine schedule) | 0.13, flat | 0.13, flat |
| bigru (original MultiStepLR) | 0.13, flat | 0.13, flat |

Read this carefully, because it is the most useful result on this branch and it
is *not* "the transformer wins":

- Under speaker-disjoint evaluation **both models are weak**. The task barely
  generalises across signers with only 4 speakers, and one held-out speaker is a
  small, noisy test set.
- The BiGRU never escaped predicting a single class: it sat at exactly 0.125,
  the majority-class rate, under both learning-rate schedules. It did not overfit
  the training speakers either (training accuracy stayed at 0.13), so this is an
  optimisation problem for this configuration rather than proof that recurrence
  cannot represent the task.
- The transformer did start fitting the training speakers (0.27) and briefly
  cleared chance on the unseen speaker before its validation loss rose. That is
  overfitting, not generalisation.

The honest conclusion: architecture is not this project's bottleneck. With 640
clips from 4 speakers, more capacity does not buy cross-signer accuracy. More
speakers — and real video rather than still images and three-second clips — are
what will move these numbers.

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

- `gesture_detector.py` and `models/Lawrence/main_yolo8n.ipynb` contain the
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
   `custom_words_dataset.py`; the same protocol should be applied here by
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
