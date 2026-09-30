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
4. Test a trained checkpoint with `webcam_custom_word_detection.ipynb`.

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
speakers appear in both train and test**. Treat any accuracy from it as a
same-speaker upper bound, not as evidence of generalisation.

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
- `models/SX/` contains the MediaPipe-landmark LSTM experiment.

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
