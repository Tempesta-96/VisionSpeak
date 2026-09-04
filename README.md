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
