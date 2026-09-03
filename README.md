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

## Environment

Install the dependencies into a dedicated environment:

```bash
pip install -r requirements.txt
```

The project intentionally uses `opencv-contrib-python` rather than installing
both OpenCV wheel variants in one environment.
