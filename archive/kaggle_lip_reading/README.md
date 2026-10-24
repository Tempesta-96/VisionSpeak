# Archived GRID/Kaggle lip-reading workflow

This directory contains the original experiments built around the
[GRID lip-reading dataset distributed on Kaggle](https://www.kaggle.com/datasets/mohamedbentalb/lipreading-dataset).
It is retained for reference and is no longer part of VisionSpeak's active
custom-word workflow.

## Contents

- `lipnet_kaggle.ipynb`: the GRID/Kaggle cells formerly at the start of the
  root `lipnet.ipynb`.
- `align_videos.py`, `aligned_dataframe.py`, `dataset.py`, and
  `validate_processed_data.py`: dataset preparation and loading.
- `models/LBY/`: isolated GRID-word training and inference.
- `models/valente/`: complete GRID-sentence CTC training and inference.
- `webcam_word_detection.ipynb`: webcam inference for the 51-word GRID
  classifier.
- `checkpoints/`: checkpoints trained on GRID.
- `data/lips-dataset/`: the locally downloaded dataset. It remains ignored by
  Git because of its size.

The generic architectures remain in the active `models/LBY/model.py` and
`models/valente/model.py` modules because the custom eight-word workflows also
use them.

## Running archived commands

Run commands from the repository root so the shared models and processing
modules remain importable:

```bash
python -m archive.kaggle_lip_reading.align_videos --crop-mouth --grayscale
python -m archive.kaggle_lip_reading.validate_processed_data
python -m archive.kaggle_lip_reading.models.LBY.train
python -m archive.kaggle_lip_reading.models.valente.train
```

This code is archived rather than maintained. New lip-reading work should use
`data/record_custom_word_dataset.ipynb`, `data/prepare_custom_word_dataset.py`,
`train_custom_words.py`, and `evaluation/lip-reading/webcam_lip_model_comparison.ipynb`.
