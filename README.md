# VisionSpeak

VisionSpeak is a lightweight multimodal text transcription prototype that routes video input through a simple processing pipeline for lip-reading or hand-gesture recognition.

## Current pipeline

```text
main.py
  -> video_loader.py
  -> preprocessor.py
  -> modality_classifier.py
  -> lip_pipeline.py / gesture_pipeline.py
  -> postprocessor.py
  -> final text output
```

## Project structure

```text
VisionSpeak/
├── Dockerfile
├── README.md
├── main.py
├── requirements.txt
└── processing/
    ├── gesture_pipeline.py
    ├── lip_pipeline.py
    ├── modality_classifier.py
    ├── postprocessor.py
    ├── preprocessor.py
    └── video_loader.py
```

## Local run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run with an MP4 file:

```bash
python main.py --source mp4 --video-path sample_video.mp4
```

Run with webcam:

```bash
python main.py --source webcam
```

Press `q` to exit the webcam preview window.

## Prepare the GRID lip-reading data

From this directory, align the MPG videos with their word timestamps:

```bash
python align_videos.py --verbose
```

This writes one archive per video plus `data/processedData/aligned_words.csv`.
The CSV uses one row per spoken word; `end_frame` is exclusive.

Load the lightweight timing index as a DataFrame and retrieve pixels lazily:

```python
from aligned_dataframe import load_aligned_dataframe, load_word_frames

aligned_words = load_aligned_dataframe()
first_word_frames = load_word_frames(aligned_words.iloc[0])
```

Validate all generated archives and their timing metadata:

```bash
python validate_processed_data.py
```
