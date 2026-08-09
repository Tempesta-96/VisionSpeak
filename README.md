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