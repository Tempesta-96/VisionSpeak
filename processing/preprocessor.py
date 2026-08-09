import cv2
import numpy as np


class Preprocessor:
    """Placeholder preprocessor for extracted video content."""

    def transform(self, video_data: dict) -> list[float]:
        if not video_data.get("loaded"):
            return []
        if video_data.get("source_type") == "webcam":
            return [0.4, 0.5, 0.6]
        return [0.1, 0.2, 0.3]

    def convert_to_greyscale(self, video_data: dict) -> dict:
        frames = video_data.get("frames", [])
        if not frames:
            return video_data

        greyscale_frames = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames
        ]
        video_data["frames"] = greyscale_frames
        return video_data
