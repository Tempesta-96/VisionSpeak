import cv2
import numpy as np


class Preprocessor:
    """Placeholder preprocessor for extracted video content."""

    def pad_frames(self, video_data: dict, value: int | float = 0) -> dict:
        """Center-pad every frame so all frames have the same spatial shape.

        The target height and width are taken from the largest frame in the
        video. The input dictionary is updated in place, consistently with the
        other preprocessing methods.
        """
        frames = video_data.get("frames", [])
        if not frames:
            return video_data

        arrays = [np.asarray(frame) for frame in frames]
        if any(frame.ndim not in (2, 3) for frame in arrays):
            raise ValueError("Each video frame must be a 2D or 3D array.")

        trailing_shapes = {frame.shape[2:] for frame in arrays}
        if len(trailing_shapes) != 1:
            raise ValueError("All video frames must have the same channel shape.")

        target_height = max(frame.shape[0] for frame in arrays)
        target_width = max(frame.shape[1] for frame in arrays)
        padded_frames = []

        for frame in arrays:
            height_padding = target_height - frame.shape[0]
            width_padding = target_width - frame.shape[1]
            top = height_padding // 2
            bottom = height_padding - top
            left = width_padding // 2
            right = width_padding - left
            padding = ((top, bottom), (left, right)) + (
                ((0, 0),) if frame.ndim == 3 else ()
            )
            padded_frames.append(
                np.pad(frame, padding, mode="constant", constant_values=value)
            )

        video_data["frames"] = padded_frames
        return video_data

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
            frame
            if np.asarray(frame).ndim == 2
            else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            for frame in frames
        ]
        video_data["frames"] = greyscale_frames
        return video_data
