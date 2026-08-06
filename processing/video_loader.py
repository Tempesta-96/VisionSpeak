from pathlib import Path
import cv2


class VideoLoader:
    """Minimal video loader for MP4 files and webcam input."""

    def load(self, source: str) -> dict:
        if source == "webcam":
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                raise RuntimeError("Unable to connect to the default webcam device.")

            frames = []
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frames.append(frame)
                cv2.imshow("VisionSpeak Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            return {
                "source_type": "webcam",
                "path": None,
                "capture": capture,
                "frames": frames,
                "loaded": len(frames) > 0,
            }

        video_path = Path(source)
        return {
            "source_type": "mp4",
            "path": str(video_path),
            "frames": [],
            "loaded": video_path.suffix.lower() == ".mp4",
        }
