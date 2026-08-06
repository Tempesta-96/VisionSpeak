from pathlib import Path


class VideoLoader:
    """Minimal video loader placeholder."""

    def load(self, video_path: str) -> dict:
        return {
            "path": str(Path(video_path)),
            "frames": [],
            "loaded": True,
        }
