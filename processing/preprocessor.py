class Preprocessor:
    """Placeholder preprocessor for extracted video content."""

    def transform(self, video_data: dict) -> list[float]:
        if not video_data.get("loaded"):
            return []
        if video_data.get("source_type") == "webcam":
            return [0.4, 0.5, 0.6]
        return [0.1, 0.2, 0.3]
