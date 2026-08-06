class Preprocessor:
    """Placeholder preprocessor for extracted video content."""

    def transform(self, video_data: dict) -> list[float]:
        if not video_data.get("loaded"):
            return []
        return [0.1, 0.2, 0.3]
