from processing.gesture_pipeline import GesturePipeline
from processing.lip_pipeline import LipPipeline


class ModalityClassifier:
    """Routes preprocessed features to the appropriate pipeline."""

    def select_pipeline(self, source: str):
        source_lower = source.lower()
        if "hand" in source_lower or "gesture" in source_lower:
            return GesturePipeline()
        return LipPipeline()
