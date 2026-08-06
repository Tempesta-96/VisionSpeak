class GesturePipeline:
    def run(self, features: list[float]) -> dict:
        prediction = "gesture_prediction" if features else "no_gesture_signal"
        return {"modality": "gesture", "prediction": prediction}
