class LipPipeline:
    def run(self, features: list[float]) -> dict:
        prediction = "lip_prediction" if features else "no_lip_signal"
        return {"modality": "lip", "prediction": prediction}
