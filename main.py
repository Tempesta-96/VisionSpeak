from processing.modality_classifier import ModalityClassifier
from processing.postprocessor import Postprocessor
from processing.preprocessor import Preprocessor
from processing.video_loader import VideoLoader


def main() -> None:
    video_path = "hand_sample_video.mp4"

    loader = VideoLoader()
    preprocessor = Preprocessor()
    classifier = ModalityClassifier()
    postprocessor = Postprocessor()

    video_data = loader.load(video_path)
    features = preprocessor.transform(video_data)
    pipeline = classifier.select_pipeline(video_path)
    result = pipeline.run(features)

    print(postprocessor.format_output(result))


if __name__ == "__main__":
    main()
