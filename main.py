import cv2
import argparse

from processing.modality_classifier import ModalityClassifier
from processing.postprocessor import Postprocessor
from processing.preprocessor import Preprocessor
from processing.video_loader import VideoLoader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run VisionSpeak with an MP4 file or webcam input."
    )
    parser.add_argument(
        "--source",
        choices=["mp4", "webcam"],
        default="mp4",
        help="Choose whether to load an MP4 file or use the webcam.",
    )
    parser.add_argument(
        "--video-path",
        default="sample_video.mp4",
        help="Path to the MP4 file when --source mp4 is selected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.video_path if args.source == "mp4" else "webcam"

    loader = VideoLoader()
    preprocessor = Preprocessor()
    classifier = ModalityClassifier()
    postprocessor = Postprocessor()

    video_data = loader.load(source)
    features = preprocessor.transform(video_data)
    pipeline = classifier.select_pipeline(source)
    result = pipeline.run(features)
    capture = video_data.get("capture")

    if capture is not None:
        capture.release()
        cv2.destroyAllWindows()

    print(postprocessor.format_output(result))


if __name__ == "__main__":
    main()
