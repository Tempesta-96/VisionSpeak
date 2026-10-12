"""
YOLOv8 Hand Gesture Detection Class
Provides reusable classes for training, testing, and real-time webcam gesture detection.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class GestureDetector:
    """Real-time hand gesture detection using YOLOv8 model."""
    
    def __init__(self, model_path: str, camera_index: int = 0, confidence: float = 0.25):
        """
        Initialize the gesture detector.
        
        Args:
            model_path: Path to the trained YOLOv8 model (.pt file)
            camera_index: Camera index (0 for default, 1 for external webcam)
            confidence: Confidence threshold for detections (0-1)
        """
        self.model = YOLO(model_path)
        self.camera_index = camera_index
        self.confidence = confidence
        self.window_name = "YOLOv8 Hand Gesture Result"
        self.camera = None
        self.class_names = self.model.names
        
    def open_camera(self) -> bool:
        """
        Open the camera connection.
        
        Returns:
            True if successful, False otherwise
        """
        self.camera = cv2.VideoCapture(self.camera_index)
        if not self.camera.isOpened():
            print(f"Error: Could not open camera at index {self.camera_index}")
            return False
        return True
    
    def close_camera(self):
        """Release camera and close windows."""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
    
    def predict_frame(self, frame) -> Tuple[str, float, cv2.Mat]:
        """
        Predict gesture in a single frame.
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple of (gesture_name, confidence, annotated_frame)
        """
        result = self.model.predict(
            source=frame,
            imgsz=640,
            conf=self.confidence,
            verbose=False,
        )[0]
        
        result_frame = result.plot()
        
        if result.boxes is not None and len(result.boxes) > 0:
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            best_index = int(confidences.argmax())
            gesture_name = self.model.names[class_ids[best_index]]
            confidence = float(confidences[best_index])
        else:
            gesture_name = "No gesture detected"
            confidence = 0.0
        
        return gesture_name, confidence, result_frame
    
    def run_with_countdown(self, countdown_seconds: int = 5) -> Optional[Dict]:
        """
        Run gesture detection with countdown timer.
        
        Args:
            countdown_seconds: Seconds to count down before capture
            
        Returns:
            Dict with keys: gesture, confidence, or None if quit requested
        """
        if not self.camera or not self.camera.isOpened():
            print("Camera not opened. Call open_camera() first.")
            return None
        
        print(f"Starting {countdown_seconds}-second countdown. Show your hand gesture to the webcam.")
        quit_requested = False

        for seconds_left in range(countdown_seconds, 0, -1):
            success, frame = self.camera.read()
            if not success:
                print("Error: Could not read frame from camera")
                return None

            frame = cv2.flip(frame, 1)
            gesture, confidence, live_frame = self.predict_frame(frame)
            
            cv2.putText(
                live_frame,
                f"Taking picture in {seconds_left}",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 255),
                3,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window_name, live_frame)
            
            if cv2.waitKey(1000) & 0xFF == ord("q"):
                quit_requested = True
                break

        if quit_requested:
            return None

        # Capture the final frame
        success, captured_frame = self.camera.read()
        if not success:
            print("Error: Could not capture frame")
            return None

        captured_frame = cv2.flip(captured_frame, 1)
        print("Photo captured. Predicting gesture...")
        
        gesture_name, confidence, result_frame = self.predict_frame(captured_frame)
        
        if confidence > 0:
            prediction_text = f"Prediction: {gesture_name} ({confidence:.1%})"
            text_color = (0, 200, 0)
        else:
            prediction_text = "Prediction: No gesture detected"
            text_color = (0, 165, 255)

        cv2.putText(
            result_frame,
            prediction_text,
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            text_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(self.window_name, result_frame)
        print(prediction_text)
        
        return {
            "gesture": gesture_name,
            "confidence": confidence,
            "frame": result_frame
        }
    
    def run_continuous(self, max_predictions: Optional[int] = None) -> bool:
        """
        Run continuous gesture detection until 'q' is pressed.
        
        Args:
            max_predictions: Max number of predictions to run, or None for unlimited
            
        Returns:
            True if exited normally, False if error occurred
        """
        if not self.camera or not self.camera.isOpened():
            print("Camera not opened. Call open_camera() first.")
            return False
        
        prediction_count = 0
        print("Running continuous detection. Press 'q' to quit.")
        
        try:
            while True:
                success, frame = self.camera.read()
                if not success:
                    print("Error: Could not read frame")
                    return False
                
                frame = cv2.flip(frame, 1)
                gesture_name, confidence, result_frame = self.predict_frame(frame)
                
                cv2.imshow(self.window_name, result_frame)
                
                if max_predictions and prediction_count >= max_predictions:
                    break
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                
                prediction_count += 1
            
            return True
        except Exception as e:
            print(f"Error during continuous detection: {e}")
            return False
    
    def run_interactive_loop(self):
        """
        Run interactive loop: countdown + prediction, repeat until quit.
        """
        if not self.open_camera():
            return
        
        try:
            while True:
                result = self.run_with_countdown()
                if result is None:
                    break
                
                print(f"Press 'n' for next prediction or 'q' to quit")
                
                while True:
                    key = cv2.waitKey(100) & 0xFF
                    if key == ord("n"):
                        break
                    if key == ord("q"):
                        return
        finally:
            self.close_camera()
            print("Gesture detection finished.")


class GestureTrainer:
    """Train YOLOv8 model on gesture dataset."""
    
    def __init__(self, data_config: str):
        """
        Initialize the trainer.
        
        Args:
            data_config: Path to data.yaml configuration file
        """
        self.data_config = data_config
        self.model = YOLO("yolov8n.pt")
    
    def train(self, epochs: int = 100, batch_size: int = 8, img_size: int = 640,
              device: int = 0, workers: int = 4) -> None:
        """
        Train the YOLOv8 model.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            device: GPU device index (0 for first GPU)
            workers: Number of workers for data loading
        """
        print(f"Starting training with {epochs} epochs...")
        self.model.train(
            data=self.data_config,
            imgsz=img_size,
            batch=batch_size,
            epochs=epochs,
            workers=workers,
            device=device
        )
        print("Training completed.")


class GestureEvaluator:
    """Evaluate model performance on test dataset."""
    
    def __init__(self, model_path: str, test_images_dir: str, test_labels_dir: str):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained YOLOv8 model
            test_images_dir: Path to test images directory
            test_labels_dir: Path to test labels directory
        """
        self.model = YOLO(model_path)
        self.test_images = Path(test_images_dir)
        self.test_labels = Path(test_labels_dir)
        self.output_dir = Path("runs/gesture_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.class_names = self.model.names
    
    def evaluate(self, confidence: float = 0.25, save_report: bool = True, device: Optional[str] = None) -> Dict:
        """
        Evaluate model on test dataset.

        Args:
            confidence: Confidence threshold for predictions
            save_report: Whether to save CSV report
            device: Optional YOLO device string such as "cpu" or "0"

        Returns:
            Dict with evaluation metrics
        """
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            path for path in self.test_images.iterdir()
            if path.suffix.lower() in image_extensions
        )
        
        if not image_paths:
            print(f"No images found in {self.test_images.resolve()}")
            return {}

        class_names = getattr(self, "class_names", getattr(self.model, "names", {}))
        rows = []
        correct = 0
        num_classes = len(class_names)
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        print(f"Evaluating {len(image_paths)} test images...")
        
        for image_path in image_paths:
            label_path = self.test_labels / f"{image_path.stem}.txt"
            expected_ids = []
            if label_path.exists():
                with label_path.open(encoding="utf-8") as label_file:
                    expected_ids = [int(line.split()[0]) for line in label_file if line.strip()]

            prediction_kwargs = {
                "source": str(image_path),
                "imgsz": 640,
                "conf": confidence,
                "verbose": False,
                "save": True,
                "project": str(self.output_dir),
                "name": "annotated",
                "exist_ok": True,
            }
            if device is not None:
                prediction_kwargs["device"] = device
            predictions = self.model.predict(**prediction_kwargs)[0]

            if predictions.boxes is not None and len(predictions.boxes):
                best_index = int(predictions.boxes.conf.argmax().item())
                predicted_id = int(predictions.boxes.cls[best_index].item())
                confidence_score = float(predictions.boxes.conf[best_index].item())
            else:
                predicted_id = None
                confidence_score = 0.0

            expected_id = expected_ids[0] if expected_ids else None
            is_correct = predicted_id == expected_id and predicted_id is not None
            correct += int(is_correct)

            if expected_id is not None and predicted_id is not None:
                confusion_matrix[expected_id, predicted_id] += 1

            rows.append({
                "image": image_path.name,
                "expected": class_names.get(expected_id, str(expected_id)) if expected_id is not None else "missing label",
                "predicted": class_names.get(predicted_id, str(predicted_id)) if predicted_id is not None else "no detection",
                "confidence": f"{confidence_score:.4f}",
                "result": "RIGHT" if is_correct else "WRONG",
            })

        if save_report:
            report_path = self.output_dir / "gesture_results.csv"
            with report_path.open("w", newline="", encoding="utf-8") as report_file:
                writer = csv.DictWriter(report_file, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"Report saved to: {report_path.resolve()}")

        result_counts = Counter(row["result"] for row in rows)
        accuracy = correct / len(rows) if rows else 0
        confusion_matrix_accuracy = (
            np.trace(confusion_matrix) / confusion_matrix.sum() if confusion_matrix.sum() else 0.0
        )

        metrics = {
            "total_images": len(rows),
            "correct": result_counts.get('RIGHT', 0),
            "wrong": result_counts.get('WRONG', 0),
            "accuracy": accuracy,
            "confusion_matrix": confusion_matrix.tolist(),
            "confusion_matrix_accuracy": float(confusion_matrix_accuracy),
            "report_path": str(self.output_dir / "gesture_results.csv")
        }
        
        print(f"\nEvaluation Results:")
        print(f"Test images: {metrics['total_images']}")
        print(f"Correct:     {metrics['correct']}")
        print(f"Wrong:       {metrics['wrong']}")
        print(f"Accuracy:    {metrics['accuracy']:.2%}")
        print("Confusion matrix metric:")
        print(np.array(metrics["confusion_matrix"]))
        print(f"Confusion-matrix accuracy: {metrics['confusion_matrix_accuracy']:.2%}")
        
        return metrics


if __name__ == "__main__":
    # Example usage
    
    # Real-time gesture detection
    detector = GestureDetector("yolov8n_custom.pt", camera_index=0)
    detector.run_interactive_loop()
    
    # Or use individually:
    # detector = GestureDetector("yolov8n_custom.pt", camera_index=0)
    # if detector.open_camera():
    #     while True:
    #         result = detector.run_with_countdown()
    #         if result is None:
    #             break
    #     detector.close_camera()
    
    # Training example (uncomment to use):
    # trainer = GestureTrainer("data.yaml")
    # trainer.train(epochs=100, batch_size=8)
    
    # Evaluation example (uncomment to use):
    # evaluator = GestureEvaluator("yolov8n_custom.pt", "test/images", "test/labels")
    # metrics = evaluator.evaluate(confidence=0.25)
