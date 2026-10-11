"""Checkpoint-compatible landmark construction shared by SX entry points."""
from contextlib import ExitStack
from pathlib import Path

import numpy as np


ASSET_DIR = Path(__file__).resolve().parents[2] / "processing"


def landmark_points(landmarks):
    """Accept Tasks landmark lists and legacy protobuf landmark containers."""
    return getattr(landmarks, "landmark", landmarks)


def create_landmarker(kind, *, video=False, model_path=None):
    """Create an IMAGE or VIDEO Tasks detector from a local model asset."""
    from processing.mediapipe_compat import python, vision

    if kind not in ("hand", "pose"):
        raise ValueError("kind must be 'hand' or 'pose'.")
    path = Path(model_path) if model_path is not None else ASSET_DIR / f"{kind}_landmarker.task"
    if not path.is_file():
        raise FileNotFoundError(f"Missing MediaPipe {kind} model asset: {path}")
    options = dict(
        base_options=python.BaseOptions(model_asset_path=str(path.resolve())),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        min_tracking_confidence=0.5,
    )
    if kind == "hand":
        return vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            **options, num_hands=2, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
        ))
    return vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        **options, num_poses=1, min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    ))


def feature_vector(hand_landmarks, pose_landmarks, *, two_hands=True, use_body_pose=False):
    """Preserve detector ordering and raw x/y/z coordinates used in training."""
    vectors = [np.asarray([(p.x, p.y, p.z) for p in landmark_points(hand)],
                          dtype=np.float32).reshape(-1) for hand in (hand_landmarks or [])[:2]]
    hand_count = 2 if two_hands else 1
    result = np.zeros(hand_count * 63 + (99 if use_body_pose else 0), dtype=np.float32)
    for i, vector in enumerate(vectors[:hand_count]):
        result[i * 63:i * 63 + min(63, len(vector))] = vector[:63]
    if use_body_pose and pose_landmarks is not None:
        body = np.asarray([(p.x, p.y, p.z) for p in landmark_points(pose_landmarks)],
                          dtype=np.float32).reshape(-1)[:99]
        result[hand_count * 63:hand_count * 63 + len(body)] = body
    return result


def extract_rgb(rgb, *, two_hands=True, use_body_pose=False, hands=None, pose=None,
                timestamp_ms=None, hand_model_path=None, pose_model_path=None):
    """Extract normalized landmarks; VIDEO detectors require a monotonic timestamp.

    Supplied detectors remain owned by the caller. Without detectors, IMAGE
    mode is used for stills, or VIDEO mode when a timestamp is supplied.
    """
    from processing.mediapipe_compat import mp

    with ExitStack() as stack:
        if hands is None:
            hands = stack.enter_context(create_landmarker(
                "hand", video=timestamp_ms is not None, model_path=hand_model_path))
        if use_body_pose and pose is None:
            pose = stack.enter_context(create_landmarker(
                "pose", video=timestamp_ms is not None, model_path=pose_model_path))
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb, dtype=np.uint8))

        def detect(detector):
            # Allow existing callers to supply legacy detectors during migration.
            if hasattr(type(detector), "detect_for_video") or hasattr(type(detector), "detect"):
                return (detector.detect(image) if timestamp_ms is None
                        else detector.detect_for_video(image, int(timestamp_ms)))
            return detector.process(rgb)

        hand_result = detect(hands)
        pose_result = detect(pose) if use_body_pose else None
        landmarks = (hand_result.hand_landmarks if hasattr(hand_result, "hand_landmarks")
                     else hand_result.multi_hand_landmarks)
        body = None
        if pose_result is not None:
            body = pose_result.pose_landmarks
            if isinstance(body, (list, tuple)):
                body = body[0] if body else None
        vector = feature_vector(landmarks,
            body,
            two_hands=two_hands, use_body_pose=use_body_pose)
        return vector, landmarks
