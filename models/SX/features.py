"""Checkpoint-compatible landmark construction shared by SX entry points."""
from contextlib import ExitStack

import numpy as np


def feature_vector(hand_landmarks, pose_landmarks, *, two_hands=True, use_body_pose=False):
    """Preserve detector ordering and raw x/y/z coordinates used in training."""
    vectors = [np.asarray([(p.x, p.y, p.z) for p in hand.landmark],
                          dtype=np.float32).reshape(-1) for hand in (hand_landmarks or [])[:2]]
    hand_count = 2 if two_hands else 1
    result = np.zeros(hand_count * 63 + (99 if use_body_pose else 0), dtype=np.float32)
    for i, vector in enumerate(vectors[:hand_count]):
        result[i * 63:i * 63 + min(63, len(vector))] = vector[:63]
    if use_body_pose and pose_landmarks is not None:
        body = np.asarray([(p.x, p.y, p.z) for p in pose_landmarks.landmark],
                          dtype=np.float32).reshape(-1)[:99]
        result[hand_count * 63:hand_count * 63 + len(body)] = body
    return result


def extract_rgb(rgb, *, two_hands=True, use_body_pose=False, hands=None, pose=None):
    from processing.mediapipe_compat import mp

    with ExitStack() as stack:
        if hands is None:
            hands = stack.enter_context(mp.solutions.hands.Hands(
                static_image_mode=True, max_num_hands=2,
                min_detection_confidence=0.5, min_tracking_confidence=0.5))
        if use_body_pose and pose is None:
            pose = stack.enter_context(mp.solutions.pose.Pose(
                static_image_mode=True, min_detection_confidence=0.5,
                min_tracking_confidence=0.5))
        hand_result = hands.process(rgb)
        pose_result = pose.process(rgb) if use_body_pose else None
        landmarks = hand_result.multi_hand_landmarks
        vector = feature_vector(landmarks,
            pose_result.pose_landmarks if pose_result else None,
            two_hands=two_hands, use_body_pose=use_body_pose)
        return vector, landmarks
