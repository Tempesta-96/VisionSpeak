"""Common controls and timing for the two gesture workers."""
from collections import deque
import time

import cv2


def run_gesture_live(detector, *, title, switch_message=""):
    if not detector.open_camera():
        detector.close_camera()
        return "error"
    timestamps = deque(maxlen=31)
    latencies = deque(maxlen=30)
    try:
        while True:
            success, frame = detector.camera.read()
            if not success:
                return "error"
            started = time.perf_counter()
            frame = cv2.flip(frame, 1)
            label, confidence, render = detector.predict_frame(frame)
            latencies.append((time.perf_counter() - started) * 1000)
            # Rendering is deliberately outside the processing timer.
            preview = render() if callable(render) else render
            now = time.perf_counter()
            timestamps.append(now)
            fps = ((len(timestamps) - 1) / (now - timestamps[0])) if len(timestamps) > 1 else 0
            lines = [title, "G model | M lip | Q quit",
                     f"{label} ({confidence:.1%})",
                     f"Processing {sum(latencies)/len(latencies):.1f} ms | FPS {fps:.1f}"]
            if switch_message:
                lines.append(switch_message)
            for i, line in enumerate(lines):
                y = 28 + i * 27
                cv2.rectangle(preview, (4, y - 22), (preview.shape[1] - 4, y + 5), (24,24,24), -1)
                cv2.putText(preview, line, (10,y), cv2.FONT_HERSHEY_SIMPLEX, .55,
                            (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow(detector.window_name, preview)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), ord('Q'), 27):
                return "quit"
            if key in (ord('m'), ord('M')):
                return "switch"
            if key in (ord('g'), ord('G')) and not switch_message:
                return "backend_switch"
            if cv2.getWindowProperty(detector.window_name, cv2.WND_PROP_VISIBLE) < 1:
                return "quit"
    finally:
        detector.close_camera()
