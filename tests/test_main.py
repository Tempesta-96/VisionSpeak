from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import main
from inference.launcher import ConfigurationError, SessionEvent, SessionSummary
from inference.protocol import EXIT_QUIT


def _resolved() -> SimpleNamespace:
    return SimpleNamespace(
        project_root=Path("/tmp/project"),
        lip_python=Path("/tmp/lip-python"),
        gesture_python=Path("/tmp/gesture-python"),
        lip_checkpoint=Path("/tmp/lip.pt"),
        gesture_checkpoint=Path("/tmp/gesture.pt"),
    )


def _summary(completed: bool) -> SessionSummary:
    event = SessionEvent(
        mode="lip",
        action="quit",
        return_code=EXIT_QUIT,
        elapsed_seconds=1.0,
    )
    error = (
        None
        if completed
        else "The lip worker stopped unexpectedly with exit code 7."
    )
    return SessionSummary(initial_mode="lip", events=(event,), error=error)


class MainCliTests(unittest.TestCase):
    def test_defaults_mirror_the_notebook_configuration(self) -> None:
        args = main.build_parser().parse_args([])
        config = main.config_from_args(args)

        self.assertEqual(config.project_root, main.PROJECT_ROOT)
        self.assertEqual(config.initial_mode, "lip")
        self.assertEqual(config.camera_index, 0)
        self.assertIsNone(config.lip_python)
        self.assertIsNone(config.gesture_python)
        self.assertEqual(config.lip_checkpoint, main.DEFAULT_LIP_CHECKPOINT)
        self.assertEqual(
            config.gesture_checkpoint, main.DEFAULT_GESTURE_CHECKPOINT
        )
        self.assertEqual(config.countdown_seconds, 2.0)
        self.assertEqual(config.capture_seconds, 3.0)
        self.assertEqual(config.min_frames, 30)
        self.assertEqual(config.min_detection_rate, 0.75)
        self.assertEqual(config.max_frames, 90)
        self.assertEqual(config.gesture_confidence, 0.5)
        self.assertEqual(config.lip_device, "auto")
        self.assertIsNone(config.gesture_device)

    def test_flags_override_every_config_field(self) -> None:
        args = main.build_parser().parse_args(
            [
                "--mode",
                "gesture",
                "--camera-index",
                "2",
                "--lip-python",
                "/opt/lip/python",
                "--gesture-python",
                "/opt/gesture/python",
                "--lip-checkpoint",
                "checkpoints/other_lip.pt",
                "--gesture-checkpoint",
                "checkpoints/other_gesture.pt",
                "--countdown-seconds",
                "0.5",
                "--capture-seconds",
                "1.5",
                "--min-frames",
                "10",
                "--min-detection-rate",
                "0.5",
                "--max-frames",
                "60",
                "--gesture-confidence",
                "0.9",
                "--lip-device",
                "cpu",
                "--gesture-device",
                "mps",
            ]
        )
        config = main.config_from_args(args)

        self.assertEqual(config.initial_mode, "gesture")
        self.assertEqual(config.camera_index, 2)
        self.assertEqual(config.lip_python, Path("/opt/lip/python"))
        self.assertEqual(config.gesture_python, Path("/opt/gesture/python"))
        self.assertEqual(
            config.lip_checkpoint, Path("checkpoints/other_lip.pt")
        )
        self.assertEqual(
            config.gesture_checkpoint, Path("checkpoints/other_gesture.pt")
        )
        self.assertEqual(config.countdown_seconds, 0.5)
        self.assertEqual(config.capture_seconds, 1.5)
        self.assertEqual(config.min_frames, 10)
        self.assertEqual(config.min_detection_rate, 0.5)
        self.assertEqual(config.max_frames, 60)
        self.assertEqual(config.gesture_confidence, 0.9)
        self.assertEqual(config.lip_device, "cpu")
        self.assertEqual(config.gesture_device, "mps")

    def test_check_only_validates_and_skips_the_webcam_session(self) -> None:
        with (
            patch(
                "main.validate_combined_config", return_value=_resolved()
            ) as validate,
            patch("main.run_combined_session") as run_session,
        ):
            exit_code = main.main(["--check-only"])

        self.assertEqual(exit_code, main.EXIT_SUCCESS)
        validate.assert_called_once()
        self.assertTrue(validate.call_args.kwargs["check_imports"])
        run_session.assert_not_called()

    def test_skip_checks_disables_import_validation(self) -> None:
        with (
            patch(
                "main.validate_combined_config", return_value=_resolved()
            ) as validate,
            patch(
                "main.run_combined_session",
                return_value=_summary(completed=True),
            ),
        ):
            exit_code = main.main(["--check-only", "--skip-checks"])

        self.assertEqual(exit_code, main.EXIT_SUCCESS)
        self.assertFalse(validate.call_args.kwargs["check_imports"])

    def test_configuration_error_returns_error_exit_code(self) -> None:
        with patch(
            "main.validate_combined_config",
            side_effect=ConfigurationError("Missing model checkpoint(s)"),
        ):
            exit_code = main.main([])

        self.assertEqual(exit_code, main.EXIT_ERROR)

    def test_completed_session_returns_success(self) -> None:
        with (
            patch("main.validate_combined_config", return_value=_resolved()),
            patch(
                "main.run_combined_session",
                return_value=_summary(completed=True),
            ),
        ):
            exit_code = main.main([])

        self.assertEqual(exit_code, main.EXIT_SUCCESS)

    def test_failed_session_returns_error_exit_code(self) -> None:
        with (
            patch("main.validate_combined_config", return_value=_resolved()),
            patch(
                "main.run_combined_session",
                return_value=_summary(completed=False),
            ),
        ):
            exit_code = main.main([])

        self.assertEqual(exit_code, main.EXIT_ERROR)

    def test_keyboard_interrupt_returns_interrupt_exit_code(self) -> None:
        with (
            patch("main.validate_combined_config", return_value=_resolved()),
            patch(
                "main.run_combined_session",
                side_effect=KeyboardInterrupt,
            ),
        ):
            exit_code = main.main([])

        self.assertEqual(exit_code, main.EXIT_INTERRUPTED)


if __name__ == "__main__":
    unittest.main()
