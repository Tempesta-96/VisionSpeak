from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from inference.launcher import (
    CombinedConfig,
    ConfigurationError,
    build_worker_command,
    resolve_interpreter,
    run_combined_session,
    validate_combined_config,
)
from inference.protocol import EXIT_QUIT, EXIT_SWITCH


class CombinedLauncherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.lip_python = self.root / "lip-python"
        self.gesture_python = self.root / "gesture-python"
        self.lip_checkpoint = self.root / "lip.pt"
        self.gesture_checkpoint = self.root / "gesture.pt"
        for path in (
            self.lip_python,
            self.gesture_python,
            self.lip_checkpoint,
            self.gesture_checkpoint,
        ):
            path.write_bytes(b"test")

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def config(self, **overrides) -> CombinedConfig:
        values = {
            "project_root": self.root,
            "lip_python": self.lip_python,
            "gesture_python": self.gesture_python,
            "lip_checkpoint": self.lip_checkpoint,
            "gesture_checkpoint": self.gesture_checkpoint,
        }
        values.update(overrides)
        return CombinedConfig(**values)

    def test_session_switches_workers_then_quits(self) -> None:
        return_codes = iter((EXIT_SWITCH, EXIT_SWITCH, EXIT_QUIT))
        invoked_modes: list[str] = []

        def runner(mode, _config):
            invoked_modes.append(mode)
            return next(return_codes)

        summary = run_combined_session(self.config(), worker_runner=runner)

        self.assertTrue(summary.completed)
        self.assertEqual(summary.switch_count, 2)
        self.assertEqual(invoked_modes, ["lip", "gesture", "lip"])
        self.assertEqual([event.action for event in summary.events], ["switch", "switch", "quit"])

    def test_worker_failure_is_reported_in_summary(self) -> None:
        summary = run_combined_session(
            self.config(initial_mode="gesture"),
            worker_runner=lambda _mode, _config: 7,
        )

        self.assertFalse(summary.completed)
        self.assertIn("exit code 7", summary.error)
        self.assertEqual(summary.events[0].action, "error")

    def test_missing_checkpoint_is_actionable(self) -> None:
        self.lip_checkpoint.unlink()
        with self.assertRaisesRegex(ConfigurationError, "Missing model checkpoint"):
            validate_combined_config(self.config(), check_imports=False)

    def test_worker_commands_use_separate_interpreters(self) -> None:
        resolved = validate_combined_config(self.config(), check_imports=False)
        lip_command = build_worker_command("lip", resolved)
        gesture_command = build_worker_command("gesture", resolved)

        self.assertEqual(Path(lip_command[0]), self.lip_python)
        self.assertEqual(Path(gesture_command[0]), self.gesture_python)
        self.assertIn("inference.lip_worker", lip_command)
        self.assertIn("inference.gesture_worker", gesture_command)

    def test_interpreter_auto_discovery_supports_both_venv_layouts(self) -> None:
        posix_python = self.root / ".venv" / "bin" / "python"
        windows_python = self.root / ".venv-yolo" / "Scripts" / "python.exe"
        posix_python.parent.mkdir(parents=True)
        windows_python.parent.mkdir(parents=True)
        posix_python.write_bytes(b"test")
        windows_python.write_bytes(b"test")

        self.assertEqual(resolve_interpreter("lip", None, self.root), posix_python)
        self.assertEqual(resolve_interpreter("gesture", None, self.root), windows_python)

    def test_interpreter_discovery_preserves_virtualenv_symlink(self) -> None:
        target = self.root / "base-python"
        target.write_bytes(b"test")
        virtualenv_python = self.root / ".venv" / "bin" / "python"
        virtualenv_python.parent.mkdir(parents=True)
        try:
            virtualenv_python.symlink_to(target)
        except OSError as error:
            self.skipTest(f"Symlinks are unavailable: {error}")

        resolved = resolve_interpreter("lip", None, self.root)
        self.assertEqual(resolved, virtualenv_python)
        self.assertNotEqual(resolved, target)


if __name__ == "__main__":
    unittest.main()
