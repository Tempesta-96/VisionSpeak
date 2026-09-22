from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from inference.environment_setup import (
    ensure_virtualenv,
    environment_has_imports,
    install_requirements,
    setup_lip_environment,
    setup_worker_environment,
    venv_python,
)

MODULE = "inference.environment_setup"


class VenvPythonTests(unittest.TestCase):
    def test_posix_layout(self) -> None:
        with patch(f"{MODULE}.sys.platform", "darwin"):
            self.assertEqual(
                venv_python(Path("/tmp/env")),
                Path("/tmp/env/bin/python"),
            )

    def test_windows_layout(self) -> None:
        with patch(f"{MODULE}.sys.platform", "win32"):
            self.assertEqual(
                venv_python(Path("C:/env")),
                Path("C:/env/Scripts/python.exe"),
            )


class EnvironmentHasImportsTests(unittest.TestCase):
    def test_returns_true_when_probe_succeeds(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=0)
        with patch(f"{MODULE}.subprocess.run", return_value=completed) as run:
            result = environment_has_imports(
                Path("/tmp/python"), ("cv2", "torch")
            )

        self.assertTrue(result)
        command = run.call_args.args[0]
        self.assertEqual(command[:2], ["/tmp/python", "-c"])
        self.assertEqual(command[2], "import cv2, torch")

    def test_returns_false_when_probe_fails(self) -> None:
        completed = subprocess.CompletedProcess(args=[], returncode=1)
        with patch(f"{MODULE}.subprocess.run", return_value=completed):
            self.assertFalse(
                environment_has_imports(Path("/tmp/python"), ("cv2",))
            )

    def test_returns_false_when_probe_cannot_run(self) -> None:
        with patch(f"{MODULE}.subprocess.run", side_effect=OSError("missing")):
            self.assertFalse(
                environment_has_imports(Path("/tmp/python"), ("cv2",))
            )


class EnsureVirtualenvTests(unittest.TestCase):
    def test_existing_environment_is_reused(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            environment_dir = Path(temporary_directory) / ".venv-yolo"
            python = venv_python(environment_dir)
            python.parent.mkdir(parents=True)
            python.write_text("")

            with patch(f"{MODULE}.subprocess.run") as run:
                result = ensure_virtualenv(environment_dir)

            self.assertEqual(result, python)
            run.assert_not_called()

    def test_missing_environment_is_created(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            environment_dir = Path(temporary_directory) / ".venv-yolo"

            with patch(f"{MODULE}.subprocess.run") as run:
                result = ensure_virtualenv(environment_dir)

            self.assertEqual(result, venv_python(environment_dir))
            command = run.call_args.args[0]
            self.assertEqual(command[1:3], ["-m", "venv"])
            self.assertEqual(command[3], str(environment_dir))

    def test_force_recreates_existing_environment(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            environment_dir = Path(temporary_directory) / ".venv-yolo"
            python = venv_python(environment_dir)
            python.parent.mkdir(parents=True)
            python.write_text("")

            with patch(f"{MODULE}.subprocess.run") as run:
                ensure_virtualenv(environment_dir, force=True)

            run.assert_called_once()


class InstallRequirementsTests(unittest.TestCase):
    def test_install_invokes_pip_with_the_requirements_file(self) -> None:
        with patch(f"{MODULE}.subprocess.run") as run:
            install_requirements(Path("/tmp/python"), Path("requirements.txt"))

        self.assertEqual(
            run.call_args.args[0],
            ["/tmp/python", "-m", "pip", "install", "-r", "requirements.txt"],
        )


class SetupWorkerEnvironmentTests(unittest.TestCase):
    def _existing_environment(self, root: Path) -> tuple[Path, Path]:
        environment_dir = root / ".venv-yolo"
        python = venv_python(environment_dir)
        python.parent.mkdir(parents=True)
        python.write_text("")
        return environment_dir, python

    def test_skips_install_when_imports_are_available(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            environment_dir, python = self._existing_environment(
                Path(temporary_directory)
            )
            with (
                patch(
                    f"{MODULE}.environment_has_imports", return_value=True
                ) as probe,
                patch(f"{MODULE}.install_requirements") as install,
            ):
                result, installed = setup_worker_environment(
                    environment_dir, Path("requirements-yolo.txt")
                )

            self.assertEqual(result, python)
            self.assertFalse(installed)
            probe.assert_called_once()
            install.assert_not_called()

    def test_installs_when_imports_are_missing(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            environment_dir, python = self._existing_environment(
                Path(temporary_directory)
            )
            requirements = Path("requirements-yolo.txt")
            with (
                patch(
                    f"{MODULE}.environment_has_imports", return_value=False
                ),
                patch(f"{MODULE}.install_requirements") as install,
            ):
                result, installed = setup_worker_environment(
                    environment_dir, requirements
                )

            self.assertEqual(result, python)
            self.assertTrue(installed)
            install.assert_called_once_with(python, requirements)

    def test_force_reinstall_skips_the_probe(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            environment_dir, _ = self._existing_environment(
                Path(temporary_directory)
            )
            with (
                patch(
                    f"{MODULE}.environment_has_imports"
                ) as probe,
                patch(f"{MODULE}.install_requirements") as install,
            ):
                _, installed = setup_worker_environment(
                    environment_dir,
                    Path("requirements-yolo.txt"),
                    force=True,
                )

            self.assertTrue(installed)
            probe.assert_not_called()
            install.assert_called_once()


class SetupLipEnvironmentTests(unittest.TestCase):
    def test_ready_kernel_reports_no_install(self) -> None:
        with (
            patch(
                f"{MODULE}.environment_has_imports", return_value=True
            ) as probe,
            patch(f"{MODULE}.install_requirements") as install,
        ):
            installed = setup_lip_environment(Path("requirements.txt"))

        self.assertFalse(installed)
        install.assert_not_called()
        self.assertEqual(probe.call_args.args[0], Path(sys.executable))

    def test_missing_imports_install_into_current_interpreter(self) -> None:
        with (
            patch(
                f"{MODULE}.environment_has_imports", return_value=False
            ),
            patch(f"{MODULE}.install_requirements") as install,
        ):
            installed = setup_lip_environment(Path("requirements.txt"))

        self.assertTrue(installed)
        install.assert_called_once_with(
            Path(sys.executable), Path("requirements.txt")
        )


if __name__ == "__main__":
    unittest.main()
