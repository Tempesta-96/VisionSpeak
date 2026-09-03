from __future__ import annotations

import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def requirement_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        for separator in ("==", ">=", "<=", "<", ">", "~=", "!=", "["):
            value = value.split(separator, 1)[0]
        names.add(value.lower())
    return names


class RequirementsProfileTests(unittest.TestCase):
    def test_opencv_profiles_are_isolated(self) -> None:
        lip = requirement_names(PROJECT_ROOT / "requirements.txt")
        yolo = requirement_names(PROJECT_ROOT / "requirements-yolo.txt")

        self.assertIn("opencv-contrib-python", lip)
        self.assertNotIn("opencv-python", lip)
        self.assertIn("opencv-python", yolo)
        self.assertNotIn("opencv-contrib-python", yolo)
        self.assertNotIn("ultralytics", lip)
        self.assertIn("ultralytics", yolo)

    def test_archive_only_imageio_is_not_an_active_dependency(self) -> None:
        self.assertNotIn(
            "imageio",
            requirement_names(PROJECT_ROOT / "requirements.txt"),
        )


if __name__ == "__main__":
    unittest.main()
