from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))

from audit_dataset import audit_split, class_names  # noqa: E402


class DatasetAuditTests(unittest.TestCase):
    def test_valid_detection_label(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "train" / "images"
            labels = root / "train" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "sample.jpg").write_bytes(b"placeholder")
            (labels / "sample.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
            result = audit_split(images, 1)
            self.assertEqual(result["images"], 1)
            self.assertEqual(result["boxes"], 1)
            self.assertEqual(result["issues"], [])

    def test_invalid_coordinate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "valid" / "images"
            labels = root / "valid" / "labels"
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            (images / "sample.png").write_bytes(b"placeholder")
            (labels / "sample.txt").write_text("0 1.5 0.5 0.2 0.2\n", encoding="utf-8")
            result = audit_split(images, 1)
            self.assertEqual(result["invalid_label_count"], 1)
            self.assertTrue(result["issues"])

    def test_mapping_class_names_are_sorted_numerically(self) -> None:
        self.assertEqual(class_names({1: "debris", 0: "obstacle"}), ["obstacle", "debris"])


if __name__ == "__main__":
    unittest.main()
