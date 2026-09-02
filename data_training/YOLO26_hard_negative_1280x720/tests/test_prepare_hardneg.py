from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.prepare_hardneg import assign_groups, parse_capture_time, validate_label


class PrepareHardNegativeTests(unittest.TestCase):
    def test_parse_capture_timestamp(self) -> None:
        value = parse_capture_time(Path("training_20260820_153412_957097.jpg"))
        self.assertEqual(value, datetime(2026, 8, 20, 15, 34, 12, 957097))

    def test_assign_groups_at_more_than_two_second_gap(self) -> None:
        files = [
            Path("training_20260820_153412_000000.jpg"),
            Path("training_20260820_153413_000000.jpg"),
            Path("training_20260820_153416_000001.jpg"),
        ]
        groups = assign_groups(files, 2.0)
        self.assertEqual(list(groups.values()), [1, 1, 2])

    def test_validate_label_rejects_bad_coordinate(self) -> None:
        with TemporaryDirectory() as directory:
            label = Path(directory) / "bad.txt"
            label.write_text("0 0.5 0.5 1.2 0.2\n", encoding="utf-8")
            count, boxes, issues = validate_label(label, 1)
        self.assertEqual(count, 0)
        self.assertEqual(boxes, [])
        self.assertIn("out-of-range", issues[0])


if __name__ == "__main__":
    unittest.main()


