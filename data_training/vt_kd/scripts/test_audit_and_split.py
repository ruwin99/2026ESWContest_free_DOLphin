from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_and_split import ALLOWED_MASK_COLORS, AuditError, audit_and_split


class AuditAndSplitTest(unittest.TestCase):
    def _make_fixture(self, root: Path) -> None:
        for partition in ("Train", "Test"):
            (root / partition / "images_512").mkdir(parents=True)
            (root / partition / "mask_512").mkdir(parents=True)

        global_index = 0
        for partition, count in (("Train", 12), ("Test", 4)):
            for local_index in range(count):
                severity = 1 + (global_index % 3)

                # A high-contrast deterministic pattern keeps every encoded
                # image unique, including after JPEG compression.
                yy, xx = np.indices((16, 16), dtype=np.uint16)
                image = np.empty((16, 16, 3), dtype=np.uint8)
                image[..., 0] = (xx * 13 + global_index * 17) % 256
                image[..., 1] = (yy * 19 + global_index * 29) % 256
                image[..., 2] = ((xx + yy) * 11 + global_index * 37) % 256

                mask = np.zeros((16, 16, 3), dtype=np.uint8)
                for class_index in range(1, severity + 1):
                    mask[class_index : class_index + 2, 2 * class_index : 2 * class_index + 2] = (
                        ALLOWED_MASK_COLORS[class_index]
                    )
                # Encode the sample number using allowed pixels so mask files
                # are unique without introducing a severity above the stratum.
                for bit in range(8):
                    if global_index & (1 << bit):
                        mask[15, bit] = ALLOWED_MASK_COLORS[severity]

                Image.fromarray(image, mode="RGB").save(
                    root / partition / "images_512" / f"{local_index}.jpeg",
                    quality=95,
                )
                Image.fromarray(mask, mode="RGB").save(
                    root / partition / "mask_512" / f"{local_index}.png"
                )
                global_index += 1

    def test_audit_split_is_portable_deterministic_and_excludes_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            dataset = temp / "512x512"
            self._make_fixture(dataset)
            output = temp / "splits" / "fixture.csv"

            stats = audit_and_split(
                dataset,
                output,
                seed=42,
                val_ratio=0.25,
                expected_train_pairs=12,
                expected_test_pairs=4,
                expected_size=(16, 16),
            )

            first_csv = output.read_bytes()
            with output.open("r", encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))

            self.assertEqual(12, len(rows))
            self.assertEqual(9, sum(row["split"] == "train" for row in rows))
            self.assertEqual(3, sum(row["split"] == "val" for row in rows))
            self.assertTrue(all(row["image_path"].startswith("Train/") for row in rows))
            self.assertTrue(all(row["mask_path"].startswith("Train/") for row in rows))
            self.assertTrue(all("\\" not in row["image_path"] for row in rows))
            self.assertTrue(all("\\" not in row["mask_path"] for row in rows))
            self.assertTrue(all(row["group_id"] == "" for row in rows))

            expected_digest = hashlib.sha256(first_csv).hexdigest()
            self.assertEqual(
                f"{expected_digest}  fixture.csv\n",
                Path(f"{output}.sha256").read_text(encoding="ascii"),
            )
            stats_file = output.with_name("fixture.stats.json")
            written_stats = json.loads(stats_file.read_text(encoding="utf-8"))
            self.assertEqual(expected_digest, written_stats["artifacts"]["split_csv_sha256"])
            self.assertEqual(4, written_stats["splits"]["test"]["pair_count"])
            self.assertEqual(3, stats["splits"]["val"]["pair_count"])

            # Re-running with the same inputs and seed must produce identical
            # CSV bytes and therefore the same sidecar digest.
            audit_and_split(
                dataset,
                output,
                seed=42,
                val_ratio=0.25,
                expected_train_pairs=12,
                expected_test_pairs=4,
                expected_size=(16, 16),
            )
            self.assertEqual(first_csv, output.read_bytes())

    def test_disallowed_mask_color_fails_before_writing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temp = Path(temporary_directory)
            dataset = temp / "512x512"
            self._make_fixture(dataset)
            bad_mask = dataset / "Train" / "mask_512" / "0.png"
            with Image.open(bad_mask) as mask:
                pixels = np.asarray(mask.convert("RGB")).copy()
            pixels[0, 0] = (1, 2, 3)
            Image.fromarray(pixels, mode="RGB").save(bad_mask)

            output = temp / "splits" / "should_not_exist.csv"
            with self.assertRaisesRegex(AuditError, "disallowed RGB color"):
                audit_and_split(
                    dataset,
                    output,
                    val_ratio=0.25,
                    expected_train_pairs=12,
                    expected_test_pairs=4,
                    expected_size=(16, 16),
                )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
