from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image


ROOT = Path(os.environ.get("RAIL_ROBOT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
PROJECT = Path(__file__).resolve().parents[1]
V1_DATASET = PROJECT / "workspace/datasets/waste_detect_hn_v8_1280_seed42"
OUTPUT_DATASET = PROJECT / "workspace/datasets/waste_detect_hn_all1410_v2_1280_seed42"
V1_MANIFEST = PROJECT / "manifests/split_manifest.csv"
OUTPUT_MANIFEST = PROJECT / "manifests/split_manifest_all1410_v2.csv"
OUTPUT_REPORT = PROJECT / "reports/prepared_all1410_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def main() -> None:
    if OUTPUT_DATASET.exists():
        raise FileExistsError(f"Refusing to overwrite prepared dataset: {OUTPUT_DATASET}")
    if not V1_DATASET.is_dir() or not V1_MANIFEST.is_file():
        raise FileNotFoundError("Prepared v1 dataset/manifest is required")

    v1_manifest_rows = list(csv.DictReader(V1_MANIFEST.open("r", encoding="utf-8-sig", newline="")))
    known_hashes = {row["source_sha256"] for row in v1_manifest_rows}
    new_records: list[dict[str, str]] = []
    new_hashes: set[str] = set()

    for version in range(2, 8):
        source_dir = ROOT / f"for model test v{version}"
        review_csv = source_dir / "review/image_review.csv"
        if not review_csv.is_file():
            raise FileNotFoundError(f"Review manifest missing: {review_csv}")
        review_rows = list(csv.DictReader(review_csv.open("r", encoding="utf-8-sig", newline="")))
        jpg_by_name = {path.name: path for path in source_dir.rglob("*.jpg")}
        if len(review_rows) != len(jpg_by_name):
            raise RuntimeError(
                f"v{version} review/image mismatch: {len(review_rows)} != {len(jpg_by_name)}"
            )
        for row in review_rows:
            source = jpg_by_name.get(row["filename"])
            if source is None:
                raise FileNotFoundError(f"Reviewed image missing: {row['filename']}")
            digest = sha256(source)
            if digest != row["sha256"]:
                raise RuntimeError(f"Reviewed SHA changed: {source}")
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                size = image.size
            if size != (1280, 720):
                raise RuntimeError(f"Camera shape mismatch {size}: {source}")
            if digest in known_hashes or digest in new_hashes:
                raise RuntimeError(f"Exact hard-negative duplicate: {source}")
            new_hashes.add(digest)
            new_records.append(
                {
                    "version": f"v{version}",
                    "source": str(source),
                    "filename": source.name,
                    "sha256": digest,
                    "burst": row.get("burst", ""),
                }
            )

    if len(new_records) != 1245:
        raise RuntimeError(f"Expected 1245 v2-v7 JPG files, found {len(new_records)}")

    images_dir = OUTPUT_DATASET / "images/train_hard_negative_v2_v7"
    labels_dir = OUTPUT_DATASET / "labels/train_hard_negative_v2_v7"
    images_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    appended_rows: list[dict[str, str]] = []
    new_training_paths: list[str] = []
    for record in new_records:
        destination_name = f"{record['version']}_{record['filename']}"
        destination_image = images_dir / destination_name
        destination_label = labels_dir / Path(destination_name).with_suffix(".txt")
        shutil.copy2(record["source"], destination_image)
        destination_label.touch()
        new_training_paths.append(str(destination_image))
        appended_rows.append(
            {
                "sample_id": destination_image.stem,
                "source": f"for_model_test_{record['version']}",
                "source_sha256": record["sha256"],
                "source_group": f"{record['version']}_burst_{record['burst']}",
                "split": "train",
                "image_path": str(destination_image),
                "label_path": str(destination_label),
                "label_mode": "user_confirmed_empty_hard_negative",
            }
        )

    old_train = read_lines(V1_DATASET / "train_images.txt")
    old_val = read_lines(V1_DATASET / "val_images.txt")
    old_test = read_lines(V1_DATASET / "test_images.txt")
    (OUTPUT_DATASET / "train_images.txt").write_text(
        "\n".join(old_train + new_training_paths) + "\n", encoding="utf-8"
    )
    (OUTPUT_DATASET / "val_images.txt").write_text("\n".join(old_val) + "\n", encoding="utf-8")
    (OUTPUT_DATASET / "test_images.txt").write_text("\n".join(old_test) + "\n", encoding="utf-8")
    data_yaml = OUTPUT_DATASET / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str(OUTPUT_DATASET / "train_images.txt"),
                "val": str(OUTPUT_DATASET / "val_images.txt"),
                "test": str(OUTPUT_DATASET / "test_images.txt"),
                "nc": 1,
                "names": ["obstacle"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    fieldnames = [
        "sample_id", "source", "source_sha256", "source_group", "split",
        "image_path", "label_path", "label_mode",
    ]
    with OUTPUT_MANIFEST.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(v1_manifest_rows + appended_rows)

    report = {
        "prepared": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256(data_yaml),
        "train_images": len(old_train) + len(new_training_paths),
        "original_deduplicated_train_images": 3390,
        "hard_negative_train_images": 110 + len(new_training_paths),
        "hard_negative_v2_v7_images": len(new_training_paths),
        "hard_negative_v8_train_images": 110,
        "hard_negative_v8_development_images": 55,
        "all_supplied_hard_negative_images": len(new_training_paths) + 165,
        "validation_images": len(old_val),
        "test_images": len(old_test),
        "review_png_files_excluded": 33,
        "exact_duplicate_count": 0,
        "camera_shape_hw": [720, 1280],
        "training_imgsz": 1280,
        "rectangular_training": True,
        "manifest": str(OUTPUT_MANIFEST),
        "manifest_sha256": sha256(OUTPUT_MANIFEST),
    }
    OUTPUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

