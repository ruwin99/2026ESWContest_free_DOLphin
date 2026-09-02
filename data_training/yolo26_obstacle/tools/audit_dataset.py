from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def class_names(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda value: int(value))]
    raise ValueError("data.yaml names must be a list or integer-keyed mapping")


def resolve_split(yaml_path: Path, config: dict[str, object], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"data.yaml is missing a string '{key}' path")
    base = Path(str(config.get("path", yaml_path.parent)))
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def label_path(image: Path, images_root: Path) -> Path:
    parts = list(images_root.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
        parts[index] = "labels"
        labels_root = Path(*parts)
    except ValueError:
        labels_root = images_root.parent / "labels"
    return labels_root / image.relative_to(images_root).with_suffix(".txt")


def audit_split(images_root: Path, classes: int) -> dict[str, object]:
    if not images_root.is_dir():
        return {"path": str(images_root), "exists": False, "images": 0, "issues": ["missing images directory"]}
    images = sorted(path for path in images_root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
    missing_labels: list[str] = []
    invalid_labels: list[str] = []
    empty_labels = 0
    boxes = 0
    for image in images:
        label = label_path(image, images_root)
        if not label.is_file():
            missing_labels.append(str(label))
            continue
        lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        for line_number, line in enumerate(lines, 1):
            fields = line.split()
            try:
                if len(fields) != 5:
                    raise ValueError("expected 5 fields")
                category = int(fields[0])
                coordinates = [float(value) for value in fields[1:]]
                if not 0 <= category < classes:
                    raise ValueError(f"class {category} outside [0,{classes - 1}]")
                if not all(0.0 <= value <= 1.0 for value in coordinates):
                    raise ValueError("coordinates outside [0,1]")
                if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
                    raise ValueError("non-positive box size")
                boxes += 1
            except ValueError as exc:
                invalid_labels.append(f"{label}:{line_number}: {exc}")
    issues: list[str] = []
    if not images:
        issues.append("no images")
    if missing_labels:
        issues.append(f"{len(missing_labels)} missing label files")
    if invalid_labels:
        issues.append(f"{len(invalid_labels)} invalid label rows")
    return {
        "path": str(images_root),
        "exists": True,
        "images": len(images),
        "boxes": boxes,
        "empty_label_images": empty_labels,
        "missing_label_count": len(missing_labels),
        "invalid_label_count": len(invalid_labels),
        "missing_label_examples": missing_labels[:10],
        "invalid_label_examples": invalid_labels[:10],
        "issues": issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed audit for a YOLO detection export.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--expected-min-images", type=int, default=8000)
    args = parser.parse_args()
    yaml_path = args.data.resolve()
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML missing: {yaml_path}")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    names = class_names(config.get("names"))
    split_keys = {"train": "train", "validation": "val", "test": "test"}
    splits = {
        label: audit_split(resolve_split(yaml_path, config, key), len(names))
        for label, key in split_keys.items()
    }
    total_images = sum(int(value["images"]) for value in splits.values())
    issues = [f"{name}: {issue}" for name, value in splits.items() for issue in value["issues"]]
    if total_images < args.expected_min_images:
        issues.append(f"only {total_images} images; expected at least {args.expected_min_images}")
    report = {
        "passed": not issues,
        "data_yaml": str(yaml_path),
        "class_names": names,
        "class_count": len(names),
        "total_images": total_images,
        "splits": splits,
        "issues": issues,
        "manual_checks": [
            "Confirm the class names are the intended obstacle classes.",
            "Confirm Roboflow Version 1 already contains the intended 640x640 Fit black-edge preprocessing.",
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
