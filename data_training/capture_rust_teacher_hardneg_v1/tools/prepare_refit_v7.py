from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from common import read_manifest, resolve_path, sha256_file, write_json, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build immutable v7-refit manifests: all v1-v7 normals in train, VT positives only in validation."
    )
    parser.add_argument("--source-train", type=Path, required=True)
    parser.add_argument("--source-validation", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, required=True)
    parser.add_argument("--output-validation", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_train = args.output_train.resolve()
    output_validation = args.output_validation.resolve()
    if output_train.exists() or output_validation.exists():
        raise FileExistsError("Refusing to overwrite existing v7-refit manifests")

    source_train = read_manifest(args.source_train.resolve(), expected_split="train")
    source_validation = read_manifest(args.source_validation.resolve(), expected_split="validation")
    train_rows = [dict(row) for row in source_train]
    validation_rows = []
    for row in source_validation:
        copied = dict(row)
        if copied["source_type"] == "hard_negative":
            copied["split"] = "train"
            copied["notes"] = (copied["notes"] + "; " if copied["notes"] else "") + "moved_from_v7_validation_to_v7_refit_train"
            train_rows.append(copied)
        else:
            validation_rows.append(copied)

    train_rows.sort(key=lambda row: row["sample_id"])
    validation_rows.sort(key=lambda row: row["sample_id"])
    sample_ids = [row["sample_id"] for row in train_rows + validation_rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("Duplicate sample_id detected while building v7-refit manifests")
    if any("v8" in row["source_dataset"].casefold() or "v8" in row["image_path"].casefold() for row in train_rows):
        raise RuntimeError("v8 must never enter v7-refit training")
    train_types = Counter(row["source_type"] for row in train_rows)
    validation_types = Counter(row["source_type"] for row in validation_rows)
    normal_sources = Counter(row["source_dataset"] for row in train_rows if row["source_type"] == "hard_negative")
    expected_sources = {
        "for model test", "for model test v2", "for model test v3", "for model test v4",
        "for model test v5", "for model test v6", "for model test v7", "jetson_normal_20260814",
    }
    if set(normal_sources) != expected_sources:
        raise RuntimeError(f"Unexpected v7-refit normal sources: {sorted(normal_sources)}")
    if train_types != Counter({"hard_negative": 1462, "positive_replay": 316}):
        raise RuntimeError(f"Unexpected v7-refit train composition: {dict(train_types)}")
    if validation_types != Counter({"positive_replay": 80}):
        raise RuntimeError(f"Unexpected v7-refit validation composition: {dict(validation_types)}")

    write_manifest(output_train, train_rows)
    write_manifest(output_validation, validation_rows)
    report = {
        "policy": "all approved normal captures through v7 are train; v8 is excluded and reserved for reused final confirmation",
        "source_train": str(args.source_train.resolve()),
        "source_validation": str(args.source_validation.resolve()),
        "train": {
            "path": str(output_train),
            "sha256": sha256_file(output_train),
            "rows": len(train_rows),
            "source_type_counts": dict(train_types),
            "hard_negative_source_counts": dict(sorted(normal_sources.items())),
        },
        "validation": {
            "path": str(output_validation),
            "sha256": sha256_file(output_validation),
            "rows": len(validation_rows),
            "source_type_counts": dict(validation_types),
            "purpose": "positive safety only; no normal false-positive selection",
        },
        "v8_in_training": False,
    }
    report_path = output_train.with_suffix(".report.json")
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
