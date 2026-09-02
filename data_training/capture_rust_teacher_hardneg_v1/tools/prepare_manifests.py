from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from common import (
    PROJECT_ROOT,
    REQUIRED_COLUMNS,
    IMAGE_EXTENSIONS,
    load_config,
    read_mask,
    relative_path,
    resolve_path,
    sha256_file,
    write_json,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inventory and build capture rust manifests without inferring labels.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("inventory", "build"), required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(f"Existing review manifest has an unexpected schema: {path}")
        return {row["image_path"]: {key: row.get(key, "") for key in REQUIRED_COLUMNS} for row in reader}


def review_hashes(folder: Path) -> dict[str, dict[str, str]]:
    candidates = (folder / "review" / "image_review.csv", folder / "image_review.csv")
    for path in candidates:
        if path.is_file():
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                return {row["filename"]: row for row in csv.DictReader(handle)}
    return {}


def capture_date(filename: str) -> str:
    match = re.search(r"(20\d{6})_(\d{6})", filename)
    if not match:
        return "TBD"
    parsed = dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    return parsed.isoformat()


def inventory(config: dict[str, Any], *, force: bool) -> dict[str, Any]:
    review_path = resolve_path(config["paths"]["review_manifest"])
    lock_path = resolve_path(config["paths"]["manifest_lock"])
    if lock_path.exists():
        raise RuntimeError(f"Manifests are locked; inventory will not modify them: {lock_path}")
    existing = load_existing(review_path)
    rows: list[dict[str, str]] = []
    source_counts: dict[str, int] = {}
    for source in config["data"]["hard_negative_sources"]:
        folder = resolve_path(source)
        if not folder.is_dir():
            source_counts[source] = 0
            continue
        known = review_hashes(folder)
        images = sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        source_counts[source] = len(images)
        for image in images:
            rel = relative_path(image)
            prior = existing.get(rel, {})
            reviewed = known.get(image.name, {})
            image_hash = reviewed.get("sha256") or sha256_file(image)
            sample_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"{source}-{image.stem}").strip("-").lower()
            default = {
                "sample_id": sample_id,
                "image_path": rel,
                "rust_mask_path": "",
                "rust_valid_mask_path": "",
                "split": "unassigned",
                "source_type": "hard_negative",
                "source_dataset": source,
                "source_url": "local:user-supplied",
                "license": "private-user-supplied",
                "license_file": relative_path(PROJECT_ROOT / "CAPTURE_RUST_TEACHER_HARD_NEGATIVE_FINETUNE_AGENT_HANDOFF.md"),
                "capture_date": capture_date(image.name),
                "camera_id": "TBD",
                "session_id": source,
                "source_print_id": "TBD",
                "placement_id": "TBD",
                "lighting_id": "TBD",
                "group_id": "TBD",
                "image_sha256": image_hash.lower(),
                "mask_sha256": "",
                "label_status": "unreviewed",
                "labeler": "",
                "reviewer": "",
                "notes": f"legacy_review={reviewed.get('human_decision', 'missing')}",
            }
            if prior:
                for key in REQUIRED_COLUMNS:
                    if key not in {"sample_id", "image_path", "source_type", "source_dataset", "image_sha256"}:
                        default[key] = prior.get(key, default[key])
            rows.append(default)
    write_manifest(review_path, rows)
    report = {
        "mode": "inventory",
        "review_manifest": str(review_path),
        "rows": len(rows),
        "source_counts": source_counts,
        "label_status_counts": dict(sorted({status: sum(row["label_status"] == status for row in rows) for status in {row["label_status"] for row in rows}}.items())),
        "warning": "Folder names and legacy review CSVs do not approve rust labels. All unreviewed rows remain blocked.",
    }
    write_json(review_path.with_suffix(".inventory.json"), report)
    return report


def vt_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    split_path = resolve_path(config["paths"]["vt_split"])
    vt_root = resolve_path(config["paths"]["vt_root"])
    with split_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, str]] = []
    for source in source_rows:
        image = (vt_root / Path(source["image_path"].replace("/", "\\"))).resolve()
        mask = (vt_root / Path(source["mask_path"].replace("/", "\\"))).resolve()
        split = "validation" if source["split"].lower() in {"val", "validation"} else "train"
        stem = image.stem
        rows.append({
            "sample_id": f"vt-cssd-{split}-{stem}",
            "image_path": relative_path(image),
            "rust_mask_path": relative_path(mask),
            "rust_valid_mask_path": "",
            "split": split,
            "source_type": "positive_replay",
            "source_dataset": "Virginia Tech CSSD 512x512",
            "source_url": "https://data.lib.vt.edu/articles/dataset/16624663",
            "license": "CC0",
            "license_file": "data_training/vt_kd/asset_manifest.json",
            "capture_date": "publisher-dataset",
            "camera_id": "publisher-dataset",
            "session_id": f"vt-{split}-{stem}",
            "source_print_id": "not-applicable",
            "placement_id": "not-applicable",
            "lighting_id": "publisher-dataset",
            "group_id": f"vt-{split}-{stem}",
            "image_sha256": source.get("image_sha256", "").lower() or sha256_file(image),
            "mask_sha256": source.get("mask_sha256", "").lower() or sha256_file(mask),
            "label_status": "official-public-label-audited",
            "labeler": "Virginia-Tech-dataset",
            "reviewer": "local-vt-kd-audit",
            "notes": f"fixed seed42 split; max_severity={source.get('max_severity', '')}",
        })
    return rows


def validate_approved_row(row: dict[str, str], config: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if row["split"] not in {"train", "validation"}:
        issues.append("split must be train or validation")
    for key in ("camera_id", "session_id", "source_print_id", "placement_id", "lighting_id", "group_id", "labeler", "reviewer"):
        if not row[key] or row[key].upper() == "TBD":
            issues.append(f"{key} is missing/TBD")
    if config["data"]["require_distinct_labeler_reviewer"] and row["labeler"].casefold() == row["reviewer"].casefold():
        issues.append("labeler and reviewer must be different")
    return issues


def materialize_good_mask(row: dict[str, str]) -> None:
    image = cv2.imread(str(resolve_path(row["image_path"])), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode approved image: {row['image_path']}")
    mask_path = WORK_PREPARED / "masks" / f"{row['sample_id']}.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    if not mask_path.is_file():
        if not cv2.imwrite(str(mask_path), np.zeros(image.shape[:2], dtype=np.uint8)):
            raise OSError(f"Failed to write zero Good mask: {mask_path}")
    mask = read_mask(mask_path)
    if mask.shape != image.shape[:2] or np.any(mask != 0):
        raise RuntimeError(f"Generated Good mask is invalid: {mask_path}")
    row["rust_mask_path"] = relative_path(mask_path)
    row["rust_valid_mask_path"] = ""
    row["mask_sha256"] = sha256_file(mask_path)


def build(config: dict[str, Any], *, force: bool) -> dict[str, Any]:
    lock_path = resolve_path(config["paths"]["manifest_lock"])
    if lock_path.exists():
        raise RuntimeError(f"Manifests are locked and cannot be rebuilt: {lock_path}")
    review_path = resolve_path(config["paths"]["review_manifest"])
    existing = load_existing(review_path)
    if not existing:
        raise RuntimeError("Run inventory first")
    approved: list[dict[str, str]] = []
    rejected: list[dict[str, Any]] = []
    for row in existing.values():
        if row["label_status"] not in {"approved_good", "approved_masked"}:
            continue
        issues = validate_approved_row(row, config)
        if issues:
            rejected.append({"sample_id": row["sample_id"], "issues": issues})
            continue
        image = resolve_path(row["image_path"])
        if not image.is_file() or sha256_file(image) != row["image_sha256"].lower():
            rejected.append({"sample_id": row["sample_id"], "issues": ["image missing or SHA-256 mismatch"]})
            continue
        if row["label_status"] == "approved_good":
            materialize_good_mask(row)
        else:
            mask = resolve_path(row["rust_mask_path"])
            if not mask.is_file():
                rejected.append({"sample_id": row["sample_id"], "issues": ["approved_masked mask missing"]})
                continue
            read_mask(mask)
            row["mask_sha256"] = sha256_file(mask)
        approved.append(row)
    all_rows = vt_rows(config) + approved
    train = sorted((row for row in all_rows if row["split"] == "train"), key=lambda row: row["sample_id"])
    validation = sorted((row for row in all_rows if row["split"] == "validation"), key=lambda row: row["sample_id"])
    train_path = resolve_path(config["paths"]["train_manifest"])
    validation_path = resolve_path(config["paths"]["validation_manifest"])
    write_manifest(train_path, train)
    write_manifest(validation_path, validation)
    report = {
        "mode": "build",
        "approved_hard_negative_rows": len(approved),
        "rejected_approved_rows": rejected,
        "train": {"rows": len(train), "positive": sum(row["source_type"] != "hard_negative" for row in train), "hard_negative": sum(row["source_type"] == "hard_negative" for row in train)},
        "validation": {"rows": len(validation), "positive": sum(row["source_type"] != "hard_negative" for row in validation), "hard_negative": sum(row["source_type"] == "hard_negative" for row in validation)},
        "ready_for_audit": bool(approved) and not rejected,
    }
    write_json(resolve_path("data_training/capture_rust_teacher_hardneg_v1/reports/manifest_build.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if rejected:
        raise RuntimeError("Approved rows failed validation; see manifest_build.json")
    return report


WORK_PREPARED = resolve_path("data_training/capture_rust_teacher_hardneg_v1/prepared")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    report = inventory(config, force=args.force) if args.mode == "inventory" else build(config, force=args.force)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

