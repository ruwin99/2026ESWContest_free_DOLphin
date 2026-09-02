from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

from common import (
    IMAGE_EXTENSIONS,
    load_config,
    read_manifest,
    relative_path,
    resolve_path,
    sha256_file,
    write_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a new hard-negative-only locked holdout without running inference.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    return parser.parse_args()


def capture_date(filename: str) -> str:
    match = re.search(r"(20\d{6})_(\d{6})", filename)
    if not match:
        return "unrecorded"
    return dt.datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S").isoformat()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock_path = resolve_path(config["paths"]["manifest_lock"])
    if lock_path.exists():
        raise RuntimeError(f"Manifest already locked: {lock_path}")
    source = args.source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(source)
    images = sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No sealed images found: {source}")

    prior_rows = []
    for split, key in (("train", "train_manifest"), ("validation", "validation_manifest")):
        prior_rows.extend(read_manifest(resolve_path(config["paths"][key]), expected_split=split))
    prior_hashes = {row["image_sha256"] for row in prior_rows}
    output_masks = resolve_path("data_training/capture_rust_teacher_hardneg_v1/prepared/sealed_masks")
    output_masks.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    group_id = f"unrecorded-demo-source|unrecorded|{source.name}"
    rows = []
    seen_hashes = set()
    for image_path in images:
        image_hash = sha256_file(image_path)
        if image_hash in prior_hashes:
            raise RuntimeError(f"Sealed image duplicates train/validation: {image_path}")
        if image_hash in seen_hashes:
            raise RuntimeError(f"Duplicate image inside sealed source: {image_path}")
        seen_hashes.add(image_hash)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.shape[:2] != (720, 1280):
            raise ValueError(f"Sealed image must decode as native 1280x720: {image_path}")
        sample_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", f"sealed-v8-{image_path.stem}").strip("-").lower()
        mask_path = output_masks / f"{sample_id}.png"
        if not cv2.imwrite(str(mask_path), np.zeros((720, 1280), dtype=np.uint8)):
            raise OSError(f"Unable to write sealed Good mask: {mask_path}")
        rows.append({
            "sample_id": sample_id,
            "image_path": relative_path(image_path),
            "rust_mask_path": relative_path(mask_path),
            "rust_valid_mask_path": "",
            "split": "sealed_test",
            "source_type": "hard_negative",
            "source_dataset": source.name,
            "source_url": "local:user-supplied",
            "license": "private-user-supplied",
            "license_file": "CAPTURE_RUST_TEACHER_HARD_NEGATIVE_FINETUNE_AGENT_HANDOFF.md",
            "capture_date": capture_date(image_path.name),
            "camera_id": "unrecorded-user-camera",
            "session_id": source.name,
            "source_print_id": "unrecorded-demo-source",
            "placement_id": "unrecorded",
            "lighting_id": source.name,
            "group_id": group_id,
            "image_sha256": image_hash,
            "mask_sha256": sha256_file(mask_path),
            "label_status": "sealed-approved",
            "labeler": args.approved_by,
            "reviewer": args.approved_by,
            "notes": "Explicit user approval as rust-free Good; hard-negative-only locked holdout; no model inference before commitment.",
        })
    manifest_path = resolve_path(config["paths"]["sealed_manifest"])
    write_manifest(manifest_path, rows)
    manifest_hash = sha256_file(manifest_path)
    commitment = {
        "format_version": 1,
        "status": "APPROVED_LOCKED",
        "created_utc": timestamp,
        "purpose": "capture-rust-hard-negative-final-evaluation",
        "group_rule": "source_print_id/placement_id/session_id never crosses split",
        "sealed_manifest": relative_path(manifest_path),
        "sealed_manifest_sha256": manifest_hash,
        "allowed_reads_before_unseal": ["file_exists", "sha256", "decode_ok", "shape", "allowed_mask_values"],
        "forbidden_before_unseal": ["model_inference", "metrics", "threshold_tuning", "sample_selection"],
        "unseal_condition": "candidate_and_baseline_hashes_and_acceptance_gates_frozen",
        "unseal_once": True,
        "positive_and_hard_negative_groups_independent_from_train_validation": "hard-negative-only independent session; positive sealed set missing",
        "acceptance_gates_approved_by": args.approved_by,
        "approved_utc": timestamp,
        "positive_sealed_set": "MISSING",
        "accuracy_scope": "false_positive_suppression_only_ACCURACY_NOT_FINAL",
        "prior_model_inference_on_sealed_images": False,
    }
    commitment_path = resolve_path(config["paths"]["sealed_commitment"])
    commitment_path.write_text(yaml.safe_dump(commitment, sort_keys=False, allow_unicode=True), encoding="utf-8")
    report = {
        "prepared_utc": timestamp,
        "source": str(source),
        "images": len(rows),
        "groups": 1,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "commitment": str(commitment_path),
        "accuracy_scope": commitment["accuracy_scope"],
    }
    write_json(resolve_path("data_training/capture_rust_teacher_hardneg_v1/reports/sealed_preparation.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

