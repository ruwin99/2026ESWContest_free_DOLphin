from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from common import (
    REQUIRED_COLUMNS,
    build_model,
    git_identity,
    load_config,
    read_manifest,
    read_mask,
    resolve_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed audit for capture rust hard-negative training.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", action="store_true")
    return parser.parse_args()


def file_identity(path: Path, expected: str | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        result.update({"bytes": path.stat().st_size, "sha256": sha256_file(path)})
        if expected:
            result["expected_sha256"] = expected.lower()
            result["sha256_matches"] = result["sha256"] == expected.lower()
    return result


def dhash(path: Path) -> int:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    reduced = cv2.resize(image, (9, 8), interpolation=cv2.INTER_AREA)
    bits = reduced[:, 1:] > reduced[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def audit_rows(rows_by_split: dict[str, list[dict[str, str]]], config: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    issues: list[str] = []
    summary: dict[str, Any] = {}
    seen_ids: dict[str, str] = {}
    hashes: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    groups: dict[str, set[str]] = defaultdict(set)
    perceptual: list[tuple[str, str, str, int]] = []
    for split, rows in rows_by_split.items():
        source_counts = Counter(row["source_type"] for row in rows)
        label_counts = Counter(row["label_status"] for row in rows)
        summary[split] = {"rows": len(rows), "source_type_counts": dict(source_counts), "label_status_counts": dict(label_counts), "groups": len({row["group_id"] for row in rows})}
        for row in rows:
            sample = row["sample_id"]
            if sample in seen_ids:
                issues.append(f"duplicate sample_id {sample}: {seen_ids[sample]} and {split}")
            seen_ids[sample] = split
            for key in REQUIRED_COLUMNS:
                if key in {"rust_valid_mask_path", "license_file", "notes"}:
                    continue
                if not row[key] or row[key].upper() == "TBD":
                    issues.append(f"{split}/{sample}: required field {key} is blank/TBD")
            if row["split"] != split:
                issues.append(f"{split}/{sample}: split field mismatch {row['split']}")
            if row["label_status"] not in {"official-public-label-audited", "approved_good", "approved_masked", "sealed-approved"}:
                issues.append(f"{split}/{sample}: unapproved label_status={row['label_status']}")
            if config["data"]["require_distinct_labeler_reviewer"] and row["labeler"].casefold() == row["reviewer"].casefold():
                issues.append(f"{split}/{sample}: labeler and reviewer are not distinct")
            image = resolve_path(row["image_path"])
            mask_path = resolve_path(row["rust_mask_path"])
            if not image.is_file() or not mask_path.is_file():
                issues.append(f"{split}/{sample}: image or mask missing")
                continue
            actual_image_hash = sha256_file(image)
            actual_mask_hash = sha256_file(mask_path)
            if actual_image_hash != row["image_sha256"].lower():
                issues.append(f"{split}/{sample}: image SHA-256 mismatch")
            if actual_mask_hash != row["mask_sha256"].lower():
                issues.append(f"{split}/{sample}: mask SHA-256 mismatch")
            decoded = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if decoded is None:
                issues.append(f"{split}/{sample}: image decode failed")
                continue
            try:
                mask = read_mask(mask_path)
            except Exception as exc:
                issues.append(f"{split}/{sample}: {exc}")
                continue
            if decoded.shape[:2] != mask.shape:
                issues.append(f"{split}/{sample}: image/mask shape mismatch")
            if row["source_type"] == "hard_negative":
                if decoded.shape[:2] != (720, 1280):
                    issues.append(f"{split}/{sample}: hard-negative must be native 1280x720")
                if np.any(mask != 0):
                    issues.append(f"{split}/{sample}: every hard-negative pixel must be Good=0 (no rust or ignore pixels)")
            if split == "sealed_test" and decoded.shape[:2] != (720, 1280):
                issues.append(f"{split}/{sample}: sealed image must be full-frame 1280x720")
            if row["rust_valid_mask_path"]:
                valid_path = resolve_path(row["rust_valid_mask_path"])
                valid = cv2.imread(str(valid_path), cv2.IMREAD_GRAYSCALE)
                if valid is None or valid.shape != mask.shape:
                    issues.append(f"{split}/{sample}: invalid rust_valid_mask")
            hashes[actual_image_hash].append((split, sample, row["group_id"]))
            groups[row["group_id"]].add(split)
            perceptual.append((split, sample, row["group_id"], dhash(image)))

    for digest, items in hashes.items():
        if len(items) > 1 and len({item[0] for item in items}) > 1:
            issues.append(f"exact image duplicate crosses splits sha256={digest}: {items}")
    for group, splits in groups.items():
        if len(splits) > 1:
            issues.append(f"group_id crosses splits: {group} -> {sorted(splits)}")
    near_candidates = []
    for left in range(len(perceptual)):
        split_a, sample_a, group_a, hash_a = perceptual[left]
        for right in range(left + 1, len(perceptual)):
            split_b, sample_b, group_b, hash_b = perceptual[right]
            if split_a == split_b or group_a == group_b:
                continue
            distance = (hash_a ^ hash_b).bit_count()
            if distance <= 2:
                near_candidates.append({"left": [split_a, sample_a, group_a], "right": [split_b, sample_b, group_b], "dhash_distance": distance})
                if len(near_candidates) >= 200:
                    break
        if len(near_candidates) >= 200:
            break
    near_policy = config["data"].get("cross_session_near_duplicate_policy")
    if near_candidates and near_policy != "user_accepted_same_demo_domain_different_session":
        issues.append(f"{len(near_candidates)} cross-split perceptual near-duplicate candidates require manual review (report capped at 200)")
    summary["near_duplicate_policy"] = near_policy or "not_approved"
    summary["near_duplicate_candidates"] = near_candidates
    return summary, issues


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    paths = config["paths"]
    integrity = config["integrity"]
    issues: list[str] = []

    assets = {
        "initial_checkpoint": file_identity(resolve_path(paths["initial_checkpoint"]), integrity["initial_checkpoint_sha256"]),
        "raw_checkpoint": file_identity(resolve_path(paths["raw_checkpoint"]), integrity["raw_checkpoint_sha256"]),
        "baseline_onnx": file_identity(resolve_path(paths["baseline_onnx"]), integrity["baseline_onnx_sha256"]),
    }
    for name, item in assets.items():
        if not item["exists"]:
            issues.append(f"required asset missing: {name} -> {item['path']}")
        elif not item.get("sha256_matches", False):
            issues.append(f"required asset SHA-256 mismatch: {name}")
    official_root = resolve_path(paths["official_training"]).parent
    official_git = git_identity(official_root)
    if official_git["commit"] != integrity["official_commit"]:
        issues.append(f"official code commit mismatch: {official_git['commit']}")

    model_report: dict[str, Any]
    try:
        model = build_model(config, resolve_path(paths["initial_checkpoint"]))
        model_report = {"strict_load": True, "parameters": sum(p.numel() for p in model.parameters())}
    except Exception as exc:
        model_report = {"strict_load": False, "error": repr(exc)}
        issues.append(f"model strict load failed: {exc}")

    manifest_paths = {
        "train": resolve_path(paths["train_manifest"]),
        "validation": resolve_path(paths["validation_manifest"]),
        "sealed_test": resolve_path(paths["sealed_manifest"]),
    }
    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split, path in manifest_paths.items():
        try:
            rows_by_split[split] = read_manifest(path, expected_split=split)
        except Exception as exc:
            issues.append(f"{split} manifest unavailable/invalid: {exc}")
    manifest_summary: dict[str, Any] = {}
    if rows_by_split:
        manifest_summary, manifest_issues = audit_rows(rows_by_split, config)
        issues.extend(manifest_issues)
        for split in ("train", "validation"):
            if split not in rows_by_split:
                continue
            types = Counter(row["source_type"] for row in rows_by_split[split])
            selection_mode = config.get("selection", {}).get("mode", "mixed_positive_and_hard_negative")
            if split == "validation" and selection_mode == "positive_safety_fixed_epoch":
                if types["positive_replay"] == 0 or types["hard_negative"] != 0:
                    issues.append("fixed-epoch validation must contain positive_replay rows only")
            elif types["hard_negative"] == 0 or types["positive_replay"] == 0:
                issues.append(f"{split} requires positive_replay and approved hard_negative rows")
        if "sealed_test" in rows_by_split:
            sealed_types = Counter(row["source_type"] for row in rows_by_split["sealed_test"])
            if sealed_types["hard_negative"] == 0:
                issues.append("sealed_test requires independent hard-negative rows")
            positive_count = sum(value for key, value in sealed_types.items() if key != "hard_negative")
            if positive_count == 0 and not bool(config.get("sealed", {}).get("allow_hard_negative_only_for_candidate", False)):
                issues.append("sealed_test positive rows are missing")
            manifest_summary["sealed_test"]["accuracy_scope"] = (
                "full_candidate_evaluation" if positive_count else config["sealed"]["accuracy_scope_without_positive"]
            )

    commitment_path = resolve_path(paths["sealed_commitment"])
    commitment: dict[str, Any] = {}
    if commitment_path.is_file():
        commitment = yaml.safe_load(commitment_path.read_text(encoding="utf-8")) or {}
        expected_commitment_status = config.get("sealed", {}).get("commitment_status", "APPROVED_LOCKED")
        if commitment.get("status") != expected_commitment_status:
            issues.append(f"sealed commitment status must be {expected_commitment_status} before manifest lock/training")
        sealed_path = manifest_paths["sealed_test"]
        expected = commitment.get("sealed_manifest_sha256")
        if not sealed_path.is_file() or not expected or expected == "TBD" or sha256_file(sealed_path) != str(expected).lower():
            issues.append("sealed commitment manifest SHA-256 is missing or does not match")
        for key in ("positive_and_hard_negative_groups_independent_from_train_validation", "acceptance_gates_approved_by", "approved_utc"):
            if not commitment.get(key) or str(commitment[key]).upper() == "TBD":
                issues.append(f"sealed commitment field is not approved: {key}")
    else:
        issues.append(f"sealed commitment missing: {commitment_path}")

    review_path = resolve_path(paths["review_manifest"])
    review_summary: dict[str, Any] = {"exists": review_path.is_file()}
    if review_path.is_file():
        review_rows = read_manifest(review_path)
        review_summary.update({"rows": len(review_rows), "label_status_counts": dict(Counter(row["label_status"] for row in review_rows))})

    qa_path = resolve_path(paths["two_person_overlay_qa"])
    qa_summary: dict[str, Any] = {"path": str(qa_path), "exists": qa_path.is_file()}
    if not qa_path.is_file():
        issues.append("independent sampled image/mask overlay QA is missing")
    else:
        with qa_path.open("r", encoding="utf-8-sig", newline="") as handle:
            qa_rows = list(csv.DictReader(handle))
        required_qa = {"sample_id", "group_id", "labeler", "reviewer", "decision", "reviewed_utc", "notes"}
        fields = set(qa_rows[0].keys()) if qa_rows else set()
        if not qa_rows or fields != required_qa:
            issues.append("two_person_overlay_qa.csv is empty or has an invalid schema")
        else:
            allow_single = bool(config["data"].get("allow_explicit_user_single_reviewer", False))
            approved_groups = {
                row["group_id"] for row in qa_rows
                if row["decision"] == "approved" and row["labeler"] and row["reviewer"]
                and (allow_single or row["labeler"].casefold() != row["reviewer"].casefold())
            }
            required_groups = {
                row["group_id"] for split in ("train", "validation")
                for row in rows_by_split.get(split, []) if row["source_type"] == "hard_negative"
            }
            missing_groups = sorted(required_groups - approved_groups)
            if missing_groups:
                issues.append("independent overlay QA does not cover hard-negative groups: " + ", ".join(missing_groups))
            qa_summary.update({"rows": len(qa_rows), "approved_groups": len(approved_groups), "required_groups": len(required_groups), "independent_second_reviewer_required": not allow_single})

    report = {
        "audited_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "passed": not issues,
        "status": {"candidate": True, "uart_status": "NOT_FOR_UART", "deployment_status": "NOT_DEPLOYED"},
        "config": file_identity(config_path),
        "assets": assets,
        "official_code": official_git,
        "model": model_report,
        "review_inventory": review_summary,
        "two_person_overlay_qa": qa_summary,
        "manifests": {name: file_identity(path) for name, path in manifest_paths.items()},
        "manifest_summary": manifest_summary,
        "sealed_commitment": file_identity(commitment_path),
        "issues": issues,
    }
    report_path = resolve_path("data_training/capture_rust_teacher_hardneg_v1/reports/data_audit.json")
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.lock:
        if issues:
            raise RuntimeError("Audit failed; manifests were not locked")
        lock_path = resolve_path(paths["manifest_lock"])
        if lock_path.exists():
            raise FileExistsError(f"Manifest lock already exists: {lock_path}")
        files = {
            "config": config_path,
            "train_manifest": manifest_paths["train"],
            "validation_manifest": manifest_paths["validation"],
            "sealed_manifest": manifest_paths["sealed_test"],
            "sealed_commitment": commitment_path,
        }
        lock = {
            "locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "files": {name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size} for name, path in files.items()},
            "status": {"candidate": True, "uart_status": "NOT_FOR_UART", "deployment_status": "NOT_DEPLOYED"},
        }
        write_json(lock_path, lock)
        print(f"Locked: {lock_path}")
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
