from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


WORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORK_ROOT.parents[1]
MODELS_ROOT = WORK_ROOT / "models"
if str(MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(MODELS_ROOT))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def onnx_shape(value: Any) -> list[int | None]:
    return [
        int(dim.dim_value) if dim.HasField("dim_value") else None
        for dim in value.type.tensor_type.shape.dim
    ]


def onnx_dtype(value: Any) -> str:
    try:
        import onnx
        return str(onnx.helper.tensor_dtype_to_np_dtype(value.type.tensor_type.elem_type))
    except Exception:
        return "unknown"


def audit_teacher(path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "expected_sha256": contract["sha256"],
        "expected_byte_size": int(contract["byte_size"]),
        "issues": [],
    }
    if not path.is_file():
        result["issues"].append("missing file")
        return result
    result["byte_size"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    if result["byte_size"] != result["expected_byte_size"]:
        result["issues"].append("byte size mismatch")
    if result["sha256"] != result["expected_sha256"]:
        result["issues"].append("SHA-256 mismatch")
    try:
        import onnx

        model = onnx.load(str(path), load_external_data=True)
        inputs = {item.name: onnx_shape(item) for item in model.graph.input}
        outputs = {item.name: onnx_shape(item) for item in model.graph.output}
        input_dtypes = {item.name: onnx_dtype(item) for item in model.graph.input}
        output_dtypes = {item.name: onnx_dtype(item) for item in model.graph.output}
        expected_input = contract["input"]
        expected_output = contract["output"]
        result["inputs"] = inputs
        result["outputs"] = outputs
        result["input_dtypes"] = input_dtypes
        result["output_dtypes"] = output_dtypes
        if inputs != {expected_input["name"]: expected_input["shape"]}:
            result["issues"].append("input contract mismatch")
        if outputs != {expected_output["name"]: expected_output["shape"]}:
            result["issues"].append("output contract mismatch")
        if input_dtypes != {expected_input["name"]: expected_input["dtype"]}:
            result["issues"].append("input dtype mismatch")
        if output_dtypes != {expected_output["name"]: expected_output["dtype"]}:
            result["issues"].append("output dtype mismatch")
    except Exception as error:
        result["issues"].append(f"ONNX audit failed: {error}")
    return result


RUBRIC_FIELDS = {
    "rubric_name",
    "rubric_version",
    "approved_by",
    "approved_at",
    "source_document_path_or_id",
    "source_document_sha256",
    "rubric_file_sha256",
    "class_0_definition",
    "class_1_fair_definition",
    "class_2_poor_definition",
    "class_3_severe_definition",
    "ambiguous_and_ignore_rules",
}


def audit_rubric(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "issues": []}
    if not path.is_file():
        result["issues"].append("approved rust rubric missing")
        return result
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = sorted(key for key in RUBRIC_FIELDS if not data.get(key))
    if missing:
        result["issues"].append(f"missing rubric fields: {missing}")
    result["sha256"] = sha256_file(path)
    if data.get("rubric_sha_scheme") == "canonical_yaml_sorted_keys_excluding_rubric_file_sha256_v1":
        canonical = dict(data)
        canonical.pop("rubric_file_sha256", None)
        payload = yaml.safe_dump(canonical, sort_keys=True, allow_unicode=True).encode("utf-8")
        canonical_sha = hashlib.sha256(payload).hexdigest()
        result["canonical_payload_sha256"] = canonical_sha
        if data.get("rubric_file_sha256") != canonical_sha:
            result["issues"].append("rubric canonical payload SHA mismatch")
    source_value = data.get("source_document_path_or_id")
    if source_value:
        source_path = resolve_path(source_value)
        if source_path.is_file():
            result["source_document_sha256"] = sha256_file(source_path)
            if result["source_document_sha256"] != str(data.get("source_document_sha256", "")).lower():
                result["issues"].append("rubric approval source SHA mismatch")
    return result


def audit_phase_a_manifest(path: Path, config: dict[str, Any], expected_split: str, allow_empty: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "split": expected_split, "issues": [], "rows": 0}
    if not path.is_file():
        result["issues"].append("Phase A manifest missing")
        return result
    rows = read_rows(path)
    result["rows"] = len(rows)
    if not rows:
        if not allow_empty:
            result["issues"].append("Phase A manifest has no rows")
        return result
    required = set(config["phase_a_manifest"]["required_columns"])
    missing = sorted(required - set(rows[0]))
    if missing:
        result["issues"].append(f"missing Phase A columns: {missing}")
        return result
    issues: list[str] = result["issues"]
    allowed_geometry = set(config["phase_a_manifest"]["allowed_geometry_contract_ids"])
    group_ids: set[str] = set()
    image_hashes: set[str] = set()
    public_mask_hashes: set[str] = set()
    source_ids: set[str] = set()
    scenarios = Counter()
    modes = {"rust": Counter(), "crack": Counter()}
    for row in rows:
        sample = row["sample_id"]
        if row["split"] != expected_split:
            issues.append(f"{sample}: split mismatch")
        if row["development_only"].lower() != "true":
            issues.append(f"{sample}: Phase A row must be development_only=true")
        if row["geometry_contract_id"] not in allowed_geometry:
            issues.append(f"{sample}: geometry contract not allowed")
        if row["scenario"] not in {"clean", "crack_only"}:
            issues.append(f"{sample}: Phase A scenario must be clean or crack_only")
        scenarios[row["scenario"]] += 1
        image = resolve_path(row["relative_image_path"])
        if not image.is_file():
            issues.append(f"{sample}: image missing")
        elif sha256_file(image) != row["image_sha256"].lower():
            issues.append(f"{sample}: image SHA mismatch")
        expected_geometry = (1280, 240) if row["geometry_contract_id"].startswith("crackseg9k") else (1280, 720)
        try:
            geometry = (int(row["native_width"]), int(row["native_height"]))
            if geometry != expected_geometry:
                issues.append(f"{sample}: geometry mismatch {geometry} != {expected_geometry}")
        except ValueError:
            issues.append(f"{sample}: invalid geometry")
        for task in ("rust", "crack"):
            mode = row[f"{task}_label_mode"]
            modes[task][mode] += 1
            if mode not in set(config["manifest"]["label_mode_values"]):
                issues.append(f"{sample}: invalid {task} label mode")
            mask_value = row[f"{task}_mask_path"]
            if mode in {"gt", "partial"} and not mask_value:
                issues.append(f"{sample}: {task} GT mask missing")
            if mask_value:
                mask = resolve_path(mask_value)
                if not mask.is_file():
                    issues.append(f"{sample}: {task} mask file missing")
                elif sha256_file(mask) != row[f"{task}_mask_sha256"].lower():
                    issues.append(f"{sample}: {task} mask SHA mismatch")
                elif row["source"] == "crackseg9k_v4_public":
                    public_mask_hashes.add(row[f"{task}_mask_sha256"].lower())
        if row["source"] == "crackseg9k_v4_public":
            if row["crack_mask_encoding"] != "binary_0_255_positive" or row["rust_label_mode"] != "teacher_only":
                issues.append(f"{sample}: public CrackSeg9k label contract mismatch")
            source_ids.update(item.split("@", 1)[0] for item in row["source_group_ids"].split("|") if item)
        else:
            if row["crack_mask_encoding"] != "indexed_0_1_ignore255" or row["scenario"] != "clean":
                issues.append(f"{sample}: normal-negative label contract mismatch")
            source_ids.add(row["normal_session_id"])
        group_ids.add(row["group_id"])
        image_hashes.add(row["image_sha256"].lower())
    result.update({
        "sha256": sha256_file(path), "groups": len(group_ids), "group_ids": sorted(group_ids),
        # Identical binary masks are legitimate in Phase A (especially all-zero
        # normal masks). Physical/source IDs and image hashes enforce leakage;
        # mask-only overlap is reported by the build report, not blocked here.
        "image_hashes": sorted(image_hashes), "mask_hashes": [],
        "scenarios": dict(scenarios), "label_modes": {key: dict(value) for key, value in modes.items()},
        "identities": {"source_id": sorted(source_ids), "print_id": [], "placement_id": [], "session_id": []},
    })
    for required_scenario in ("clean", "crack_only"):
        if not scenarios[required_scenario]:
            issues.append(f"Phase A missing scenario: {required_scenario}")
    return result


def _float(row: dict[str, str], name: str, issues: list[str], sample: str) -> float | None:
    try:
        return float(row[name])
    except (KeyError, TypeError, ValueError):
        issues.append(f"{sample}: invalid {name}")
        return None


def audit_manifest(path: Path, config: dict[str, Any], expected_split: str, allow_empty: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
        "split": expected_split,
        "issues": [],
        "rows": 0,
    }
    if not path.is_file():
        result["issues"].append("manifest missing")
        return result
    rows = read_rows(path)
    result["rows"] = len(rows)
    if not rows:
        if not allow_empty:
            result["issues"].append("manifest has no rows")
        return result
    required = list(config["manifest"]["required_columns"])
    missing_columns = sorted(set(required) - set(rows[0]))
    if missing_columns:
        result["issues"].append(f"missing columns: {missing_columns}")
        return result
    issues: list[str] = result["issues"]
    scenarios = Counter()
    modes = {"rust": Counter(), "crack": Counter()}
    groups: set[str] = set()
    identities = {name: set() for name in ("source_id", "print_id", "placement_id", "session_id")}
    image_hashes: set[str] = set()
    mask_hashes: set[str] = set()
    frame_keys: set[tuple[str, int]] = set()
    label_modes = set(config["manifest"]["label_mode_values"])
    scenario_values = set(config["manifest"]["scenario_values"])
    for index, row in enumerate(rows, 2):
        sample = row.get("sample_id") or f"line-{index}"
        if row.get("split") != expected_split:
            issues.append(f"{sample}: split mismatch")
        if row.get("scenario") not in scenario_values:
            issues.append(f"{sample}: invalid scenario")
        else:
            scenarios[row["scenario"]] += 1
        if row.get("thin_print", "").lower() not in {"true", "false"}:
            issues.append(f"{sample}: thin_print must be true or false")
        for task in ("rust", "crack"):
            mode = row.get(f"{task}_label_mode", "")
            if mode not in label_modes:
                issues.append(f"{sample}: invalid {task}_label_mode")
            modes[task][mode] += 1
        if row.get("stable_window_review_status") != "approved":
            issues.append(f"{sample}: evaluation/training row is not an approved stable window")
        start = _float(row, "section_start_timestamp", issues, sample)
        end = _float(row, "section_end_timestamp", issues, sample)
        frame = _float(row, "frame_timestamp", issues, sample)
        if start is not None and end is not None and end - start < 2.0:
            issues.append(f"{sample}: stable section is shorter than 2.0 seconds")
        if start is not None and end is not None and frame is not None and not (start <= frame <= end):
            issues.append(f"{sample}: frame timestamp outside section")
        try:
            frame_index = int(row["frame_index"])
            key = (row["session_id"], frame_index)
            if key in frame_keys:
                issues.append(f"{sample}: duplicate frame_index in session")
            frame_keys.add(key)
        except (KeyError, ValueError):
            issues.append(f"{sample}: invalid frame_index")
        crop = tuple(row.get(name) for name in (
            "canonical_crop_x0", "canonical_crop_y0", "canonical_crop_x1", "canonical_crop_y1"
        ))
        if crop != ("0", "0", "1280", "240"):
            issues.append(f"{sample}: canonical crop must be 0,0,1280,240")
        if row.get("width") != "1280" or row.get("height") != "720":
            issues.append(f"{sample}: source image must be 1280x720")
        groups.add(row["group_id"])
        image_hashes.add(row["image_sha256"].lower())
        for name in identities:
            identities[name].add(row[name])
        image_path = resolve_path(row["relative_image_path"])
        if not image_path.is_file():
            issues.append(f"{sample}: image missing")
        elif sha256_file(image_path) != row["image_sha256"].lower():
            issues.append(f"{sample}: image SHA mismatch")
        for task in ("rust", "crack"):
            mode = row[f"{task}_label_mode"]
            mask_value = row[f"{task}_mask_path"]
            if mode in {"gt", "partial"} and not mask_value:
                issues.append(f"{sample}: {task} mask required for {mode}")
            if mask_value:
                mask_path = resolve_path(mask_value)
                if not mask_path.is_file():
                    issues.append(f"{sample}: {task} mask missing")
                elif not row.get(f"{task}_mask_sha256"):
                    issues.append(f"{sample}: {task} mask SHA missing")
                elif sha256_file(mask_path) != row[f"{task}_mask_sha256"].lower():
                    issues.append(f"{sample}: {task} mask SHA mismatch")
                else:
                    mask_hashes.add(row[f"{task}_mask_sha256"].lower())
            if mode == "gt" and not all(row.get(name) for name in (
                "reviewer_a_result", "reviewer_b_result", "review_resolution", "review_status"
            )):
                issues.append(f"{sample}: double review metadata missing")
        if row.get("review_status") not in {"double_review_agreed", "resolved_after_disagreement"}:
            issues.append(f"{sample}: review_status is not approved")
    result.update({
        "sha256": sha256_file(path),
        "groups": len(groups),
        "scenarios": dict(scenarios),
        "label_modes": {task: dict(value) for task, value in modes.items()},
        "group_ids": sorted(groups),
        "image_hashes": sorted(image_hashes),
        "mask_hashes": sorted(mask_hashes),
        "identities": {name: sorted(value) for name, value in identities.items()},
    })
    missing_scenarios = sorted(scenario_values - set(scenarios))
    if missing_scenarios:
        issues.append(f"missing scenarios: {missing_scenarios}")
    return result


def readiness(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    teachers = {
        name: audit_teacher(resolve_path(config["paths"][f"{name if name == 'rust' else 'hrseg'}_teacher_onnx"]), contract)
        for name, contract in config["teachers"].items()
    }
    rubric = audit_rubric(resolve_path(config["paths"]["rust_rubric"]))
    phase_a = config.get("status", {}).get("phase") == "PHASE_A_DEVELOPMENT_ONLY"
    audit_function = audit_phase_a_manifest if phase_a else audit_manifest
    manifests = {
        split: audit_function(
            resolve_path(config["paths"][f"{split}_manifest"]), config, split,
            allow_empty=(split == "development_calibration" and config["evaluation"]["threshold_calibration_status"] == "UNCALIBRATED_BASELINE")
        ) for split in ("train", "development_calibration", "validation")
    }
    issues: list[str] = []
    for name, result in teachers.items():
        issues.extend(f"teacher/{name}: {item}" for item in result["issues"])
    issues.extend(f"rubric: {item}" for item in rubric["issues"])
    for split, result in manifests.items():
        issues.extend(f"manifest/{split}: {item}" for item in result["issues"])
    issues.extend(cross_split_issues(manifests))
    sealed_path = resolve_path(config["paths"]["sealed_commitment"])
    sealed = {"path": str(sealed_path), "exists": sealed_path.is_file(), "issues": []}
    if not sealed_path.is_file():
        if config.get("status", {}).get("sealed_commitment_required_for_training", True):
            sealed["issues"].append("sealed manifest commitment missing")
        else:
            sealed["deferred_to_phase_b"] = True
    else:
        sealed["sha256"] = sha256_file(sealed_path)
        try:
            sealed_data = json.loads(sealed_path.read_text(encoding="utf-8"))
            for field in ("commitment_id", "manifest_sha256", "created_at", "held_by"):
                if not sealed_data.get(field):
                    sealed["issues"].append(f"sealed commitment missing {field}")
            if sealed_data.get("status") != "COMMITTED_BY_INDEPENDENT_EVALUATOR":
                sealed["issues"].append("sealed commitment status is not independently committed")
        except (OSError, json.JSONDecodeError) as error:
            sealed["issues"].append(f"invalid sealed commitment JSON: {error}")
    issues.extend(f"sealed: {item}" for item in sealed["issues"])
    status = config.get("status", {})
    if not status.get("training_authorized"):
        issues.append("config status.training_authorized is false")
    return {
        "schema_version": 1,
        "ready_for_training": not issues,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "teachers": teachers,
        "rust_rubric": rubric,
        "manifests": manifests,
        "sealed_commitment": sealed,
        "issues": issues,
        "actuator_authorization": "PROHIBITED",
        "result_labels": config.get("status", {}).get("required_result_labels", []),
    }


def cross_split_issues(manifests: dict[str, dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    valid = [value for value in manifests.values() if value.get("rows")]
    for left_index, left in enumerate(valid):
        for right in valid[left_index + 1 :]:
            for field in ("group_ids", "image_hashes", "mask_hashes"):
                overlap = set(left.get(field, [])) & set(right.get(field, []))
                if overlap:
                    issues.append(f"cross-split {field} overlap: {left['split']} / {right['split']} ({len(overlap)})")
            for field in ("source_id", "print_id", "placement_id", "session_id"):
                overlap = set(left.get("identities", {}).get(field, [])) & set(right.get("identities", {}).get(field, []))
                if overlap:
                    issues.append(f"cross-split {field} overlap: {left['split']} / {right['split']} ({len(overlap)})")
    return issues


def assert_ready(config_path: Path) -> dict[str, Any]:
    report = readiness(config_path)
    if not report["ready_for_training"]:
        raise RuntimeError("Training blocked:\n - " + "\n - ".join(report["issues"]))
    return report


def assert_teacher_cache(config: dict[str, Any], readiness_report: dict[str, Any], split: str) -> dict[str, Any]:
    cache_root = resolve_path(config["paths"]["teacher_cache"]) / split
    manifest_path = cache_root / "cache_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Teacher cache manifest missing: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_manifest_sha = readiness_report["manifests"][split].get("sha256")
    issues: list[str] = []
    if data.get("split") != split:
        issues.append("split mismatch")
    if data.get("manifest_sha256") != expected_manifest_sha:
        issues.append("source manifest SHA mismatch")
    if data.get("config_sha256") != readiness_report.get("config_sha256"):
        issues.append("config SHA mismatch")
    expected_rows = int(readiness_report["manifests"][split].get("rows", 0))
    if int(data.get("samples", -1)) != expected_rows:
        issues.append("sample count mismatch")
    entries = data.get("entries") or []
    if len(entries) != expected_rows:
        issues.append("entry count mismatch")
    expected_teachers = {
        "rust_teacher_sha256": readiness_report["teachers"]["rust"].get("sha256"),
        "hrseg_teacher_sha256": readiness_report["teachers"]["crack"].get("sha256"),
    }
    for entry in entries:
        for field, expected in expected_teachers.items():
            if entry.get(field) != expected:
                issues.append(f"{entry.get('sample_id')}: {field} mismatch")
        cache_path = Path(str(entry.get("cache_path", "")))
        if not cache_path.is_file():
            issues.append(f"{entry.get('sample_id')}: cache file missing")
        elif entry.get("cache_file_sha256") != sha256_file(cache_path):
            issues.append(f"{entry.get('sample_id')}: cache file SHA mismatch")
    if issues:
        raise RuntimeError("Teacher cache rejected:\n - " + "\n - ".join(issues[:50]))
    return data
