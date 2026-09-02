from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image


WORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RUST_COLORS = np.asarray(
    ((0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0)), dtype=np.uint8
)


class ReadinessError(RuntimeError):
    pass


@dataclass(frozen=True)
class ManifestAudit:
    path: Path
    split: str
    rows: list[dict[str, str]]
    issues: list[str]
    warnings: list[str]
    stats: dict[str, Any]


def load_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    temp_path.replace(path)


def parse_bool(value: str, *, field: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"{field} must be true or false, got {value!r}")


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    import torch

    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _onnx_shape(value_info: Any) -> list[int | None]:
    dimensions: list[int | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        dimensions.append(int(dim.dim_value) if dim.HasField("dim_value") else None)
    return dimensions


def audit_onnx_contract(
    path: Path,
    expected_sha256: str,
    expected_input_name: str,
    expected_input_shape: list[int],
    expected_output_name: str,
    expected_output_shape: list[int],
) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "issues": []}
    if not path.is_file():
        result["issues"].append(f"teacher ONNX missing: {path}")
        return result

    actual_sha = sha256_file(path)
    result["sha256"] = actual_sha
    if actual_sha.lower() != expected_sha256.lower():
        result["issues"].append(
            f"teacher SHA mismatch: expected={expected_sha256}, actual={actual_sha}, path={path}"
        )

    import onnx

    model = onnx.load(str(path), load_external_data=False)
    inputs = {value.name: _onnx_shape(value) for value in model.graph.input}
    outputs = {value.name: _onnx_shape(value) for value in model.graph.output}
    result["inputs"] = inputs
    result["outputs"] = outputs
    result["opsets"] = [
        {"domain": item.domain, "version": int(item.version)} for item in model.opset_import
    ]
    if inputs.get(expected_input_name) != expected_input_shape:
        result["issues"].append(
            f"teacher input contract mismatch: expected {expected_input_name}={expected_input_shape}, got {inputs}"
        )
    if outputs.get(expected_output_name) != expected_output_shape:
        result["issues"].append(
            f"teacher output contract mismatch: expected {expected_output_name}={expected_output_shape}, got {outputs}"
        )
    return result


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    return _read_rows(path)[1]


def cache_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _crack_mask_stats(path: Path, crop: tuple[int, int, int, int]) -> tuple[int, int]:
    with Image.open(path) as image:
        array = np.asarray(image.convert("L"))
    x0, y0, x1, y1 = crop
    if array.shape == (y1 - y0, x1 - x0):
        roi = array[112:240, :]
    else:
        roi = array[y0 + 112 : y0 + 240, x0:x1]
    return int(np.count_nonzero(roi > 0)), int(roi.size)


def audit_manifest(config: dict[str, Any], path: Path, split: str) -> ManifestAudit:
    issues: list[str] = []
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    stats: dict[str, Any] = {
        "rows": 0,
        "rust_valid": 0,
        "crack_valid": 0,
        "crack_positive": 0,
        "crack_negative": 0,
        "crack_positive_pixels": 0,
        "crack_valid_pixels": 0,
        "camera_native": 0,
        "groups": 0,
        "capture_sessions": 0,
        "physical_specimens": 0,
    }
    if not path.is_file():
        issues.append(f"manifest missing: {path}")
        return ManifestAudit(path, split, rows, issues, warnings, stats)

    columns, rows = _read_rows(path)
    required = list(config["data"]["required_manifest_columns"])
    missing = [name for name in required if name not in columns]
    extra = [name for name in columns if name not in required]
    if missing:
        issues.append(f"manifest columns missing: {missing}")
    if extra:
        warnings.append(f"manifest has extra columns: {extra}")
    if missing:
        return ManifestAudit(path, split, rows, issues, warnings, stats)

    expected_geometry = config["contracts"]["geometry_contract_id"]
    allowed_geometries = set(
        config.get("data", {}).get(
            "allowed_geometry_contract_ids", [expected_geometry]
        )
    )
    expected_crop = tuple(int(v) for v in config["contracts"]["crop"])
    seen_ids: set[str] = set()
    groups: set[str] = set()
    sessions: set[str] = set()
    specimens: set[str] = set()
    for index, row in enumerate(rows, start=2):
        prefix = f"{path.name}:{index}"
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in seen_ids:
            issues.append(f"{prefix}: blank or duplicate sample_id={sample_id!r}")
        seen_ids.add(sample_id)
        if row["split"].strip() != split:
            issues.append(f"{prefix}: split must be {split!r}")
        if row["geometry_contract_id"].strip() not in allowed_geometries:
            issues.append(f"{prefix}: wrong geometry_contract_id")
        for identity in (
            "physical_specimen_id",
            "capture_session_id",
            "encoder_section_id",
            "group_id",
        ):
            if not row[identity].strip():
                issues.append(f"{prefix}: {identity} is required")

        try:
            rust_valid = parse_bool(row["rust_valid"], field="rust_valid")
            crack_valid = parse_bool(row["crack_valid"], field="crack_valid")
            camera_native = parse_bool(row["camera_native"], field="camera_native")
            synthetic = parse_bool(row["synthetic"], field="synthetic")
        except ValueError as error:
            issues.append(f"{prefix}: {error}")
            continue
        if not rust_valid and not crack_valid:
            issues.append(f"{prefix}: at least one label domain must be valid")
        if crack_valid and (not camera_native or synthetic):
            public_bootstrap = bool(
                config.get("data", {}).get("allow_public_crackseg_bootstrap", False)
            )
            is_public_panel = (
                row["source"].strip() == "crackseg9k_v4"
                and synthetic
                and not camera_native
            )
            if public_bootstrap and is_public_panel:
                message = (
                    "public CrackSeg9k panels are training/dev evidence, "
                    "not camera-native validation"
                )
                if message not in warnings:
                    warnings.append(message)
            else:
                issues.append(
                    f"{prefix}: crack-valid training/evaluation must be camera_native=true and synthetic=false"
                )

        image_path = resolve_path(row["image_path"])
        if not image_path.is_file():
            issues.append(f"{prefix}: image missing: {image_path}")
            continue
        try:
            width, height = _image_size(image_path)
            declared = (int(row["native_width"]), int(row["native_height"]))
            crop = tuple(
                int(row[name]) for name in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")
            )
        except (ValueError, OSError) as error:
            issues.append(f"{prefix}: invalid image geometry: {error}")
            continue
        if declared != (width, height):
            issues.append(
                f"{prefix}: declared native size {declared} != actual {(width, height)}"
            )
        if crop != expected_crop:
            issues.append(f"{prefix}: crop must be {expected_crop}, got {crop}")
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            issues.append(f"{prefix}: crop lies outside native image")
        expected_sha = row["image_sha256"].strip().lower()
        actual_sha = sha256_file(image_path)
        if expected_sha != actual_sha:
            issues.append(f"{prefix}: image SHA mismatch")

        for valid, field in ((rust_valid, "rust_mask_path"), (crack_valid, "crack_mask_path")):
            raw_path = row[field].strip()
            if valid and not raw_path:
                issues.append(f"{prefix}: {field} required when its valid flag is true")
            if not valid and raw_path:
                warnings.append(f"{prefix}: {field} is ignored because its valid flag is false")
            if valid and raw_path:
                mask_path = resolve_path(raw_path)
                if not mask_path.is_file():
                    issues.append(f"{prefix}: mask missing: {mask_path}")
                else:
                    mask_size = _image_size(mask_path)
                    crop_size = (x1 - x0, y1 - y0)
                    if mask_size not in {(width, height), crop_size}:
                        issues.append(
                            f"{prefix}: mask size {mask_size} must equal native {(width, height)} or crop {crop_size}"
                        )
        if crack_valid and row["crack_mask_path"].strip():
            crack_path = resolve_path(row["crack_mask_path"])
            if crack_path.is_file():
                positive_pixels, valid_pixels = _crack_mask_stats(crack_path, crop)
                stats["crack_positive"] += int(positive_pixels > 0)
                stats["crack_negative"] += int(positive_pixels == 0)
                stats["crack_positive_pixels"] += positive_pixels
                stats["crack_valid_pixels"] += valid_pixels

        stats["rust_valid"] += int(rust_valid)
        stats["crack_valid"] += int(crack_valid)
        stats["camera_native"] += int(camera_native)
        groups.add(row["group_id"].strip())
        sessions.add(row["capture_session_id"].strip())
        specimens.add(row["physical_specimen_id"].strip())

    stats.update(
        rows=len(rows),
        groups=len(groups - {""}),
        capture_sessions=len(sessions - {""}),
        physical_specimens=len(specimens - {""}),
    )
    if not rows:
        issues.append(f"manifest contains no samples: {path}")
    elif bool(config.get("data", {}).get("require_both_tasks_per_split", True)) and (
        stats["rust_valid"] == 0 or stats["crack_valid"] == 0
    ):
        issues.append(
            f"{path.name} must contain both rust-valid and crack-valid samples: {stats}"
        )
    return ManifestAudit(path, split, rows, issues, warnings, stats)


def _unset_preregistration(config: dict[str, Any]) -> list[str]:
    required = config.get("preregistered_acceptance", {})
    return [f"preregistered_acceptance.{key}" for key, value in required.items() if value is None]


def _unset_loss_decisions(config: dict[str, Any]) -> list[str]:
    keys = (
        "boundary_loss",
        "boundary_width_px",
        "crack_pixel_imbalance_strategy",
        "crack_pos_weight",
        "crack_source_sampler_ratio",
    )
    return [f"loss.{key}" for key in keys if config.get("loss", {}).get(key) is None]


def audit_readiness(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = load_config(config_path)
    issues: list[str] = []
    warnings: list[str] = []

    teacher_reports: dict[str, Any] = {}
    for name in ("rust", "crack"):
        teacher = config["teachers"][name]
        path_key = "rust_teacher_onnx" if name == "rust" else "hrseg_teacher_onnx"
        report = audit_onnx_contract(
            resolve_path(config["paths"][path_key]),
            str(teacher["sha256"]),
            str(teacher["input_name"]),
            [int(v) for v in teacher["input_shape"]],
            str(teacher["output_name"]),
            [int(v) for v in teacher["output_shape"]],
        )
        teacher_reports[name] = report
        issues.extend(report["issues"])

    rust_checkpoint = resolve_path(config["paths"]["rust_checkpoint"])
    if not rust_checkpoint.is_file():
        issues.append(f"rust checkpoint missing: {rust_checkpoint}")

    train = audit_manifest(config, resolve_path(config["paths"]["train_manifest"]), "train")
    val = audit_manifest(config, resolve_path(config["paths"]["val_manifest"]), "val")
    issues.extend(train.issues)
    issues.extend(val.issues)
    warnings.extend(train.warnings)
    warnings.extend(val.warnings)

    if train.rows and val.rows:
        train_groups = {row["group_id"].strip() for row in train.rows}
        val_groups = {row["group_id"].strip() for row in val.rows}
        overlap = sorted((train_groups & val_groups) - {""})
        if overlap:
            issues.append(f"train/val group leakage: {overlap[:20]}")
        train_hashes = {row["image_sha256"].strip().lower() for row in train.rows}
        val_hashes = {row["image_sha256"].strip().lower() for row in val.rows}
        duplicate_hashes = sorted((train_hashes & val_hashes) - {""})
        if duplicate_hashes:
            issues.append(f"train/val image SHA leakage: {duplicate_hashes[:20]}")

    unset = _unset_preregistration(config) + _unset_loss_decisions(config)
    if unset:
        issues.append("unresolved locked fields: " + ", ".join(unset))
    if not bool(config.get("status", {}).get("training_authorized", False)):
        issues.append("status.training_authorized is false")

    acceptance = config.get("preregistered_acceptance", {})
    combined_sessions = train.stats["capture_sessions"] + val.stats["capture_sessions"]
    combined_specimens = train.stats["physical_specimens"] + val.stats["physical_specimens"]
    combined_positive = train.stats["crack_positive"] + val.stats["crack_positive"]
    combined_normal = train.stats["crack_negative"] + val.stats["crack_negative"]
    for key, actual in (
        ("min_capture_sessions", combined_sessions),
        ("min_physical_specimens", combined_specimens),
        ("min_positive_sections", combined_positive),
        ("min_normal_sections", combined_normal),
    ):
        required = acceptance.get(key)
        if required is not None and actual < int(required):
            issues.append(f"{key} not met: required={required}, actual={actual}")
    coverage_required = acceptance.get("min_crack_pixel_coverage")
    valid_pixels = train.stats["crack_valid_pixels"] + val.stats["crack_valid_pixels"]
    positive_pixels = train.stats["crack_positive_pixels"] + val.stats["crack_positive_pixels"]
    coverage = positive_pixels / valid_pixels if valid_pixels else 0.0
    if coverage_required is not None and coverage < float(coverage_required):
        issues.append(
            f"min_crack_pixel_coverage not met: required={coverage_required}, actual={coverage}"
        )

    report = {
        "schema_version": 1,
        "ready_for_training": not issues,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "project_root": str(PROJECT_ROOT),
        "work_root": str(WORK_ROOT),
        "teachers": teacher_reports,
        "rust_checkpoint": {
            "path": str(rust_checkpoint),
            "exists": rust_checkpoint.is_file(),
            "sha256": sha256_file(rust_checkpoint) if rust_checkpoint.is_file() else None,
        },
        "manifests": {
            "train": {
                "path": str(train.path),
                "sha256": sha256_file(train.path) if train.path.is_file() else None,
                "stats": train.stats,
            },
            "val": {
                "path": str(val.path),
                "sha256": sha256_file(val.path) if val.path.is_file() else None,
                "stats": val.stats,
            },
        },
        "issues": issues,
        "warnings": warnings,
    }
    return report


def assert_training_ready(config_path: Path) -> dict[str, Any]:
    report = audit_readiness(config_path)
    if not report["ready_for_training"]:
        joined = "\n - ".join(report["issues"])
        raise ReadinessError(f"Training is blocked by the readiness audit:\n - {joined}")
    return report


def crop_native_rgb(image_path: Path, crop: Iterable[int]) -> np.ndarray:
    x0, y0, x1, y1 = (int(value) for value in crop)
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    if not (0 <= x0 < x1 <= rgb.shape[1] and 0 <= y0 < y1 <= rgb.shape[0]):
        raise ValueError(f"Crop {(x0, y0, x1, y1)} outside image {rgb.shape[:2]}")
    return np.ascontiguousarray(rgb[y0:y1, x0:x1])


def rust_mask_to_indices(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 2:
        values = set(int(v) for v in np.unique(mask))
        if values.issubset({0, 1, 2, 3}):
            return mask.astype(np.int64)
    if mask.ndim == 3:
        output = np.full(mask.shape[:2], 255, dtype=np.int64)
        for index, color in enumerate(RUST_COLORS):
            output[np.all(mask[..., :3] == color, axis=-1)] = index
        if not np.any(output == 255):
            return output
    raise ValueError("Rust mask must be indexed 0..3 or use the four locked RGB colors")
