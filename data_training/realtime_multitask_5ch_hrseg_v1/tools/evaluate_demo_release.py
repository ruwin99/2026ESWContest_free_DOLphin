from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from common import load_config, resolve_path, sha256_file, write_json
from model import build_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select on historical demo normals and evaluate once on the v7 normal holdout."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--positive-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--locked-dir", type=Path, default=Path("for model test v7"))
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-locked-false-positive-frames", type=int, default=3)
    return parser.parse_args()


def image_paths(root: Path) -> list[Path]:
    paths = sorted(
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No top-level images in {root}")
    return paths


def prepare(path: Path, mean: np.ndarray, std: np.ndarray) -> torch.Tensor:
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        if rgb.size != (1280, 720):
            raise ValueError(f"Expected 1280x720 image: {path} ({rgb.size})")
        array = np.asarray(rgb.crop((0, 0, 1280, 240)), dtype=np.float32) / 255.0
    array = (array - mean) / std
    return torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)))


def infer(
    model: torch.nn.Module,
    device: torch.device,
    roots: list[Path],
    thresholds: list[float],
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    paths: list[Path] = []
    folder_counts: dict[str, int] = {}
    for root in roots:
        current = image_paths(root)
        folder_counts[str(root)] = len(current)
        paths.extend(current)
    records: list[dict[str, Any]] = []
    with torch.inference_mode():
        for offset in tqdm(range(0, len(paths), batch_size), desc="camera normal audit"):
            current = paths[offset : offset + batch_size]
            batch = torch.stack([prepare(path, mean, std) for path in current]).to(device)
            logits = model(batch)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model logits contain NaN or Inf")
            crack = logits[:, 4, 112:240].float().cpu().numpy()
            for path, values in zip(current, crack, strict=True):
                largest: dict[str, int] = {}
                raw_pixels: dict[str, int] = {}
                for threshold in thresholds:
                    tau = np.float32(math.log(threshold / (1.0 - threshold)))
                    binary = np.ascontiguousarray(values >= tau, dtype=np.uint8)
                    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    areas = stats[1:count, cv2.CC_STAT_AREA]
                    key = format(threshold, ".9g")
                    largest[key] = int(areas.max()) if areas.size else 0
                    raw_pixels[key] = int(binary.sum())
                records.append(
                    {
                        "path": str(path),
                        "folder": str(path.parent),
                        "filename": path.name,
                        "sha256": sha256_file(path),
                        "largest_component_pixels": largest,
                        "raw_candidate_pixels": raw_pixels,
                    }
                )
    return records, folder_counts


def false_positives(records: list[dict[str, Any]], threshold: float, minimum: int) -> list[dict[str, Any]]:
    key = format(threshold, ".9g")
    return [record for record in records if record["largest_component_pixels"][key] >= minimum]


def main() -> None:
    args = parse_args()
    if args.batch < 1 or args.max_locked_false_positive_frames < 0:
        raise ValueError("Invalid evaluation settings")
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    positive_path = args.positive_report.resolve()
    config = load_config(config_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config_sha = sha256_file(config_path)
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint.get("config_sha256") != config_sha:
        raise ValueError("Checkpoint/config SHA mismatch")

    positive = json.loads(positive_path.read_text(encoding="utf-8"))
    if positive.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("Positive validation report/checkpoint SHA mismatch")
    candidates = [row for row in positive.get("results", []) if row.get("passed")]
    if not candidates:
        raise RuntimeError("Positive validation has no passing operating point")
    thresholds = sorted({float(row["probability_threshold"]) for row in candidates})

    configured = [
        resolve_path(Path(value))
        for value in config["data"]["candidate_sources"]["demo_normal_roots"]
    ]
    locked_root = resolve_path(args.locked_dir)
    selection_roots = [root for root in configured if root.resolve() != locked_root.resolve()]
    if not locked_root.is_dir() or any(not root.is_dir() for root in selection_roots):
        raise FileNotFoundError("One or more configured camera-normal folders are missing")

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    mean = np.asarray(config["contracts"]["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config["contracts"]["std"], dtype=np.float32).reshape(1, 1, 3)
    selection_records, selection_counts = infer(
        model, device, selection_roots, thresholds, mean, std, args.batch
    )

    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        threshold = float(candidate["probability_threshold"])
        minimum = int(candidate["min_component_pixels"])
        failures = false_positives(selection_records, threshold, minimum)
        ranked.append(
            {
                "probability_threshold": threshold,
                "logit_threshold_fp32": float(np.float32(math.log(threshold / (1.0 - threshold)))),
                "min_component_pixels": minimum,
                "positive_dice": float(candidate["dice"]),
                "positive_recall": float(candidate["recall"]),
                "positive_precision": float(candidate["precision"]),
                "selection_normal_frames": len(selection_records),
                "selection_false_positive_frames": len(failures),
                "selection_false_positive_rate": len(failures) / len(selection_records),
            }
        )
    ranked.sort(
        key=lambda row: (
            row["selection_false_positive_frames"],
            -row["positive_dice"],
            -row["positive_recall"],
            row["min_component_pixels"],
        )
    )
    selected = ranked[0]

    locked_records, locked_counts = infer(
        model, device, [locked_root], [selected["probability_threshold"]], mean, std, args.batch
    )
    locked_failures = false_positives(
        locked_records, selected["probability_threshold"], selected["min_component_pixels"]
    )
    selection_hashes = {record["sha256"] for record in selection_records}
    locked_hash_counts = Counter(record["sha256"] for record in locked_records)
    overlap = sorted(
        record["filename"] for record in locked_records if record["sha256"] in selection_hashes
    )
    locked_duplicates = sorted(digest for digest, count in locked_hash_counts.items() if count > 1)
    passed = (
        len(locked_failures) <= args.max_locked_false_positive_frames
        and not overlap
        and not locked_duplicates
    )
    report = {
        "schema_version": 1,
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "passed_for_demo_onnx_export" if passed else "failed_demo_release_audit",
        "scope": "demo-camera normal holdout plus public CrackSeg9k positive validation",
        "limitations": [
            "No camera-native crack-positive ground truth was available.",
            "This does not establish real-rail crack accuracy.",
        ],
        "config": str(config_path),
        "config_sha256": config_sha,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "positive_validation_report": str(positive_path),
        "positive_validation_report_sha256": sha256_file(positive_path),
        "selection_normal_folders": selection_counts,
        "selection_operating_points": ranked,
        "selected_operating_point": selected,
        "locked_normal": {
            "folders": locked_counts,
            "frames": len(locked_records),
            "maximum_allowed_false_positive_frames": args.max_locked_false_positive_frames,
            "false_positive_frames": len(locked_failures),
            "false_positive_rate": len(locked_failures) / len(locked_records),
            "false_positive_files": [record["filename"] for record in locked_failures],
            "largest_component_pixels": max(
                record["largest_component_pixels"][format(selected["probability_threshold"], ".9g")]
                for record in locked_records
            ),
            "selection_overlap_files": overlap,
            "within_locked_duplicate_sha256": locked_duplicates,
            "passed": passed,
        },
    }
    write_json(args.output.resolve(), report)
    print(json.dumps({
        "status": report["status"],
        "selected_operating_point": selected,
        "locked_normal": report["locked_normal"],
    }, ensure_ascii=False, indent=2))
    if not passed:
        raise RuntimeError("Demo release audit failed; ONNX export is not authorized")


if __name__ == "__main__":
    main()
