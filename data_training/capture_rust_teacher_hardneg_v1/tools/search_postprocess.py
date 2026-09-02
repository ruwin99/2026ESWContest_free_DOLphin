from __future__ import annotations

import argparse
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    CLASS_NAMES,
    RustManifestDataset,
    build_model,
    load_config,
    metrics_from_confusion,
    read_manifest,
    resolve_path,
    sha256_file,
    validation_gate,
    verify_manifest_lock,
    write_json,
)


DEFAULT_THRESHOLDS = "0,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90"
DEFAULT_MIN_COMPONENTS = "0,4,8,16,32,64,128,256"


def parse_float_list(value: str) -> list[float]:
    result = sorted(set(float(item.strip()) for item in value.split(",") if item.strip()))
    if not result or result[0] < 0.0 or result[-1] > 1.0:
        raise ValueError("Probability thresholds must be within [0, 1]")
    return result


def parse_int_list(value: str) -> list[int]:
    result = sorted(set(int(item.strip()) for item in value.split(",") if item.strip()))
    if not result or result[0] < 0:
        raise ValueError("Minimum component sizes must be non-negative")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search Fair-only probability and connected-component suppression on validation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--probability-thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--min-components", default=DEFAULT_MIN_COMPONENTS)
    parser.add_argument("--max-samples", type=int, default=0, help="Smoke only; never writes a selectable policy.")
    return parser.parse_args()


def empty_group() -> dict[str, Any]:
    return {
        "valid_pixels": 0,
        "candidate_prediction_counts": [0, 0, 0, 0],
        "baseline_prediction_counts": [0, 0, 0, 0],
        "samples": 0,
        "candidate_frames_with_fp": 0,
        "baseline_frames_with_fp": 0,
    }


def fair_keep_masks(raw_prediction: np.ndarray, fair_probability: np.ndarray, threshold: float, minimums: list[int]) -> dict[int, np.ndarray]:
    eligible = (raw_prediction == 1) & (fair_probability >= threshold)
    if not eligible.any():
        return {minimum: eligible for minimum in minimums}
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        eligible.astype(np.uint8), connectivity=8
    )
    areas = stats[:, cv2.CC_STAT_AREA]
    areas[0] = 0
    result: dict[int, np.ndarray] = {}
    for minimum in minimums:
        if minimum <= 1:
            result[minimum] = eligible
        else:
            keep_label = np.zeros(labels_count, dtype=bool)
            keep_label[1:] = areas[1:] >= minimum
            result[minimum] = keep_label[labels]
    return result


def add_confusion(matrix: np.ndarray, target: np.ndarray, prediction: np.ndarray) -> None:
    valid = target != 255
    counts = np.bincount(
        target[valid].astype(np.int64) * 4 + prediction[valid].astype(np.int64),
        minlength=16,
    )
    matrix += counts.reshape(4, 4)


def counts4(prediction: np.ndarray, valid: np.ndarray) -> list[int]:
    return np.bincount(prediction[valid].astype(np.int64), minlength=4).astype(np.int64).tolist()


def add_counts(destination: list[int], source: list[int]) -> None:
    for index, value in enumerate(source):
        destination[index] += int(value)


def hard_report(groups: dict[str, dict[str, Any]]) -> dict[str, Any]:
    group_reports: dict[str, Any] = {}
    for name, item in groups.items():
        valid = int(item["valid_pixels"])
        candidate_counts = item["candidate_prediction_counts"]
        baseline_counts = item["baseline_prediction_counts"]
        candidate_fp = int(sum(candidate_counts[1:]))
        baseline_fp = int(sum(baseline_counts[1:]))
        group_reports[name] = {
            **item,
            "candidate_fp": candidate_fp,
            "baseline_fp": baseline_fp,
            "candidate_fp_rate": candidate_fp / valid if valid else None,
            "baseline_fp_rate": baseline_fp / valid if valid else None,
            "worsened": candidate_fp > baseline_fp,
        }
    valid = sum(int(item["valid_pixels"]) for item in groups.values())
    candidate_counts = [sum(item["candidate_prediction_counts"][index] for item in groups.values()) for index in range(4)]
    baseline_counts = [sum(item["baseline_prediction_counts"][index] for item in groups.values()) for index in range(4)]
    candidate_fp = int(sum(candidate_counts[1:]))
    baseline_fp = int(sum(baseline_counts[1:]))
    return {
        "valid_pixels": valid,
        "candidate_fp": candidate_fp,
        "baseline_fp": baseline_fp,
        "candidate_fp_rate": candidate_fp / valid if valid else None,
        "baseline_fp_rate": baseline_fp / valid if valid else None,
        "all_pixels_expected_class": "Good",
        "class_order": list(CLASS_NAMES),
        "candidate_prediction_counts": candidate_counts,
        "baseline_prediction_counts": baseline_counts,
        "candidate_false_positive_by_class": dict(zip(CLASS_NAMES[1:], candidate_counts[1:])),
        "baseline_false_positive_by_class": dict(zip(CLASS_NAMES[1:], baseline_counts[1:])),
        "candidate_frames_with_fp": sum(int(item["candidate_frames_with_fp"]) for item in groups.values()),
        "baseline_frames_with_fp": sum(int(item["baseline_frames_with_fp"]) for item in groups.values()),
        "groups": group_reports,
    }


def make_policy_state() -> dict[str, Any]:
    return {
        "candidate_cm": np.zeros((4, 4), dtype=np.int64),
        "groups": defaultdict(empty_group),
        "new_severe_under": 0,
        "new_poor_good": 0,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-frame postprocess search")
    config = load_config(args.config)
    lock = verify_manifest_lock(config)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Unsupported candidate checkpoint")
    if not checkpoint.get("validation_gate", {}).get("passed", False):
        raise RuntimeError("Only a validation-gated best candidate may be postprocessed")
    if checkpoint.get("provenance", {}).get("manifest_lock", {}).get("files") != lock.get("files"):
        raise RuntimeError("Checkpoint manifest provenance does not match the current lock")

    thresholds = parse_float_list(args.probability_thresholds)
    minimums = parse_int_list(args.min_components)
    if 0.0 not in thresholds or 0 not in minimums:
        raise ValueError("Grid must include probability=0 and min_component=0 as the raw-output control")
    policies = [(threshold, minimum) for threshold in thresholds for minimum in minimums]
    states = {policy: make_policy_state() for policy in policies}

    initial = resolve_path(config["paths"]["initial_checkpoint"])
    candidate = build_model(config, initial)
    candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    baseline = build_model(config, initial)
    device = torch.device("cuda:0")
    candidate.to(device).eval()
    baseline.to(device).eval()

    manifest_path = resolve_path(config["paths"]["validation_manifest"])
    rows = read_manifest(manifest_path, expected_split="validation")
    if args.max_samples:
        rows = rows[: args.max_samples]
    loader = DataLoader(
        RustManifestDataset(rows, config, training=False),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    baseline_cm = np.zeros((4, 4), dtype=np.int64)

    with torch.inference_mode():
        for item_index, batch in enumerate(loader, start=1):
            image = batch["image"].to(device, non_blocking=True)
            target = batch["target"][0].numpy().astype(np.uint8, copy=False)
            candidate_logits = candidate(image)
            baseline_logits = baseline(image)
            if not torch.isfinite(candidate_logits).all() or not torch.isfinite(baseline_logits).all():
                raise FloatingPointError("Validation logits contain NaN or Inf")
            raw_prediction = candidate_logits.argmax(1)[0].cpu().numpy().astype(np.uint8, copy=False)
            baseline_prediction = baseline_logits.argmax(1)[0].cpu().numpy().astype(np.uint8, copy=False)
            fair_probability = F.softmax(candidate_logits, dim=1)[0, 1].cpu().numpy()
            source_type = batch["source_type"][0]
            group_id = batch["group_id"][0]
            if source_type != "hard_negative":
                add_confusion(baseline_cm, target, baseline_prediction)

            for threshold in thresholds:
                masks = fair_keep_masks(raw_prediction, fair_probability, threshold, minimums)
                for minimum in minimums:
                    prediction = raw_prediction.copy()
                    prediction[(raw_prediction == 1) & ~masks[minimum]] = 0
                    state = states[(threshold, minimum)]
                    if source_type == "hard_negative":
                        valid = target == 0
                        group = state["groups"][group_id]
                        group["valid_pixels"] += int(valid.sum())
                        candidate_counts = counts4(prediction, valid)
                        baseline_counts = counts4(baseline_prediction, valid)
                        add_counts(group["candidate_prediction_counts"], candidate_counts)
                        add_counts(group["baseline_prediction_counts"], baseline_counts)
                        group["samples"] += 1
                        group["candidate_frames_with_fp"] += int(sum(candidate_counts[1:]) > 0)
                        group["baseline_frames_with_fp"] += int(sum(baseline_counts[1:]) > 0)
                    else:
                        add_confusion(state["candidate_cm"], target, prediction)
                        valid = target != 255
                        severe = valid & (target == 3)
                        poor = valid & (target == 2)
                        state["new_severe_under"] += int(
                            np.count_nonzero(severe & (prediction <= 1) & (baseline_prediction > 1))
                        )
                        state["new_poor_good"] += int(
                            np.count_nonzero(poor & (prediction == 0) & (baseline_prediction != 0))
                        )
            if item_index % 25 == 0 or item_index == len(rows):
                print(json.dumps({"processed": item_index, "total": len(rows)}, ensure_ascii=False), flush=True)

    baseline_positive = metrics_from_confusion(torch.from_numpy(baseline_cm))
    positive_samples = sum(row["source_type"] != "hard_negative" for row in rows)
    hard_samples = len(rows) - positive_samples
    results = []
    for threshold, minimum in policies:
        state = states[(threshold, minimum)]
        report = {
            "positive_samples": positive_samples,
            "hard_negative_samples": hard_samples,
            "candidate_positive": metrics_from_confusion(torch.from_numpy(state["candidate_cm"])),
            "baseline_positive": baseline_positive,
            "hard_negative": hard_report(state["groups"]),
            "new_major_under_calls_vs_baseline": {
                "severe_to_good_or_fair": int(state["new_severe_under"]),
                "poor_to_good": int(state["new_poor_good"]),
            },
        }
        gate = validation_gate(report, config) if not args.max_samples else {
            "passed": False,
            "reasons": ["smoke subset cannot select a policy"],
            "selection_score": None,
        }
        hard = report["hard_negative"]
        results.append({
            "policy": {
                "fair_probability_threshold": threshold,
                "fair_min_component_pixels": minimum,
                "connectivity": 8,
                "other_classes": "unchanged",
            },
            "gate": gate,
            "summary": {
                "candidate_fp": hard["candidate_fp"],
                "baseline_fp": hard["baseline_fp"],
                "candidate_frames_with_fp": hard["candidate_frames_with_fp"],
                "baseline_frames_with_fp": hard["baseline_frames_with_fp"],
                "candidate_macro_iou_corrosion3": report["candidate_positive"]["macro_iou_corrosion3"],
                "baseline_macro_iou_corrosion3": report["baseline_positive"]["macro_iou_corrosion3"],
                "new_severe_to_good_or_fair": report["new_major_under_calls_vs_baseline"]["severe_to_good_or_fair"],
                "new_poor_to_good": report["new_major_under_calls_vs_baseline"]["poor_to_good"],
                "candidate_false_positive_by_class": hard["candidate_false_positive_by_class"],
            },
            "metrics": report,
        })

    passing = [item for item in results if item["gate"]["passed"]]
    passing.sort(key=lambda item: (
        item["summary"]["candidate_fp"],
        item["summary"]["candidate_frames_with_fp"],
        item["policy"]["fair_probability_threshold"],
        item["policy"]["fair_min_component_pixels"],
    ))
    best = passing[0] if passing else None
    raw = next(
        item for item in results
        if item["policy"]["fair_probability_threshold"] == 0.0
        and item["policy"]["fair_min_component_pixels"] == 0
    )
    output = args.output.resolve()
    report = {
        "searched_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "search_scope": "validation_v7_only_never_sealed_v8",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_manifest": str(manifest_path),
        "validation_manifest_sha256": sha256_file(manifest_path),
        "grid": {
            "fair_probability_thresholds": thresholds,
            "fair_min_component_pixels": minimums,
            "policies": len(policies),
        },
        "samples": len(rows),
        "smoke_only": bool(args.max_samples),
        "raw_control": raw,
        "passing_policies": len(passing),
        "best": best,
        "all_results": results,
        "restrictions": {
            "sealed_v8_used": False,
            "temporal_filter_selected": False,
            "deployment_status": "NOT_DEPLOYED",
            "accuracy": "NOT_FINAL",
        },
    }
    write_json(output, report)
    if best is not None and not args.max_samples:
        policy_path = output.with_suffix(".policy.json")
        write_json(policy_path, {
            "format_version": 1,
            "source_report": str(output),
            "source_report_sha256": sha256_file(output),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "validation_manifest_sha256": sha256_file(manifest_path),
            "selected": best,
            "status": {
                "candidate": True,
                "uart_status": "NOT_FOR_UART",
                "deployment_status": "NOT_DEPLOYED",
                "accuracy": "NOT_FINAL",
            },
        })
    print(json.dumps({
        "output": str(output),
        "passing_policies": len(passing),
        "raw": raw["summary"],
        "best": None if best is None else {"policy": best["policy"], "summary": best["summary"], "gate": best["gate"]},
    }, ensure_ascii=False, indent=2))
    if not args.max_samples and best is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
