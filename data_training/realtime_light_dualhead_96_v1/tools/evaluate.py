from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader

from common import assert_ready, assert_teacher_cache, load_config, read_rows, resolve_path, sha256_file, write_json
from dataset import DemoManifestDataset
from light_dualhead_96 import LightDualHead96


STRUCTURE8 = np.ones((3, 3), dtype=np.uint8)


def remove_small(binary: np.ndarray, minimum: int) -> tuple[np.ndarray, int]:
    labels, count = ndimage.label(binary, structure=STRUCTURE8)
    areas = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.flatnonzero(areas >= minimum)
    keep = keep[keep != 0]
    return np.isin(labels, keep), int(keep.size)


def rust_grade(class_map: np.ndarray, minimum: int) -> int:
    for grade in (3, 2, 1):
        _, count = remove_small(class_map == grade, minimum)
        if count:
            return grade
    return 0


def section_grade(frame_grades: list[int]) -> int:
    if sum(value == 3 for value in frame_grades) >= 2:
        return 3
    if sum(value >= 2 for value in frame_grades) >= 2:
        return 2
    if sum(value >= 1 for value in frame_grades) >= 2:
        return 1
    return 0


def select_five(rows: list[dict[str, str]], targets: list[float], tolerance: float) -> list[dict[str, str]]:
    available = rows.copy()
    selected: list[dict[str, str]] = []
    for target in targets:
        candidates = []
        for row in available:
            relative = float(row["frame_timestamp"]) - float(row["section_start_timestamp"])
            delta = abs(relative - target)
            if delta <= tolerance:
                candidates.append((delta, float(row["frame_timestamp"]), int(row["frame_index"]), row))
        if not candidates:
            return []
        chosen = min(candidates, key=lambda item: item[:3])[3]
        selected.append(chosen)
        available.remove(chosen)
    return selected


def component_counts(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray, minimum: int, threshold: float) -> tuple[int, int, int]:
    pred, _ = remove_small(prediction & valid, minimum)
    pred_labels, pred_count = ndimage.label(pred, structure=STRUCTURE8)
    gt_labels, gt_count = ndimage.label(target & valid, structure=STRUCTURE8)
    if not pred_count:
        return 0, 0, int(gt_count)
    if not gt_count:
        return 0, int(pred_count), 0
    matrix = np.zeros((pred_count, gt_count), dtype=np.float64)
    for p in range(1, pred_count + 1):
        p_mask = pred_labels == p
        for g in range(1, gt_count + 1):
            g_mask = gt_labels == g
            union = np.count_nonzero(p_mask | g_mask)
            matrix[p - 1, g - 1] = np.count_nonzero(p_mask & g_mask) / union if union else 0.0
    p_indices, g_indices = linear_sum_assignment(-matrix)
    matched = sum(matrix[p, g] >= threshold for p, g in zip(p_indices, g_indices))
    return int(matched), int(pred_count - matched), int(gt_count - matched)


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline printed-demo validation; never opens sealed-test content.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("development_calibration", "validation"), default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    readiness = assert_ready(config_path)
    config = load_config(config_path)
    assert_teacher_cache(config, readiness, args.split)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("config_sha256") != readiness["config_sha256"]:
        raise ValueError("Checkpoint/config SHA mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = LightDualHead96(resolve_path(config["paths"]["official_training"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    manifest = resolve_path(config["paths"][f"{args.split}_manifest"])
    dataset = DemoManifestDataset(manifest, resolve_path(config["paths"]["teacher_cache"]), False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    rows = read_rows(manifest)
    predictions: dict[str, dict[str, Any]] = {}
    rust_confusion = np.zeros((4, 4), dtype=np.int64)
    crack_tp = crack_fp = crack_fn = 0
    comp_tp = comp_fp = comp_fn = 0
    rust_teacher_equal = rust_teacher_total = 0
    nonfinite = 0
    with torch.inference_mode():
        for row, batch in zip(rows, loader):
            output = model(batch["image"].to(device)).float().cpu()
            if not torch.isfinite(output).all():
                nonfinite += 1
                continue
            rust_map = output[0, :4].argmax(0).numpy().astype(np.uint8)
            crack_logit = output[0, 4, 112:240].numpy()
            crack_map = crack_logit >= float(config["evaluation"]["crack_raw_logit_threshold"])
            rust_gt = batch["rust_gt"][0].numpy()
            crack_gt = batch["crack_gt"][0, 112:240].numpy()
            if row["rust_label_mode"] in {"gt", "partial"}:
                valid = rust_gt != 255
                np.add.at(rust_confusion, (rust_gt[valid], rust_map[valid]), 1)
            teacher_map = batch["rust_teacher"][0].argmax(0).numpy()
            rust_teacher_equal += int(np.count_nonzero(teacher_map == rust_map))
            rust_teacher_total += rust_map.size
            if row["crack_label_mode"] in {"gt", "partial"}:
                valid = crack_gt != 255
                target = crack_gt == 1
                crack_tp += int(np.count_nonzero(crack_map & target & valid))
                crack_fp += int(np.count_nonzero(crack_map & ~target & valid))
                crack_fn += int(np.count_nonzero(~crack_map & target & valid))
                tp, fp, fn = component_counts(
                    crack_map, target, valid, int(config["evaluation"]["crack_min_component_pixels"]),
                    float(config["evaluation"]["component_match_iou_threshold"])
                )
                comp_tp += tp; comp_fp += fp; comp_fn += fn
            filtered_crack, crack_components = remove_small(
                crack_map, int(config["evaluation"]["crack_min_component_pixels"])
            )
            predictions[row["sample_id"]] = {
                "rust_grade": rust_grade(rust_map, int(config["evaluation"]["rust_min_component_pixels"])),
                "crack_positive": bool(crack_components),
                "rust_gt_grade": rust_grade(rust_gt, int(config["evaluation"]["rust_min_component_pixels"]))
                    if row["rust_label_mode"] == "gt" else None,
                "crack_gt_positive": bool(np.any(crack_gt == 1)) if row["crack_label_mode"] == "gt" else None,
            }
    ious: list[float | None] = []
    for class_index in range(4):
        tp = rust_confusion[class_index, class_index]
        union = rust_confusion[class_index].sum() + rust_confusion[:, class_index].sum() - tp
        ious.append(ratio(tp, union))
    rust_macro_iou = float(np.mean([value for value in ious if value is not None])) if all(v is not None for v in ious) else None
    if config.get("status", {}).get("phase") == "PHASE_A_DEVELOPMENT_ONLY":
        normal_rows = [row for row in rows if row["scenario"] == "clean"]
        public_rows = [row for row in rows if row["scenario"] == "crack_only"]
        normal_false = ratio(sum(predictions[row["sample_id"]]["crack_positive"] for row in normal_rows), len(normal_rows))
        public_recall = ratio(sum(predictions[row["sample_id"]]["crack_positive"] for row in public_rows), len(public_rows))
        phase_metrics = {
            "rust_class_iou_development": ious,
            "rust_teacher_argmax_agreement_diagnostic": ratio(rust_teacher_equal, rust_teacher_total),
            "crack_pixel_precision": ratio(crack_tp, crack_tp + crack_fp),
            "crack_pixel_recall": ratio(crack_tp, crack_tp + crack_fn),
            "crack_pixel_iou": ratio(crack_tp, crack_tp + crack_fp + crack_fn),
            "crack_pixel_dice": ratio(2 * crack_tp, 2 * crack_tp + crack_fp + crack_fn),
            "crack_component_micro_f1": ratio(2 * comp_tp, 2 * comp_tp + comp_fp + comp_fn),
            "public_crack_positive_frame_recall": public_recall,
            "normal_negative_false_positive_frame_rate": normal_false,
            "nonfinite_outputs": nonfinite,
        }
        report = {
            "phase": "PHASE_A_DEVELOPMENT_ONLY", "scope": "public_plus_normal_negative_development_only",
            "result_labels": ["ACCURACY_NOT_FINAL", "NOT_FOR_UART"], "split": args.split,
            "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "config_sha256": readiness["config_sha256"], "metrics": phase_metrics,
            "development_checks_passed": nonfinite == 0 and bool(public_rows) and bool(normal_rows),
            "status": "ACCURACY_NOT_FINAL", "actuator_authorization": "PROHIBITED",
            "notes": ["No final accuracy gate is evaluated in Phase A.", "Phase B requires independent printed-demo sealed data."],
        }
        output = resolve_path(config["paths"]["reports"]) / f"phase_a_{args.split}_{args.checkpoint.stem}_evaluation.json"
        write_json(output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["development_checks_passed"]:
            raise SystemExit(2)
        return
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["section_id"]].append(row)
    invalid_sections: list[str] = []
    section_results: list[dict[str, Any]] = []
    targets = [float(v) for v in config["evaluation"]["frame_relative_seconds"]]
    for section_id, section_rows in grouped.items():
        selected = select_five(section_rows, targets, float(config["evaluation"]["frame_tolerance_seconds"]))
        if len(selected) != int(config["evaluation"]["frames_per_section"]):
            invalid_sections.append(section_id)
            continue
        pred = [predictions[row["sample_id"]] for row in selected]
        rust_gt_values = [value["rust_gt_grade"] for value in pred]
        crack_gt_values = [value["crack_gt_positive"] for value in pred]
        section_results.append({
            "section_id": section_id,
            "scenario": selected[0]["scenario"],
            "thin_print": selected[0]["thin_print"].lower() == "true",
            "pred_rust_grade": section_grade([int(value["rust_grade"]) for value in pred]),
            "pred_crack": sum(bool(value["crack_positive"]) for value in pred) >= 2,
            "gt_rust_grade": section_grade([int(value) for value in rust_gt_values]) if all(v is not None for v in rust_gt_values) else None,
            "gt_crack": sum(bool(value) for value in crack_gt_values) >= 2 if all(v is not None for v in crack_gt_values) else None,
        })
    rust_positive = [s for s in section_results if s["gt_rust_grade"] is not None and s["gt_rust_grade"] >= 2]
    crack_positive = [s for s in section_results if s["gt_crack"] is True]
    thin_positive = [s for s in crack_positive if s["thin_print"]]
    clean = [s for s in section_results if s["scenario"] == "clean"]
    artifact = [s for s in section_results if s["scenario"] == "artifact_hard_negative"]
    rust_ps_recall = ratio(sum(s["pred_rust_grade"] >= 2 for s in rust_positive), len(rust_positive))
    crack_section_recall = ratio(sum(s["pred_crack"] for s in crack_positive), len(crack_positive))
    thin_recall = ratio(sum(s["pred_crack"] for s in thin_positive), len(thin_positive))
    clean_rust_false = ratio(sum(s["pred_rust_grade"] >= 2 for s in clean), len(clean))
    clean_crack_false = ratio(sum(s["pred_crack"] for s in clean), len(clean))
    artifact_crack_false = ratio(sum(s["pred_crack"] for s in artifact), len(artifact))
    crack_iou = ratio(crack_tp, crack_tp + crack_fp + crack_fn)
    component_f1 = ratio(2 * comp_tp, 2 * comp_tp + comp_fp + comp_fn)
    metrics = {
        "rust_class_iou": ious, "rust_macro_iou": rust_macro_iou,
        "rust_poor_severe_section_recall": rust_ps_recall,
        "rust_clean_false_poor_severe_rate": clean_rust_false,
        "rust_teacher_argmax_agreement_diagnostic": ratio(rust_teacher_equal, rust_teacher_total),
        "crack_pixel_precision": ratio(crack_tp, crack_tp + crack_fp),
        "crack_pixel_recall": ratio(crack_tp, crack_tp + crack_fn), "crack_pixel_iou": crack_iou,
        "crack_pixel_dice": ratio(2 * crack_tp, 2 * crack_tp + crack_fp + crack_fn),
        "crack_component_micro_f1": component_f1,
        "crack_positive_section_recall": crack_section_recall,
        "thin_print_crack_section_recall": thin_recall,
        "crack_clean_false_stop_rate": clean_crack_false,
        "artifact_false_stop_rate": artifact_crack_false,
        "nonfinite_outputs": nonfinite,
    }
    gates = config["evaluation"]["gates"]
    checks = {
        "rust_macro_iou": rust_macro_iou is not None and rust_macro_iou >= float(gates["rust_macro_iou_min"]),
        "rust_ps_recall": rust_ps_recall is not None and rust_ps_recall >= float(gates["rust_poor_severe_section_recall_min"]),
        "rust_clean_false": clean_rust_false is not None and clean_rust_false <= float(gates["rust_clean_false_poor_severe_max"]),
        "crack_section_recall": crack_section_recall is not None and crack_section_recall >= float(gates["crack_positive_section_recall_min"]),
        "thin_print_recall": thin_recall is not None and thin_recall >= float(gates["thin_print_crack_section_recall_min"]),
        "crack_clean_false": clean_crack_false is not None and clean_crack_false <= float(gates["crack_clean_false_stop_max"]),
        "artifact_false": artifact_crack_false is not None and artifact_crack_false <= float(gates["artifact_false_stop_max"]),
        "crack_pixel_iou": crack_iou is not None and crack_iou >= float(gates["crack_pixel_iou_min"]),
        "valid_sections": not invalid_sections,
        "finite": nonfinite == 0,
        "ten_second_stability": False,
    }
    required_composite = [rust_macro_iou, rust_ps_recall, clean_rust_false, crack_section_recall, component_f1, clean_crack_false, artifact_crack_false]
    composite = None
    if all(value is not None for value in required_composite):
        negative_false = (sum(s["pred_crack"] for s in clean + artifact) / len(clean + artifact)) if clean + artifact else None
        if negative_false is not None:
            composite = (0.20 * rust_macro_iou + 0.20 * rust_ps_recall + 0.15 * (1-clean_rust_false)
                         + 0.20 * crack_section_recall + 0.10 * component_f1 + 0.15 * (1-negative_false))
    report = {
        "scope": "printed_defect_demo_offline_only", "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "config_sha256": readiness["config_sha256"], "metrics": metrics, "hard_gate_checks": checks,
        "hard_gates_passed": all(checks.values()), "demo_control_composite": composite,
        "invalid_sections": invalid_sections,
        "status": "BLOCKED" if invalid_sections or composite is None or not all(checks.values()) else "VALIDATION_PASSED_NOT_DEPLOYMENT_APPROVED",
        "actuator_authorization": "PROHIBITED",
        "notes": ["ten_second_stability requires a separately reviewed >=10 s stable sequence and is not inferred from a 2 s section"],
    }
    output = resolve_path(config["paths"]["reports"]) / f"{args.split}_{args.checkpoint.stem}_evaluation.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
