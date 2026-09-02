from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    audit_manifest,
    load_config,
    read_manifest_rows,
    resolve_path,
    sha256_file,
    write_json,
)
from data import MultitaskDataset
from model import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search preregistered crack probability and connected-component "
            "thresholds on validation data only."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def metrics(counts: dict[str, int]) -> dict[str, float | int]:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    return {
        **counts,
        "dice": (2.0 * tp) / max(1, 2 * tp + fp + fn),
        "recall": tp / max(1, tp + fn),
        "precision": tp / max(1, tp + fp),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    manifest = args.manifest.resolve() if args.manifest else resolve_path(config["paths"]["val_manifest"])
    if "locked" in manifest.name.lower() or "test" in manifest.name.lower():
        raise PermissionError("This tool is restricted to validation manifests")
    audit = audit_manifest(config, manifest, "val")
    if audit.issues:
        raise ValueError("Validation manifest audit failed:\n - " + "\n - ".join(audit.issues))

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config_sha256") != sha256_file(config_path):
        raise ValueError("Checkpoint/config SHA mismatch")
    if checkpoint.get("val_manifest_sha256") != sha256_file(manifest):
        raise ValueError("Checkpoint/validation manifest SHA mismatch")

    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    loader = DataLoader(
        MultitaskDataset(read_manifest_rows(manifest), config, training=False, use_cache=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    search = config["postprocess_search"]
    thresholds = sorted({float(value) for value in search["probability_thresholds"]})
    component_mins = [0] + sorted({int(value) for value in search["min_component_pixels"]})
    counters: dict[tuple[float, int], dict[str, int]] = {
        (threshold, minimum): {"tp": 0, "fp": 0, "fn": 0, "kept_components": 0}
        for threshold in thresholds
        for minimum in component_mins
    }
    start, end = (int(value) for value in config["contracts"]["crack_valid_rows"])
    samples = 0
    for raw in tqdm(loader, desc="postprocess validation"):
        images = raw["image"].to(device, non_blocking=True)
        probabilities = model(images)[:, 4, start:end].sigmoid().cpu().numpy()
        targets = raw["crack_target"][:, 0, start:end].numpy() > 0.5
        valid = raw["crack_valid"].numpy().astype(bool)
        for probability, target, is_valid in zip(probabilities, targets, valid, strict=True):
            if not is_valid:
                continue
            samples += 1
            target_positive = int(target.sum())
            for threshold in thresholds:
                binary = np.ascontiguousarray(probability >= threshold, dtype=np.uint8)
                labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                    binary, connectivity=int(search.get("connectivity", 8))
                )
                areas = stats[:, cv2.CC_STAT_AREA]
                for minimum in component_mins:
                    keep = areas >= minimum
                    keep[0] = False
                    prediction = keep[labels]
                    tp = int(np.logical_and(prediction, target).sum())
                    fp = int(np.logical_and(prediction, np.logical_not(target)).sum())
                    counts = counters[(threshold, minimum)]
                    counts["tp"] += tp
                    counts["fp"] += fp
                    counts["fn"] += target_positive - tp
                    counts["kept_components"] += int(keep[:labels_count].sum())

    acceptance = config["preregistered_acceptance"]
    dice_min = float(acceptance["crack_dice_min"])
    recall_min = float(acceptance["crack_pixel_recall_min"])
    results: list[dict[str, Any]] = []
    for (threshold, minimum), counts in counters.items():
        result: dict[str, Any] = {
            "probability_threshold": threshold,
            "min_component_pixels": minimum,
            **metrics(counts),
        }
        result["passed"] = result["dice"] >= dice_min and result["recall"] >= recall_min
        results.append(result)
    results.sort(key=lambda item: (item["probability_threshold"], item["min_component_pixels"]))
    passing = sorted(
        (item for item in results if item["passed"]),
        key=lambda item: (item["dice"], item["recall"], item["precision"]),
        reverse=True,
    )
    best_overall = max(results, key=lambda item: (item["dice"], item["recall"]))
    payload = {
        "scope": "validation_postprocess_selection_not_locked_test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "samples": samples,
        "acceptance": {"crack_dice_min": dice_min, "crack_pixel_recall_min": recall_min},
        "searched_probability_thresholds": thresholds,
        "searched_min_component_pixels": component_mins,
        "passing_combinations": len(passing),
        "selected": passing[0] if passing else None,
        "best_overall": best_overall,
        "results": results,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps({key: payload[key] for key in (
        "scope", "checkpoint_epoch", "samples", "acceptance", "passing_combinations", "selected", "best_overall"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
