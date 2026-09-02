from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from common import (
    CAPTURE_ROOT,
    CLASS_NAMES,
    EXTERNAL_HEIGHT,
    EXTERNAL_WIDTH,
    MASK_COLORS_RGB,
    CaptureBGCrack,
    DeviceSafeMapToGrad,
    build_capture_bgcrack,
    build_corrosion_model,
    checkpoint_state,
    load_config,
    project_path,
    sha256_file,
    write_json,
)


PUBLIC_SIZE = 512
TOP = (EXTERNAL_HEIGHT - PUBLIC_SIZE) // 2
LEFT = (EXTERNAL_WIDTH - PUBLIC_SIZE) // 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare original 512 inference with centered 720x1280 capture inference."
    )
    parser.add_argument("--model", choices=("corrosion", "crack"), required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--padding-policy", choices=("zero", "reflect", "replicate"), default="reflect"
    )
    parser.add_argument("--allow-test-rerun", action="store_true")
    return parser.parse_args()


def rust_items(split: str) -> list[tuple[Path, Path, str]]:
    root = (
        CAPTURE_ROOT.parent
        / "data"
        / "raw"
        / "virginia_tech_cssd"
        / "dataset"
        / "512x512"
    )
    if split == "validation":
        csv_path = CAPTURE_ROOT.parent / "vt_kd" / "splits" / "vt_train316_val80_seed42.csv"
        items: list[tuple[Path, Path, str]] = []
        with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                if row["split"] == "val":
                    items.append(
                        (
                            root / row["image_path"],
                            root / row["mask_path"],
                            Path(row["image_path"]).stem,
                        )
                    )
        return items
    images = {path.stem: path for path in (root / "Test" / "images_512").iterdir() if path.is_file()}
    masks = {path.stem: path for path in (root / "Test" / "mask_512").iterdir() if path.is_file()}
    if images.keys() != masks.keys():
        raise RuntimeError("Virginia Tech Test image/mask mismatch")
    return [(images[name], masks[name], name) for name in sorted(images)]


def crack_items(split: str) -> list[tuple[Path, Path, str]]:
    official_split = "Validation" if split == "validation" else "Test"
    root = CAPTURE_ROOT.parent / "steelcrack" / "data" / "Steelcrack" / official_split
    images = {path.stem: path for path in (root / "images").iterdir() if path.is_file()}
    masks = {path.stem: path for path in (root / "masks").iterdir() if path.is_file()}
    if images.keys() != masks.keys():
        raise RuntimeError(f"Steelcrack {official_split} image/mask mismatch")
    return [(images[name], masks[name], name) for name in sorted(images)]


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        array = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    if array.shape != (PUBLIC_SIZE, PUBLIC_SIZE, 3):
        raise ValueError(f"Expected 512x512 RGB image: {path}")
    return array


def read_rust_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        raw = np.asarray(source).copy()
    if raw.shape[:2] != (PUBLIC_SIZE, PUBLIC_SIZE):
        raise ValueError(f"Expected 512x512 rust mask: {path}")
    if raw.ndim == 2:
        values = set(np.unique(raw).tolist())
        if values.issubset({0, 1, 2, 3, 255}):
            return torch.from_numpy(raw.astype(np.int64))
    rgb = raw[..., :3]
    target = np.full((PUBLIC_SIZE, PUBLIC_SIZE), 255, dtype=np.int64)
    for index, color in enumerate(MASK_COLORS_RGB):
        target[np.all(rgb == color, axis=2)] = index
    if np.any(target == 255):
        raise ValueError(f"Unsupported rust mask palette: {path}")
    return torch.from_numpy(target)


def read_crack_mask(path: Path) -> torch.Tensor:
    with Image.open(path) as source:
        raw = np.asarray(source.convert("L"), dtype=np.uint8).copy()
    if raw.shape != (PUBLIC_SIZE, PUBLIC_SIZE):
        raise ValueError(f"Expected 512x512 crack mask: {path}")
    return torch.from_numpy((raw > 0).astype(np.float32))


def centered_canvas(image: torch.Tensor, policy: str) -> torch.Tensor:
    padding = (
        LEFT,
        EXTERNAL_WIDTH - PUBLIC_SIZE - LEFT,
        TOP,
        EXTERNAL_HEIGHT - PUBLIC_SIZE - TOP,
    )
    if policy == "zero":
        return torch.nn.functional.pad(image, padding, mode="constant", value=0.0)
    return torch.nn.functional.pad(image, padding, mode=policy)


def update_confusion(
    confusion: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
) -> None:
    valid = target.ne(255)
    encoded = target[valid] * 4 + prediction[valid]
    confusion += torch.bincount(encoded.cpu(), minlength=16).reshape(4, 4)


def rust_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    diagonal = confusion.diag().float()
    target = confusion.sum(1).float()
    predicted = confusion.sum(0).float()
    union = target + predicted - diagonal
    iou = torch.where(union > 0, diagonal / union, torch.nan)
    recall = torch.where(target > 0, diagonal / target, torch.nan)
    precision = torch.where(predicted > 0, diagonal / predicted, torch.nan)
    return {
        "macro_iou": float(torch.nanmean(iou)),
        "per_class_iou": {name: float(iou[i]) for i, name in enumerate(CLASS_NAMES)},
        "per_class_recall": {name: float(recall[i]) for i, name in enumerate(CLASS_NAMES)},
        "per_class_precision": {name: float(precision[i]) for i, name in enumerate(CLASS_NAMES)},
        "valid_pixels": int(confusion.sum()),
    }


class BinarySums:
    def __init__(self) -> None:
        self.samples = 0
        self.soft_dice = 0.0
        self.hard_dice = 0.0
        self.hard_iou = 0.0
        self.precision = 0.0
        self.recall = 0.0

    def update(self, probability: torch.Tensor, target: torch.Tensor) -> None:
        probability = probability.flatten(1)
        target = target.flatten(1)
        prediction = probability.ge(0.5).float()
        soft_intersection = (probability * target).sum(1)
        intersection = (prediction * target).sum(1)
        predicted_count = prediction.sum(1)
        target_count = target.sum(1)
        union = predicted_count + target_count - intersection
        self.samples += probability.shape[0]
        self.soft_dice += float(
            ((2 * soft_intersection + 1e-6) / (probability.sum(1) + target_count + 1e-6)).sum()
        )
        self.hard_dice += float(
            ((2 * intersection + 1e-6) / (predicted_count + target_count + 1e-6)).sum()
        )
        self.hard_iou += float(((intersection + 1e-6) / (union + 1e-6)).sum())
        self.precision += float(((intersection + 1e-6) / (predicted_count + 1e-6)).sum())
        self.recall += float(((intersection + 1e-6) / (target_count + 1e-6)).sum())

    def result(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "soft_dice": self.soft_dice / self.samples,
            "hard_dice": self.hard_dice / self.samples,
            "hard_iou": self.hard_iou / self.samples,
            "precision": self.precision / self.samples,
            "recall": self.recall / self.samples,
            "valid_pixels": self.samples * PUBLIC_SIZE * PUBLIC_SIZE,
        }


def build_original_bgcrack(checkpoint: Path) -> torch.nn.Module:
    official_root = CAPTURE_ROOT.parent / "steelcrack" / "official_bgcrack"
    text = str(official_root.resolve())
    if text not in sys.path:
        sys.path.insert(0, text)
    import Model.Module.utils as official_utils

    official_utils.Map_2_Grad = DeviceSafeMapToGrad
    from Model.BGCrack import BGCrack

    model = BGCrack()
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return model


@torch.inference_mode()
def evaluate_corrosion(
    checkpoint: Path, split: str, device: torch.device, padding_policy: str
) -> dict[str, Any]:
    model = build_corrosion_model(checkpoint).to(device).eval()
    original_confusion = torch.zeros((4, 4), dtype=torch.int64)
    capture_confusion = torch.zeros((4, 4), dtype=torch.int64)
    matching = 0
    total = 0
    items = rust_items(split)
    started = time.perf_counter()
    for image_path, mask_path, _ in items:
        rgb = read_rgb(image_path)
        bgr = np.ascontiguousarray(rgb[..., ::-1].transpose(2, 0, 1)).astype(np.float32)
        image = torch.from_numpy(bgr).unsqueeze(0).to(device)
        target = read_rust_mask(mask_path)
        original = model(image).argmax(1).cpu()
        capture = model(centered_canvas(image, padding_policy))[
            ..., TOP : TOP + PUBLIC_SIZE, LEFT : LEFT + PUBLIC_SIZE
        ]
        capture = capture.argmax(1).cpu()
        update_confusion(original_confusion, target.unsqueeze(0), original)
        update_confusion(capture_confusion, target.unsqueeze(0), capture)
        matching += int((original == capture).sum())
        total += original.numel()
    return {
        "samples": len(items),
        "original_512_metrics": rust_metrics(original_confusion),
        "capture_center_roi_metrics": rust_metrics(capture_confusion),
        "shape_change_pixel_agreement": matching / total,
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_crack(
    checkpoint: Path, split: str, device: torch.device, padding_policy: str
) -> dict[str, Any]:
    original_model = build_original_bgcrack(checkpoint).to(device).eval()
    capture_model = CaptureBGCrack(build_capture_bgcrack(checkpoint)).to(device).eval()
    original_sums = BinarySums()
    capture_sums = BinarySums()
    matching = 0
    total = 0
    items = crack_items(split)
    started = time.perf_counter()
    for image_path, mask_path, _ in items:
        rgb = read_rgb(image_path).transpose(2, 0, 1).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(np.ascontiguousarray(rgb)).unsqueeze(0).to(device)
        target = read_crack_mask(mask_path).unsqueeze(0).unsqueeze(0)
        original = original_model(image)[0].cpu()
        capture = capture_model(centered_canvas(image, padding_policy))[
            ..., TOP : TOP + PUBLIC_SIZE, LEFT : LEFT + PUBLIC_SIZE
        ].cpu()
        original_sums.update(original, target)
        capture_sums.update(capture, target)
        matching += int((original.ge(0.5) == capture.ge(0.5)).sum())
        total += original.numel()
    return {
        "samples": len(items),
        "original_512_metrics": original_sums.result(),
        "capture_center_roi_metrics": capture_sums.result(),
        "shape_change_pixel_agreement": matching / total,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    marker = output.parent / f"public_test_regression_{args.model}.json"
    if args.split == "test" and marker.exists() and not args.allow_test_rerun:
        raise FileExistsError(f"Public Test regression already exists: {marker}")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0")
    result = (
        evaluate_corrosion(checkpoint, args.split, device, args.padding_policy)
        if args.model == "corrosion"
        else evaluate_crack(checkpoint, args.split, device, args.padding_policy)
    )
    if args.model == "corrosion":
        original_metrics = result["original_512_metrics"]
        capture_metrics = result["capture_center_roi_metrics"]
        metric_drops = {
            "macro_iou": original_metrics["macro_iou"] - capture_metrics["macro_iou"],
            "severe_recall": (
                original_metrics["per_class_recall"]["Severe"]
                - capture_metrics["per_class_recall"]["Severe"]
            ),
        }
        acceptance = {
            "macro_iou_drop_max": 0.015,
            "severe_recall_drop_max": 0.01,
            "passed": (
                metric_drops["macro_iou"] <= 0.015
                and metric_drops["severe_recall"] <= 0.01
            ),
        }
    else:
        original_metrics = result["original_512_metrics"]
        capture_metrics = result["capture_center_roi_metrics"]
        metric_drops = {
            "hard_dice": original_metrics["hard_dice"] - capture_metrics["hard_dice"],
            "hard_iou": original_metrics["hard_iou"] - capture_metrics["hard_iou"],
            "recall": original_metrics["recall"] - capture_metrics["recall"],
        }
        acceptance = {
            "hard_dice_drop_max": 0.015,
            "hard_iou_drop_max": 0.015,
            "recall_drop_max": 0.02,
            "passed": (
                metric_drops["hard_dice"] <= 0.015
                and metric_drops["hard_iou"] <= 0.015
                and metric_drops["recall"] <= 0.02
            ),
        }
    report = {
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model": args.model,
        "split": args.split,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "public_source": (
            "Virginia Tech CSSD 512x512"
            if args.model == "corrosion"
            else "Steelcrack 512x512"
        ),
        "placement_policy": {
            "resize": False,
            "stretch": False,
            "canvas": [EXTERNAL_WIDTH, EXTERNAL_HEIGHT],
            "source": [PUBLIC_SIZE, PUBLIC_SIZE],
            "top": TOP,
            "left": LEFT,
            "padding_policy": args.padding_policy,
            "padding_value": 0.0 if args.padding_policy == "zero" else None,
            "evaluated_pixels": "center source ROI only",
            "pixels_per_image": PUBLIC_SIZE * PUBLIC_SIZE,
        },
        **result,
        "metric_drops": metric_drops,
        "provisional_acceptance": acceptance,
        "max_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "real_camera_accuracy": "unverified",
    }
    write_json(output, report)
    if args.split == "test":
        write_json(marker, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
