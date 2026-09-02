from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    CAPTURE_ROOT,
    CLASS_NAMES,
    CaptureBGCrack,
    CaptureDataset,
    LabelToGrad,
    atomic_torch_save,
    build_capture_bgcrack,
    build_corrosion_model,
    charbonnier_loss,
    dice_loss_binary,
    dice_loss_multiclass,
    load_config,
    project_path,
    seed_everything,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 1280x720 capture model in FP32.")
    parser.add_argument("--model", choices=("corrosion", "crack"), required=True)
    parser.add_argument("--data-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--initial-checkpoint", type=Path)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def build_model(task: str, initial_checkpoint: Path) -> torch.nn.Module:
    if task == "corrosion":
        return build_corrosion_model(initial_checkpoint)
    return CaptureBGCrack(build_capture_bgcrack(initial_checkpoint))


def corrosion_loss(
    logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor
) -> tuple[torch.Tensor, dict[str, float]]:
    cross_entropy = F.cross_entropy(
        logits, target, weight=class_weights, ignore_index=255
    )
    dice = dice_loss_multiclass(logits, target)
    total = cross_entropy + dice
    return total, {"cross_entropy": float(cross_entropy.detach()), "dice_loss": float(dice.detach())}


def crack_loss(
    model: CaptureBGCrack,
    images: torch.Tensor,
    mask: torch.Tensor,
    edge: torch.Tensor,
    label_to_grad: LabelToGrad,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    body, predicted_edge, gradient = model.forward_all(images)
    target_gradient = label_to_grad(mask)
    body_bce = F.binary_cross_entropy(body, mask)
    edge_bce = F.binary_cross_entropy(predicted_edge, edge)
    body_dice = dice_loss_binary(body, mask)
    edge_dice = dice_loss_binary(predicted_edge, edge)
    grad = charbonnier_loss(gradient, target_gradient)
    total = body_bce + edge_bce + body_dice + edge_dice + grad
    return (
        total,
        {
            "body_bce": float(body_bce.detach()),
            "edge_bce": float(edge_bce.detach()),
            "body_dice_loss": float(body_dice.detach()),
            "edge_dice_loss": float(edge_dice.detach()),
            "gradient_loss": float(grad.detach()),
        },
        body,
    )


def update_confusion(
    confusion: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
) -> None:
    valid = target.ne(255)
    encoded = target[valid] * 4 + prediction[valid]
    confusion += torch.bincount(encoded.cpu(), minlength=16).reshape(4, 4)


def corrosion_metrics(confusion: torch.Tensor) -> dict[str, Any]:
    true_positive = confusion.diag().float()
    target_total = confusion.sum(dim=1).float()
    predicted_total = confusion.sum(dim=0).float()
    union = target_total + predicted_total - true_positive
    iou = torch.where(union > 0, true_positive / union, torch.nan)
    recall = torch.where(target_total > 0, true_positive / target_total, torch.nan)
    precision = torch.where(predicted_total > 0, true_positive / predicted_total, torch.nan)
    return {
        "macro_iou": float(torch.nanmean(iou)),
        "per_class_iou": {name: float(iou[i]) for i, name in enumerate(CLASS_NAMES)},
        "per_class_recall": {name: float(recall[i]) for i, name in enumerate(CLASS_NAMES)},
        "per_class_precision": {
            name: float(precision[i]) for i, name in enumerate(CLASS_NAMES)
        },
        "valid_pixels": int(confusion.sum()),
    }


@torch.inference_mode()
def evaluate(
    task: str, model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> dict[str, Any]:
    model.eval()
    if task == "corrosion":
        confusion = torch.zeros((4, 4), dtype=torch.int64)
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            logits = model(images)
            update_confusion(confusion, target, logits.argmax(dim=1))
        return corrosion_metrics(confusion)

    sample_count = 0
    soft_dice_sum = 0.0
    hard_dice_sum = 0.0
    hard_iou_sum = 0.0
    precision_sum = 0.0
    recall_sum = 0.0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["mask"].to(device, non_blocking=True)
        probability = model(images).flatten(1)
        target = target.flatten(1)
        predicted = probability.ge(0.5).float()
        intersection_soft = (probability * target).sum(dim=1)
        intersection = (predicted * target).sum(dim=1)
        predicted_count = predicted.sum(dim=1)
        target_count = target.sum(dim=1)
        union = predicted_count + target_count - intersection
        soft_dice = (2 * intersection_soft + 1e-6) / (
            probability.sum(dim=1) + target_count + 1e-6
        )
        hard_dice = (2 * intersection + 1e-6) / (
            predicted_count + target_count + 1e-6
        )
        hard_iou = (intersection + 1e-6) / (union + 1e-6)
        precision = (intersection + 1e-6) / (predicted_count + 1e-6)
        recall = (intersection + 1e-6) / (target_count + 1e-6)
        count = images.shape[0]
        sample_count += count
        soft_dice_sum += float(soft_dice.sum())
        hard_dice_sum += float(hard_dice.sum())
        hard_iou_sum += float(hard_iou.sum())
        precision_sum += float(precision.sum())
        recall_sum += float(recall.sum())
    return {
        "samples": sample_count,
        "soft_dice": soft_dice_sum / sample_count,
        "hard_dice": hard_dice_sum / sample_count,
        "hard_iou": hard_iou_sum / sample_count,
        "precision": precision_sum / sample_count,
        "recall": recall_sum / sample_count,
        "valid_pixels": sample_count * 1280 * 720,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for capture-model training")
    if min(args.epochs, args.batch_size, args.accumulate) < 1 or args.workers < 0:
        raise ValueError("Invalid epochs, batch-size, accumulate, or workers")
    seed_everything(args.seed)
    config = load_config()
    defaults = config["training_defaults"]
    initial_checkpoint = (
        args.initial_checkpoint.resolve()
        if args.initial_checkpoint
        else project_path(config[args.model]["initial_checkpoint"])
    )
    learning_rate = args.lr or float(defaults[f"{args.model}_learning_rate"])
    data_root = args.data_root.resolve()
    run_dir = args.runs_root.resolve() / args.run_name
    weights_dir = run_dir / "weights"
    if run_dir.exists() and not args.resume and any(run_dir.iterdir()):
        raise FileExistsError(f"Run directory already exists; use --resume: {run_dir}")
    weights_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = CaptureDataset(data_root, args.model.replace("corrosion", "rust"), "train", True)
    val_dataset = CaptureDataset(
        data_root, args.model.replace("corrosion", "rust"), "validation", False
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    model = build_model(args.model, initial_checkpoint).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    start_epoch = 1
    best_metric = -math.inf
    if args.resume:
        resume = torch.load(args.resume.resolve(), map_location="cpu", weights_only=True)
        if resume.get("task") != args.model:
            raise ValueError("Resume checkpoint task mismatch")
        model.load_state_dict(resume["model_state_dict"], strict=True)
        optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume["epoch"]) + 1
        best_metric = float(resume["best_metric"])

    class_weights = torch.tensor(
        config["corrosion"]["class_weights"], dtype=torch.float32, device=device
    )
    label_to_grad = LabelToGrad().to(device)
    config_report = {
        "task": args.model,
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": sha256_file(initial_checkpoint),
        "data_root": str(data_root),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulate,
        "effective_batch_size": args.batch_size * args.accumulate,
        "workers": args.workers,
        "learning_rate": learning_rate,
        "seed": args.seed,
        "precision": "fp32",
        "best_metric": "macro_iou" if args.model == "corrosion" else "soft_dice",
    }
    write_json(run_dir / "run_config.json", config_report)
    metrics_path = run_dir / "metrics.csv"

    for epoch in range(start_epoch, args.epochs + 1):
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["mask"].to(device, non_blocking=True)
            if args.model == "corrosion":
                loss, parts = corrosion_loss(model(images), target, class_weights)
            else:
                edge = batch["edge"].to(device, non_blocking=True)
                loss, parts, _ = crack_loss(model, images, target, edge, label_to_grad)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
            (loss / args.accumulate).backward()
            if batch_index % args.accumulate == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            sums["loss"] = sums.get("loss", 0.0) + float(loss.detach())
            for name, value in parts.items():
                sums[name] = sums.get(name, 0.0) + value
            batches += 1
            if args.smoke_batches and batches >= args.smoke_batches:
                break

        if args.smoke_batches:
            validation = {"skipped_for_smoke": True}
            current_metric = -math.inf
        else:
            validation = evaluate(args.model, model, val_loader, device)
            current_metric = float(
                validation["macro_iou"]
                if args.model == "corrosion"
                else validation["soft_dice"]
            )
        row = {
            "epoch": epoch,
            **{f"train_{name}": value / batches for name, value in sums.items()},
            **{f"validation_{name}": value for name, value in validation.items() if not isinstance(value, dict)},
            "epoch_seconds": time.perf_counter() - started,
            "max_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        }
        exists = metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, ensure_ascii=False))

        improved = current_metric > best_metric
        if improved:
            best_metric = current_metric
        payload = {
            "format_version": 1,
            "task": args.model,
            "architecture": (
                "deeplabv3plus_resnet101_os8"
                if args.model == "corrosion"
                else "bgcrack_v1_capture_ext720_int736_minpad"
            ),
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "run_config": config_report,
            "validation": validation,
        }
        atomic_torch_save(payload, weights_dir / "last_resume.pt")
        if improved:
            best_payload = dict(payload)
            best_payload.pop("optimizer_state_dict")
            atomic_torch_save(best_payload, weights_dir / "best.pt")
        if args.smoke_batches:
            break


if __name__ == "__main__":
    main()
