from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OFFICIAL_CODE,
    DEFAULT_RUNS_ROOT,
    LabelToGrad,
    MetricSums,
    SteelCrackDataset,
    atomic_torch_save,
    build_bgcrack,
    charbonnier_loss,
    configure_cuda_for_speed,
    dice_loss,
    json_dump,
    seed_everything,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows single-GPU trainer for the official Steelcrack BGCrack model."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--official-code", type=Path, default=DEFAULT_OFFICIAL_CODE)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--run-name", default="bgcrack-seed42")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=9)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def worker_init_fn(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    import random
    import numpy as np

    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: SteelCrackDataset,
    batch_size: int,
    workers: int,
    shuffle: bool,
    generator: torch.Generator | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )


def official_commit(official_code: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(official_code), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    label_to_grad: LabelToGrad,
    device: torch.device,
    amp: bool,
    epoch_number: int,
    total_epochs: int,
    max_batches: int = 0,
) -> dict[str, Any]:
    model.train()
    sums = {"loss": 0.0, "body_bce": 0.0, "edge_bce": 0.0, "grad": 0.0, "body_dice": 0.0, "edge_dice": 0.0}
    samples = 0
    batches = 0
    last_shapes: dict[str, list[int]] = {}
    started = time.perf_counter()

    for batch_index, (images, masks, edges, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        edges = edges.to(device, non_blocking=True)
        target_grad = label_to_grad(masks)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp):
            body, predicted_edge, predicted_grad = model(images)

        # The official model returns sigmoid probabilities rather than raw logits.
        # PyTorch intentionally rejects BCELoss inside an autocast region, so keep
        # the forward pass mixed precision and calculate the original five losses
        # in FP32 without changing their mathematical meaning.
        body = body.float()
        predicted_edge = predicted_edge.float()
        predicted_grad = predicted_grad.float()
        masks = masks.float()
        edges = edges.float()
        target_grad = target_grad.float()
        body_bce = torch.nn.functional.binary_cross_entropy(body, masks)
        edge_bce = torch.nn.functional.binary_cross_entropy(predicted_edge, edges)
        grad_loss = charbonnier_loss(predicted_grad, target_grad)
        body_dice = dice_loss(body, masks)
        edge_dice = dice_loss(predicted_edge, edges)
        loss = body_bce + edge_bce + grad_loss + body_dice + edge_dice

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at batch {batch_index}: {loss}")
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        current_batch = images.shape[0]
        samples += current_batch
        batches += 1
        sums["loss"] += float(loss.detach().cpu()) * current_batch
        sums["body_bce"] += float(body_bce.detach().cpu()) * current_batch
        sums["edge_bce"] += float(edge_bce.detach().cpu()) * current_batch
        sums["grad"] += float(grad_loss.detach().cpu()) * current_batch
        sums["body_dice"] += float(body_dice.detach().cpu()) * current_batch
        sums["edge_dice"] += float(edge_dice.detach().cpu()) * current_batch
        last_shapes = {
            "images": list(images.shape),
            "body": list(body.shape),
            "edge": list(predicted_edge.shape),
            "gradient": list(predicted_grad.shape),
        }

        if batch_index % 100 == 0 or batch_index == len(loader) or max_batches:
            print(
                json.dumps(
                    {
                        "epoch": epoch_number,
                        "epochs": total_epochs,
                        "batch": batch_index,
                        "batches": len(loader),
                        "loss": float(loss.detach().cpu()),
                    },
                    ensure_ascii=False,
                )
            )
        if max_batches and batch_index >= max_batches:
            break

    return {
        **{name: value / samples for name, value in sums.items()},
        "samples": samples,
        "batches": batches,
        "seconds": time.perf_counter() - started,
        "last_shapes": last_shapes,
    }


@torch.inference_mode()
def validate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
    max_batches: int = 0,
) -> dict[str, float]:
    model.eval()
    metrics = MetricSums()
    for batch_index, (images, masks, _) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            body, _, _ = model(images)
        metrics.update(body, masks)
        if max_batches and batch_index >= max_batches:
            break
    return metrics.averages()


def restore_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    train_generator: torch.Generator,
) -> tuple[int, float, int]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "BGCrack":
        raise ValueError("Resume checkpoint is not a BGCrack training checkpoint")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint["scaler_state_dict"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
    train_generator.set_state(checkpoint["loader_generator_state"])
    return (
        int(checkpoint["epoch"]) + 1,
        float(checkpoint.get("best_soft_dice", -math.inf)),
        int(checkpoint.get("best_epoch", 0)),
    )


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.workers < 0:
        raise ValueError("epochs and batch-size must be positive; workers must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for BGCrack training")

    data_root = args.data_root.resolve()
    official_code = args.official_code.resolve()
    runs_root = args.runs_root.resolve()
    seed_everything(args.seed)
    configure_cuda_for_speed()
    device = torch.device("cuda:0")
    amp = bool(args.amp)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    train_generator = torch.Generator().manual_seed(args.seed)
    train_dataset = SteelCrackDataset(data_root, "Train", include_edge=True)
    validation_dataset = SteelCrackDataset(data_root, "Validation", include_edge=False)
    train_loader = make_loader(
        train_dataset, args.batch_size, args.workers, True, train_generator
    )
    validation_loader = make_loader(validation_dataset, 1, args.workers, False)

    model = build_bgcrack(official_code).to(device)
    label_to_grad = LabelToGrad().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    print(
        json.dumps(
            {
                "device": str(device),
                "gpu": torch.cuda.get_device_name(device),
                "model": "BGCrack",
                "parameters": parameter_count,
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "batch_size": args.batch_size,
                "workers": args.workers,
                "amp": amp,
                "official_commit": official_commit(official_code),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.smoke_batches > 0:
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            label_to_grad,
            device,
            amp,
            1,
            1,
            max_batches=args.smoke_batches,
        )
        validation_result = validate(
            model, validation_loader, device, amp, max_batches=1
        )
        report = {
            "smoke": True,
            "train": train_result,
            "validation_one_batch": validation_result,
            "max_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "all_losses_finite": all(
                math.isfinite(float(train_result[name]))
                for name in ("loss", "body_bce", "edge_bce", "grad", "body_dice", "edge_dice")
            ),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["all_losses_finite"]:
            raise FloatingPointError("Smoke test produced a non-finite loss")
        return

    if args.resume is not None:
        resume_path = args.resume.resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        run_dir = resume_path.parent.parent
    else:
        run_dir = runs_root / args.run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already exists; use --resume or another --run-name: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    config = {
        "model": "BGCrack",
        "data_root": str(data_root),
        "official_code": str(official_code),
        "official_commit": official_commit(official_code),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "lr": args.lr,
        "seed": args.seed,
        "amp": amp,
        "validation_starts_at_epoch": max(1, math.ceil(args.epochs * 0.6)),
        "best_metric": "validation_soft_dice",
    }
    if args.resume is None:
        json_dump(run_dir / "run_config.json", config)

    start_epoch = 0
    best_soft_dice = -math.inf
    best_epoch = 0
    if args.resume is not None:
        start_epoch, best_soft_dice, best_epoch = restore_checkpoint(
            resume_path, model, optimizer, scaler, train_generator
        )

    fields = [
        "epoch",
        "train_loss",
        "train_body_bce",
        "train_edge_bce",
        "train_grad_loss",
        "train_body_dice_loss",
        "train_edge_dice_loss",
        "validation_soft_dice",
        "validation_hard_dice",
        "validation_hard_iou",
        "epoch_seconds",
        "max_cuda_memory_mib",
    ]
    validation_start = max(1, math.ceil(args.epochs * 0.6))

    for epoch_index in range(start_epoch, args.epochs):
        epoch_number = epoch_index + 1
        epoch_started = time.perf_counter()
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            label_to_grad,
            device,
            amp,
            epoch_number,
            args.epochs,
        )
        validation_result: dict[str, float] | None = None
        if epoch_number >= validation_start:
            validation_result = validate(model, validation_loader, device, amp)
            if validation_result["soft_dice"] > best_soft_dice:
                best_soft_dice = validation_result["soft_dice"]
                best_epoch = epoch_number
                atomic_torch_save(model.state_dict(), weights_dir / "best.pth")

        checkpoint = {
            "format_version": 1,
            "model_name": "BGCrack",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "epoch": epoch_index,
            "best_soft_dice": best_soft_dice,
            "best_epoch": best_epoch,
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all(),
            "loader_generator_state": train_generator.get_state(),
            "config": config,
        }
        atomic_torch_save(checkpoint, weights_dir / "last_resume.pt")

        row = {
            "epoch": epoch_number,
            "train_loss": train_result["loss"],
            "train_body_bce": train_result["body_bce"],
            "train_edge_bce": train_result["edge_bce"],
            "train_grad_loss": train_result["grad"],
            "train_body_dice_loss": train_result["body_dice"],
            "train_edge_dice_loss": train_result["edge_dice"],
            "validation_soft_dice": validation_result["soft_dice"] if validation_result else "",
            "validation_hard_dice": validation_result["hard_dice"] if validation_result else "",
            "validation_hard_iou": validation_result["hard_iou"] if validation_result else "",
            "epoch_seconds": time.perf_counter() - epoch_started,
            "max_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
        }
        write_header = not metrics_path.exists()
        with metrics_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(json.dumps(row, ensure_ascii=False))

    best_path = weights_dir / "best.pth"
    summary = {
        "run_dir": str(run_dir),
        "completed_epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_soft_dice": best_soft_dice if best_epoch else None,
        "best_checkpoint": str(best_path) if best_path.is_file() else None,
        "best_checkpoint_sha256": sha256_file(best_path) if best_path.is_file() else None,
        "resume_checkpoint": str(weights_dir / "last_resume.pt"),
    }
    json_dump(run_dir / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
