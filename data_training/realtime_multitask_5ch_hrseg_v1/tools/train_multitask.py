from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from common import (
    assert_training_ready,
    load_config,
    read_manifest_rows,
    resolve_path,
    seed_everything,
    seed_worker,
    sha256_file,
    write_json,
)
from data import MultitaskDataset
from losses import crack_kd_loss, crack_supervised_loss, rust_kd_loss, rust_supervised_loss
from model import (
    build_model,
    enforce_frozen_batchnorm_eval,
    load_rust_checkpoint_strict,
    set_training_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the audited dual-head 5-channel student.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("rust_restore", "crack_bootstrap", "joint"), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--accumulate", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--initialize", type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def identity(config_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    cache_manifest = resolve_path(config["paths"]["teacher_cache"]) / "cache_manifest.json"
    if not cache_manifest.is_file():
        raise FileNotFoundError(f"Teacher cache manifest missing: {cache_manifest}")
    values = {
        "config_sha256": sha256_file(config_path),
        "train_manifest_sha256": sha256_file(resolve_path(config["paths"]["train_manifest"])),
        "val_manifest_sha256": sha256_file(resolve_path(config["paths"]["val_manifest"])),
        "cache_manifest_sha256": sha256_file(cache_manifest),
        "rust_teacher_sha256": sha256_file(resolve_path(config["paths"]["rust_teacher_onnx"])),
        "crack_teacher_sha256": sha256_file(resolve_path(config["paths"]["hrseg_teacher_onnx"])),
    }
    cache = json.loads(cache_manifest.read_text(encoding="utf-8"))
    if cache.get("config_sha256") != values["config_sha256"]:
        raise ValueError("Teacher cache/config SHA mismatch; rebuild the cache")
    if cache.get("manifest_sha256") != {
        "train": values["train_manifest_sha256"],
        "val": values["val_manifest_sha256"],
    }:
        raise ValueError("Teacher cache/manifest SHA mismatch; rebuild the cache")
    return values


def make_loader(
    rows: list[dict[str, str]],
    config: dict[str, Any],
    *,
    training: bool,
    batch: int,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        MultitaskDataset(rows, config, training=training, use_cache=True),
        batch_size=batch,
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=training,
    )


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def compute_loss(
    logits: torch.Tensor, batch: dict[str, Any], config: dict[str, Any], stage: str
) -> tuple[torch.Tensor, dict[str, float]]:
    loss_config = config["loss"]
    valid_rows = tuple(int(v) for v in config["contracts"]["crack_valid_rows"])
    rust_sup, rust_parts = rust_supervised_loss(
        logits[:, :4], batch["rust_target"], batch["rust_valid"], loss_config
    )
    crack_sup, crack_parts = crack_supervised_loss(
        logits[:, 4:5], batch["crack_target"], batch["crack_valid"], valid_rows, loss_config
    )
    rust_kd = rust_kd_loss(
        logits[:, :4], batch["rust_teacher"], batch["rust_teacher_available"], float(loss_config["rust_kd_temperature"])
    )
    crack_kd = crack_kd_loss(
        logits[:, 4:5], batch["crack_teacher"], batch["crack_teacher_available"], valid_rows, float(loss_config["crack_kd_temperature"])
    )
    rust_total = rust_sup + float(loss_config["rust_kd_weight"]) * rust_kd
    crack_total = crack_sup + float(loss_config["crack_kd_weight"]) * crack_kd
    total = rust_total if stage == "rust_restore" else crack_total if stage == "crack_bootstrap" else rust_total + crack_total
    parts = {**rust_parts, **crack_parts, "rust_kd": rust_kd, "crack_kd": crack_kd}
    return total, {key: float(value.detach()) for key, value in parts.items()}


@torch.no_grad()
def evaluate_epoch(model, loader, device, config) -> dict[str, float]:
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.float64, device=device)
    crack_tp = crack_fp = crack_fn = 0.0
    start, end = (int(v) for v in config["contracts"]["crack_valid_rows"])
    for raw in loader:
        batch = move(raw, device)
        logits = model(batch["image"])
        for index in torch.where(batch["rust_valid"].bool())[0]:
            target = batch["rust_target"][index].reshape(-1)
            prediction = logits[index, :4].argmax(0).reshape(-1)
            confusion += torch.bincount(target * 4 + prediction, minlength=16).reshape(4, 4)
        for index in torch.where(batch["crack_valid"].bool())[0]:
            target = batch["crack_target"][index, 0, start:end] > 0.5
            prediction = logits[index, 4, start:end].sigmoid() >= 0.5
            crack_tp += float((prediction & target).sum())
            crack_fp += float((prediction & ~target).sum())
            crack_fn += float((~prediction & target).sum())
    intersection = confusion.diag()
    union = confusion.sum(0) + confusion.sum(1) - intersection
    rust_iou = float((intersection / union.clamp_min(1)).mean())
    crack_dice = (2 * crack_tp) / max(1.0, 2 * crack_tp + crack_fp + crack_fn)
    crack_recall = crack_tp / max(1.0, crack_tp + crack_fn)
    return {"rust_macro_iou": rust_iou, "crack_dice": crack_dice, "crack_pixel_recall": crack_recall}


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    readiness = assert_training_ready(config_path)
    config = load_config(config_path)
    run_identity = identity(config_path, config)
    seed = int(args.seed if args.seed is not None else config["train"]["seed"])
    epochs = int(args.epochs if args.epochs is not None else config["train"]["epochs"])
    batch_size = int(args.batch if args.batch is not None else config["train"]["physical_batch_size"])
    accumulate = int(args.accumulate if args.accumulate is not None else config["train"]["gradient_accumulation_steps"])
    workers = int(args.workers if args.workers is not None else config["train"]["workers"])
    if batch_size < 1 or accumulate < 1 or epochs < 1 or workers < 0:
        raise ValueError("Invalid training dimensions")
    seed_everything(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    device = torch.device("cuda:0")
    model = build_model(config)
    load_rust_checkpoint_strict(model, resolve_path(config["paths"]["rust_checkpoint"]))

    if args.initialize:
        initial = torch.load(args.initialize.resolve(), map_location="cpu", weights_only=True)
        if initial.get("architecture") != config["student"]["architecture"]:
            raise ValueError("Initialization checkpoint architecture mismatch")
        model.load_state_dict(initial["model_state_dict"], strict=True)
    set_training_stage(model, args.stage)
    model.to(device)

    groups = []
    for module, lr in (
        (model.backbone, config["train"]["backbone_lr"]),
        (model.classifier, config["train"]["rust_head_lr"]),
        (model.crack_head, config["train"]["crack_head_lr"]),
    ):
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": float(lr)})
    optimizer = torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))
    scaler = GradScaler(enabled=True)
    start_epoch = 1
    best_score = -math.inf

    run_dir = resolve_path(config["paths"]["runs"]) / args.run_name
    weights_dir = run_dir / "weights"
    if run_dir.exists() and not args.resume:
        raise FileExistsError(f"Run directory exists; use --resume or a new name: {run_dir}")
    weights_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        resumed = torch.load(args.resume.resolve(), map_location="cpu", weights_only=True)
        for key, value in run_identity.items():
            if resumed.get(key) != value:
                raise ValueError(f"Resume provenance mismatch for {key}")
        if resumed.get("stage") != args.stage or int(resumed.get("seed", -1)) != seed:
            raise ValueError("Resume stage/seed mismatch")
        model.load_state_dict(resumed["model_state_dict"], strict=True)
        optimizer.load_state_dict(resumed["optimizer_state_dict"])
        scaler.load_state_dict(resumed["scaler_state_dict"])
        start_epoch = int(resumed["epoch"]) + 1
        best_score = float(resumed["best_score"])

    train_rows = read_manifest_rows(resolve_path(config["paths"]["train_manifest"]))
    val_rows = read_manifest_rows(resolve_path(config["paths"]["val_manifest"]))
    train_loader = make_loader(train_rows, config, training=True, batch=batch_size, workers=workers, seed=seed)
    val_loader = make_loader(val_rows, config, training=False, batch=batch_size, workers=workers, seed=seed)
    acceptance = config["preregistered_acceptance"]
    for epoch in range(start_epoch, epochs + 1):
        started = time.perf_counter()
        model.train()
        enforce_frozen_batchnorm_eval(model)
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, raw in enumerate(train_loader, start=1):
            batch_data = move(raw, device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(batch_data["image"])
                loss, _ = compute_loss(logits, batch_data, config, args.stage)
                loss = loss / accumulate
            scaler.scale(loss).backward()
            if step % accumulate == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running += float(loss.detach()) * accumulate
        metrics = evaluate_epoch(model, val_loader, device, config)
        eligible = (
            metrics["rust_macro_iou"] >= float(acceptance["rust_macro_iou_min"])
            and metrics["crack_pixel_recall"] >= float(acceptance["crack_pixel_recall_min"])
            and metrics["crack_dice"] >= float(acceptance["crack_dice_min"])
        )
        score = metrics["rust_macro_iou"] if args.stage == "rust_restore" else (
            metrics["rust_macro_iou"] + metrics["crack_dice"] + 0.25 * metrics["crack_pixel_recall"] if eligible else -math.inf
        )
        payload = {
            "format_version": 1,
            "architecture": config["student"]["architecture"],
            "stage": args.stage,
            "seed": seed,
            "epoch": epoch,
            "best_score": max(best_score, score),
            "metrics": metrics,
            **run_identity,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        }
        torch.save(payload, weights_dir / "last.pt")
        if score > best_score:
            best_score = score
            payload["best_score"] = best_score
            torch.save(payload, weights_dir / "best.pt")
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            **metrics,
            "eligible": eligible,
            "best_score": best_score if math.isfinite(best_score) else None,
            "epoch_seconds": time.perf_counter() - started,
            "max_cuda_memory_mib": torch.cuda.max_memory_allocated() / 1024**2,
        }
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
