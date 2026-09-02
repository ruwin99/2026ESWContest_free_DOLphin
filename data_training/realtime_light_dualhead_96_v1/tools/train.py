from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from common import assert_ready, assert_teacher_cache, load_config, resolve_path, seed_everything, sha256_file, write_json
from dataset import DemoManifestDataset
from light_dualhead_96 import (
    LightDualHead96,
    freeze_batchnorm_running_stats,
    load_encoder_from_rust_checkpoint_strict,
    set_stage,
    stage4_encoder_modules,
)
from samplers import HalfNegativeBatchSampler
from training_core import compute_losses, finite_losses, move_batch


def trainable_groups(model: LightDualHead96, stage: int, config: dict[str, Any]) -> list[dict[str, Any]]:
    rates = config["train"]["stage_learning_rates"][f"stage{stage}"]
    if stage == 1:
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": float(rates["shared"])}]
    if stage == 2:
        return [{"params": model.crack_detail.parameters(), "lr": float(rates["crack_head"])}]
    shared = [*model.low_project.parameters(), *model.lraspp.parameters(), *model.shared_decoder.parameters()]
    heads = [*model.rust_head.parameters(), *model.crack_detail.parameters()]
    groups = [{"params": shared, "lr": float(rates["shared"])}, {"params": heads, "lr": float(rates["heads"])}]
    if stage == 4:
        encoder = [p for module in stage4_encoder_modules(model) for p in module.parameters()]
        groups.append({"params": encoder, "lr": float(rates["encoder"])})
    return groups


@torch.no_grad()
def validate_loss(model: LightDualHead96, loader: DataLoader, config: dict[str, Any], device: torch.device, stage: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["image"])
        losses = compute_losses(output, batch, config)
        if not finite_losses(losses):
            raise FloatingPointError("Non-finite validation loss")
        for key, value in losses.items():
            totals[key] = totals.get(key, 0.0) + float(value)
        count += 1
    if not count:
        raise RuntimeError("Validation loader is empty")
    result = {f"val_{key}": value / count for key, value in totals.items()}
    selection = result["val_rust_total"] if stage == 1 else result["val_crack_total"] if stage == 2 else result["val_total"]
    result["selection_loss"] = selection
    return result


def selection_state_from_history(history_path: Path, seed: int, stage: int, through_epoch: int) -> tuple[float, int]:
    """Recover stage-local early-stopping state without trusting a reset resume run."""
    if not history_path.exists():
        return math.inf, 0
    by_epoch: dict[int, float] = {}
    with history_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                epoch = int(record["epoch"])
                if int(record["seed"]) == seed and int(record["stage"]) == stage and epoch <= through_epoch:
                    by_epoch[epoch] = float(record["selection_loss"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue
    if not by_epoch:
        return math.inf, 0
    ordered = sorted(by_epoch.items())
    best_position = min(range(len(ordered)), key=lambda index: ordered[index][1])
    return ordered[best_position][1], len(ordered) - best_position - 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Four-stage fail-closed LightDualHead96 trainer.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--stage", choices=("1", "2", "3", "4", "all"), default="all")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int, default=0, help="Runtime-only physical batch override")
    parser.add_argument("--workers", type=int, default=0, help="Runtime-only DataLoader worker override")
    parser.add_argument("--accumulation", type=int, default=0, help="Runtime-only gradient accumulation override")
    args = parser.parse_args()
    for name in ("batch_size", "workers", "accumulation"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be zero or greater")
    config_path = args.config.resolve()
    readiness = assert_ready(config_path)
    config = load_config(config_path)
    assert_teacher_cache(config, readiness, "train")
    assert_teacher_cache(config, readiness, "validation")
    if not torch.cuda.is_available():
        raise RuntimeError("BLOCKED_RTX5070TI_PREFLIGHT: CUDA is required; CPU fallback is forbidden")
    if "5070 Ti" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"BLOCKED_RTX5070TI_PREFLIGHT: expected RTX 5070 Ti, got {torch.cuda.get_device_name(0)}")
    seed_everything(args.seed)
    device = torch.device("cuda:0")
    model = LightDualHead96(resolve_path(config["paths"]["official_training"]))
    initialization = resolve_path(config["paths"]["rust_initialization_checkpoint"])
    init_checkpoint = load_encoder_from_rust_checkpoint_strict(model, initialization)
    model.to(device)
    stages = [1, 2, 3, 4] if args.stage == "all" else [int(args.stage)]
    start_stage, start_epoch = stages[0], 1
    resume_checkpoint: dict[str, Any] | None = None
    if args.resume:
        resume_checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if resume_checkpoint.get("config_sha256") != sha256_file(config_path) or resume_checkpoint.get("seed") != args.seed:
            raise ValueError("Resume config SHA or seed mismatch")
        model.load_state_dict(resume_checkpoint["model_state_dict"], strict=True)
        start_stage = int(resume_checkpoint["stage"])
        start_epoch = int(resume_checkpoint["epoch"]) + 1
        stages = [stage for stage in stages if stage >= start_stage]
    train_dataset = DemoManifestDataset(
        resolve_path(config["paths"]["train_manifest"]), resolve_path(config["paths"]["teacher_cache"]), True
    )
    validation_dataset = DemoManifestDataset(
        resolve_path(config["paths"]["validation_manifest"]), resolve_path(config["paths"]["teacher_cache"]), False
    )
    batch_size = args.batch_size or int(config["train"]["physical_batch_size"])
    workers = args.workers or int(config["train"]["workers"])
    accumulation = args.accumulation or int(config["train"]["gradient_accumulation"])
    if batch_size < 1 or accumulation < 1:
        raise ValueError("Batch size and gradient accumulation must be positive")
    loader_runtime = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False, **loader_runtime)
    output_root = resolve_path(config["paths"]["checkpoints"]) / f"seed{args.seed}"
    output_root.mkdir(parents=True, exist_ok=True)
    history_path = output_root / "history.jsonl"
    for stage in stages:
        set_stage(model, stage)
        groups = trainable_groups(model, stage, config)
        optimizer = torch.optim.AdamW(groups, weight_decay=float(config["train"]["weight_decay"]))
        if stage == 2:
            sampler = HalfNegativeBatchSampler([row["scenario"] for row in train_dataset.rows], batch_size, args.seed)
            train_loader = DataLoader(train_dataset, batch_sampler=sampler, **loader_runtime)
        else:
            generator = torch.Generator().manual_seed(args.seed + stage)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, generator=generator,
                drop_last=False, **loader_runtime
            )
            sampler = None
        scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]))
        best_loss = math.inf
        patience = 0
        if resume_checkpoint is not None and stage == start_stage:
            optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if isinstance(value, torch.Tensor):
                        state[key] = value.to(device)
            if "scaler_state_dict" in resume_checkpoint:
                scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
            best_loss, patience = selection_state_from_history(
                history_path, args.seed, stage, int(resume_checkpoint["epoch"])
            )
        first_epoch = start_epoch if stage == start_stage else 1
        for epoch in range(first_epoch, int(config["train"]["epochs_per_stage"]) + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            model.train()
            freeze_batchnorm_running_stats(model)
            optimizer.zero_grad(set_to_none=True)
            epoch_totals: dict[str, float] = {}
            steps = 0
            begin = time.perf_counter()
            for step, batch in enumerate(train_loader, 1):
                batch = move_batch(batch, device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=bool(config["train"]["amp"])):
                    output = model(batch["image"])
                    losses = compute_losses(output, batch, config)
                    objective = losses["rust_total"] if stage == 1 else losses["crack_total"] if stage == 2 else losses["total"]
                    objective = objective / accumulation
                if not finite_losses(losses):
                    raise FloatingPointError(f"Non-finite loss at stage={stage}, epoch={epoch}, step={step}")
                scaler.scale(objective).backward()
                if step % accumulation == 0 or step == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], float(config["train"]["gradient_clip_norm"])
                    )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                for key, value in losses.items():
                    epoch_totals[key] = epoch_totals.get(key, 0.0) + float(value.detach())
                steps += 1
            metrics = {f"train_{key}": value / steps for key, value in epoch_totals.items()}
            metrics.update(validate_loss(model, validation_loader, config, device, stage))
            record = {
                "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "stage": stage,
                "epoch": epoch, "epoch_seconds": time.perf_counter() - begin,
                "max_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20, **metrics,
                "runtime": {"batch_size": batch_size, "workers": workers, "accumulation": accumulation},
            }
            with history_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False))
            payload = {
                "model_name": config["architecture"]["name"], "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(), "seed": args.seed, "stage": stage, "epoch": epoch,
                "scaler_state_dict": scaler.state_dict(),
                "config_sha256": readiness["config_sha256"], "manifest_sha256": {
                    split: readiness["manifests"][split]["sha256"] for split in ("train", "validation")
                }, "rust_initialization_path": str(initialization), "rust_initialization_sha256": sha256_file(initialization),
                "metrics": metrics, "phase": config["status"].get("phase"),
                "runtime": {"batch_size": batch_size, "workers": workers, "accumulation": accumulation},
                "accuracy_status": "ACCURACY_NOT_FINAL", "deployment_status": "NOT_FOR_UART",
            }
            torch.save(payload, output_root / "last.pt")
            if metrics["selection_loss"] < best_loss:
                best_loss = metrics["selection_loss"]
                patience = 0
                torch.save(payload, output_root / f"stage{stage}_best.pt")
            else:
                patience += 1
            if patience >= int(config["train"]["early_stopping_patience"]):
                break
        # Automatic stage progression remains provisional until full offline gates are
        # calculated by evaluate.py; never label a loss-selected checkpoint final.
        start_epoch = 1
    write_json(output_root / "training_complete.json", {
        "seed": args.seed, "stages": stages, "status": "PROVISIONAL_REQUIRES_OFFLINE_GATE_EVALUATION",
        "accuracy_status": "ACCURACY_NOT_FINAL", "deployment_status": "NOT_FOR_UART",
        "actuator_authorization": "PROHIBITED",
    })


if __name__ == "__main__":
    main()
