from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    BalancedGroupBatchSampler,
    RustManifestDataset,
    atomic_torch_save,
    baseline_anchored_stage_a_loss,
    build_model,
    canonical_evaluation_backend_report,
    class_weights_from_rows,
    enforce_frozen_batchnorm,
    enforce_frozen_dropout,
    evaluate_models,
    freeze_for_stage,
    load_config,
    optimizer_for_stage,
    read_manifest,
    resolve_path,
    seed_everything,
    sha256_file,
    configured_validation_gate,
    verify_manifest_lock,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-wise hard-negative fine-tuning for the capture rust teacher.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("a", "b"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 43, 44), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke-batches", type=int, default=0)
    return parser.parse_args()


def provenance(config_path: Path, config: dict[str, Any], lock: dict[str, Any]) -> dict[str, Any]:
    checkpoint = resolve_path(config["paths"]["initial_checkpoint"])
    return {
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "initial_checkpoint": str(checkpoint),
        "initial_checkpoint_sha256": sha256_file(checkpoint),
        "manifest_lock": lock,
    }


def assert_resume_provenance(payload: dict[str, Any], current: dict[str, Any]) -> None:
    previous = payload.get("provenance")
    if not isinstance(previous, dict):
        raise RuntimeError("Resume checkpoint lacks provenance")
    for key in ("config_sha256", "initial_checkpoint_sha256"):
        if previous.get(key) != current.get(key):
            raise RuntimeError(f"Resume provenance mismatch: {key}")
    old_files = previous.get("manifest_lock", {}).get("files")
    new_files = current.get("manifest_lock", {}).get("files")
    if old_files != new_files:
        raise RuntimeError("Resume manifest lock mismatch")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full capture teacher training")
    config_path = args.config.resolve()
    config = load_config(config_path)
    if args.stage == "b" and not bool(config.get("stage_b", {}).get("enabled", True)):
        raise RuntimeError("Stage B is disabled for this training profile")
    lock = verify_manifest_lock(config)
    seed_everything(args.seed)
    stage_config = config[f"stage_{args.stage}"]
    train_rows = read_manifest(resolve_path(config["paths"]["train_manifest"]), expected_split="train")
    validation_rows = read_manifest(resolve_path(config["paths"]["validation_manifest"]), expected_split="validation")
    batch_size = int(stage_config["physical_batch_size"])
    accumulate = int(stage_config["accumulation_steps"])
    workers = int(stage_config["workers"])
    sampler = BalancedGroupBatchSampler(train_rows, batch_size, args.seed)
    train_loader = DataLoader(
        RustManifestDataset(train_rows, config, training=True),
        batch_sampler=sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        RustManifestDataset(validation_rows, config, training=False),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    class_weights, class_counts = class_weights_from_rows(train_rows, config)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    initial = resolve_path(config["paths"]["initial_checkpoint"])
    model = build_model(config, initial)
    baseline = build_model(config, initial).to(device).eval()
    freeze_report = freeze_for_stage(model, args.stage, config)
    model.to(device)
    optimizer = optimizer_for_stage(model, config, args.stage)
    current_provenance = provenance(config_path, config, lock)

    run_dir = resolve_path(config["paths"]["runs_root"]) / args.run_name
    weights_dir = run_dir / "weights"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(f"Run directory exists; use its last.pt only for same-stage resume: {run_dir}")
    weights_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_score = math.inf
    transfer_from: dict[str, Any] | None = None
    if args.resume:
        resume_path = args.resume.resolve()
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise ValueError("Resume checkpoint is not a capture rust hard-negative checkpoint")
        assert_resume_provenance(payload, current_provenance)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        previous_stage = payload.get("stage")
        if previous_stage == args.stage:
            optimizer.load_state_dict(payload["optimizer_state_dict"])
            start_epoch = int(payload["epoch"]) + 1
            best_score = float(payload.get("best_score", math.inf))
        elif args.stage == "b" and previous_stage == "a":
            if not payload.get("validation_gate", {}).get("passed", False):
                raise RuntimeError("Stage B requires a Stage A checkpoint that passed the validation gate")
            transfer_from = {"path": str(resume_path), "sha256": sha256_file(resume_path), "stage": "a", "epoch": payload.get("epoch")}
            start_epoch = 1
            best_score = math.inf
        else:
            raise RuntimeError(f"Invalid stage transition: {previous_stage} -> {args.stage}")
        freeze_report = freeze_for_stage(model, args.stage, config)

    run_config = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "project": config["project_name"],
        "stage": args.stage,
        "seed": args.seed,
        "epochs": int(stage_config["epochs"]),
        "physical_batch_size": batch_size,
        "accumulation_steps": accumulate,
        "effective_batch_size": batch_size * accumulate,
        "workers": workers,
        "precision": "fp32",
        "evaluation_backend": canonical_evaluation_backend_report(),
        "class_pixel_counts_train_only": class_counts,
        "class_weights_train_only": class_weights.tolist(),
        "freeze_audit": freeze_report,
        "provenance": current_provenance,
        "transfer_from": transfer_from,
        "status": {"candidate": True, "uart_status": "NOT_FOR_UART", "deployment_status": "NOT_DEPLOYED"},
    }
    write_json(run_dir / "run_config.json", run_config)
    epochs = int(stage_config["epochs"])
    class_weights = class_weights.to(device)

    for epoch in range(start_epoch, epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        enforce_frozen_batchnorm(model)
        if args.stage == "a" and bool(stage_config.get("freeze_dropout", False)):
            enforce_frozen_dropout(model)
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        loss_sum = 0.0
        component_sums: dict[str, float] = {}
        batches = 0
        for batch_index, batch in enumerate(train_loader, start=1):
            images = batch["image"].to(device, non_blocking=True)
            target = batch["target"].to(device, non_blocking=True)
            logits = model(images)
            if not torch.isfinite(logits).all():
                raise FloatingPointError(f"Non-finite logits at epoch={epoch}, batch={batch_index}")
            if args.stage == "a" and config["loss"]["name"] == "baseline_anchored_hard_negative_ce":
                with torch.inference_mode():
                    baseline_logits = baseline(images)
                loss, components = baseline_anchored_stage_a_loss(
                    logits,
                    baseline_logits,
                    target,
                    batch["source_type"],
                    class_weights,
                    config,
                )
            else:
                loss = F.cross_entropy(logits, target, weight=class_weights, ignore_index=int(config["data"]["ignore_index"]))
                components = {"cross_entropy": loss}
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={batch_index}")
            (loss / accumulate).backward()
            if batch_index % accumulate == 0 or batch_index == len(train_loader):
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_sum += float(loss.detach())
            for name, value in components.items():
                component_sums[name] = component_sums.get(name, 0.0) + float(value.detach())
            batches += 1
            if args.smoke_batches and batches >= args.smoke_batches:
                break

        if args.smoke_batches:
            validation = {"skipped_for_smoke": True}
            gate = {"passed": False, "reasons": ["validation skipped for smoke"], "selection_score": None}
        else:
            validation = evaluate_models(model, baseline, validation_loader, device)
            gate = configured_validation_gate(validation, config)
        score = gate["selection_score"]
        selection_mode = config.get("selection", {}).get("mode", "mixed_positive_and_hard_negative")
        if selection_mode == "positive_safety_fixed_epoch":
            fixed_epoch = int(config["selection"]["fixed_checkpoint_epoch"])
            improved = bool(gate["passed"] and epoch == fixed_epoch)
            score = float(epoch) if improved else None
        else:
            improved = bool(gate["passed"] and score is not None and float(score) < best_score)
        if improved:
            best_score = float(score)
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(1, batches),
            "train_loss_components": {name: value / max(1, batches) for name, value in component_sums.items()},
            "validation_gate_passed": gate["passed"],
            "validation_selection_score": score,
            "epoch_seconds": time.perf_counter() - started,
            "max_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
        }
        append_jsonl(run_dir / "metrics.jsonl", row)
        validation_report = {"epoch": epoch, "metrics": validation, "gate": gate}
        write_json(run_dir / "validation_latest.json", validation_report)
        write_json(run_dir / f"validation_epoch_{epoch:03d}.json", validation_report)
        print(json.dumps(row, ensure_ascii=False))
        checkpoint = {
            "format_version": 1,
            "architecture": "deeplabv3plus_resnet101_os8",
            "stage": args.stage,
            "seed": args.seed,
            "epoch": epoch,
            "best_score": best_score,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation": validation,
            "validation_gate": gate,
            "selection_mode": selection_mode,
            "evaluation_backend": canonical_evaluation_backend_report(),
            "freeze_audit": freeze_report,
            "provenance": current_provenance,
            "status": {"candidate": True, "uart_status": "NOT_FOR_UART", "deployment_status": "NOT_DEPLOYED"},
        }
        atomic_torch_save(checkpoint, weights_dir / "last.pt")
        if improved:
            best = dict(checkpoint)
            best.pop("optimizer_state_dict")
            atomic_torch_save(best, weights_dir / "best.pt")
        if args.smoke_batches:
            break


if __name__ == "__main__":
    main()
