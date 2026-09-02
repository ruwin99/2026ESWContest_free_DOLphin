from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    BalancedGroupBatchSampler,
    RustManifestDataset,
    baseline_anchored_stage_a_loss,
    build_model,
    enforce_frozen_batchnorm,
    enforce_frozen_dropout,
    freeze_for_stage,
    load_config,
    optimizer_for_stage,
    read_manifest,
    resolve_path,
    seed_everything,
    verify_manifest_lock,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke tests for strict load, freezing and one real batch.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("structure", "data"), required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def bn_snapshot(model: nn.Module) -> dict[str, tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]]:
    return {
        name: (
            None if module.running_mean is None else module.running_mean.detach().cpu().clone(),
            None if module.running_var is None else module.running_var.detach().cpu().clone(),
            None if module.num_batches_tracked is None else module.num_batches_tracked.detach().cpu().clone(),
        )
        for name, module in model.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    }


def equal_bn(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if left.keys() != right.keys():
        return False
    for name in left:
        for a, b in zip(left[name], right[name], strict=True):
            if (a is None) != (b is None) or (a is not None and not torch.equal(a, b)):
                return False
    return True


def one_step(
    config: dict[str, Any],
    stage: str,
    images: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    source_types: list[str] | None = None,
) -> dict[str, Any]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    initial = resolve_path(config["paths"]["initial_checkpoint"])
    model = build_model(config, initial).to(device)
    freeze_report = freeze_for_stage(model, stage, config)
    optimizer = optimizer_for_stage(model, config, stage)
    frozen_before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad}
    trainable_before = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
    bn_before = bn_snapshot(model)
    model.train()
    enforce_frozen_batchnorm(model)
    if stage == "a" and bool(config["stage_a"].get("freeze_dropout", False)):
        enforce_frozen_dropout(model)
    dropout_frozen = all(
        not module.training
        for module in model.modules()
        if isinstance(module, nn.modules.dropout._DropoutNd)
    ) if stage == "a" and bool(config["stage_a"].get("freeze_dropout", False)) else True
    logits = model(images.to(device))
    target_device = target.to(device)
    if stage == "a" and config["loss"]["name"] == "baseline_anchored_hard_negative_ce":
        if source_types is None:
            source_types = ["positive_replay" if item % 2 == 0 else "hard_negative" for item in range(images.shape[0])]
        baseline = build_model(config, initial).to(device).eval()
        with torch.inference_mode():
            baseline_logits = baseline(images.to(device))
        loss, loss_components = baseline_anchored_stage_a_loss(
            logits,
            baseline_logits,
            target_device,
            source_types,
            torch.ones(4, dtype=torch.float32, device=device),
            config,
        )
    else:
        loss = F.cross_entropy(logits, target_device, ignore_index=255)
        loss_components = {"cross_entropy": loss}
    if not torch.isfinite(loss) or not torch.isfinite(logits).all():
        raise FloatingPointError("Smoke forward/loss is non-finite")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    frozen_unchanged = all(torch.equal(before, dict(model.named_parameters())[name].detach().cpu()) for name, before in frozen_before.items())
    trainable_changed = [name for name, before in trainable_before.items() if not torch.equal(before, dict(model.named_parameters())[name].detach().cpu())]
    bn_unchanged = equal_bn(bn_before, bn_snapshot(model))
    passed = frozen_unchanged and bool(trainable_changed) and bn_unchanged and dropout_frozen
    return {
        "stage": stage,
        "passed": passed,
        "loss": float(loss.detach()),
        "loss_components": {name: float(value.detach()) for name, value in loss_components.items()},
        "logits_shape": list(logits.shape),
        "finite_logits": bool(torch.isfinite(logits).all()),
        "frozen_parameters_unchanged": frozen_unchanged,
        "trainable_parameter_tensors_changed": trainable_changed,
        "batchnorm_running_stats_unchanged": bn_unchanged,
        "dropout_frozen_eval": dropout_frozen,
        "max_cuda_memory_mib": (
            float(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0.0
        ),
        "freeze_audit": freeze_report,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.mode == "structure":
        height, width = 96, 128
        images = torch.linspace(0, 255, steps=2 * 3 * height * width).reshape(2, 3, height, width)
        target = torch.arange(height * width).reshape(1, height, width).remainder(4).repeat(2, 1, 1)
        reports = [one_step(config, stage, images, target, device) for stage in ("a", "b")]
    else:
        verify_manifest_lock(config)
        rows = read_manifest(resolve_path(config["paths"]["train_manifest"]), expected_split="train")
        batch_size = int(config["stage_a"]["physical_batch_size"])
        sampler = BalancedGroupBatchSampler(rows, batch_size, args.seed)
        loader = DataLoader(RustManifestDataset(rows, config, training=True), batch_sampler=sampler, num_workers=0)
        batch = next(iter(loader))
        source_types = sorted(set(batch["source_type"]))
        if source_types != ["hard_negative", "positive_replay"]:
            raise RuntimeError(f"Smoke batch is not 1:1 positive/hard-negative: {source_types}")
        reports = [one_step(config, "a", batch["image"], batch["target"], device, list(batch["source_type"]))]
        reports[0]["source_types"] = list(batch["source_type"])
        reports[0]["sample_ids"] = list(batch["sample_id"])
        reports[0]["physical_batch_size"] = batch_size
    report = {
        "mode": args.mode,
        "device": str(device),
        "passed": all(item["passed"] for item in reports),
        "reports": reports,
        "status": {"candidate": True, "uart_status": "NOT_FOR_UART", "deployment_status": "NOT_DEPLOYED"},
    }
    output = resolve_path(f"data_training/capture_rust_teacher_hardneg_v1/reports/smoke_{args.mode}.json")
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
