from __future__ import annotations

from typing import Any

import torch

from losses import crack_kd, crack_supervised, rust_kd, masked_rust_supervised


def mode_valid_mask(modes: list[str], target: torch.Tensor, roi: slice | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    selected = target if roi is None else target[:, roi, :]
    supervised = torch.zeros_like(selected, dtype=torch.bool)
    kd = torch.zeros_like(selected, dtype=torch.bool)
    for index, mode in enumerate(modes):
        if mode in {"gt", "partial"}:
            valid = selected[index] != 255
            supervised[index] = valid
            kd[index] = valid
            valid_count = int(valid.sum())
            if mode == "gt" and valid_count != valid.numel():
                raise AssertionError(f"gt mode has ignore pixels at batch index {index}")
            if mode == "partial" and (valid_count == 0 or valid_count == valid.numel()):
                raise AssertionError(f"partial mode must contain both valid and ignore pixels at batch index {index}")
        elif mode == "teacher_only":
            kd[index] = True
        elif mode != "unlabeled":
            raise ValueError(f"Unknown label mode: {mode}")
    return supervised, kd


def compute_losses(output: torch.Tensor, batch: dict[str, Any], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    rust_logits = output[:, :4]
    crack_logits = output[:, 4:5, 112:240]
    rust_target = batch["rust_gt"]
    crack_target = batch["crack_gt"][:, None, 112:240]
    rust_supervised_valid, rust_kd_valid = mode_valid_mask(batch["rust_label_mode"], rust_target)
    crack_supervised_valid, crack_kd_valid = mode_valid_mask(
        batch["crack_label_mode"], batch["crack_gt"], slice(112, 240)
    )
    crack_supervised_valid = crack_supervised_valid[:, None]
    crack_kd_valid = crack_kd_valid[:, None]
    rust_ce, rust_dice = masked_rust_supervised(rust_logits, rust_target, rust_supervised_valid)
    rust_distill = rust_kd(
        rust_logits, batch["rust_teacher"], rust_kd_valid, float(config["loss"]["rust_temperature"])
    )
    crack_bce, crack_dice, crack_boundary = crack_supervised(
        crack_logits,
        crack_target,
        crack_supervised_valid,
        float(config["loss"]["crack_pos_weight"]),
        int(config["loss"]["crack_boundary_radius"]),
    )
    crack_distill = crack_kd(
        crack_logits,
        batch["crack_teacher"],
        crack_kd_valid,
        float(config["loss"]["crack_temperature"]),
        float(config["loss"]["crack_kd_epsilon"]),
    )
    rust_total = (
        float(config["loss"]["rust_ce_weight"]) * rust_ce
        + float(config["loss"]["rust_dice_weight"]) * rust_dice
        + float(config["loss"]["rust_kd_weight"]) * rust_distill
    )
    crack_total = (
        float(config["loss"]["crack_bce_weight"]) * crack_bce
        + float(config["loss"]["crack_dice_weight"]) * crack_dice
        + float(config["loss"]["crack_boundary_weight"]) * crack_boundary
        + float(config["loss"]["crack_kd_weight"]) * crack_distill
    )
    return {
        "rust_ce": rust_ce,
        "rust_dice": rust_dice,
        "rust_kd": rust_distill,
        "rust_total": rust_total,
        "crack_bce": crack_bce,
        "crack_dice": crack_dice,
        "crack_boundary": crack_boundary,
        "crack_kd": crack_distill,
        "crack_total": crack_total,
        "total": rust_total + crack_total,
    }


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def finite_losses(losses: dict[str, torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in losses.values())
