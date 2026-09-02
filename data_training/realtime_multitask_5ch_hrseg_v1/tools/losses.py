from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def multiclass_soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, classes: int = 4
) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target.long(), num_classes=classes).permute(0, 3, 1, 2).float()
    numerator = 2.0 * (probabilities * one_hot).sum(dim=(0, 2, 3))
    denominator = (probabilities + one_hot).sum(dim=(0, 2, 3))
    dice = (numerator + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def binary_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probability = logits.sigmoid()
    numerator = 2.0 * (probability * target).sum(dim=(1, 2, 3))
    denominator = (probability + target).sum(dim=(1, 2, 3))
    return 1.0 - ((numerator + 1.0) / (denominator + 1.0)).mean()


def rust_supervised_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_valid: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = sample_valid.bool()
    if not selected.any():
        zero = _zero(logits)
        return zero, {"rust_ce": zero, "rust_dice": zero}
    selected_logits = logits[selected]
    selected_target = target[selected]
    weights = torch.as_tensor(
        config["rust_class_weights"], dtype=logits.dtype, device=logits.device
    )
    ce = F.cross_entropy(selected_logits, selected_target.long(), weight=weights)
    dice = multiclass_soft_dice_loss(selected_logits, selected_target)
    total = float(config["rust_ce_weight"]) * ce + float(config["rust_dice_weight"]) * dice
    return total, {"rust_ce": ce, "rust_dice": dice}


def crack_supervised_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    sample_valid: torch.Tensor,
    valid_rows: tuple[int, int],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    selected = sample_valid.bool()
    if not selected.any():
        zero = _zero(logits)
        return zero, {"crack_bce": zero, "crack_dice": zero, "crack_boundary": zero}
    start, end = valid_rows
    selected_logits = logits[selected, :, start:end, :]
    selected_target = target[selected, :, start:end, :].float()
    pos_weight = torch.as_tensor(
        [float(config["crack_pos_weight"])], dtype=logits.dtype, device=logits.device
    )
    bce = F.binary_cross_entropy_with_logits(
        selected_logits, selected_target, pos_weight=pos_weight
    )
    dice = binary_soft_dice_loss(selected_logits, selected_target)
    boundary = _zero(logits)
    if config["boundary_loss"] == "sobel_l1":
        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            device=logits.device,
            dtype=logits.dtype,
        ).view(1, 1, 3, 3)
        kernel_y = kernel_x.transpose(-1, -2)
        probability = selected_logits.sigmoid()
        predicted_edge = torch.sqrt(
            F.conv2d(probability, kernel_x, padding=1).square()
            + F.conv2d(probability, kernel_y, padding=1).square()
            + 1e-6
        )
        target_edge = torch.sqrt(
            F.conv2d(selected_target, kernel_x, padding=1).square()
            + F.conv2d(selected_target, kernel_y, padding=1).square()
            + 1e-6
        )
        boundary = F.l1_loss(predicted_edge, target_edge)
    elif config["boundary_loss"] != "none":
        raise ValueError("loss.boundary_loss must be 'none' or 'sobel_l1'")
    total = (
        float(config["crack_bce_weight"]) * bce
        + float(config["crack_dice_weight"]) * dice
        + (0.25 * boundary if config["boundary_loss"] == "sobel_l1" else 0.0)
    )
    return total, {"crack_bce": bce, "crack_dice": dice, "crack_boundary": boundary}


def rust_kd_loss(
    student: torch.Tensor, teacher: torch.Tensor, available: torch.Tensor, temperature: float
) -> torch.Tensor:
    selected = available.bool()
    if not selected.any():
        return _zero(student)
    t = float(temperature)
    return (
        F.kl_div(
            F.log_softmax(student[selected] / t, dim=1),
            F.softmax(teacher[selected] / t, dim=1),
            reduction="batchmean",
        )
        * (t * t)
        / float(student.shape[-2] * student.shape[-1])
    )


def crack_kd_loss(
    student: torch.Tensor,
    teacher_margin: torch.Tensor,
    available: torch.Tensor,
    valid_rows: tuple[int, int],
    temperature: float,
) -> torch.Tensor:
    selected = available.bool()
    if not selected.any():
        return _zero(student)
    start, end = valid_rows
    t = float(temperature)
    student_roi = student[selected, :, start:end, :] / t
    teacher_roi = teacher_margin[selected] / t
    target_probability = teacher_roi.sigmoid()
    return F.binary_cross_entropy_with_logits(student_roi, target_probability) * (t * t)
