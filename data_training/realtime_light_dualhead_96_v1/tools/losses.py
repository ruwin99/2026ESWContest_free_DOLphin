from __future__ import annotations

import torch
from torch.nn import functional as F


def connected_zero(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().sum() * 0.0


def masked_rust_supervised(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if not bool(valid.any()):
        zero = connected_zero(logits)
        return zero, zero
    safe_target = target.masked_fill(~valid, 255)
    ce_map = F.cross_entropy(logits.float(), safe_target, ignore_index=255, reduction="none")
    ce = ce_map[valid].mean()
    probabilities = logits.float().softmax(dim=1)
    dice_terms: list[torch.Tensor] = []
    for class_index in range(4):
        expected = (target == class_index) & valid
        predicted = probabilities[:, class_index] * valid
        denominator = predicted.sum() + expected.float().sum()
        if float(denominator.detach()) > 0:
            dice_terms.append((2.0 * (predicted * expected).sum() + 1e-6) / (denominator + 1e-6))
    dice = 1.0 - torch.stack(dice_terms).mean() if dice_terms else connected_zero(logits)
    return ce, dice


def rust_kd(
    student: torch.Tensor,
    teacher: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 2.0,
) -> torch.Tensor:
    student_pixels = student.permute(0, 2, 3, 1)[valid]
    teacher_pixels = teacher.detach().permute(0, 2, 3, 1)[valid]
    if student_pixels.shape[0] == 0:
        return connected_zero(student)
    with torch.autocast(device_type=student.device.type, enabled=False):
        student_pixels = student_pixels.float()
        teacher_pixels = teacher_pixels.float()
        return F.kl_div(
            F.log_softmax(student_pixels / temperature, dim=1),
            F.softmax(teacher_pixels / temperature, dim=1),
            reduction="batchmean",
        ) * temperature**2


def binary_soft_dice(logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    if not bool(valid.any()):
        return connected_zero(logits)
    probability = logits.float().sigmoid()
    valid_float = valid.float()
    intersection = (probability * target * valid_float).sum()
    denominator = (probability * valid_float).sum() + (target * valid_float).sum()
    return 1.0 - (2.0 * intersection + 1e-6) / (denominator + 1e-6)


def boundary_band_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    radius: int = 2,
) -> torch.Tensor:
    """BCE restricted to a radius-r morphological GT boundary band."""
    if radius < 1:
        raise ValueError("Boundary radius must be positive")
    kernel = 2 * radius + 1
    # Ignore pixels must not leak into the morphology neighborhood.
    target_float = torch.where(valid, target.float(), torch.zeros_like(target, dtype=torch.float32))
    dilated = F.max_pool2d(target_float, kernel, stride=1, padding=radius)
    eroded = -F.max_pool2d(-target_float, kernel, stride=1, padding=radius)
    band = (dilated - eroded > 0) & valid
    if not bool(band.any()):
        return connected_zero(logits)
    values = F.binary_cross_entropy_with_logits(logits.float(), target_float, reduction="none")
    return values[band].mean()


def crack_supervised(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    pos_weight: float = 4.0,
    boundary_radius: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not bool(valid.any()):
        zero = connected_zero(logits)
        return zero, zero, zero
    values = F.binary_cross_entropy_with_logits(
        logits.float(), target.float(), reduction="none", pos_weight=torch.tensor(pos_weight, device=logits.device)
    )
    bce = values[valid].mean()
    return bce, binary_soft_dice(logits, target, valid), boundary_band_bce(
        logits, target, valid, boundary_radius
    )


def crack_kd(
    student: torch.Tensor,
    teacher_margin: torch.Tensor,
    valid: torch.Tensor,
    temperature: float = 2.0,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    if student.shape != teacher_margin.shape or valid.shape != student.shape:
        raise ValueError(
            f"Crack KD shape mismatch: student={student.shape}, teacher={teacher_margin.shape}, valid={valid.shape}"
        )
    if not bool(valid.any()):
        return connected_zero(student)
    with torch.autocast(device_type=student.device.type, enabled=False):
        s = student.float() / temperature
        t = teacher_margin.detach().float() / temperature
        q = torch.sigmoid(t).clamp(epsilon, 1.0 - epsilon)
        kl = q * (torch.log(q) - F.logsigmoid(s)) + (1.0 - q) * (
            torch.log1p(-q) - F.logsigmoid(-s)
        )
        valid_float = valid.float()
        return (kl * valid_float).sum() / valid_float.sum() * temperature**2
