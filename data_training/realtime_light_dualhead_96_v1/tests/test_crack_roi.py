from __future__ import annotations

import torch

from losses import crack_kd


def test_crack_kd_roi_shape_is_exact_and_finite() -> None:
    student = torch.randn((2, 1, 128, 1280), requires_grad=True)
    teacher = torch.randn_like(student)
    valid = torch.ones_like(student, dtype=torch.bool)
    value = crack_kd(student, teacher, valid, temperature=2.0)
    value.backward()
    assert torch.isfinite(value)
    assert student.grad is not None and torch.isfinite(student.grad).all()
