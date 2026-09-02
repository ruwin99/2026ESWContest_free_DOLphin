from __future__ import annotations

import torch

from losses import boundary_band_bce, crack_kd, crack_supervised, rust_kd


def test_empty_masks_return_finite_graph_connected_zero() -> None:
    student_rust = torch.randn(1, 4, 4, 5, requires_grad=True)
    teacher_rust = torch.randn_like(student_rust)
    empty_rust = torch.zeros((1, 4, 5), dtype=torch.bool)
    first = rust_kd(student_rust, teacher_rust, empty_rust)
    student_crack = torch.randn(1, 1, 4, 5, requires_grad=True)
    empty_crack = torch.zeros_like(student_crack, dtype=torch.bool)
    second = crack_kd(student_crack, torch.randn_like(student_crack), empty_crack)
    total = first + second
    total.backward()
    assert torch.isfinite(total)
    assert student_rust.grad is not None and student_crack.grad is not None


def test_crack_loss_ignores_rows_above_112() -> None:
    first = torch.zeros((1, 1, 240, 16), requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    second.data[:, :, :112] = 100.0
    target = torch.zeros_like(first)
    valid = torch.zeros_like(first, dtype=torch.bool)
    valid[:, :, 112:240] = True
    losses_a = crack_supervised(first, target, valid)
    losses_b = crack_supervised(second, target, valid)
    for left, right in zip(losses_a, losses_b, strict=True):
        torch.testing.assert_close(left, right)


def test_boundary_loss_is_zero_for_empty_negative_mask() -> None:
    logits = torch.zeros((1, 1, 16, 16), requires_grad=True)
    target = torch.zeros_like(logits)
    valid = torch.ones_like(logits, dtype=torch.bool)
    loss = boundary_band_bce(logits, target, valid, radius=2)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None
