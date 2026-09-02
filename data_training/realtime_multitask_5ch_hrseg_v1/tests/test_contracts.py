from __future__ import annotations

import sys
from pathlib import Path

import torch


WORK = Path(__file__).resolve().parents[1]
TOOLS = WORK / "tools"
sys.path.insert(0, str(TOOLS))

from common import load_config, resolve_path  # noqa: E402
from losses import binary_soft_dice_loss, crack_supervised_loss  # noqa: E402
from model import build_model, load_rust_checkpoint_strict  # noqa: E402


CONFIG_PATH = WORK / "configs" / "mnv2_os8_dualhead_w1280_h240.yaml"


def test_crack_rows_above_112_are_ignored() -> None:
    config = load_config(CONFIG_PATH)["loss"]
    config = {**config, "crack_pos_weight": 2.0, "boundary_loss": "none"}
    target = torch.zeros((1, 1, 240, 16))
    target[:, :, 120:124] = 1.0
    first = torch.zeros_like(target, requires_grad=True)
    second = first.detach().clone()
    second[:, :, :112] = 100.0
    second.requires_grad_(True)
    loss_a, _ = crack_supervised_loss(first, target, torch.tensor([True]), (112, 240), config)
    loss_b, _ = crack_supervised_loss(second, target, torch.tensor([True]), (112, 240), config)
    assert torch.equal(loss_a, loss_b)
    loss_b.backward()
    assert torch.count_nonzero(second.grad[:, :, :112]) == 0


def test_empty_binary_dice_is_finite() -> None:
    logits = torch.zeros((2, 1, 32, 32), requires_grad=True)
    target = torch.zeros_like(logits)
    loss = binary_soft_dice_loss(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_strict_rust_initialization_and_output_contract() -> None:
    config = load_config(CONFIG_PATH)
    model = build_model(config)
    checkpoint = load_rust_checkpoint_strict(model, resolve_path(config["paths"]["rust_checkpoint"]))
    assert checkpoint["model_name"] == "deeplabv3plus_mobilenet"
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 3, 32, 64)))
    assert output.shape == (1, 5, 32, 64)
    assert torch.isfinite(output).all()


def test_model_contains_no_resize_or_padding_modules() -> None:
    model = build_model(load_config(CONFIG_PATH))
    forbidden = (torch.nn.Upsample, torch.nn.ZeroPad2d, torch.nn.ReflectionPad2d)
    assert not any(isinstance(module, forbidden) for module in model.modules())
