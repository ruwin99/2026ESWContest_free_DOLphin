from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from light_dualhead_96 import LightDualHead96, load_encoder_from_rust_checkpoint_strict, set_stage, stage4_encoder_modules


ROOT = Path(__file__).resolve().parents[3]
OFFICIAL = ROOT / "models" / "virginia_tech_cssd" / "official_code" / "Training - Testing"
RUST_CHECKPOINT = ROOT / "outputs" / "training" / "vt_kd" / "kd-seed42" / "weights" / "best.pt"


def test_exact_96_channel_structure_and_encoder_import() -> None:
    model = LightDualHead96(OFFICIAL)
    load_encoder_from_rust_checkpoint_strict(model, RUST_CHECKPOINT)
    convolutions = [module for module in model.modules() if isinstance(module, nn.Conv2d)]
    assert not any(tuple(module.weight.shape) == (256, 304, 3, 3) for module in convolutions)
    shared_dw = model.shared_decoder[0]
    shared_pw = model.shared_decoder[3]
    assert (shared_dw.in_channels, shared_dw.out_channels, shared_dw.groups) == (128, 128, 128)
    assert (shared_pw.in_channels, shared_pw.out_channels) == (128, 96)
    assert (model.rust_head.in_channels, model.rust_head.out_channels) == (96, 4)
    assert model.crack_detail[-1].out_channels == 1


def test_forward_has_raw_five_channel_output() -> None:
    model = LightDualHead96(OFFICIAL).eval()
    with torch.inference_mode():
        output = model(torch.zeros((1, 3, 64, 128)))
    assert output.shape == (1, 5, 64, 128)
    assert torch.isfinite(output).all()


def test_stage4_unfreezes_exact_final_two_inverted_residual_stages_and_bn_stays_eval() -> None:
    model = LightDualHead96(OFFICIAL)
    set_stage(model, 4)
    selected = stage4_encoder_modules(model)
    assert [module.conv[-1].num_features for module in selected] == [160, 160, 160, 320]
    assert all(parameter.requires_grad for module in selected for parameter in module.parameters())
    assert all(not module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))
