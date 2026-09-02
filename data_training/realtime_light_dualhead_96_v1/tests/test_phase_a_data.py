from __future__ import annotations

from pathlib import Path

import torch

from common import read_rows, resolve_path
from dataset import load_canonical_rgb, load_mask


def test_phase_a_public_and_normal_mask_contracts() -> None:
    rows = read_rows(resolve_path("data_training/realtime_light_dualhead_96_v1/manifests/train.csv"))
    public = next(row for row in rows if row["source"] == "crackseg9k_v4_public")
    normal = next(row for row in rows if row["scenario"] == "clean")
    public_image, _ = load_canonical_rgb(resolve_path(public["relative_image_path"]))
    public_crack = load_mask(
        public["crack_mask_path"], encoding=public["crack_mask_encoding"], crack=True
    )
    normal_image, _ = load_canonical_rgb(resolve_path(normal["relative_image_path"]))
    normal_rust = load_mask(normal["rust_mask_path"])
    normal_crack = load_mask(normal["crack_mask_path"], crack=True)
    assert public_image.shape == normal_image.shape == (3, 240, 1280)
    assert torch.all(public_crack[:112] == 255)
    assert set(torch.unique(public_crack[112:]).tolist()) <= {0, 1}
    assert torch.all(normal_rust == 0)
    assert torch.all(normal_crack[:112] == 255)
    assert torch.all(normal_crack[112:] == 0)
