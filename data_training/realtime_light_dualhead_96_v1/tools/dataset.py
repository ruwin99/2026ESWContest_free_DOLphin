from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import read_rows, resolve_path


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[:, None, None]


def load_canonical_rgb(path: Path) -> tuple[torch.Tensor, np.ndarray]:
    source = np.asarray(Image.open(path).convert("RGB"))
    if source.shape not in {(720, 1280, 3), (240, 1280, 3)}:
        raise ValueError(f"Expected 1280x720 camera or 1280x240 Phase-A panel, got {source.shape}: {path}")
    canonical = np.ascontiguousarray(source[:240, :, :])
    tensor = torch.from_numpy(canonical.copy()).permute(2, 0, 1).float().div_(255.0)
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor, canonical


def load_mask(path_value: str, default: int = 255, *, encoding: str = "indexed_0_1_ignore255", crack: bool = False) -> torch.Tensor:
    if not path_value:
        return torch.full((240, 1280), default, dtype=torch.long)
    path = resolve_path(path_value)
    mask = np.asarray(Image.open(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    if mask.shape == (720, 1280):
        mask = mask[:240]
    if mask.shape != (240, 1280):
        raise ValueError(f"Mask must resolve to [240,1280], got {mask.shape}: {path}")
    if encoding == "binary_0_255_positive":
        mask = (mask > 0).astype(np.uint8)
        if crack:
            mask[:112] = 255
    elif encoding != "indexed_0_1_ignore255":
        raise ValueError(f"Unsupported mask encoding: {encoding}")
    return torch.from_numpy(np.ascontiguousarray(mask).copy()).long()


class DemoManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: Path, cache_root: Path, augment: bool = False) -> None:
        self.rows = read_rows(manifest)
        self.cache_root = cache_root
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image, _ = load_canonical_rgb(resolve_path(row["relative_image_path"]))
        encoding = row.get("crack_mask_encoding", "indexed_0_1_ignore255")
        rust = load_mask(row["rust_mask_path"])
        crack = load_mask(row["crack_mask_path"], encoding=encoding, crack=True)
        if row["rust_label_mode"] in {"gt", "partial"}:
            values = set(torch.unique(rust).tolist())
            if not values <= {0, 1, 2, 3, 255}:
                raise ValueError(f"Invalid rust labels {values}: {row['sample_id']}")
        if row["crack_label_mode"] in {"gt", "partial"}:
            values = set(torch.unique(crack).tolist())
            if not values <= {0, 1, 255}:
                raise ValueError(f"Invalid crack labels {values}: {row['sample_id']}")
            if torch.any(crack[:112] != 255):
                raise ValueError(f"Crack mask rows 0:112 must be ignore=255: {row['sample_id']}")
        cache_path = self.cache_root / row["split"] / f"{row['sample_id']}.npz"
        if not cache_path.is_file():
            raise FileNotFoundError(f"Teacher cache missing: {cache_path}")
        with np.load(cache_path, allow_pickle=False) as cached:
            rust_teacher = torch.from_numpy(cached["rust_teacher_logits"].copy()).float()
            crack_teacher = torch.from_numpy(cached["hrseg_logit_margin"].copy()).float()
        if rust_teacher.shape != (4, 240, 1280):
            raise ValueError(f"Bad rust teacher shape: {cache_path}")
        if crack_teacher.shape != (1, 128, 1280):
            raise ValueError(f"Bad crack teacher shape: {cache_path}")
        if not torch.isfinite(rust_teacher).all() or not torch.isfinite(crack_teacher).all():
            raise ValueError(f"Non-finite teacher cache: {cache_path}")
        if self.augment and torch.rand(()) < 0.5:
            image = image.flip(-1)
            rust = rust.flip(-1)
            crack = crack.flip(-1)
            rust_teacher = rust_teacher.flip(-1)
            crack_teacher = crack_teacher.flip(-1)
        return {
            "sample_id": row["sample_id"],
            "image": image,
            "rust_gt": rust,
            "crack_gt": crack,
            "rust_teacher": rust_teacher,
            "crack_teacher": crack_teacher,
            "rust_label_mode": row["rust_label_mode"],
            "crack_label_mode": row["crack_label_mode"],
            "scenario": row["scenario"],
            "group_id": row["group_id"],
        }
