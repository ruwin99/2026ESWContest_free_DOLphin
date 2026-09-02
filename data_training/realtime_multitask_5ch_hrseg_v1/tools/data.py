from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from common import cache_key, crop_native_rgb, parse_bool, resolve_path, rust_mask_to_indices


def _crop_mask(path: Path, crop: tuple[int, int, int, int], mode: str) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image.convert(mode))
    x0, y0, x1, y1 = crop
    crop_size = (y1 - y0, x1 - x0)
    if array.shape[:2] == crop_size:
        return np.ascontiguousarray(array)
    return np.ascontiguousarray(array[y0:y1, x0:x1])


class MultitaskDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        rows: list[dict[str, str]],
        config: dict[str, Any],
        *,
        training: bool,
        use_cache: bool,
    ) -> None:
        self.rows = rows
        self.training = training
        self.flip_probability = float(config["train"]["horizontal_flip_probability"])
        self.mean = np.asarray(config["contracts"]["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(config["contracts"]["std"], dtype=np.float32).reshape(1, 1, 3)
        self.cache_root = resolve_path(config["paths"]["teacher_cache"])
        self.use_cache = use_cache

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        crop = tuple(
            int(row[name]) for name in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")
        )
        rgb = crop_native_rgb(resolve_path(row["image_path"]), crop)
        normalized = (rgb.astype(np.float32) / 255.0 - self.mean) / self.std
        image = torch.from_numpy(normalized.transpose(2, 0, 1).copy())

        rust_valid = parse_bool(row["rust_valid"], field="rust_valid")
        crack_valid = parse_bool(row["crack_valid"], field="crack_valid")
        rust_target = np.zeros(rgb.shape[:2], dtype=np.int64)
        crack_target = np.zeros((1, *rgb.shape[:2]), dtype=np.float32)
        if rust_valid:
            rust_raw = _crop_mask(resolve_path(row["rust_mask_path"]), crop, "RGB")
            rust_target = rust_mask_to_indices(rust_raw)
        if crack_valid:
            crack_raw = _crop_mask(resolve_path(row["crack_mask_path"]), crop, "L")
            crack_target[0] = (crack_raw > 0).astype(np.float32)

        rust_teacher = np.zeros((4, *rgb.shape[:2]), dtype=np.float32)
        crack_teacher = np.zeros((1, 128, rgb.shape[1]), dtype=np.float32)
        rust_teacher_available = False
        crack_teacher_available = False
        if self.use_cache:
            path = self.cache_root / f"{cache_key(row['sample_id'])}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Teacher cache missing for {row['sample_id']}: {path}")
            with np.load(path, allow_pickle=False) as cached:
                if "rust_logits" in cached:
                    rust_teacher = cached["rust_logits"].astype(np.float32, copy=True)
                    rust_teacher_available = True
                if "crack_margin" in cached:
                    crack_teacher = cached["crack_margin"].astype(np.float32, copy=True)
                    crack_teacher_available = True

        if self.training and torch.rand(()) < self.flip_probability:
            image = image.flip(-1)
            rust_target = np.flip(rust_target, axis=-1).copy()
            crack_target = np.flip(crack_target, axis=-1).copy()
            rust_teacher = np.flip(rust_teacher, axis=-1).copy()
            crack_teacher = np.flip(crack_teacher, axis=-1).copy()

        return {
            "sample_id": row["sample_id"],
            "image": image,
            "rust_target": torch.from_numpy(rust_target),
            "crack_target": torch.from_numpy(crack_target),
            "rust_valid": torch.tensor(rust_valid),
            "crack_valid": torch.tensor(crack_valid),
            "rust_teacher": torch.from_numpy(rust_teacher),
            "crack_teacher": torch.from_numpy(crack_teacher),
            "rust_teacher_available": torch.tensor(rust_teacher_available),
            "crack_teacher_available": torch.tensor(crack_teacher_available),
        }
