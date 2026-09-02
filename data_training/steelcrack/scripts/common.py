from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
TRAINING_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = TRAINING_ROOT / "data" / "Steelcrack"
DEFAULT_OFFICIAL_CODE = TRAINING_ROOT / "official_bgcrack"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "outputs" / "training" / "steelcrack"
EXPECTED_IMAGE_SIZE = (512, 512)
EXPECTED_COUNTS = {"Train": 3300, "Validation": 525, "Test": 530}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_cuda_for_speed() -> None:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


class DeviceSafeMapToGrad(nn.Module):
    """Device-safe equivalent of the official Map_2_Grad implementation."""

    def __init__(self) -> None:
        super().__init__()
        fx = torch.tensor(
            [[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]]
        ).unsqueeze(0)
        fy = torch.tensor(
            [[[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]]]
        ).unsqueeze(0)
        self.register_buffer("fx", fx, persistent=False)
        self.register_buffer("fy", fy, persistent=False)

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        probability = prediction.sigmoid()
        probability = F.pad(probability, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(probability, self.fx)
        grad_y = F.conv2d(probability, self.fy)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


class LabelToGrad(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        fx = torch.tensor(
            [[[-3.0, 0.0, 3.0], [-10.0, 0.0, 10.0], [-3.0, 0.0, 3.0]]]
        ).unsqueeze(0)
        fy = torch.tensor(
            [[[-3.0, -10.0, -3.0], [0.0, 0.0, 0.0], [3.0, 10.0, 3.0]]]
        ).unsqueeze(0)
        self.register_buffer("fx", fx, persistent=False)
        self.register_buffer("fy", fy, persistent=False)

    def forward(self, label: torch.Tensor) -> torch.Tensor:
        binary = label.gt(0.5).float()
        binary = F.pad(binary, (1, 1, 1, 1), mode="replicate")
        grad_x = F.conv2d(binary, self.fx)
        grad_y = F.conv2d(binary, self.fy)
        return torch.sqrt(grad_x.square() + grad_y.square() + 1e-6)


def build_bgcrack(official_code: Path = DEFAULT_OFFICIAL_CODE) -> nn.Module:
    official_code = official_code.resolve()
    if not (official_code / "Model" / "BGCrack.py").is_file():
        raise FileNotFoundError(f"Official BGCrack code not found: {official_code}")
    code_text = str(official_code)
    if code_text not in sys.path:
        sys.path.insert(0, code_text)

    # The official helper constructs CUDA tensors inside __init__. Replacing only
    # that helper keeps the official architecture/state_dict intact while allowing
    # normal .to(device) handling on Windows and CPU-based checkpoint inspection.
    import Model.Module.utils as official_utils

    official_utils.Map_2_Grad = DeviceSafeMapToGrad
    from Model.BGCrack import BGCrack

    return BGCrack()


def _png_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    files = {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() == ".png"
    }
    return files


@dataclass(frozen=True)
class SteelCrackItem:
    image: Path
    mask: Path
    edge: Path
    name: str


def paired_items(data_root: Path, split: str) -> list[SteelCrackItem]:
    split_root = data_root.resolve() / split
    images = _png_map(split_root / "images")
    masks = _png_map(split_root / "masks")
    edges = _png_map(split_root / "edges")
    if not (images.keys() == masks.keys() == edges.keys()):
        missing_mask = sorted(images.keys() - masks.keys())[:10]
        missing_edge = sorted(images.keys() - edges.keys())[:10]
        extra_mask = sorted(masks.keys() - images.keys())[:10]
        extra_edge = sorted(edges.keys() - images.keys())[:10]
        raise RuntimeError(
            "Image/mask/edge stem mismatch: "
            f"missing_mask={missing_mask}, missing_edge={missing_edge}, "
            f"extra_mask={extra_mask}, extra_edge={extra_edge}"
        )
    return [
        SteelCrackItem(images[name], masks[name], edges[name], name)
        for name in sorted(images)
    ]


class SteelCrackDataset(Dataset):
    def __init__(self, data_root: Path, split: str, include_edge: bool) -> None:
        self.items = paired_items(data_root, split)
        self.include_edge = include_edge

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _image_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
        tensor = torch.from_numpy(image).permute(2, 0, 1).div_(255.0)
        return tensor.sub_(0.5).div_(0.5)

    @staticmethod
    def _label_tensor(path: Path) -> torch.Tensor:
        with Image.open(path) as source:
            label = np.asarray(source.convert("L"), dtype=np.float32).copy()
        return torch.from_numpy(label).unsqueeze(0).div_(255.0)

    def __getitem__(self, index: int):
        item = self.items[index]
        image = self._image_tensor(item.image)
        mask = self._label_tensor(item.mask)
        if self.include_edge:
            edge = self._label_tensor(item.edge)
            return image, mask, edge, item.name
        return image, mask, item.name


def dice_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    smooth = 1e-6
    prediction_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    intersection = (prediction_flat * target_flat).sum(dim=1)
    score = (2.0 * intersection + smooth) / (
        prediction_flat.sum(dim=1) + target_flat.sum(dim=1) + smooth
    )
    return 1.0 - score.mean()


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + 1e-6).mean()


@dataclass
class MetricSums:
    samples: int = 0
    soft_dice: float = 0.0
    hard_dice: float = 0.0
    hard_iou: float = 0.0

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        prediction = prediction.float().flatten(1)
        target = target.float().flatten(1)
        epsilon = 1e-6

        soft_intersection = (prediction * target).sum(dim=1)
        soft_dice = (2.0 * soft_intersection + epsilon) / (
            prediction.sum(dim=1) + target.sum(dim=1) + epsilon
        )

        prediction_binary = prediction.ge(0.5).float()
        target_binary = target.ge(0.5).float()
        intersection = (prediction_binary * target_binary).sum(dim=1)
        union = (prediction_binary + target_binary - prediction_binary * target_binary).sum(
            dim=1
        )
        hard_dice = (2.0 * intersection + epsilon) / (
            prediction_binary.sum(dim=1) + target_binary.sum(dim=1) + epsilon
        )
        hard_iou = (intersection + epsilon) / (union + epsilon)

        self.samples += prediction.shape[0]
        self.soft_dice += float(soft_dice.sum().cpu())
        self.hard_dice += float(hard_dice.sum().cpu())
        self.hard_iou += float(hard_iou.sum().cpu())

    def averages(self) -> dict[str, float]:
        if self.samples == 0:
            raise RuntimeError("No samples were evaluated")
        return {
            "samples": self.samples,
            "soft_dice": self.soft_dice / self.samples,
            "hard_dice": self.hard_dice / self.samples,
            "hard_iou": self.hard_iou / self.samples,
        }


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return dict(state_dict)


def atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
