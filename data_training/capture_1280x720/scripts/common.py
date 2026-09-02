from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CAPTURE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = CAPTURE_ROOT / "config.json"
EXTERNAL_HEIGHT = 720
EXTERNAL_WIDTH = 1280
CRACK_INTERNAL_HEIGHT = 736
CRACK_INTERNAL_WIDTH = 1280
CRACK_PAD_TOP = 8
CRACK_PAD_BOTTOM = 8
CLASS_NAMES = ("Good", "Fair", "Poor", "Severe")
MASK_COLORS_RGB = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DCT_BUFFER_KEYS = {
    "HFIE1_S.dct_layer.weight",
    "HFIE2_S.dct_layer.weight",
    "HFIE1_C.dct_layer.weight",
    "HFIE2_C.dct_layer.weight",
}


def load_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def normalize_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key.removeprefix("module."): value for key, value in state_dict.items()}
    return dict(state_dict)


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Checkpoint does not contain a state_dict: {path}")
    return normalize_state_dict_keys(payload)


def build_corrosion_model(checkpoint: Path) -> nn.Module:
    official_root = (
        PROJECT_ROOT
        / "models"
        / "virginia_tech_cssd"
        / "official_code"
        / "Training - Testing"
    )
    if not (official_root / "network" / "modeling.py").is_file():
        raise FileNotFoundError(f"Virginia Tech model code not found: {official_root}")
    text = str(official_root)
    if text not in sys.path:
        sys.path.insert(0, text)
    os.environ["TORCH_HOME"] = str(
        PROJECT_ROOT / "models" / "virginia_tech_cssd" / "torch_cache"
    )
    from network.modeling import deeplabv3plus_resnet101

    model = deeplabv3plus_resnet101(
        num_classes=4, output_stride=8, pretrained_backbone=False
    )
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)
    return model


class DeviceSafeMapToGrad(nn.Module):
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


def _replace_dct_buffer(module: nn.Module, height: int, width: int, spatial: bool) -> None:
    if spatial:
        from Model.Module.Spatial_maxsum import get_freq_indices

        channel = int(module.dct_layer.weight.shape[1])
        mapper_x = get_freq_indices("top16")
        mapper_x = [value * (channel // 8) for value in mapper_x]
        weight = module.dct_layer.get_dct_filter(height, width, mapper_x, channel)
    else:
        from Model.Module.Channel_maxsum import get_freq_indices

        channel = int(module.dct_layer.c0.in_channels)
        mapper_x, mapper_y = get_freq_indices("top16")
        mapper_x = [value * (height // 8) for value in mapper_x]
        mapper_y = [value * (width // 8) for value in mapper_y]
        module.dct_layer.height = height
        module.dct_layer.width = width
        weight = module.dct_layer.get_dct_filter(
            height, width, mapper_x, mapper_y, channel
        )
    module.dct_h = height
    module.dct_w = width
    module.dct_layer.weight = weight


def adapt_bgcrack_dct(model: nn.Module) -> None:
    _replace_dct_buffer(model.HFIE1_S, 184, 320, spatial=True)
    _replace_dct_buffer(model.HFIE2_S, 92, 160, spatial=True)
    _replace_dct_buffer(model.HFIE1_C, 184, 320, spatial=False)
    _replace_dct_buffer(model.HFIE2_C, 92, 160, spatial=False)


class PatchCompatibleMobileViT(nn.Module):
    """Pad only an odd feature-map edge for a 2x2 MobileViT patch, then crop it."""

    def __init__(
        self, module: nn.Module, output_height: int, output_width: int
    ) -> None:
        super().__init__()
        self.module = module
        self.output_height = output_height
        self.output_width = output_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 1), value=0.0)
        output = self.module(x)
        return output[..., : self.output_height, : self.output_width]


def build_capture_bgcrack(checkpoint: Path) -> nn.Module:
    official_root = CAPTURE_ROOT.parent / "steelcrack" / "official_bgcrack"
    text = str(official_root.resolve())
    if not (official_root / "Model" / "BGCrack.py").is_file():
        raise FileNotFoundError(f"BGCrack model code not found: {official_root}")
    if text not in sys.path:
        sys.path.insert(0, text)
    import Model.Module.utils as official_utils

    official_utils.Map_2_Grad = DeviceSafeMapToGrad
    from Model.BGCrack import BGCrack

    model = BGCrack()
    adapt_bgcrack_dct(model)
    state = checkpoint_state(checkpoint)
    filtered = {key: value for key, value in state.items() if key not in DCT_BUFFER_KEYS}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if set(missing) != DCT_BUFFER_KEYS or unexpected:
        raise RuntimeError(
            "Unexpected BGCrack checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    model.MVT4 = PatchCompatibleMobileViT(model.MVT4, 23, 40)
    return model


class CaptureBGCrack(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    @staticmethod
    def _crop(tensor: torch.Tensor) -> torch.Tensor:
        return tensor[..., CRACK_PAD_TOP : CRACK_PAD_TOP + EXTERNAL_HEIGHT, :]

    def forward_all(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        padded = F.pad(images, (0, 0, CRACK_PAD_TOP, CRACK_PAD_BOTTOM), value=0.0)
        body, edge, gradient = self.core(padded)
        return self._crop(body), self._crop(edge), self._crop(gradient)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        body, _, _ = self.forward_all(images)
        return body


def _file_map(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def paired_paths(root: Path, task: str, split: str) -> list[dict[str, Path]]:
    base = root / task / split
    images = _file_map(base / "images")
    masks = _file_map(base / "masks")
    groups = {"image": images, "mask": masks}
    if task == "crack":
        groups["edge"] = _file_map(base / "edges")
    stems = set(images)
    if not stems or any(set(items) != stems for items in groups.values()):
        counts = {name: len(items) for name, items in groups.items()}
        raise RuntimeError(f"Empty or mismatched {task}/{split} data: {counts}")
    return [
        {name: items[stem] for name, items in groups.items()} | {"name": Path(stem)}
        for stem in sorted(stems)
    ]


def _read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
    if image.shape != (EXTERNAL_HEIGHT, EXTERNAL_WIDTH, 3):
        raise ValueError(f"Expected 1280x720 RGB image, got {image.shape}: {path}")
    return image


def _read_rust_mask(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        raw = np.asarray(source).copy()
    if raw.shape[:2] != (EXTERNAL_HEIGHT, EXTERNAL_WIDTH):
        raise ValueError(f"Expected 1280x720 rust mask: {path}")
    if raw.ndim == 2:
        values = set(np.unique(raw).tolist())
        if not values.issubset({0, 1, 2, 3, 255}):
            raise ValueError(f"Unsupported rust mask values {sorted(values)}: {path}")
        return raw.astype(np.int64)
    rgb = raw[..., :3]
    target = np.full(rgb.shape[:2], 255, dtype=np.int64)
    for index, color in enumerate(MASK_COLORS_RGB):
        target[np.all(rgb == color, axis=2)] = index
    if np.any(target == 255):
        raise ValueError(f"Unsupported rust mask palette: {path}")
    return target


def _read_binary(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        raw = np.asarray(source.convert("L"), dtype=np.uint8).copy()
    if raw.shape != (EXTERNAL_HEIGHT, EXTERNAL_WIDTH):
        raise ValueError(f"Expected 1280x720 binary label: {path}")
    values = set(np.unique(raw).tolist())
    if not values.issubset({0, 1, 255}):
        raise ValueError(f"Unsupported binary values {sorted(values)}: {path}")
    return (raw > 0).astype(np.float32)


class CaptureDataset(Dataset):
    def __init__(self, root: Path, task: str, split: str, augment: bool = False) -> None:
        if task not in {"rust", "crack"}:
            raise ValueError(f"Unsupported task: {task}")
        self.task = task
        self.items = paired_paths(root, task, split)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        image = _read_rgb(item["image"])
        flip = self.augment and random.random() < 0.5
        if self.task == "rust":
            mask = _read_rust_mask(item["mask"])
            if flip:
                image = image[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            bgr = np.ascontiguousarray(image[..., ::-1].transpose(2, 0, 1))
            return {
                "image": torch.from_numpy(bgr.astype(np.float32)),
                "mask": torch.from_numpy(mask),
                "name": item["name"].name,
            }
        mask = _read_binary(item["mask"])
        edge = _read_binary(item["edge"])
        if flip:
            image = image[:, ::-1].copy()
            mask = mask[:, ::-1].copy()
            edge = edge[:, ::-1].copy()
        rgb = np.ascontiguousarray(image.transpose(2, 0, 1)).astype(np.float32)
        rgb = rgb / 127.5 - 1.0
        return {
            "image": torch.from_numpy(rgb),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "edge": torch.from_numpy(edge).unsqueeze(0),
            "name": item["name"].name,
        }


def dice_loss_binary(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.flatten(1)
    target = target.flatten(1)
    intersection = (prediction * target).sum(dim=1)
    return 1.0 - (
        (2.0 * intersection + 1e-6)
        / (prediction.sum(dim=1) + target.sum(dim=1) + 1e-6)
    ).mean()


def dice_loss_multiclass(
    logits: torch.Tensor, target: torch.Tensor, ignore_index: int = 255
) -> torch.Tensor:
    valid = target.ne(ignore_index)
    safe_target = target.masked_fill(~valid, 0)
    one_hot = F.one_hot(safe_target, num_classes=4).permute(0, 3, 1, 2).float()
    valid_float = valid.unsqueeze(1).float()
    probability = logits.softmax(dim=1) * valid_float
    one_hot = one_hot * valid_float
    intersection = (probability * one_hot).sum(dim=(0, 2, 3))
    denominator = probability.sum(dim=(0, 2, 3)) + one_hot.sum(dim=(0, 2, 3))
    return 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()


def charbonnier_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt((prediction - target).square() + 1e-6).mean()


def atomic_torch_save(payload: Any, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)


def _dft_basis(length: int, frequency_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = torch.arange(frequency_count, dtype=torch.float32).unsqueeze(1)
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(0)
    angles = (2.0 * math.pi / length) * frequencies * positions
    return torch.cos(angles), torch.sin(angles)


class TensorRTFriendlyFFTBlock(nn.Module):
    def __init__(self, original: nn.Module, height: int, width: int) -> None:
        super().__init__()
        if width % 2 or getattr(original, "norm", None) != "forward":
            raise ValueError("Unsupported FFT block shape or normalization")
        self.former3 = original.former3
        self.conv4 = original.conv4
        self.height = height
        self.width = width
        cos_h, sin_h = _dft_basis(height, height)
        cos_w_forward, sin_w_forward = _dft_basis(width, width // 2 + 1)
        cos_w_full, sin_w_full = _dft_basis(width, width)
        negative_height = torch.cat(
            (torch.zeros(1, dtype=torch.long), torch.arange(height - 1, 0, -1))
        )
        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w_forward", cos_w_forward, persistent=False)
        self.register_buffer("sin_w_forward", sin_w_forward, persistent=False)
        self.register_buffer("cos_w_full", cos_w_full, persistent=False)
        self.register_buffer("sin_w_full", sin_w_full, persistent=False)
        self.register_buffer("negative_height", negative_height, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        real_w = torch.matmul(x, self.cos_w_forward.transpose(0, 1))
        imag_w = -torch.matmul(x, self.sin_w_forward.transpose(0, 1))
        real_w, imag_w = real_w.transpose(-2, -1), imag_w.transpose(-2, -1)
        real = (
            torch.matmul(real_w, self.cos_h.transpose(0, 1))
            + torch.matmul(imag_w, self.sin_h.transpose(0, 1))
        ).transpose(-2, -1)
        imag = (
            torch.matmul(imag_w, self.cos_h.transpose(0, 1))
            - torch.matmul(real_w, self.sin_h.transpose(0, 1))
        ).transpose(-2, -1)
        real = real / (self.height * self.width)
        imag = imag / (self.height * self.width)
        frequency = self.former3(torch.cat((real, imag), dim=1))
        real, imag = torch.chunk(frequency, 2, dim=1)
        real_tail = real.index_select(-2, self.negative_height)[..., 1:-1].flip(-1)
        imag_tail = -imag.index_select(-2, self.negative_height)[..., 1:-1].flip(-1)
        real = torch.cat((real, real_tail), dim=-1)
        imag = torch.cat((imag, imag_tail), dim=-1)
        real_w = torch.matmul(real, self.cos_w_full) - torch.matmul(imag, self.sin_w_full)
        imag_w = torch.matmul(real, self.sin_w_full) + torch.matmul(imag, self.cos_w_full)
        real_w, imag_w = real_w.transpose(-2, -1), imag_w.transpose(-2, -1)
        output = (
            torch.matmul(real_w, self.cos_h) - torch.matmul(imag_w, self.sin_h)
        ).transpose(-2, -1)
        return self.conv4(output)


def rewrite_bgcrack_fft_for_onnx(model: CaptureBGCrack) -> None:
    model.core.b3_FFT = TensorRTFriendlyFFTBlock(model.core.b3_FFT, 46, 80)
    model.core.b4_FFT = TensorRTFriendlyFFTBlock(model.core.b4_FFT, 23, 40)
