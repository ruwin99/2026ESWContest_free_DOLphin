from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASS_NAMES = ("Good", "Fair", "Poor", "Severe")
MASK_COLORS_RGB = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def resolve_project_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def configure_model_imports(config: dict[str, Any]) -> None:
    paths = config.get("paths", {})
    official_root = resolve_project_path(paths["official_training"])
    if not (official_root / "network" / "modeling.py").is_file():
        raise FileNotFoundError(
            f"Virginia Tech official model code not found: {official_root}"
        )
    official_text = str(official_root)
    if official_text not in sys.path:
        sys.path.insert(0, official_text)

    torch_home = resolve_project_path(paths["torch_home"])
    torch_home.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_HOME"] = str(torch_home)


def build_student(config: dict[str, Any], *, pretrained_backbone: bool) -> torch.nn.Module:
    configure_model_imports(config)
    from network.modeling import deeplabv3plus_mobilenet

    student_cfg = config["student"]
    model = deeplabv3plus_mobilenet(
        num_classes=int(student_cfg["num_classes"]),
        output_stride=int(student_cfg["output_stride"]),
        pretrained_backbone=pretrained_backbone,
    )
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = int(student_cfg["expected_parameters"])
    if actual != expected:
        raise RuntimeError(
            f"Unexpected student parameter count: expected={expected}, actual={actual}"
        )
    return model


def build_teacher(
    config: dict[str, Any], architecture: str, *, pretrained_backbone: bool = False
) -> torch.nn.Module:
    configure_model_imports(config)
    from network import modeling

    factories = {
        "deeplabv3plus_resnet50": modeling.deeplabv3plus_resnet50,
        "deeplabv3plus_resnet101": modeling.deeplabv3plus_resnet101,
    }
    if architecture not in factories:
        raise ValueError(f"Unsupported teacher architecture: {architecture}")
    return factories[architecture](
        num_classes=4,
        output_stride=8,
        pretrained_backbone=pretrained_backbone,
    )


def mask_rgb_to_indices(mask_rgb: np.ndarray) -> np.ndarray:
    if mask_rgb.ndim != 3 or mask_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB mask, got {mask_rgb.shape}")
    indices = np.full(mask_rgb.shape[:2], 255, dtype=np.uint8)
    for class_index, color in enumerate(MASK_COLORS_RGB):
        indices[np.all(mask_rgb == color, axis=2)] = class_index
    if np.any(indices == 255):
        invalid = np.unique(mask_rgb[indices == 255].reshape(-1, 3), axis=0)
        sample = invalid[:10].tolist()
        raise ValueError(f"Mask contains unsupported RGB colors: {sample}")
    return indices


def discover_pairs(dataset_root: Path, source_split: str) -> list[tuple[Path, Path]]:
    base = dataset_root / source_split
    image_dir = base / "images_512"
    mask_dir = base / "mask_512"
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError(
            f"Expected images_512 and mask_512 under: {base}"
        )
    images = {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    masks = {
        path.stem: path
        for path in mask_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if images.keys() != masks.keys():
        missing_masks = sorted(images.keys() - masks.keys())[:10]
        missing_images = sorted(masks.keys() - images.keys())[:10]
        raise RuntimeError(
            "Image/mask stem mismatch: "
            f"missing_masks={missing_masks}, missing_images={missing_images}"
        )
    return [(images[stem], masks[stem]) for stem in sorted(images)]


def _record_path(value: str, dataset_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (dataset_root / Path(value.replace("/", os.sep))).resolve()


def load_split_pairs(
    split_csv: Path, dataset_root: Path, requested_split: str
) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "mask_path", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"Split CSV must contain {sorted(required)}: {split_csv}"
            )
        for row in reader:
            if row["split"].strip().lower() != requested_split.lower():
                continue
            pairs.append(
                (
                    _record_path(row["image_path"], dataset_root),
                    _record_path(row["mask_path"], dataset_root),
                )
            )
    if not pairs:
        raise RuntimeError(f"No {requested_split!r} rows found in {split_csv}")
    return pairs


class VTCSSDDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        *,
        input_size: tuple[int, int] = (512, 512),
        augment_horizontal_flip: bool = False,
    ) -> None:
        self.pairs = list(pairs)
        self.input_size = input_size
        self.augment_horizontal_flip = augment_horizontal_flip

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        image_path, mask_path = self.pairs[index]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        with Image.open(mask_path) as source:
            mask = source.convert("RGB")

        width, height = self.input_size[1], self.input_size[0]
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        if mask.size != (width, height):
            mask = mask.resize((width, height), Image.Resampling.NEAREST)

        if self.augment_horizontal_flip and bool(torch.rand(()) < 0.5):
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        image_rgb_u8 = np.asarray(image, dtype=np.uint8)
        mask_rgb_u8 = np.asarray(mask, dtype=np.uint8)
        target = mask_rgb_to_indices(mask_rgb_u8)

        student_array = image_rgb_u8.astype(np.float32) / 255.0
        student_array = (student_array - IMAGENET_MEAN) / IMAGENET_STD
        student_array = np.ascontiguousarray(student_array.transpose(2, 0, 1))

        teacher_array = image_rgb_u8[..., ::-1].astype(np.float32)
        teacher_array = np.ascontiguousarray(teacher_array.transpose(2, 0, 1))

        return {
            "student_view": torch.from_numpy(student_array),
            "teacher_view": torch.from_numpy(teacher_array),
            "target": torch.from_numpy(target.astype(np.int64, copy=False)),
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def update_confusion_matrix(
    matrix: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor
) -> None:
    valid = (target >= 0) & (target < 4)
    encoded = target[valid].to(torch.int64) * 4 + prediction[valid].to(torch.int64)
    matrix += torch.bincount(encoded.cpu(), minlength=16).reshape(4, 4)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _mean_defined(values: Iterable[float | None]) -> float | None:
    defined = [value for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def metrics_from_confusion(matrix: torch.Tensor | np.ndarray) -> dict[str, Any]:
    cm = np.asarray(matrix, dtype=np.int64)
    if cm.shape != (4, 4):
        raise ValueError(f"Expected 4x4 confusion matrix, got {cm.shape}")

    per_class: dict[str, dict[str, float | int | None]] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(cm[index, index])
        fp = int(cm[:, index].sum() - tp)
        fn = int(cm[index, :].sum() - tp)
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        iou = _ratio(tp, tp + fp + fn)
        per_class[name] = {
            "support_pixels": int(cm[index, :].sum()),
            "predicted_pixels": int(cm[:, index].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou,
        }

    severe_total = int(cm[3, :].sum())
    poor_total = int(cm[2, :].sum())
    corrosion_indices = (1, 2, 3)
    poor_severe_target = int(cm[2:4, :].sum())
    poor_severe_correct = int(cm[2:4, 2:4].sum())
    severe_under_call = int(cm[3, 0] + cm[3, 1])
    poor_to_good = int(cm[2, 0])

    return {
        "confusion_matrix": cm.tolist(),
        "total_pixels": int(cm.sum()),
        "pixel_accuracy": _ratio(int(np.trace(cm)), int(cm.sum())),
        "per_class": per_class,
        "macro_f1_4": _mean_defined(
            per_class[name]["f1"] for name in CLASS_NAMES
        ),
        "macro_iou_4": _mean_defined(
            per_class[name]["iou"] for name in CLASS_NAMES
        ),
        "macro_f1_corrosion3": _mean_defined(
            per_class[CLASS_NAMES[index]]["f1"] for index in corrosion_indices
        ),
        "macro_iou_corrosion3": _mean_defined(
            per_class[CLASS_NAMES[index]]["iou"] for index in corrosion_indices
        ),
        "poor_severe_recall": _ratio(poor_severe_correct, poor_severe_target),
        "severe_recall": per_class["Severe"]["recall"],
        "severe_under_call_count": severe_under_call,
        "severe_under_call_rate": _ratio(severe_under_call, severe_total),
        "poor_to_good_count": poor_to_good,
        "poor_to_good_rate": _ratio(poor_to_good, poor_total),
    }


@dataclass(frozen=True)
class EvaluationResult:
    loss: float | None
    metrics: dict[str, Any]
    sample_count: int


@torch.inference_mode()
def evaluate_student(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    class_weights: torch.Tensor | None = None,
    amp: bool = False,
) -> EvaluationResult:
    model.eval()
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    total_loss = 0.0
    sample_count = 0
    batch_count = 0
    for batch in loader:
        inputs = batch["student_view"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp and device.type == "cuda",
        ):
            logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("Student logits contain NaN or Inf during evaluation")
        if class_weights is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.float(), target, weight=class_weights
            )
            total_loss += float(loss)
        prediction = logits.argmax(dim=1)
        update_confusion_matrix(confusion, target, prediction)
        sample_count += int(target.shape[0])
        batch_count += 1
    return EvaluationResult(
        loss=(total_loss / batch_count) if class_weights is not None and batch_count else None,
        metrics=metrics_from_confusion(confusion),
        sample_count=sample_count,
    )

