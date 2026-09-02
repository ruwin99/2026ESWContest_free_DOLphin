from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, Sampler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORK_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("Good", "Fair", "Poor", "Severe")
MASK_COLORS_RGB = ((0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0))
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REQUIRED_COLUMNS = (
    "sample_id", "image_path", "rust_mask_path", "rust_valid_mask_path", "split",
    "source_type", "source_dataset", "source_url", "license", "license_file",
    "capture_date", "camera_id", "session_id", "source_print_id", "placement_id",
    "lighting_id", "group_id", "image_sha256", "mask_sha256", "label_status",
    "labeler", "reviewer", "notes",
)


def load_config(path: Path) -> dict[str, Any]:
    with path.resolve().open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Invalid config: {path}")
    return config


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


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


def atomic_torch_save(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def git_identity(path: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            result = subprocess.run(
                ["git", "-C", str(path), *args], capture_output=True, text=True,
                encoding="utf-8", errors="replace", check=False,
            )
        except FileNotFoundError:
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = run("status", "--short")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def checkpoint_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata: dict[str, Any] = {}
    if isinstance(payload, dict):
        metadata = {key: value for key, value in payload.items() if key not in {"state_dict", "model_state_dict"}}
        for key in ("model_state_dict", "state_dict"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Checkpoint does not contain a state_dict: {path}")
    state = dict(payload)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state, metadata


def build_model(config: dict[str, Any], checkpoint: Path | None = None) -> nn.Module:
    official_root = resolve_path(config["paths"]["official_training"])
    if not (official_root / "network" / "modeling.py").is_file():
        raise FileNotFoundError(f"Official model code missing: {official_root}")
    if str(official_root) not in sys.path:
        sys.path.insert(0, str(official_root))
    os.environ["TORCH_HOME"] = str(resolve_path("models/virginia_tech_cssd/torch_cache"))
    from network.modeling import deeplabv3plus_resnet101

    model = deeplabv3plus_resnet101(
        num_classes=int(config["model"]["num_classes"]),
        output_stride=int(config["model"]["output_stride"]),
        pretrained_backbone=False,
    )
    actual = sum(parameter.numel() for parameter in model.parameters())
    expected = int(config["model"]["expected_parameters"])
    if actual != expected:
        raise RuntimeError(f"Unexpected parameter count: expected={expected}, actual={actual}")
    if checkpoint is not None:
        state, _ = checkpoint_state(checkpoint)
        model.load_state_dict(state, strict=True)
    return model


def read_manifest(path: Path, *, expected_split: str | None = None) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or tuple(reader.fieldnames) != REQUIRED_COLUMNS:
            raise ValueError(f"Manifest columns must exactly match required schema: {path}")
        rows = [{key: (value or "").strip() for key, value in row.items()} for row in reader]
    if not rows:
        raise RuntimeError(f"Manifest is empty: {path}")
    if expected_split and any(row["split"] != expected_split for row in rows):
        raise ValueError(f"Manifest contains a row outside split={expected_split}: {path}")
    return rows


def write_manifest(path: Path, rows: Sequence[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in REQUIRED_COLUMNS})


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def read_mask(path: Path) -> np.ndarray:
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"Unable to decode mask: {path}")
    if raw.ndim == 2:
        mask = raw.astype(np.uint8, copy=False)
    else:
        rgb = cv2.cvtColor(raw[..., :3], cv2.COLOR_BGR2RGB)
        mask = np.full(rgb.shape[:2], 255, dtype=np.uint8)
        for index, color in enumerate(MASK_COLORS_RGB):
            mask[np.all(rgb == color, axis=2)] = index
    values = set(np.unique(mask).tolist())
    if not values.issubset({0, 1, 2, 3, 255}):
        raise ValueError(f"Unsupported mask values {sorted(values)}: {path}")
    return mask


class RustManifestDataset(Dataset[dict[str, Any]]):
    def __init__(self, rows: Sequence[dict[str, str]], config: dict[str, Any], *, training: bool) -> None:
        self.rows = list(rows)
        self.training = training
        self.crop_h = int(config["data"]["train_crop_height"])
        self.crop_w = int(config["data"]["train_crop_width"])
        self.augmentation = config["data"]["augmentation"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image = read_bgr(resolve_path(row["image_path"]))
        mask = read_mask(resolve_path(row["rust_mask_path"]))
        if image.shape[:2] != mask.shape:
            raise ValueError(f"Image/mask shape mismatch for {row['sample_id']}: {image.shape[:2]} != {mask.shape}")
        valid_path = row["rust_valid_mask_path"]
        if valid_path:
            valid = cv2.imread(str(resolve_path(valid_path)), cv2.IMREAD_GRAYSCALE)
            if valid is None or valid.shape != mask.shape:
                raise ValueError(f"Invalid valid mask for {row['sample_id']}")
            mask = mask.copy()
            mask[valid == 0] = 255

        if self.training:
            height, width = mask.shape
            if height < self.crop_h or width < self.crop_w:
                raise ValueError(f"Training image smaller than native crop for {row['sample_id']}: {(height, width)}")
            top = random.randint(0, height - self.crop_h)
            left = random.randint(0, width - self.crop_w)
            image = image[top : top + self.crop_h, left : left + self.crop_w]
            mask = mask[top : top + self.crop_h, left : left + self.crop_w]
            if random.random() < float(self.augmentation["horizontal_flip_probability"]):
                image = image[:, ::-1].copy()
                mask = mask[:, ::-1].copy()
            exposure = random.uniform(*map(float, self.augmentation["exposure_range"]))
            wb_low, wb_high = map(float, self.augmentation["white_balance_range"])
            white_balance = np.asarray([random.uniform(wb_low, wb_high) for _ in range(3)], dtype=np.float32)
            image_float = image.astype(np.float32) * exposure * white_balance.reshape(1, 1, 3)
            noise_max = float(self.augmentation["gaussian_noise_std_max"])
            if noise_max > 0:
                sigma = random.uniform(0.0, noise_max)
                image_float += np.random.normal(0.0, sigma, image_float.shape).astype(np.float32)
            image = np.clip(image_float, 0.0, 255.0)
        else:
            image = image.astype(np.float32)

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        return {
            "image": tensor,
            "target": torch.from_numpy(np.ascontiguousarray(mask)).long(),
            "sample_id": row["sample_id"],
            "source_type": row["source_type"],
            "group_id": row["group_id"],
        }


class BalancedGroupBatchSampler(Sampler[list[int]]):
    """Equal positive/hard-negative exposure with round-robin group sampling."""

    def __init__(self, rows: Sequence[dict[str, str]], batch_size: int, seed: int) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.rows = list(rows)
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        by_type: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, row in enumerate(rows):
            kind = "hard_negative" if row["source_type"] == "hard_negative" else "positive"
            by_type[kind][row["group_id"]].append(index)
        if not by_type["positive"] or not by_type["hard_negative"]:
            raise RuntimeError("Training requires both positive replay and approved hard-negative groups")
        self.by_type = by_type
        largest = max(sum(map(len, groups.values())) for groups in by_type.values())
        self.epoch_samples = 2 * largest

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.epoch_samples / self.batch_size)

    @staticmethod
    def _stream(groups: dict[str, list[int]], rng: random.Random) -> Iterable[int]:
        names = list(groups)
        while True:
            rng.shuffle(names)
            for name in names:
                candidates = groups[name]
                yield candidates[rng.randrange(len(candidates))]

    def __iter__(self) -> Iterable[list[int]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        streams = {kind: self._stream(groups, rng) for kind, groups in self.by_type.items()}
        batch: list[int] = []
        for offset in range(self.epoch_samples):
            kind = "positive" if offset % 2 == 0 else "hard_negative"
            batch.append(next(streams[kind]))
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def freeze_for_stage(model: nn.Module, stage: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "a":
        train_modules = ["classifier"] if config is None else list(config["stage_a"].get("train_modules", ["classifier"]))
        if train_modules == ["classifier"]:
            for parameter in model.classifier.parameters():
                parameter.requires_grad = True
        elif train_modules == ["classifier.classifier.3"]:
            for parameter in model.classifier.classifier[3].parameters():
                parameter.requires_grad = True
        else:
            raise ValueError(f"Unsupported Stage A train_modules: {train_modules}")
    elif stage == "b":
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.layer4.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unsupported stage: {stage}")
    bn_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            bn_names.append(name)
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad = False
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    return {
        "stage": stage,
        "trainable_parameter_tensors": trainable,
        "frozen_parameter_tensors": frozen,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "frozen_parameters": sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad),
        "batchnorm_modules_frozen": bn_names,
    }


def enforce_frozen_batchnorm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def enforce_frozen_dropout(model: nn.Module) -> None:
    """Disable stochastic dropout when all layers feeding the final logits are frozen."""
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.eval()


def optimizer_for_stage(model: nn.Module, config: dict[str, Any], stage: str) -> torch.optim.Optimizer:
    if stage == "a":
        groups = [{"params": [p for p in model.classifier.parameters() if p.requires_grad], "lr": float(config["stage_a"]["classifier_learning_rate"]), "name": "classifier"}]
    else:
        classifier = [p for p in model.classifier.parameters() if p.requires_grad]
        layer4 = [p for p in model.backbone.layer4.parameters() if p.requires_grad]
        groups = [
            {"params": classifier, "lr": float(config["stage_b"]["classifier_learning_rate"]), "name": "classifier"},
            {"params": layer4, "lr": float(config["stage_b"]["backbone_learning_rate"]), "name": "backbone.layer4"},
        ]
    if any(not group["params"] for group in groups):
        raise RuntimeError("A configured optimizer parameter group is empty")
    return torch.optim.Adam(groups, weight_decay=float(config["optimizer_reference"]["weight_decay"]))


def baseline_anchored_stage_a_loss(
    logits: torch.Tensor,
    baseline_logits: torch.Tensor,
    target: torch.Tensor,
    source_types: Sequence[str],
    class_weights: torch.Tensor,
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Conservative Stage A loss: preserve positive behavior and mine only hard normal pixels."""
    settings = config["loss"]
    ignore_index = int(config["data"]["ignore_index"])
    ce_map = F.cross_entropy(
        logits,
        target,
        weight=class_weights,
        ignore_index=ignore_index,
        reduction="none",
    )
    is_hard_negative = torch.tensor(
        [kind == "hard_negative" for kind in source_types],
        dtype=torch.bool,
        device=logits.device,
    )
    positive_pixels = target.ne(ignore_index) & ~is_hard_negative[:, None, None]
    if not positive_pixels.any():
        raise RuntimeError("Stage A v2 batch has no positive replay pixels")
    positive_ce = ce_map[positive_pixels].mean()

    # Select the most rust-like approved-Good pixels independently per image.
    # A configured focus class allows suppressing the observed false-positive
    # class without unnecessarily pushing every corrosion logit down.
    focus_name = settings.get("hard_negative_focus_class")
    if focus_name:
        if focus_name not in CLASS_NAMES[1:]:
            raise ValueError(f"hard_negative_focus_class must be one of {CLASS_NAMES[1:]}, got {focus_name}")
        focus_index = CLASS_NAMES.index(focus_name)
        rust_margin = logits[:, focus_index] - logits[:, 0]
        focused_good_loss = F.softplus(rust_margin)
    else:
        rust_margin = logits[:, 1:].amax(dim=1) - logits[:, 0]
        focused_good_loss = ce_map
    hard_losses: list[torch.Tensor] = []
    hard_all_losses: list[torch.Tensor] = []
    fraction = float(settings["hard_negative_top_fraction"])
    minimum = int(settings["hard_negative_min_pixels_per_sample"])
    for item, is_negative in enumerate(is_hard_negative.tolist()):
        if not is_negative:
            continue
        eligible = target[item].eq(0)
        count = int(eligible.sum())
        if count == 0:
            continue
        take = min(count, max(minimum, math.ceil(count * fraction)))
        candidate_loss = ce_map[item][eligible]
        hard_all_losses.append(candidate_loss)
        candidate_margin = rust_margin[item][eligible]
        focused_loss = focused_good_loss[item][eligible]
        selected = torch.topk(candidate_margin.detach(), k=take, sorted=False).indices
        hard_losses.append(focused_loss[selected])
    if not hard_losses:
        raise RuntimeError("Stage A v2 batch has no approved-Good hard-negative pixels")
    hard_negative_ce = torch.cat(hard_losses).mean()
    hard_negative_all_pixel_ce = torch.cat(hard_all_losses).mean()

    temperature = float(settings["distillation_temperature"])
    with torch.no_grad():
        teacher_probability = F.softmax(baseline_logits / temperature, dim=1)
        teacher_log_probability = F.log_softmax(baseline_logits / temperature, dim=1)
    student_log_probability = F.log_softmax(logits / temperature, dim=1)
    kl_map = (teacher_probability * (teacher_log_probability - student_log_probability)).sum(dim=1)
    positive_teacher_kl = kl_map[positive_pixels].mean() * (temperature * temperature)

    # Preserve the teacher's decision margins exactly where the locked safety
    # gate would count a new major under-call.
    baseline_prediction = baseline_logits.argmax(dim=1)
    candidate_severe_margin = logits[:, 2:].amax(dim=1) - logits[:, :2].amax(dim=1)
    baseline_severe_margin = baseline_logits[:, 2:].amax(dim=1) - baseline_logits[:, :2].amax(dim=1)
    severe_guard = positive_pixels & target.eq(3) & baseline_prediction.gt(1)
    severe_teacher_margin = (
        F.relu(baseline_severe_margin[severe_guard] - candidate_severe_margin[severe_guard]).mean()
        if severe_guard.any()
        else logits.new_zeros(())
    )

    candidate_poor_margin = logits[:, 1:].amax(dim=1) - logits[:, 0]
    baseline_poor_margin = baseline_logits[:, 1:].amax(dim=1) - baseline_logits[:, 0]
    poor_guard = positive_pixels & target.eq(2) & baseline_prediction.ne(0)
    poor_teacher_margin = (
        F.relu(baseline_poor_margin[poor_guard] - candidate_poor_margin[poor_guard]).mean()
        if poor_guard.any()
        else logits.new_zeros(())
    )

    candidate_fair_margin = logits[:, 1] - logits[:, 0]
    baseline_fair_margin = baseline_logits[:, 1] - baseline_logits[:, 0]
    fair_guard = positive_pixels & target.eq(1) & baseline_prediction.eq(1)
    fair_teacher_margin = (
        F.relu(baseline_fair_margin[fair_guard] - candidate_fair_margin[fair_guard]).mean()
        if fair_guard.any()
        else logits.new_zeros(())
    )

    total = (
        float(settings["positive_ce_weight"]) * positive_ce
        + float(settings["hard_negative_ce_weight"]) * hard_negative_ce
        + float(settings.get("hard_negative_all_pixel_ce_weight", 0.0)) * hard_negative_all_pixel_ce
        + float(settings["positive_teacher_kl_weight"]) * positive_teacher_kl
        + float(settings.get("severe_teacher_margin_weight", 0.0)) * severe_teacher_margin
        + float(settings.get("poor_teacher_margin_weight", 0.0)) * poor_teacher_margin
        + float(settings.get("fair_teacher_margin_weight", 0.0)) * fair_teacher_margin
    )
    return total, {
        "positive_ce": positive_ce,
        "hard_negative_ce": hard_negative_ce,
        "hard_negative_all_pixel_ce": hard_negative_all_pixel_ce,
        "positive_teacher_kl": positive_teacher_kl,
        "severe_teacher_margin": severe_teacher_margin,
        "poor_teacher_margin": poor_teacher_margin,
        "fair_teacher_margin": fair_teacher_margin,
    }


def class_weights_from_rows(rows: Sequence[dict[str, str]], config: dict[str, Any]) -> tuple[torch.Tensor, list[int]]:
    counts = np.zeros(4, dtype=np.int64)
    for row in rows:
        mask = read_mask(resolve_path(row["rust_mask_path"]))
        if row["rust_valid_mask_path"]:
            valid = cv2.imread(str(resolve_path(row["rust_valid_mask_path"])), cv2.IMREAD_GRAYSCALE)
            if valid is None or valid.shape != mask.shape:
                raise ValueError(f"Invalid valid mask for {row['sample_id']}")
            mask = mask.copy()
            mask[valid == 0] = 255
        valid_values = mask[mask != 255]
        counts += np.bincount(valid_values, minlength=4)[:4]
    if np.any(counts == 0):
        raise RuntimeError(f"Every rust class must occur in train data; pixel counts={counts.tolist()}")
    frequency = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequency)
    weights /= weights.mean()
    low, high = map(float, config["loss"]["class_weight_clip"])
    weights = np.clip(weights, low, high)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts.tolist()


def update_confusion(matrix: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor) -> None:
    valid = target.ne(255)
    encoded = target[valid].to(torch.int64) * 4 + prediction[valid].to(torch.int64)
    matrix += torch.bincount(encoded.cpu(), minlength=16).reshape(4, 4)


def _ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def metrics_from_confusion(matrix: torch.Tensor) -> dict[str, Any]:
    cm = matrix.numpy().astype(np.int64)
    per_class: dict[str, Any] = {}
    for index, name in enumerate(CLASS_NAMES):
        tp = int(cm[index, index])
        support = int(cm[index, :].sum())
        predicted = int(cm[:, index].sum())
        union = support + predicted - tp
        per_class[name] = {
            "support_pixels": support,
            "predicted_pixels": predicted,
            "precision": _ratio(tp, predicted),
            "recall": _ratio(tp, support),
            "iou": _ratio(tp, union),
        }
    corrosion_ious = [per_class[name]["iou"] for name in CLASS_NAMES[1:] if per_class[name]["iou"] is not None]
    return {
        "confusion_matrix": cm.tolist(),
        "valid_pixels": int(cm.sum()),
        "pixel_accuracy": _ratio(int(np.trace(cm)), int(cm.sum())),
        "macro_iou_corrosion3": float(np.mean(corrosion_ious)) if corrosion_ious else None,
        "per_class": per_class,
        "severe_to_good_or_fair": int(cm[3, 0] + cm[3, 1]),
        "poor_to_good": int(cm[2, 0]),
    }


@contextmanager
def canonical_evaluation_backend() -> Iterable[None]:
    """Make in-training and standalone FP32 evaluation use the same CUDA math path."""
    previous = {
        "benchmark": torch.backends.cudnn.benchmark,
        "deterministic": torch.backends.cudnn.deterministic,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "matmul_precision": torch.get_float32_matmul_precision(),
    }
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous["benchmark"]
        torch.backends.cudnn.deterministic = previous["deterministic"]
        torch.backends.cudnn.allow_tf32 = previous["cudnn_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = previous["matmul_tf32"]
        torch.set_float32_matmul_precision(previous["matmul_precision"])


def canonical_evaluation_backend_report() -> dict[str, Any]:
    return {
        "precision": "fp32",
        "float32_matmul_precision": "highest",
        "cuda_matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
    }


@torch.inference_mode()
def _evaluate_models_impl(
    candidate: nn.Module,
    baseline: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    candidate.eval()
    baseline.eval()
    candidate_cm = torch.zeros((4, 4), dtype=torch.int64)
    baseline_cm = torch.zeros((4, 4), dtype=torch.int64)
    hard_groups: dict[str, dict[str, Any]] = defaultdict(lambda: {
        "valid_pixels": 0,
        "candidate_fp": 0,
        "baseline_fp": 0,
        "samples": 0,
        "candidate_prediction_counts": [0, 0, 0, 0],
        "baseline_prediction_counts": [0, 0, 0, 0],
        "baseline_to_candidate_transition": [[0, 0, 0, 0] for _ in range(4)],
    })
    new_severe_under = 0
    new_poor_good = 0
    positive_samples = 0
    hard_samples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        candidate_logits = candidate(images)
        baseline_logits = baseline(images)
        if not torch.isfinite(candidate_logits).all() or not torch.isfinite(baseline_logits).all():
            raise FloatingPointError("Validation logits contain NaN or Inf")
        candidate_pred = candidate_logits.argmax(1)
        baseline_pred = baseline_logits.argmax(1)
        for item in range(images.shape[0]):
            kind = batch["source_type"][item]
            if kind == "hard_negative":
                hard_samples += 1
                # A hard-negative frame may still contain genuine rust.  Only
                # approved Good pixels are eligible for false-positive counts;
                # labelled rust and ignore pixels are never relabelled Good.
                valid = target[item].eq(0)
                group = hard_groups[batch["group_id"][item]]
                group["valid_pixels"] += int(valid.sum())
                group["candidate_fp"] += int((candidate_pred[item][valid] != 0).sum())
                group["baseline_fp"] += int((baseline_pred[item][valid] != 0).sum())
                group["samples"] += 1
                candidate_counts = torch.bincount(candidate_pred[item][valid].cpu(), minlength=4).tolist()
                baseline_counts = torch.bincount(baseline_pred[item][valid].cpu(), minlength=4).tolist()
                transitions = torch.bincount(
                    (baseline_pred[item][valid].to(torch.int64) * 4 + candidate_pred[item][valid].to(torch.int64)).cpu(),
                    minlength=16,
                ).reshape(4, 4).tolist()
                for predicted_class in range(4):
                    group["candidate_prediction_counts"][predicted_class] += int(candidate_counts[predicted_class])
                    group["baseline_prediction_counts"][predicted_class] += int(baseline_counts[predicted_class])
                    for candidate_class in range(4):
                        group["baseline_to_candidate_transition"][predicted_class][candidate_class] += int(
                            transitions[predicted_class][candidate_class]
                        )
            else:
                positive_samples += 1
                update_confusion(candidate_cm, target[item], candidate_pred[item])
                update_confusion(baseline_cm, target[item], baseline_pred[item])
                valid = target[item].ne(255)
                severe = valid & target[item].eq(3)
                poor = valid & target[item].eq(2)
                new_severe_under += int((severe & candidate_pred[item].le(1) & baseline_pred[item].gt(1)).sum())
                new_poor_good += int((poor & candidate_pred[item].eq(0) & baseline_pred[item].ne(0)).sum())
    group_report = {}
    for name, values in hard_groups.items():
        valid = values["valid_pixels"]
        group_report[name] = {
            **values,
            "candidate_fp_rate": _ratio(values["candidate_fp"], valid),
            "baseline_fp_rate": _ratio(values["baseline_fp"], valid),
            "worsened": values["candidate_fp"] > values["baseline_fp"],
        }
    total_valid = sum(item["valid_pixels"] for item in hard_groups.values())
    total_candidate_fp = sum(item["candidate_fp"] for item in hard_groups.values())
    total_baseline_fp = sum(item["baseline_fp"] for item in hard_groups.values())
    total_candidate_counts = [sum(item["candidate_prediction_counts"][index] for item in hard_groups.values()) for index in range(4)]
    total_baseline_counts = [sum(item["baseline_prediction_counts"][index] for item in hard_groups.values()) for index in range(4)]
    total_transitions = [
        [sum(item["baseline_to_candidate_transition"][left][right] for item in hard_groups.values()) for right in range(4)]
        for left in range(4)
    ]
    return {
        "positive_samples": positive_samples,
        "hard_negative_samples": hard_samples,
        "candidate_positive": metrics_from_confusion(candidate_cm),
        "baseline_positive": metrics_from_confusion(baseline_cm),
        "hard_negative": {
            "valid_pixels": total_valid,
            "candidate_fp": total_candidate_fp,
            "baseline_fp": total_baseline_fp,
            "candidate_fp_rate": _ratio(total_candidate_fp, total_valid),
            "baseline_fp_rate": _ratio(total_baseline_fp, total_valid),
            "all_pixels_expected_class": "Good",
            "class_order": list(CLASS_NAMES),
            "candidate_prediction_counts": total_candidate_counts,
            "baseline_prediction_counts": total_baseline_counts,
            "candidate_false_positive_by_class": dict(zip(CLASS_NAMES[1:], total_candidate_counts[1:])),
            "baseline_false_positive_by_class": dict(zip(CLASS_NAMES[1:], total_baseline_counts[1:])),
            "baseline_to_candidate_transition": total_transitions,
            "groups": group_report,
        },
        "new_major_under_calls_vs_baseline": {
            "severe_to_good_or_fair": new_severe_under,
            "poor_to_good": new_poor_good,
        },
    }


def evaluate_models(
    candidate: nn.Module,
    baseline: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    with canonical_evaluation_backend():
        return _evaluate_models_impl(candidate, baseline, loader, device)


def validation_gate(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    selection = config["selection"]
    candidate = report["candidate_positive"]
    baseline = report["baseline_positive"]
    reasons: list[str] = []
    if report["positive_samples"] == 0 or report["hard_negative_samples"] == 0:
        reasons.append("validation must contain both positive replay and hard-negative samples")
    candidate_miou = candidate["macro_iou_corrosion3"]
    baseline_miou = baseline["macro_iou_corrosion3"]
    if candidate_miou is None or baseline_miou is None or baseline_miou - candidate_miou > float(selection["positive_macro_iou_corrosion3_drop_max"]):
        reasons.append("positive macro IoU corrosion3 regression exceeded limit")
    for name in CLASS_NAMES[1:]:
        c_recall = candidate["per_class"][name]["recall"]
        b_recall = baseline["per_class"][name]["recall"]
        if c_recall is None or b_recall is None or b_recall - c_recall > float(selection["per_class_recall_drop_max"]):
            reasons.append(f"{name} recall regression exceeded limit")
    under = report["new_major_under_calls_vs_baseline"]
    if under["severe_to_good_or_fair"] > int(selection["new_severe_to_good_or_fair_max"]):
        reasons.append("new Severe->Good/Fair under-call pixels detected")
    if under["poor_to_good"] > int(selection["new_poor_to_good_max"]):
        reasons.append("new Poor->Good under-call pixels detected")
    worsened = [name for name, item in report["hard_negative"]["groups"].items() if item["worsened"]]
    if worsened and not bool(selection["hard_negative_group_may_worsen"]):
        reasons.append("hard-negative groups worsened: " + ", ".join(worsened))
    return {"passed": not reasons, "reasons": reasons, "selection_score": report["hard_negative"]["candidate_fp_rate"]}


def positive_safety_gate(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Safety-only gate for a fixed-epoch refit that uses every normal session for training."""
    selection = config["selection"]
    candidate = report["candidate_positive"]
    baseline = report["baseline_positive"]
    reasons: list[str] = []
    if report["positive_samples"] == 0:
        reasons.append("positive safety validation is empty")
    if report["hard_negative_samples"] != 0:
        reasons.append("fixed-epoch positive safety validation must not contain hard-negative samples")
    candidate_miou = candidate["macro_iou_corrosion3"]
    baseline_miou = baseline["macro_iou_corrosion3"]
    if candidate_miou is None or baseline_miou is None or baseline_miou - candidate_miou > float(selection["positive_macro_iou_corrosion3_drop_max"]):
        reasons.append("positive macro IoU corrosion3 regression exceeded limit")
    for name in CLASS_NAMES[1:]:
        c_recall = candidate["per_class"][name]["recall"]
        b_recall = baseline["per_class"][name]["recall"]
        if c_recall is None or b_recall is None or b_recall - c_recall > float(selection["per_class_recall_drop_max"]):
            reasons.append(f"{name} recall regression exceeded limit")
    under = report["new_major_under_calls_vs_baseline"]
    if under["severe_to_good_or_fair"] > int(selection["new_severe_to_good_or_fair_max"]):
        reasons.append("new Severe->Good/Fair under-call pixels detected")
    if under["poor_to_good"] > int(selection["new_poor_to_good_max"]):
        reasons.append("new Poor->Good under-call pixels detected")
    return {"passed": not reasons, "reasons": reasons, "selection_score": None}


def configured_validation_gate(report: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    mode = config.get("selection", {}).get("mode", "mixed_positive_and_hard_negative")
    if mode == "mixed_positive_and_hard_negative":
        return validation_gate(report, config)
    if mode == "positive_safety_fixed_epoch":
        return positive_safety_gate(report, config)
    raise ValueError(f"Unsupported selection mode: {mode}")


def verify_manifest_lock(config: dict[str, Any]) -> dict[str, Any]:
    lock_path = resolve_path(config["paths"]["manifest_lock"])
    if not lock_path.is_file():
        raise RuntimeError(f"Manifest lock missing; audit and lock before data smoke/training: {lock_path}")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    for key in ("config", "train_manifest", "validation_manifest", "sealed_manifest", "sealed_commitment"):
        item = lock["files"].get(key)
        if not item:
            raise RuntimeError(f"Manifest lock missing entry: {key}")
        path = resolve_path(item["path"])
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise RuntimeError(f"Locked file changed or missing: {path}")
    return lock
