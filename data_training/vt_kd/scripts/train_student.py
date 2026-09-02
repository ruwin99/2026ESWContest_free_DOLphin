from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from common import (
    CLASS_NAMES,
    PROJECT_ROOT,
    VTCSSDDataset,
    build_student,
    build_teacher,
    evaluate_student,
    load_config,
    load_split_pairs,
    metrics_from_confusion,
    resolve_project_path,
    seed_everything,
    seed_worker,
    sha256_file,
    update_confusion_matrix,
    write_json,
)


def parse_args() -> argparse.Namespace:
    default_config = PROJECT_ROOT / "data_training" / "vt_kd" / "configs" / "student_mnv2_os8.yaml"
    parser = argparse.ArgumentParser(
        description="Train the VT CSSD MobileNetV2 + DeepLabV3+ student."
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--mode", choices=("supervised", "kd"), required=True)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-name")
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--kd-weight", type=float)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False")
    return device


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("KD temperature must be greater than zero")
    per_class = F.kl_div(
        F.log_softmax(student_logits.float() / temperature, dim=1),
        F.softmax(teacher_logits.float() / temperature, dim=1),
        reduction="none",
    )
    return per_class.sum(dim=1).mean() * (temperature * temperature)


def make_loader(
    dataset: VTCSSDDataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    drop_last: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=pin_memory,
        persistent_workers=workers > 0,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer, epochs: int, warmup_epochs: int
) -> torch.optim.lr_scheduler.LambdaLR:
    def multiplier(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / float(warmup_epochs)
        decay_epochs = max(1, epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch_index - warmup_epochs) / decay_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def package_versions() -> dict[str, str | None]:
    packages = (
        "torch",
        "torchvision",
        "numpy",
        "Pillow",
        "PyYAML",
        "tensorboard",
        "onnx",
        "onnxruntime",
    )
    result: dict[str, str | None] = {}
    for package in packages:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def official_git_state(config: dict[str, Any]) -> dict[str, Any]:
    training_path = resolve_project_path(config["paths"]["official_training"])
    repository = training_path.parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        diff = subprocess.check_output(
            ["git", "-C", str(repository), "diff", "--", "Training - Testing/network/backbone"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return {
            "repository": str(repository),
            "commit": commit,
            "compatibility_patch_sha256": __import__("hashlib").sha256(
                diff.encode("utf-8")
            ).hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"repository": str(repository), "commit": None, "compatibility_patch_sha256": None}


def capture_rng_states() -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": numpy_state[1].tolist(),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_states(states: dict[str, Any]) -> None:
    if not states:
        return
    random.setstate(states["python"])
    numpy_state = states["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            np.asarray(numpy_state["state"], dtype=np.uint32),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(states["torch_cpu"])
    if torch.cuda.is_available() and states.get("torch_cuda"):
        torch.cuda.set_rng_state_all(states["torch_cuda"])


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_teacher(
    path: Path, config: dict[str, Any], device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Converted teacher checkpoint not found: {path}\n"
            "Run scripts/convert_teacher.py after explicitly reviewing the legacy pickle risk."
        )
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    required = {"state_dict", "architecture", "num_classes", "output_stride", "source_sha256"}
    missing = required - bundle.keys()
    if missing:
        raise ValueError(f"Teacher bundle is missing keys: {sorted(missing)}")
    if int(bundle["num_classes"]) != 4 or int(bundle["output_stride"]) != 8:
        raise ValueError("Teacher bundle must use 4 classes and output_stride=8")
    teacher = build_teacher(
        config, str(bundle["architecture"]), pretrained_backbone=False
    )
    teacher.load_state_dict(bundle["state_dict"], strict=True)
    teacher.to(device).eval()
    teacher.requires_grad_(False)
    return teacher, bundle


def smoke_step(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    class_weights: torch.Tensor,
    temperature: float,
    kd_weight: float,
    use_amp: bool,
    max_batches: int,
) -> dict[str, Any]:
    student.train()
    optimizer.zero_grad(set_to_none=True)
    start = time.perf_counter()
    seen = 0
    last_report: dict[str, Any] = {}
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        student_view = batch["student_view"].to(device, non_blocking=True)
        teacher_view = batch["teacher_view"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        teacher_logits = None
        if teacher is not None:
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                teacher_logits = teacher(teacher_view)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            student_logits = student(student_view)

        ce = F.cross_entropy(student_logits.float(), target, weight=class_weights)
        if teacher_logits is None:
            kd = torch.zeros((), device=device, dtype=torch.float32)
            total = ce
        else:
            if student_logits.shape != teacher_logits.shape:
                raise RuntimeError(
                    f"Teacher/student shape mismatch: {teacher_logits.shape} vs {student_logits.shape}"
                )
            kd = kd_loss(student_logits, teacher_logits, temperature)
            total = (1.0 - kd_weight) * ce + kd_weight * kd
        if target.dtype != torch.int64 or int(target.min()) < 0 or int(target.max()) > 3:
            raise RuntimeError("Target must be int64 with class values 0..3")
        if not all(bool(torch.isfinite(item)) for item in (ce, kd, total)):
            raise FloatingPointError("CE/KD/total loss contains NaN or Inf")

        scaler.scale(total).backward()
        if teacher is not None and any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("Teacher unexpectedly received gradients")
        if not any(parameter.grad is not None for parameter in student.parameters()):
            raise RuntimeError("Student did not receive gradients")
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        seen += int(target.shape[0])
        last_report = {
            "student_logits_shape": list(student_logits.shape),
            "teacher_logits_shape": list(teacher_logits.shape) if teacher_logits is not None else None,
            "target_shape": list(target.shape),
            "target_classes": sorted(int(value) for value in torch.unique(target).cpu()),
            "ce_loss": float(ce.detach().cpu()),
            "kd_loss": float(kd.detach().cpu()),
            "total_loss": float(total.detach().cpu()),
        }
    elapsed = time.perf_counter() - start
    last_report.update(
        {
            "batches": min(max_batches, len(loader)),
            "samples": seen,
            "seconds": elapsed,
            "max_cuda_memory_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    return last_report


def train_one_epoch(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module | None,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    class_weights: torch.Tensor,
    temperature: float,
    kd_weight: float,
    accumulation_steps: int,
    use_amp: bool,
) -> dict[str, Any]:
    student.train()
    if teacher is not None:
        teacher.eval()
    optimizer.zero_grad(set_to_none=True)
    confusion = torch.zeros((4, 4), dtype=torch.int64)
    totals = {"ce": 0.0, "kd": 0.0, "loss": 0.0, "samples": 0}
    start = time.perf_counter()

    for batch_index, batch in enumerate(loader):
        student_view = batch["student_view"].to(device, non_blocking=True)
        teacher_view = batch["teacher_view"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)

        teacher_logits = None
        if teacher is not None:
            with torch.inference_mode(), torch.amp.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                teacher_logits = teacher(teacher_view)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=use_amp,
        ):
            student_logits = student(student_view)

        ce = F.cross_entropy(student_logits.float(), target, weight=class_weights)
        if teacher_logits is None:
            kd = torch.zeros((), device=device, dtype=torch.float32)
            total = ce
        else:
            if student_logits.shape != teacher_logits.shape:
                raise RuntimeError(
                    f"Teacher/student shape mismatch: {teacher_logits.shape} vs {student_logits.shape}"
                )
            kd = kd_loss(student_logits, teacher_logits, temperature)
            total = (1.0 - kd_weight) * ce + kd_weight * kd
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(
                f"Non-finite training loss at batch {batch_index}: ce={float(ce)}, kd={float(kd)}"
            )

        scaler.scale(total / accumulation_steps).backward()
        is_last = batch_index + 1 == len(loader)
        if (batch_index + 1) % accumulation_steps == 0 or is_last:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        prediction = student_logits.detach().argmax(dim=1)
        update_confusion_matrix(confusion, target, prediction)
        batch_size = int(target.shape[0])
        totals["ce"] += float(ce.detach().cpu()) * batch_size
        totals["kd"] += float(kd.detach().cpu()) * batch_size
        totals["loss"] += float(total.detach().cpu()) * batch_size
        totals["samples"] += batch_size

    samples = int(totals["samples"])
    metrics = metrics_from_confusion(confusion)
    return {
        "loss": totals["loss"] / samples,
        "ce_loss": totals["ce"] / samples,
        "kd_loss": totals["kd"] / samples,
        "sample_count": samples,
        "seconds": time.perf_counter() - start,
        "metrics": metrics,
    }


def checkpoint_payload(
    *,
    student: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_value: float,
    best_severe_recall: float,
    config: dict[str, Any],
    mode: str,
    teacher_bundle: dict[str, Any] | None,
    split_hash: str,
    config_hash: str,
    official_git: dict[str, Any],
    metrics: dict[str, Any],
    epochs_without_improvement: int,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "model_name": "deeplabv3plus_mobilenet",
        "model_state_dict": student.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "best_metric_name": "val_macro_iou_corrosion3",
        "best_metric_value": best_value,
        "best_severe_recall": best_severe_recall,
        "epochs_without_improvement": epochs_without_improvement,
        "mode": mode,
        "benchmark_label": (
            "teacher-contaminated transfer benchmark"
            if mode == "kd"
            else "supervised student baseline"
        ),
        "config": config,
        "class_names": list(CLASS_NAMES),
        "num_classes": 4,
        "output_stride": 8,
        "parameter_count": 5_221_348,
        "student_preprocessing": {
            "color": "RGB",
            "scale": "0..1",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "resize": [512, 512],
        },
        "teacher_source_sha256": (
            teacher_bundle.get("source_sha256") if teacher_bundle else None
        ),
        "teacher_architecture": (
            teacher_bundle.get("architecture") if teacher_bundle else None
        ),
        "split_csv_sha256": split_hash,
        "config_sha256": config_hash,
        "official_code": official_git,
        "python_packages": package_versions(),
        "python_version": platform.python_version(),
        "pytorch_cuda_runtime": torch.version.cuda,
        "validation_metrics": metrics,
        "rng_states": capture_rng_states(),
    }


def append_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    fields = list(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    train_cfg = config["train"]
    data_cfg = config["data"]
    loss_cfg = config["loss"]

    epochs = int(args.epochs if args.epochs is not None else train_cfg["epochs"])
    seed = int(args.seed if args.seed is not None else config["seed"])
    batch_size = int(
        args.batch_size if args.batch_size is not None else train_cfg["physical_batch_size"]
    )
    workers = int(args.workers if args.workers is not None else train_cfg["num_workers"])
    temperature = float(
        args.temperature if args.temperature is not None else loss_cfg["kd_temperature"]
    )
    kd_weight = float(
        args.kd_weight if args.kd_weight is not None else loss_cfg["kd_weight"]
    )
    if args.mode == "supervised":
        kd_weight = 0.0
    if not 0.0 <= kd_weight <= 1.0:
        raise ValueError("kd_weight must be between 0 and 1")
    if epochs <= 0 or batch_size < 2 or workers < 0:
        raise ValueError(
            "epochs must be positive, workers non-negative, and physical batch-size must be "
            "at least 2 because the original DeepLabV3+ ASPP head uses BatchNorm after global pooling"
        )

    device = select_device(args.device)
    use_amp = str(train_cfg["amp"]).lower() == "fp16" and device.type == "cuda"
    seed_everything(seed)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()

    dataset_root = resolve_project_path(data_cfg["root"])
    split_csv = resolve_project_path(data_cfg["split_csv"])
    if not split_csv.is_file():
        raise FileNotFoundError(
            f"Split CSV not found: {split_csv}\nRun scripts/audit_and_split.py first."
        )
    train_pairs = load_split_pairs(split_csv, dataset_root, "train")
    val_pairs = load_split_pairs(split_csv, dataset_root, "val")
    input_size = (int(data_cfg["input_height"]), int(data_cfg["input_width"]))
    train_dataset = VTCSSDDataset(
        train_pairs, input_size=input_size, augment_horizontal_flip=True
    )
    val_dataset = VTCSSDDataset(
        val_pairs, input_size=input_size, augment_horizontal_flip=False
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        workers=workers,
        drop_last=True,
        seed=seed,
        pin_memory=device.type == "cuda",
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        workers=workers,
        drop_last=False,
        seed=seed,
        pin_memory=device.type == "cuda",
    )

    student = build_student(config, pretrained_backbone=True).to(device)
    parameter_count = sum(parameter.numel() for parameter in student.parameters())
    class_weights = torch.tensor(
        loss_cfg["class_weights"], dtype=torch.float32, device=device
    )

    teacher: torch.nn.Module | None = None
    teacher_bundle: dict[str, Any] | None = None
    if args.mode == "kd":
        teacher_path = (
            args.teacher.resolve()
            if args.teacher is not None
            else resolve_project_path(config["teacher"]["checkpoint"])
        )
        teacher, teacher_bundle = load_teacher(teacher_path, config, device)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": student.backbone.parameters(),
                "lr": float(train_cfg["backbone_lr"]),
                "name": "backbone",
            },
            {
                "params": student.classifier.parameters(),
                "lr": float(train_cfg["classifier_lr"]),
                "name": "classifier",
            },
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = make_scheduler(optimizer, epochs, int(train_cfg["warmup_epochs"]))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "device": str(device),
                "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
                "student_parameters": parameter_count,
                "train_samples": len(train_dataset),
                "val_samples": len(val_dataset),
                "batch_size": batch_size,
                "workers": workers,
                "amp": use_amp,
                "temperature": temperature,
                "kd_weight": kd_weight,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    if args.smoke_batches:
        report = smoke_step(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            class_weights=class_weights,
            temperature=temperature,
            kd_weight=kd_weight,
            use_amp=use_amp,
            max_batches=args.smoke_batches,
        )
        report["student_parameters"] = parameter_count
        report["teacher_has_gradients"] = (
            any(parameter.grad is not None for parameter in teacher.parameters())
            if teacher is not None
            else None
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    run_name = args.run_name or f"mnv2-os8-{args.mode}-seed{seed}"
    runs_root = resolve_project_path(config["paths"]["runs"])
    if args.resume is not None:
        resume_path = args.resume.resolve()
        run_dir = resume_path.parent.parent
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    else:
        run_dir = runs_root / run_name
        if run_dir.exists() and any(run_dir.iterdir()):
            raise FileExistsError(
                f"Run directory already exists; choose another --run-name or use --resume: {run_dir}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = run_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = run_dir / "config.yaml"
    if not config_snapshot.exists():
        shutil.copy2(config_path, config_snapshot)
    config_hash = sha256_file(config_snapshot)
    split_hash = sha256_file(split_csv)
    official_git = official_git_state(config)
    write_json(
        run_dir / "environment.json",
        {
            "python": platform.python_version(),
            "packages": package_versions(),
            "torch_cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "official_code": official_git,
            "split_csv": str(split_csv),
            "split_csv_sha256": split_hash,
            "config_sha256": config_hash,
        },
    )

    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(run_dir / "tensorboard"))
    except Exception as exc:
        print(f"WARNING: TensorBoard disabled: {exc}", file=sys.stderr)
        writer = None

    start_epoch = 0
    best_value = float("-inf")
    best_severe_recall = float("-inf")
    epochs_without_improvement = 0
    if args.resume is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if checkpoint.get("model_name") != "deeplabv3plus_mobilenet":
            raise ValueError("Resume checkpoint has the wrong model architecture")
        if checkpoint.get("mode") != args.mode:
            raise ValueError(
                f"Resume mode mismatch: checkpoint={checkpoint.get('mode')} requested={args.mode}"
            )
        if checkpoint.get("split_csv_sha256") != split_hash:
            raise ValueError("Split CSV hash changed; refusing to resume")
        student.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_value = float(checkpoint["best_metric_value"])
        best_severe_recall = float(checkpoint.get("best_severe_recall", -math.inf))
        epochs_without_improvement = int(
            checkpoint.get("epochs_without_improvement", 0)
        )
        restore_rng_states(checkpoint.get("rng_states", {}))

    patience = int(train_cfg["early_stopping_patience"])
    accumulation_steps = int(train_cfg["gradient_accumulation_steps"])
    for epoch in range(start_epoch, epochs):
        epoch_start = time.perf_counter()
        train_result = train_one_epoch(
            student=student,
            teacher=teacher,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            class_weights=class_weights,
            temperature=temperature,
            kd_weight=kd_weight,
            accumulation_steps=accumulation_steps,
            use_amp=use_amp,
        )
        validation = evaluate_student(
            student,
            val_loader,
            device,
            class_weights=class_weights,
            amp=use_amp,
        )
        if validation.sample_count != len(val_dataset):
            raise RuntimeError(
                f"Validation sample count mismatch: expected={len(val_dataset)} actual={validation.sample_count}"
            )
        scheduler.step()

        val_metric = validation.metrics["macro_iou_corrosion3"]
        severe_recall = validation.metrics["severe_recall"]
        if val_metric is None or severe_recall is None:
            raise RuntimeError("Validation corrosion3 mIoU or Severe recall is undefined")
        improved = val_metric > best_value + 1e-12 or (
            abs(val_metric - best_value) <= 1e-12
            and severe_recall > best_severe_recall
        )
        if improved:
            best_value = float(val_metric)
            best_severe_recall = float(severe_recall)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        payload = checkpoint_payload(
            student=student,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_value=best_value,
            best_severe_recall=best_severe_recall,
            config=config,
            mode=args.mode,
            teacher_bundle=teacher_bundle,
            split_hash=split_hash,
            config_hash=config_hash,
            official_git=official_git,
            metrics=validation.metrics,
            epochs_without_improvement=epochs_without_improvement,
        )
        atomic_torch_save(payload, weights_dir / "last.pt")
        if improved:
            atomic_torch_save(payload, weights_dir / "best.pt")

        row = {
            "epoch": epoch + 1,
            "train_loss": train_result["loss"],
            "train_ce_loss": train_result["ce_loss"],
            "train_kd_loss": train_result["kd_loss"],
            "val_loss": validation.loss,
            "val_macro_iou_corrosion3": val_metric,
            "val_macro_f1_corrosion3": validation.metrics["macro_f1_corrosion3"],
            "val_severe_recall": severe_recall,
            "val_poor_severe_recall": validation.metrics["poor_severe_recall"],
            "val_severe_under_call_count": validation.metrics["severe_under_call_count"],
            "backbone_lr": optimizer.param_groups[0]["lr"],
            "classifier_lr": optimizer.param_groups[1]["lr"],
            "epoch_seconds": time.perf_counter() - epoch_start,
            "max_cuda_memory_mib": (
                torch.cuda.max_memory_allocated() / 1024**2
                if device.type == "cuda"
                else 0.0
            ),
        }
        append_metrics_csv(run_dir / "metrics.csv", row)
        write_json(run_dir / "validation_latest.json", validation.metrics)
        if writer is not None:
            for key, value in row.items():
                if isinstance(value, (int, float)) and value is not None:
                    writer.add_scalar(key, value, epoch + 1)
            writer.flush()

        print(json.dumps(row, ensure_ascii=False))
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch + 1}: no improvement for {patience} epochs"
            )
            break

    if writer is not None:
        writer.close()
    print(f"Run complete: {run_dir}")
    print(f"Best validation corrosion3 mIoU: {best_value:.6f}")


if __name__ == "__main__":
    main()
