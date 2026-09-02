from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    PROJECT_ROOT,
    VTCSSDDataset,
    build_student,
    discover_pairs,
    evaluate_student,
    load_config,
    load_split_pairs,
    resolve_project_path,
    seed_worker,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    default_config = PROJECT_ROOT / "data_training" / "vt_kd" / "configs" / "student_mnv2_os8.yaml"
    parser = argparse.ArgumentParser(description="Evaluate a VT CSSD student checkpoint.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--split-csv", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-test-rerun",
        action="store_true",
        help="Permit reusing an output folder that already records a final Test evaluation.",
    )
    return parser.parse_args()


def select_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def write_confusion_csv(path: Path, matrix: list[list[int]]) -> None:
    class_names = ["Good", "Fair", "Poor", "Severe"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["target\\prediction", *class_names])
        for name, row in zip(class_names, matrix, strict=True):
            writer.writerow([name, *row])


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "TEST_EVALUATION_PERFORMED.json"
    if args.split == "test" and marker.exists() and not args.allow_test_rerun:
        raise RuntimeError(
            f"This output folder already contains a Test evaluation: {marker}\n"
            "Do not tune on Test. Use --allow-test-rerun only for an intentional reproducibility rerun."
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "deeplabv3plus_mobilenet":
        raise ValueError("Checkpoint is not a MobileNetV2 + DeepLabV3+ student bundle")

    dataset_root = (
        args.dataset_root.resolve()
        if args.dataset_root is not None
        else resolve_project_path(config["data"]["root"])
    )
    if args.split == "test":
        pairs = discover_pairs(dataset_root, "Test")
    else:
        split_csv = (
            args.split_csv.resolve()
            if args.split_csv is not None
            else resolve_project_path(config["data"]["split_csv"])
        )
        if checkpoint.get("split_csv_sha256") != sha256_file(split_csv):
            raise RuntimeError("Validation split CSV hash does not match the checkpoint")
        pairs = load_split_pairs(split_csv, dataset_root, "val")

    dataset = VTCSSDDataset(pairs, augment_horizontal_flip=False)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        drop_last=False,
        worker_init_fn=seed_worker,
    )
    device = select_device(args.device)
    student = build_student(config, pretrained_backbone=False)
    student.load_state_dict(checkpoint["model_state_dict"], strict=True)
    student.to(device).eval()
    class_weights = torch.tensor(
        config["loss"]["class_weights"], dtype=torch.float32, device=device
    )
    result = evaluate_student(
        student,
        loader,
        device,
        class_weights=class_weights,
        amp=False,
    )
    if result.sample_count != len(dataset):
        raise RuntimeError(
            f"Evaluation skipped samples: expected={len(dataset)} actual={result.sample_count}"
        )

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "benchmark_label": checkpoint.get("benchmark_label"),
        "sample_count": result.sample_count,
        "loss": result.loss,
        "metrics": result.metrics,
        "warning": (
            "The public teacher may have been selected using this Test set; KD results are a "
            "teacher-contaminated transfer benchmark, not an independent generalization claim."
            if checkpoint.get("mode") == "kd"
            else None
        ),
    }
    write_json(output_dir / "evaluation.json", report)
    write_confusion_csv(
        output_dir / "confusion_matrix.csv", result.metrics["confusion_matrix"]
    )
    if args.split == "test":
        write_json(
            marker,
            {
                "evaluated_at": report["evaluated_at"],
                "checkpoint_sha256": report["checkpoint_sha256"],
                "sample_count": result.sample_count,
                "policy": "Test is one-shot and must not be used for tuning.",
            },
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
