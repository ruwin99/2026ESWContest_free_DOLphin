from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    CLASS_NAMES,
    PROJECT_ROOT,
    VTCSSDDataset,
    build_student,
    load_config,
    load_split_pairs,
    metrics_from_confusion,
    resolve_project_path,
    sha256_file,
    update_confusion_matrix,
    write_json,
)


def parse_args() -> argparse.Namespace:
    default_config = PROJECT_ROOT / "data_training" / "vt_kd" / "configs" / "student_mnv2_os8.yaml"
    parser = argparse.ArgumentParser(description="Export a VT CSSD student to FP32 ONNX.")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--input-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--verify-samples", type=int, default=8)
    parser.add_argument("--skip-runtime-verification", action="store_true")
    return parser.parse_args()


def verify_runtime(
    *,
    model: torch.nn.Module,
    onnx_path: Path,
    config: dict[str, Any],
    sample_limit: int,
) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required for export equivalence checks") from exc

    dataset_root = resolve_project_path(config["data"]["root"])
    split_csv = resolve_project_path(config["data"]["split_csv"])
    pairs = load_split_pairs(split_csv, dataset_root, "val")[:sample_limit]
    dataset = VTCSSDDataset(pairs, augment_horizontal_flip=False)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    max_abs_error = 0.0
    sum_abs_error = 0.0
    value_count = 0
    matching_pixels = 0
    total_pixels = 0
    pytorch_confusion = torch.zeros((4, 4), dtype=torch.int64)
    onnx_confusion = torch.zeros((4, 4), dtype=torch.int64)

    model.cpu().eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            sample = dataset[index]
            input_tensor = sample["student_view"].unsqueeze(0).float()
            target = sample["target"].unsqueeze(0)
            pytorch_logits = model(input_tensor).cpu().numpy()
            onnx_logits = session.run(
                ["logits"], {"images": input_tensor.numpy()}
            )[0]
            difference = np.abs(pytorch_logits - onnx_logits)
            max_abs_error = max(max_abs_error, float(difference.max()))
            sum_abs_error += float(difference.sum())
            value_count += int(difference.size)
            pytorch_prediction = torch.from_numpy(pytorch_logits.argmax(axis=1))
            onnx_prediction = torch.from_numpy(onnx_logits.argmax(axis=1))
            matching_pixels += int((pytorch_prediction == onnx_prediction).sum())
            total_pixels += int(pytorch_prediction.numel())
            update_confusion_matrix(pytorch_confusion, target, pytorch_prediction)
            update_confusion_matrix(onnx_confusion, target, onnx_prediction)

    pytorch_metrics = metrics_from_confusion(pytorch_confusion)
    onnx_metrics = metrics_from_confusion(onnx_confusion)
    metric_keys = (
        "macro_iou_corrosion3",
        "severe_recall",
        "poor_severe_recall",
    )
    metric_differences: dict[str, float | None] = {}
    for key in metric_keys:
        left = pytorch_metrics[key]
        right = onnx_metrics[key]
        metric_differences[key] = (
            abs(float(left) - float(right))
            if left is not None and right is not None
            else None
        )
    defined_differences = [
        value for value in metric_differences.values() if value is not None
    ]
    maximum_metric_difference = max(defined_differences, default=0.0)
    agreement = matching_pixels / total_pixels
    new_severe_under_calls = max(
        0,
        int(onnx_metrics["severe_under_call_count"])
        - int(pytorch_metrics["severe_under_call_count"]),
    )
    passed = (
        agreement >= 0.999
        and maximum_metric_difference <= 0.001
        and new_severe_under_calls == 0
    )
    return {
        "samples": len(dataset),
        "max_abs_logit_error": max_abs_error,
        "mean_abs_logit_error": sum_abs_error / value_count,
        "argmax_pixel_agreement": agreement,
        "metric_differences": metric_differences,
        "maximum_metric_difference": maximum_metric_difference,
        "new_severe_under_calls": new_severe_under_calls,
        "gate": {
            "argmax_pixel_agreement_min": 0.999,
            "metric_difference_max": 0.001,
            "new_severe_under_calls_max": 0,
            "passed": passed,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    config = load_config(args.config.resolve())
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("--output must end with .onnx")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Student checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("model_name") != "deeplabv3plus_mobilenet":
        raise ValueError("Checkpoint architecture is not deeplabv3plus_mobilenet")

    model = build_student(config, pretrained_backbone=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.cpu().eval()
    height, width = args.input_size
    dummy = torch.zeros((1, 3, height, width), dtype=torch.float32)
    with torch.inference_mode():
        probe = model(dummy)
    if tuple(probe.shape) != (1, 4, height, width):
        raise RuntimeError(f"Unexpected model output shape: {tuple(probe.shape)}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        output_path,
        input_names=["images"],
        output_names=["logits"],
        opset_version=args.opset,
        dynamo=True,
    )
    try:
        import onnx

        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
    except ImportError as exc:
        raise RuntimeError("onnx is required to validate the exported graph") from exc

    actual_opsets = {
        item.domain or "ai.onnx": int(item.version)
        for item in onnx_model.opset_import
    }
    exporter_mode = "dynamo"
    if actual_opsets.get("ai.onnx") != args.opset:
        # PyTorch 2.12's dynamo exporter may emit opset 18 and fail its automatic
        # downgrade because ONNX has no Pad adapter to opset 17. Keep the guide's
        # fixed opset-17 deployment contract by using the maintained legacy path
        # only when the dynamo path demonstrably did not honor the request.
        print(
            "WARNING: dynamo exporter produced "
            f"opset {actual_opsets.get('ai.onnx')} instead of {args.opset}; "
            "retrying with the legacy exporter for exact opset control.",
            file=sys.stderr,
        )
        torch.onnx.export(
            model,
            (dummy,),
            output_path,
            input_names=["images"],
            output_names=["logits"],
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
        )
        onnx_model = onnx.load(str(output_path))
        onnx.checker.check_model(onnx_model)
        actual_opsets = {
            item.domain or "ai.onnx": int(item.version)
            for item in onnx_model.opset_import
        }
        exporter_mode = "legacy_fallback_after_dynamo_opset_mismatch"
    if actual_opsets.get("ai.onnx") != args.opset:
        raise RuntimeError(
            f"Exported ONNX opset mismatch: requested={args.opset}, actual={actual_opsets}"
        )

    equivalence = None
    if not args.skip_runtime_verification:
        equivalence = verify_runtime(
            model=model,
            onnx_path=output_path,
            config=config,
            sample_limit=args.verify_samples,
        )
        if not equivalence["gate"]["passed"]:
            raise RuntimeError(
                "ONNX export failed the PyTorch equivalence gate: "
                + json.dumps(equivalence, ensure_ascii=False)
            )

    package_dir = output_path.parent
    metadata = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "model_name": "deeplabv3plus_mobilenet",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": output_path.name,
        "onnx_sha256": sha256_file(output_path),
        "opset": actual_opsets,
        "requested_opset": args.opset,
        "exporter_mode": exporter_mode,
        "input": {"name": "images", "shape": [1, 3, height, width], "dtype": "float32"},
        "output": {"name": "logits", "shape": [1, 4, height, width], "dtype": "float32"},
        "softmax_argmax_inside_model": False,
        "benchmark_label": checkpoint.get("benchmark_label"),
        "equivalence": equivalence,
    }
    preprocessing = {
        "color": "RGB",
        "scale": "0..1",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "resize": [height, width],
        "resize_policy": "fixed shape; deployment crop/letterbox/tile policy must be validated separately",
    }
    write_json(package_dir / "mnv2-deeplabv3plus-os8-metadata.json", metadata)
    write_json(package_dir / "classes.json", {"classes": list(CLASS_NAMES)})
    write_json(package_dir / "preprocessing.json", preprocessing)
    write_json(package_dir / "evaluation-summary.json", equivalence or {"skipped": True})
    (package_dir / "sha256.txt").write_text(
        f"{sha256_file(output_path)}  {output_path.name}\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
