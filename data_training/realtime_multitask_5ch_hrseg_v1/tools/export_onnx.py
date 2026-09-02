from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import (
    assert_training_ready,
    load_config,
    resolve_path,
    sha256_file,
    write_json,
)
from model import build_model, load_rust_checkpoint_strict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export fixed-shape raw 5-channel logits ONNX.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--evaluation-report", type=Path)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--structure-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite: {output}")
    model = build_model(config)

    if args.structure_only:
        rust_checkpoint = resolve_path(config["paths"]["rust_checkpoint"])
        rust_metadata = load_rust_checkpoint_strict(model, rust_checkpoint)
        checkpoint_sha = None
        label = "STRUCTURE_ONLY_NOT_FOR_ACCURACY_OR_CONTROL"
    else:
        readiness = assert_training_ready(config_path)
        status = config.get("status", {})
        checkpoint_path = args.checkpoint.resolve()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("architecture") != config["student"]["architecture"]:
            raise ValueError("Checkpoint architecture mismatch")
        if checkpoint.get("config_sha256") != readiness["config_sha256"]:
            raise ValueError("Checkpoint/config SHA mismatch")
        evaluation = None
        if not status.get("locked_test_passed") or not status.get("final_export_authorized"):
            if args.evaluation_report is None:
                raise PermissionError(
                    "Final export is not authorized. A passed demo evaluation report is "
                    "required for a clearly labelled demo-only export."
                )
            evaluation_path = args.evaluation_report.resolve()
            evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
            if evaluation.get("status") != "passed_for_demo_onnx_export":
                raise PermissionError("Demo evaluation report did not pass")
            if evaluation.get("checkpoint_sha256") != sha256_file(checkpoint_path):
                raise ValueError("Demo evaluation/checkpoint SHA mismatch")
            if evaluation.get("config_sha256") != readiness["config_sha256"]:
                raise ValueError("Demo evaluation/config SHA mismatch")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        rust_metadata = None
        checkpoint_sha = sha256_file(checkpoint_path)
        label = (
            "FINAL_LOCKED_TEST_APPROVED"
            if status.get("locked_test_passed") and status.get("final_export_authorized")
            else "DEMO_ONLY_CAMERA_NORMAL_AND_PUBLIC_CRACK_VALIDATED"
        )

    model.eval()
    example = torch.zeros((1, 3, 240, 1280), dtype=torch.float32)
    with torch.no_grad():
        output_tensor = model(example)
    if tuple(output_tensor.shape) != (1, 5, 240, 1280):
        raise RuntimeError(f"Model output shape mismatch: {tuple(output_tensor.shape)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        example,
        str(output),
        input_names=["images"],
        output_names=["logits"],
        opset_version=int(config["export"]["opset"]),
        do_constant_folding=True,
        dynamo=False,
    )
    import onnx

    graph = onnx.load(str(output))
    for dimension, value in zip(
        graph.graph.input[0].type.tensor_type.shape.dim, (1, 3, 240, 1280)
    ):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    for dimension, value in zip(
        graph.graph.output[0].type.tensor_type.shape.dim, (1, 5, 240, 1280)
    ):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))
    payload = {
        "schema_version": 1,
        "label": label,
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "checkpoint_sha256": checkpoint_sha,
        "rust_initialization_checkpoint": config["paths"]["rust_checkpoint"] if args.structure_only else None,
        "input": {"name": "images", "shape": [1, 3, 240, 1280], "dtype": "float32"},
        "output": {"name": "logits", "shape": [1, 5, 240, 1280], "dtype": "float32"},
        "channels": ["Good", "Fair", "Poor", "Severe", "Crack"],
        "raw_logits_only": True,
        "evaluation_report": (
            str(args.evaluation_report.resolve()) if args.evaluation_report is not None else None
        ),
        "evaluation_report_sha256": (
            sha256_file(args.evaluation_report.resolve()) if args.evaluation_report is not None else None
        ),
        "operating_point": (
            evaluation.get("selected_operating_point") if not args.structure_only and evaluation else None
        ),
        "limitations": (
            evaluation.get("limitations", []) if not args.structure_only and evaluation else []
        ),
    }
    write_json(output.with_suffix(".metadata.json"), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
