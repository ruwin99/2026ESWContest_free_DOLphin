from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import assert_ready, load_config, resolve_path, sha256_file, write_json
from light_dualhead_96 import LightDualHead96


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the selected FP32 raw-logit model with static axes.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    readiness = assert_ready(config_path)
    config = load_config(config_path)
    if not config.get("status", {}).get("export_authorized"):
        raise RuntimeError("Export blocked: set status.export_authorized only after validation and independent approval")
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint.get("config_sha256") != readiness["config_sha256"]:
        raise ValueError("Checkpoint/config SHA mismatch")
    model = LightDualHead96(resolve_path(config["paths"]["official_training"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.float().eval()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, 3, 240, 1280, dtype=torch.float32)
    torch.onnx.export(
        model, example, output, input_names=["images"], output_names=["multitask_logits"],
        opset_version=17, do_constant_folding=True, dynamic_axes=None, dynamo=False,
    )
    import onnx

    exported = onnx.load(str(output), load_external_data=True)
    expected_output_shape = (1, 5, 240, 1280)
    graph_output = next((value for value in exported.graph.output if value.name == "multitask_logits"), None)
    if graph_output is None:
        raise RuntimeError("Exported ONNX is missing the multitask_logits output")
    dimensions = graph_output.type.tensor_type.shape.dim
    if len(dimensions) != len(expected_output_shape):
        raise RuntimeError(f"Unexpected ONNX output rank: {len(dimensions)}")
    # Legacy TorchScript export leaves Resize-derived output dimensions symbolic
    # even though this deployment contract is fully static. Record the verified
    # fixed contract explicitly so TensorRT and downstream tooling see the real
    # [1, 5, 240, 1280] output shape.
    for dimension, value in zip(dimensions, expected_output_shape):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    onnx.checker.check_model(exported)
    onnx.save_model(exported, str(output))
    sidecar = {
        "model_role": "fixed_camera_printed-defect_demo_only",
        "input": config["contract"]["input"],
        "preprocess": {key: config["contract"][key] for key in ("color", "scale", "mean", "std")},
        "output": {**config["contract"]["output"], "activation": "none"},
        "rust_channels": {"0": "Good", "1": "Fair", "2": "Poor", "3": "Severe"},
        "crack_channel": 4, "crack_valid_rows": config["contract"]["crack_valid_rows"],
        "output_contract_version": "light-dualhead96-raw5-v1",
        "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
        "config_sha256": readiness["config_sha256"], "onnx_sha256": sha256_file(output),
        "thresholds": config["evaluation"],
        "phase": config["status"].get("phase"), "accuracy_status": "ACCURACY_NOT_FINAL",
        "deployment_status": "NOT_FOR_UART", "actuator_authorization": "PROHIBITED",
    }
    write_json(output.with_suffix(output.suffix + ".json"), sidecar)
    print(json.dumps(sidecar, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
