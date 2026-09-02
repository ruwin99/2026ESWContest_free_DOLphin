from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import load_config, onnx_shape, resolve_path, sha256_file, write_json
from light_dualhead_96 import LightDualHead96


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    import onnx
    import onnxruntime as ort

    path = args.onnx.resolve()
    graph = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(graph)
    issues: list[str] = []
    inputs = {value.name: onnx_shape(value) for value in graph.graph.input}
    outputs = {value.name: onnx_shape(value) for value in graph.graph.output}
    if inputs != {"images": [1, 3, 240, 1280]}:
        issues.append(f"input mismatch: {inputs}")
    if outputs != {"multitask_logits": [1, 5, 240, 1280]}:
        issues.append(f"output mismatch: {outputs}")
    opsets = [item.version for item in graph.opset_import if item.domain in {"", "ai.onnx"}]
    if opsets != [17]:
        issues.append(f"opset mismatch: {opsets}")
    # Sigmoid is required inside the LR-ASPP attention gate. Only output-level
    # activations/post-processing are forbidden by the raw-logit contract.
    forbidden = {"Softmax", "ArgMax", "NonMaxSuppression"}
    forbidden_found = sorted({node.op_type for node in graph.graph.node} & forbidden)
    if forbidden_found:
        issues.append(f"forbidden final/postprocess ops found: {forbidden_found}")
    output_name = "multitask_logits"
    output_producers = [node.op_type for node in graph.graph.node if output_name in node.output]
    if any(value in {"Sigmoid", "Softmax", "ArgMax"} for value in output_producers):
        issues.append(f"output activation is forbidden: {output_producers}")
    for initializer in graph.graph.initializer:
        values = onnx.numpy_helper.to_array(initializer)
        if not np.isfinite(values).all():
            issues.append(f"non-finite initializer: {initializer.name}")
    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=True)
    model = LightDualHead96(resolve_path(config["paths"]["official_training"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.float().eval()
    rng = np.random.default_rng(96)
    sample = rng.standard_normal((1, 3, 240, 1280), dtype=np.float32)
    with torch.inference_mode():
        reference = model(torch.from_numpy(sample)).numpy()
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["multitask_logits"], {"images": sample})[0]
    error = np.abs(reference - actual)
    rust_argmax = float(np.mean(reference[:, :4].argmax(1) == actual[:, :4].argmax(1)))
    crack_mask = float(np.mean((reference[:, 4] >= 0.0) == (actual[:, 4] >= 0.0)))
    if not np.isfinite(actual).all():
        issues.append("non-finite ONNX output")
    if float(error[:, :4].max()) > 1e-4 or float(error[:, 4:].max()) > 1e-4:
        issues.append("PyTorch/ONNX max abs error exceeds 1e-4")
    if rust_argmax < 0.99999:
        issues.append("rust argmax pixel agreement below 99.999%")
    if crack_mask < 0.99999:
        issues.append("crack binary mask agreement below 99.999%")
    report = {
        "onnx": str(path), "onnx_sha256": sha256_file(path), "inputs": inputs, "outputs": outputs,
        "opset": opsets, "forbidden_ops": forbidden_found,
        "rust_max_abs_error": float(error[:, :4].max()), "rust_mean_abs_error": float(error[:, :4].mean()),
        "crack_max_abs_error": float(error[:, 4:].max()), "crack_mean_abs_error": float(error[:, 4:].mean()),
        "rust_argmax_pixel_agreement": rust_argmax, "crack_mask_agreement": crack_mask,
        "issues": issues, "passed": not issues,
    }
    output = path.with_suffix(path.suffix + ".verification.json")
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
