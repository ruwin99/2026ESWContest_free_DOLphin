from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import build_model, load_config, resolve_path, sha256_file, verify_manifest_lock, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a validation-selected candidate as fixed FP32 ONNX.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def compare(reference: np.ndarray, actual: np.ndarray, config: dict[str, Any]) -> dict[str, Any]:
    difference = np.abs(reference - actual)
    agreement = float(np.mean(reference.argmax(1) == actual.argmax(1)))
    maximum = float(difference.max())
    mean = float(difference.mean())
    export = config["export"]
    passed = maximum <= float(export["max_abs_error"]) and mean <= float(export["mean_abs_error"]) and agreement >= float(export["argmax_agreement_min"])
    return {"max_abs_error": maximum, "mean_abs_error": mean, "argmax_agreement": agreement, "passed": passed}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = verify_manifest_lock(config)
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate ONNX: {output}")
    rejection_path = checkpoint_path.parent.parent / "BEST_PT_REJECTED.json"
    if rejection_path.is_file():
        rejection = json.loads(rejection_path.read_text(encoding="utf-8"))
        if rejection.get("checkpoint_sha256") == sha256_file(checkpoint_path):
            raise RuntimeError(f"Checkpoint is explicitly rejected and cannot be exported: {rejection_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Unsupported candidate checkpoint")
    if not checkpoint.get("validation_gate", {}).get("passed", False):
        raise RuntimeError("Only a validation-gated best candidate may be exported")
    if checkpoint.get("provenance", {}).get("manifest_lock", {}).get("files") != lock.get("files"):
        raise RuntimeError("Checkpoint manifest provenance does not match the current lock")
    initial = resolve_path(config["paths"]["initial_checkpoint"])
    model = build_model(config, initial)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval().cpu()
    height = int(config["export"]["height"])
    width = int(config["export"]["width"])
    y = torch.linspace(0.0, 255.0, height).reshape(1, 1, height, 1)
    x = torch.linspace(0.0, 255.0, width).reshape(1, 1, 1, width)
    probe = torch.cat((x.expand(1, 1, height, width), y.expand(1, 1, height, width), ((x + y) * 0.5)), dim=1).float()
    with torch.inference_mode():
        reference = model(probe).numpy()
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnx and onnxruntime are required in the training environment") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.onnx")
    temporary.unlink(missing_ok=True)
    try:
        torch.onnx.export(
            model, (probe,), temporary,
            input_names=["images"], output_names=["logits"],
            opset_version=int(config["export"]["opset"]),
            dynamo=False, do_constant_folding=True,
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        input_shape = [item.dim_value for item in graph.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [item.dim_value for item in graph.graph.output[0].type.tensor_type.shape.dim]
        if input_shape != [1, 3, height, width] or output_shape != [1, 4, height, width]:
            raise RuntimeError(f"Unexpected ONNX shapes: input={input_shape}, output={output_shape}")
        session = ort.InferenceSession(str(temporary), providers=["CPUExecutionProvider"])
        actual = session.run(["logits"], {"images": probe.numpy()})[0]
        if not np.isfinite(actual).all():
            raise FloatingPointError("ONNX output contains NaN or Inf")
        equivalence = compare(reference, actual, config)
        if not equivalence["passed"]:
            raise RuntimeError("PyTorch/ONNX equivalence gate failed: " + json.dumps(equivalence))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    metadata = {
        "exported_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "architecture": "DeepLabV3+ ResNet-101 output-stride 8",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "onnx_bytes": output.stat().st_size,
        "opset": int(config["export"]["opset"]),
        "input": config["model"]["input"],
        "output": config["model"]["output"],
        "pytorch_onnx_equivalence": equivalence,
        "manifest_lock": lock,
        "interpretation": "validation-selected false-positive suppression candidate; sealed accuracy is not final",
        "status": {"candidate": True, "uart_status": "NOT_FOR_UART", "deployment_status": "NOT_DEPLOYED", "accuracy": "NOT_FINAL", "tensorrt_plan": "NOT_BUILT"},
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    output.with_suffix(".sha256.txt").write_text(f"{metadata['onnx_sha256']}  {output.name}\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
