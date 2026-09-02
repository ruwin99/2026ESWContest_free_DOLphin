from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import sha256_file


def shape(value) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the static 5-channel ONNX contract.")
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    if bool(args.config) != bool(args.checkpoint):
        raise ValueError("--config and --checkpoint must be provided together")
    if args.image and not args.checkpoint:
        raise ValueError("--image requires --config and --checkpoint")
    path = args.onnx.resolve()
    import onnx

    model = onnx.load(str(path), load_external_data=True)
    onnx.checker.check_model(model)
    inputs = {item.name: shape(item) for item in model.graph.input}
    outputs = {item.name: shape(item) for item in model.graph.output}
    if inputs != {"images": [1, 3, 240, 1280]}:
        raise ValueError(f"Input contract mismatch: {inputs}")
    if outputs != {"logits": [1, 5, 240, 1280]}:
        raise ValueError(f"Output contract mismatch: {outputs}")
    result = {
        "passed": True,
        "onnx": str(path),
        "sha256": sha256_file(path),
        "inputs": inputs,
        "outputs": outputs,
        "opsets": [{"domain": item.domain, "version": int(item.version)} for item in model.opset_import],
    }
    if args.run_inference:
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        output = session.run(["logits"], {"images": np.zeros((1, 3, 240, 1280), np.float32)})[0]
        result["runtime_shape"] = list(output.shape)
        result["runtime_finite"] = bool(np.isfinite(output).all())
        if output.shape != (1, 5, 240, 1280) or not result["runtime_finite"]:
            raise RuntimeError(f"ONNX Runtime smoke failed: {result}")
    if args.checkpoint:
        import torch
        from PIL import Image

        from common import load_config, sha256_file as common_sha256
        from model import build_model

        config_path = args.config.resolve()
        checkpoint_path = args.checkpoint.resolve()
        config = load_config(config_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("config_sha256") != common_sha256(config_path):
            raise ValueError("Checkpoint/config SHA mismatch")
        candidate = build_model(config)
        candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
        candidate.eval()
        if args.image:
            with Image.open(args.image.resolve()) as source:
                rgb = source.convert("RGB")
                if rgb.size != (1280, 720):
                    raise ValueError(f"Expected 1280x720 parity image, got {rgb.size}")
                values = np.asarray(rgb.crop((0, 0, 1280, 240)), dtype=np.float32) / 255.0
            mean = np.asarray(config["contracts"]["mean"], dtype=np.float32).reshape(1, 1, 3)
            std = np.asarray(config["contracts"]["std"], dtype=np.float32).reshape(1, 1, 3)
            values = ((values - mean) / std).transpose(2, 0, 1)[None]
        else:
            values = np.zeros((1, 3, 240, 1280), dtype=np.float32)
        values = np.ascontiguousarray(values, dtype=np.float32)
        with torch.inference_mode():
            torch_output = candidate(torch.from_numpy(values)).numpy()
        import onnxruntime as ort

        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_output = session.run(["logits"], {"images": values})[0]
        difference = np.abs(torch_output - ort_output)
        max_abs = float(difference.max())
        mean_abs = float(difference.mean())
        parity_passed = max_abs <= 5e-4 and mean_abs <= 1e-5
        result["pytorch_onnx_parity"] = {
            "passed": parity_passed,
            "image": str(args.image.resolve()) if args.image else None,
            "max_abs_error": max_abs,
            "mean_abs_error": mean_abs,
            "max_abs_tolerance": 5e-4,
            "mean_abs_tolerance": 1e-5,
        }
        if not parity_passed:
            raise RuntimeError(f"PyTorch/ONNX parity failed: {result['pytorch_onnx_parity']}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
