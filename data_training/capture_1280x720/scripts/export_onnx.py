from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    CAPTURE_ROOT,
    EXTERNAL_HEIGHT,
    EXTERNAL_WIDTH,
    CaptureBGCrack,
    build_capture_bgcrack,
    build_corrosion_model,
    load_config,
    project_path,
    rewrite_bgcrack_fft_for_onnx,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a fixed 1280x720 capture ONNX.")
    parser.add_argument("--model", choices=("corrosion", "crack"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--skip-runtime-verification", action="store_true")
    return parser.parse_args()


def load_model(task: str, checkpoint: Path) -> torch.nn.Module:
    config = load_config()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    is_capture = isinstance(payload, dict) and "model_state_dict" in payload
    if task == "corrosion":
        initial = project_path(config["corrosion"]["initial_checkpoint"])
        model = build_corrosion_model(initial if is_capture else checkpoint)
    else:
        initial = project_path(config["crack"]["initial_checkpoint"])
        model = CaptureBGCrack(build_capture_bgcrack(initial if is_capture else checkpoint))
    if is_capture:
        if payload.get("task") != task:
            raise ValueError("Capture checkpoint task mismatch")
        model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def patterned_input(task: str) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, EXTERNAL_HEIGHT).reshape(1, 1, EXTERNAL_HEIGHT, 1)
    x = torch.linspace(0.0, 1.0, EXTERNAL_WIDTH).reshape(1, 1, 1, EXTERNAL_WIDTH)
    image = torch.cat(
        (
            (x + 0.0 * y).expand(1, 1, EXTERNAL_HEIGHT, EXTERNAL_WIDTH),
            (y + 0.0 * x).expand(1, 1, EXTERNAL_HEIGHT, EXTERNAL_WIDTH),
            ((x + y) * 0.5),
        ),
        dim=1,
    )
    return image * (255.0 if task == "corrosion" else 2.0) - (
        0.0 if task == "corrosion" else 1.0
    )


def compare(left: np.ndarray, right: np.ndarray, task: str) -> dict[str, Any]:
    difference = np.abs(left - right)
    if task == "corrosion":
        agreement = float(np.mean(left.argmax(axis=1) == right.argmax(axis=1)))
    else:
        agreement = float(np.mean((left >= 0.5) == (right >= 0.5)))
    maximum = float(difference.max())
    mean = float(difference.mean())
    passed = maximum <= 5e-4 and mean <= 1e-5 and agreement >= 0.999
    return {
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "pixel_decision_agreement": agreement,
        "gate": {
            "max_abs_error_max": 5e-4,
            "mean_abs_error_max": 1e-5,
            "pixel_decision_agreement_min": 0.999,
            "passed": passed,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.suffix.lower() != ".onnx":
        raise ValueError("Output must end with .onnx")
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required") from exc
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required") from exc

    model = load_model(args.model, checkpoint)
    probe = patterned_input(args.model)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    with torch.inference_mode():
        reference = model(probe.to(device)).float().cpu().numpy()

    fft_rewrite = None
    if args.model == "crack":
        rewrite_bgcrack_fft_for_onnx(model)
        model.to(device).eval()
        with torch.inference_mode():
            rewritten = model(probe.to(device)).float().cpu().numpy()
        fft_rewrite = compare(reference, rewritten, args.model)
        if not fft_rewrite["gate"]["passed"]:
            raise RuntimeError("FFT rewrite failed: " + json.dumps(fft_rewrite))

    model.cpu().eval()
    with torch.inference_mode():
        export_reference = model(probe).float().cpu().numpy()
    output_name = "logits" if args.model == "corrosion" else "crack_probability"
    expected_output_shape = (
        (1, 4, EXTERNAL_HEIGHT, EXTERNAL_WIDTH)
        if args.model == "corrosion"
        else (1, 1, EXTERNAL_HEIGHT, EXTERNAL_WIDTH)
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.stem + ".tmp.onnx")
    temporary.unlink(missing_ok=True)
    try:
        torch.onnx.export(
            model,
            (probe,),
            temporary,
            input_names=["images"],
            output_names=[output_name],
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
        )
        graph = onnx.load(str(temporary))
        onnx.checker.check_model(graph)
        opsets = {
            item.domain or "ai.onnx": int(item.version) for item in graph.opset_import
        }
        if opsets.get("ai.onnx") != args.opset:
            raise RuntimeError(f"Unexpected opset: {opsets}")
        input_shape = [dimension.dim_value for dimension in graph.graph.input[0].type.tensor_type.shape.dim]
        output_shape = [dimension.dim_value for dimension in graph.graph.output[0].type.tensor_type.shape.dim]
        if input_shape != [1, 3, EXTERNAL_HEIGHT, EXTERNAL_WIDTH]:
            raise RuntimeError(f"Unexpected fixed input shape: {input_shape}")
        output_shape_annotation_repaired = False
        if output_shape != list(expected_output_shape) and args.model == "crack":
            output_dimensions = graph.graph.output[0].type.tensor_type.shape.dim
            if len(output_dimensions) != len(expected_output_shape):
                raise RuntimeError(f"Unexpected output rank: {output_shape}")
            for dimension, value in zip(output_dimensions, expected_output_shape, strict=True):
                dimension.ClearField("dim_param")
                dimension.dim_value = value
            onnx.save(graph, str(temporary))
            graph = onnx.load(str(temporary))
            onnx.checker.check_model(graph)
            output_shape = [
                dimension.dim_value
                for dimension in graph.graph.output[0].type.tensor_type.shape.dim
            ]
            output_shape_annotation_repaired = True
        if output_shape != list(expected_output_shape):
            raise RuntimeError(f"Unexpected fixed output shape: {output_shape}")
        operators = sorted({node.op_type for node in graph.graph.node})
        if "DFT" in operators:
            raise RuntimeError("ONNX graph still contains DFT")

        runtime = None
        if not args.skip_runtime_verification:
            session = ort.InferenceSession(
                str(temporary), providers=["CPUExecutionProvider"]
            )
            actual = session.run([output_name], {"images": probe.numpy()})[0]
            if actual.shape != expected_output_shape or not np.isfinite(actual).all():
                raise RuntimeError(f"Invalid ONNX Runtime output: {actual.shape}")
            runtime = compare(export_reference, actual, args.model)
            if not runtime["gate"]["passed"]:
                raise RuntimeError("ONNX equivalence failed: " + json.dumps(runtime))
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task": args.model,
        "architecture": (
            "DeepLabV3+ ResNet-101 output-stride 8"
            if args.model == "corrosion"
            else "BGCrack V1 external720 internal736 minpad8"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "onnx_bytes": output.stat().st_size,
        "opset": opsets,
        "input": {
            "name": "images",
            "dtype": "float32",
            "shape": [1, 3, EXTERNAL_HEIGHT, EXTERNAL_WIDTH],
            "preprocessing": (
                "OpenCV BGR 0..255" if args.model == "corrosion" else "RGB [-1,1]"
            ),
        },
        "output": {
            "name": output_name,
            "dtype": "float32",
            "shape": list(expected_output_shape),
            "class_order": list(load_config()["corrosion"]["classes"])
            if args.model == "corrosion"
            else None,
            "sigmoid_already_applied": args.model == "crack",
            "threshold": 0.5 if args.model == "crack" else None,
        },
        "fixed_shape": True,
        "output_shape_annotation_repaired": output_shape_annotation_repaired,
        "valid_output_pixels": EXTERNAL_HEIGHT * EXTERNAL_WIDTH,
        "contains_dft": False,
        "operator_types": operators,
        "fft_rewrite_equivalence": fft_rewrite,
        "onnx_runtime_equivalence": runtime,
        "tensor_rt_policy": "Build the plan on the target Jetson; verify FP32 first.",
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    output.with_suffix(".sha256.txt").write_text(
        f"{metadata['onnx_sha256']}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
