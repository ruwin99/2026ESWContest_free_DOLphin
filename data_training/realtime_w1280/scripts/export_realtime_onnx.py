from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
import warnings
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUST_HEIGHT = 240
CRACK_HEIGHT = 128
INPUT_WIDTH = 1280
DCT_BUFFER_KEYS = {
    "HFIE1_S.dct_layer.weight",
    "HFIE2_S.dct_layer.weight",
    "HFIE1_C.dct_layer.weight",
    "HFIE2_C.dct_layer.weight",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a fixed-shape realtime strip model to a single FP32 ONNX file."
    )
    parser.add_argument("--model", choices=("rust", "crack"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def load_local_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in payload:
                payload = payload[key]
                break
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"Checkpoint does not contain a state_dict: {path}")
    state = dict(payload)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    return state


def build_rust(checkpoint: Path) -> nn.Module:
    common = load_local_module(
        "_realtime_vt_common",
        PROJECT_ROOT / "data_training" / "vt_kd" / "scripts" / "common.py",
    )
    config = common.load_config(
        PROJECT_ROOT
        / "data_training"
        / "vt_kd"
        / "configs"
        / "student_mnv2_os8.yaml"
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("model_name") != "deeplabv3plus_mobilenet":
        raise ValueError("Rust checkpoint is not a MobileNetV2 DeepLabV3+ student bundle")
    model = common.build_student(config, pretrained_backbone=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval()


def build_crack(checkpoint: Path, capture_common: ModuleType) -> nn.Module:
    official_root = (
        PROJECT_ROOT / "data_training" / "steelcrack" / "official_bgcrack"
    )
    if not (official_root / "Model" / "BGCrack.py").is_file():
        raise FileNotFoundError(f"BGCrack source not found: {official_root}")
    official_text = str(official_root.resolve())
    if official_text not in sys.path:
        sys.path.insert(0, official_text)

    import Model.Module.utils as official_utils

    official_utils.Map_2_Grad = capture_common.DeviceSafeMapToGrad
    from Model.BGCrack import BGCrack

    model = BGCrack()
    capture_common._replace_dct_buffer(model.HFIE1_S, 32, 320, spatial=True)
    capture_common._replace_dct_buffer(model.HFIE2_S, 16, 160, spatial=True)
    capture_common._replace_dct_buffer(model.HFIE1_C, 32, 320, spatial=False)
    capture_common._replace_dct_buffer(model.HFIE2_C, 16, 160, spatial=False)

    state = checkpoint_state(checkpoint)
    filtered = {key: value for key, value in state.items() if key not in DCT_BUFFER_KEYS}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if set(missing) != DCT_BUFFER_KEYS or unexpected:
        raise RuntimeError(
            "Unexpected BGCrack checkpoint mismatch: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.eval()


class CrackProbability(nn.Module):
    def __init__(self, core: nn.Module) -> None:
        super().__init__()
        self.core = core

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        body_probability, _, _ = self.core(images)
        return body_probability


def patterned_input(model_name: str, height: int) -> torch.Tensor:
    y = torch.linspace(0.0, 1.0, height).reshape(1, 1, height, 1)
    x = torch.linspace(0.0, 1.0, INPUT_WIDTH).reshape(1, 1, 1, INPUT_WIDTH)
    rgb01 = torch.cat(
        (
            (x + 0.0 * y).expand(1, 1, height, INPUT_WIDTH),
            (y + 0.0 * x).expand(1, 1, height, INPUT_WIDTH),
            (x + y) * 0.5,
        ),
        dim=1,
    )
    if model_name == "crack":
        return rgb01 * 2.0 - 1.0
    mean = torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1)
    return (rgb01 - mean) / std


def compare(left: np.ndarray, right: np.ndarray, model_name: str) -> dict[str, Any]:
    difference = np.abs(left - right)
    if model_name == "rust":
        agreement = float(np.mean(left.argmax(axis=1) == right.argmax(axis=1)))
        agreement_name = "argmax_pixel_agreement"
    else:
        agreement = float(np.mean((left >= 0.5) == (right >= 0.5)))
        agreement_name = "binary_pixel_agreement_at_0_5"
    maximum = float(difference.max())
    mean = float(difference.mean())
    passed = maximum <= 5e-4 and mean <= 1e-5 and agreement >= 0.999
    return {
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        agreement_name: agreement,
        "gate": {
            "max_abs_error_max": 5e-4,
            "mean_abs_error_max": 1e-5,
            "pixel_decision_agreement_min": 0.999,
            "passed": passed,
        },
    }


@torch.inference_mode()
def run_model(model: nn.Module, inputs: torch.Tensor, device: torch.device) -> np.ndarray:
    model.to(device).eval()
    output = model(inputs.to(device)).float()
    if not torch.isfinite(output).all():
        raise FloatingPointError("Model output contains NaN or Inf")
    return output.cpu().numpy()


def shape_of(value: Any) -> list[int]:
    return [dimension.dim_value for dimension in value.type.tensor_type.shape.dim]


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    warnings.filterwarnings(
        "ignore",
        message=r"`nn\.functional\.upsample` is deprecated.*",
        category=UserWarning,
    )
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if output.suffix.lower() != ".onnx":
        raise ValueError("--output must end in .onnx")

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnx and onnxruntime are required") from exc

    height = RUST_HEIGHT if args.model == "rust" else CRACK_HEIGHT
    output_channels = 4 if args.model == "rust" else 1
    output_name = "logits" if args.model == "rust" else "crack_probability"
    probe = patterned_input(args.model, height)
    capture_common = None
    if args.model == "rust":
        model = build_rust(checkpoint)
    else:
        capture_common = load_local_module(
            "_realtime_capture_common",
            PROJECT_ROOT
            / "data_training"
            / "capture_1280x720"
            / "scripts"
            / "common.py",
        )
        model = CrackProbability(build_crack(checkpoint, capture_common))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    original_reference = run_model(model, probe, device)
    expected_shape = (1, output_channels, height, INPUT_WIDTH)
    if original_reference.shape != expected_shape:
        raise RuntimeError(f"Unexpected model output shape: {original_reference.shape}")

    fft_rewrite = None
    if args.model == "crack":
        assert capture_common is not None
        core = model.core
        core.b3_FFT = capture_common.TensorRTFriendlyFFTBlock(core.b3_FFT, 8, 80)
        core.b4_FFT = capture_common.TensorRTFriendlyFFTBlock(core.b4_FFT, 4, 40)
        rewritten_reference = run_model(model, probe, device)
        fft_rewrite = compare(original_reference, rewritten_reference, args.model)
        if not fft_rewrite["gate"]["passed"]:
            raise RuntimeError("FFT rewrite equivalence failed: " + json.dumps(fft_rewrite))
    else:
        rewritten_reference = original_reference

    peak_cuda_memory_mib = (
        float(torch.cuda.max_memory_allocated() / (1024**2))
        if device.type == "cuda"
        else None
    )
    model.cpu().eval()
    with torch.inference_mode():
        export_reference = model(probe).float().numpy()

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
            raise RuntimeError(f"Unexpected ONNX opset: {opsets}")
        input_shape = shape_of(graph.graph.input[0])
        output_shape = shape_of(graph.graph.output[0])
        output_shape_annotation_repaired = False
        if input_shape != [1, 3, height, INPUT_WIDTH]:
            raise RuntimeError(f"Unexpected ONNX input shape: {input_shape}")
        if output_shape != list(expected_shape):
            output_dimensions = graph.graph.output[0].type.tensor_type.shape.dim
            if len(output_dimensions) != len(expected_shape):
                raise RuntimeError(f"Unexpected ONNX output rank: {output_shape}")
            for dimension, value in zip(output_dimensions, expected_shape, strict=True):
                dimension.ClearField("dim_param")
                dimension.dim_value = value
            onnx.save(graph, str(temporary))
            graph = onnx.load(str(temporary))
            onnx.checker.check_model(graph)
            output_shape = shape_of(graph.graph.output[0])
            output_shape_annotation_repaired = True
        if output_shape != list(expected_shape):
            raise RuntimeError(f"Unexpected repaired ONNX output shape: {output_shape}")
        operators = sorted({node.op_type for node in graph.graph.node})
        forbidden = sorted(set(operators) & {"DFT", "Mod"})
        if forbidden:
            raise RuntimeError(f"Forbidden ONNX operators remain: {forbidden}")

        session = ort.InferenceSession(
            str(temporary), providers=["CPUExecutionProvider"]
        )
        actual = session.run([output_name], {"images": probe.numpy()})[0]
        if actual.shape != expected_shape or not np.isfinite(actual).all():
            raise RuntimeError(f"Invalid ONNX Runtime output: {actual.shape}")
        runtime_equivalence = compare(export_reference, actual, args.model)
        if not runtime_equivalence["gate"]["passed"]:
            raise RuntimeError(
                "ONNX Runtime equivalence failed: "
                + json.dumps(runtime_equivalence, ensure_ascii=False)
            )
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    metadata = {
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "role": "realtime bottom-strip inference",
        "model": args.model,
        "architecture": (
            "DeepLabV3+ MobileNetV2 output-stride 8"
            if args.model == "rust"
            else "BGCrack V1 H128 no-input-padding"
        ),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "onnx_bytes": output.stat().st_size,
        "single_file_onnx": True,
        "opset": opsets,
        "input": {
            "name": "images",
            "dtype": "float32",
            "shape": [1, 3, height, INPUT_WIDTH],
            "camera_crop": "frame[480:720, :]" if args.model == "rust" else "frame[592:720, :]",
            "external_resize": False,
            "external_padding": False,
            "preprocessing": (
                "RGB /255 then ImageNet mean/std"
                if args.model == "rust"
                else "RGB /127.5 - 1.0"
            ),
        },
        "output": {
            "name": output_name,
            "dtype": "float32",
            "shape": list(expected_shape),
            "class_order": ["Good", "Fair", "Poor", "Severe"]
            if args.model == "rust"
            else None,
            "sigmoid_already_applied": args.model == "crack",
            "threshold": 0.5 if args.model == "crack" else None,
        },
        "crack_dct_shapes": {
            "HFIE1": [32, 320],
            "HFIE2": [16, 160],
        }
        if args.model == "crack"
        else None,
        "crack_feature_shapes": {
            "stride16": [8, 80],
            "stride32": [4, 40],
        }
        if args.model == "crack"
        else None,
        "fixed_shape": True,
        "output_shape_annotation_repaired": output_shape_annotation_repaired,
        "contains_onnx_dft": False,
        "contains_onnx_mod": False,
        "operator_types": operators,
        "fft_rewrite_equivalence": fft_rewrite,
        "onnx_runtime_equivalence": runtime_equivalence,
        "max_cuda_memory_mib": peak_cuda_memory_mib,
        "real_camera_accuracy": "unverified",
        "tensor_rt_policy": "Build and validate the TensorRT plan on the target Jetson.",
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    output.with_suffix(".sha256.txt").write_text(
        f"{metadata['onnx_sha256']}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
