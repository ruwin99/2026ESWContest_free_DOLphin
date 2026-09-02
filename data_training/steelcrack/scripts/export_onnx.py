from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OFFICIAL_CODE,
    SteelCrackDataset,
    build_bgcrack,
    json_dump,
    normalize_state_dict_keys,
    sha256_file,
)


DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[3]
    / "output"
    / "models"
    / "steelcrack"
    / "bgcrack-steelcrack-512-fp32.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Steelcrack BGCrack checkpoint to TensorRT-friendly ONNX."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--official-code", type=Path, default=DEFAULT_OFFICIAL_CODE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--input-size", type=int, nargs=2, default=(512, 512))
    parser.add_argument("--verify-samples", type=int, default=1)
    parser.add_argument("--skip-runtime-verification", action="store_true")
    return parser.parse_args()


def _dft_basis(length: int, frequency_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    frequencies = torch.arange(frequency_count, dtype=torch.float32).unsqueeze(1)
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(0)
    angles = (2.0 * math.pi / length) * frequencies * positions
    return torch.cos(angles), torch.sin(angles)


class TensorRTFriendlyFFTBlock(nn.Module):
    """Exact fixed-size real-valued replacement for the official FFT block.

    ONNX/TensorRT do not reliably support the complex-valued rfft2/irfft2 path.
    The two transforms are therefore expressed as real MatMul operations while
    the trained frequency-domain layers and convolution layers remain unchanged.
    """

    def __init__(self, original: nn.Module, height: int, width: int) -> None:
        super().__init__()
        if height % 2 or width % 2:
            raise ValueError("The TensorRT-friendly FFT replacement requires even sizes")
        if getattr(original, "norm", None) != "forward":
            raise ValueError("Only the official norm='forward' FFT block is supported")

        self.former3 = original.former3
        self.conv4 = original.conv4
        self.height = height
        self.width = width
        self.width_frequencies = width // 2 + 1

        cos_h, sin_h = _dft_basis(height, height)
        cos_w_forward, sin_w_forward = _dft_basis(width, self.width_frequencies)
        cos_w_full, sin_w_full = _dft_basis(width, width)
        negative_height = torch.cat(
            (torch.zeros(1, dtype=torch.long), torch.arange(height - 1, 0, -1))
        )

        self.register_buffer("cos_h", cos_h, persistent=False)
        self.register_buffer("sin_h", sin_h, persistent=False)
        self.register_buffer("cos_w_forward", cos_w_forward, persistent=False)
        self.register_buffer("sin_w_forward", sin_w_forward, persistent=False)
        self.register_buffer("cos_w_full", cos_w_full, persistent=False)
        self.register_buffer("sin_w_full", sin_w_full, persistent=False)
        self.register_buffer("negative_height", negative_height, persistent=False)

    def _rfft2(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        real_width = torch.matmul(x, self.cos_w_forward.transpose(0, 1))
        imag_width = -torch.matmul(x, self.sin_w_forward.transpose(0, 1))

        real_width = real_width.transpose(-2, -1)
        imag_width = imag_width.transpose(-2, -1)
        real = (
            torch.matmul(real_width, self.cos_h.transpose(0, 1))
            + torch.matmul(imag_width, self.sin_h.transpose(0, 1))
        ).transpose(-2, -1)
        imag = (
            torch.matmul(imag_width, self.cos_h.transpose(0, 1))
            - torch.matmul(real_width, self.sin_h.transpose(0, 1))
        ).transpose(-2, -1)
        scale = 1.0 / (self.height * self.width)
        return real * scale, imag * scale

    def _restore_full_spectrum(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        real_tail = real.index_select(-2, self.negative_height)[..., 1:-1].flip(-1)
        imag_tail = -imag.index_select(-2, self.negative_height)[..., 1:-1].flip(-1)
        return torch.cat((real, real_tail), dim=-1), torch.cat((imag, imag_tail), dim=-1)

    def _irfft2(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        real, imag = self._restore_full_spectrum(real, imag)
        real_width = (
            torch.matmul(real, self.cos_w_full)
            - torch.matmul(imag, self.sin_w_full)
        )
        imag_width = (
            torch.matmul(real, self.sin_w_full)
            + torch.matmul(imag, self.cos_w_full)
        )

        real_width = real_width.transpose(-2, -1)
        imag_width = imag_width.transpose(-2, -1)
        output = (
            torch.matmul(real_width, self.cos_h)
            - torch.matmul(imag_width, self.sin_h)
        ).transpose(-2, -1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != (self.height, self.width):
            raise ValueError(
                f"Expected FFT feature size {(self.height, self.width)}, "
                f"got {tuple(x.shape[-2:])}"
            )
        real, imag = self._rfft2(x)
        frequency = self.former3(torch.cat((real, imag), dim=1))
        real, imag = torch.chunk(frequency, 2, dim=1)
        return self.conv4(self._irfft2(real, imag))


class CrackProbabilityExport(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        body, _, _ = self.model(images)
        return body


def rewrite_fft_for_tensorrt(model: nn.Module, height: int, width: int) -> None:
    if height % 32 or width % 32:
        raise ValueError("Input height and width must both be divisible by 32")
    model.b3_FFT = TensorRTFriendlyFFTBlock(model.b3_FFT, height // 16, width // 16)
    model.b4_FFT = TensorRTFriendlyFFTBlock(model.b4_FFT, height // 32, width // 32)


def load_state_dict(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        payload = payload["model_state_dict"]
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Checkpoint does not contain a model state_dict")
    return normalize_state_dict_keys(payload)


def validation_inputs(
    data_root: Path, sample_count: int
) -> tuple[list[torch.Tensor], list[str]]:
    if sample_count < 1:
        raise ValueError("--verify-samples must be at least 1")
    dataset = SteelCrackDataset(data_root, "Validation", include_edge=False)
    count = min(sample_count, len(dataset))
    tensors: list[torch.Tensor] = []
    names: list[str] = []
    for index in range(count):
        image, _, name = dataset[index]
        tensors.append(image.unsqueeze(0).float())
        names.append(name)
    return tensors, names


def run_pytorch(model: nn.Module, inputs: list[torch.Tensor]) -> list[np.ndarray]:
    results: list[np.ndarray] = []
    model.cpu().eval()
    with torch.inference_mode():
        for item in inputs:
            output = model(item)
            if isinstance(output, tuple):
                output = output[0]
            results.append(output.detach().cpu().numpy())
    return results


def compare_outputs(left: list[np.ndarray], right: list[np.ndarray]) -> dict[str, Any]:
    max_abs_error = 0.0
    sum_abs_error = 0.0
    value_count = 0
    matching_pixels = 0
    total_pixels = 0
    for left_item, right_item in zip(left, right, strict=True):
        difference = np.abs(left_item - right_item)
        max_abs_error = max(max_abs_error, float(difference.max()))
        sum_abs_error += float(difference.sum())
        value_count += int(difference.size)
        left_mask = left_item >= 0.5
        right_mask = right_item >= 0.5
        matching_pixels += int(np.count_nonzero(left_mask == right_mask))
        total_pixels += int(left_mask.size)
    mean_abs_error = sum_abs_error / value_count
    binary_pixel_agreement = matching_pixels / total_pixels
    passed = (
        max_abs_error <= 5e-4
        and mean_abs_error <= 1e-5
        and binary_pixel_agreement >= 0.999
    )
    return {
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "binary_pixel_agreement_at_0_5": binary_pixel_agreement,
        "gate": {
            "max_abs_error_max": 5e-4,
            "mean_abs_error_max": 1e-5,
            "binary_pixel_agreement_min": 0.999,
            "passed": passed,
        },
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_path = args.output.resolve()
    official_code = args.official_code.resolve()
    data_root = args.data_root.resolve()
    height, width = args.input_size

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError("--output must end with .onnx")
    if (height, width) != (512, 512):
        raise ValueError("This trained BGCrack deployment contract requires 512x512 input")

    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("onnx is required; rerun setup_steelcrack.ps1") from exc
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("onnxruntime is required; rerun setup_steelcrack.ps1") from exc

    state_dict = load_state_dict(checkpoint_path)
    model = build_bgcrack(official_code)
    model.load_state_dict(state_dict, strict=True)
    model.cpu().eval()

    inputs, sample_names = validation_inputs(data_root, args.verify_samples)
    original_outputs = run_pytorch(model, inputs)

    rewrite_fft_for_tensorrt(model, height, width)
    export_model = CrackProbabilityExport(model).cpu().eval()
    rewritten_outputs = run_pytorch(export_model, inputs)
    fft_rewrite_equivalence = compare_outputs(original_outputs, rewritten_outputs)
    if not fft_rewrite_equivalence["gate"]["passed"]:
        raise RuntimeError(
            "TensorRT-friendly FFT rewrite failed equivalence: "
            + json.dumps(fft_rewrite_equivalence, ensure_ascii=False)
        )

    dummy = torch.zeros((1, 3, height, width), dtype=torch.float32)
    with torch.inference_mode():
        probe = export_model(dummy)
    if tuple(probe.shape) != (1, 1, height, width):
        raise RuntimeError(f"Unexpected export output shape: {tuple(probe.shape)}")
    if not torch.isfinite(probe).all():
        raise RuntimeError("Export model produced NaN or Inf")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(output_path.stem + ".tmp.onnx")
    temporary_path.unlink(missing_ok=True)
    try:
        torch.onnx.export(
            export_model,
            (dummy,),
            temporary_path,
            input_names=["images"],
            output_names=["crack_probability"],
            opset_version=args.opset,
            dynamo=False,
            do_constant_folding=True,
        )

        onnx_model = onnx.load(str(temporary_path))
        onnx.checker.check_model(onnx_model)
        actual_opsets = {
            item.domain or "ai.onnx": int(item.version)
            for item in onnx_model.opset_import
        }
        if actual_opsets.get("ai.onnx") != args.opset:
            raise RuntimeError(
                f"ONNX opset mismatch: requested={args.opset}, actual={actual_opsets}"
            )

        operator_types = sorted({node.op_type for node in onnx_model.graph.node})
        forbidden_operators = sorted(set(operator_types) & {"DFT"})
        if forbidden_operators:
            raise RuntimeError(
                "TensorRT-incompatible complex FFT operators remain: "
                + ", ".join(forbidden_operators)
            )

        runtime_equivalence = None
        if not args.skip_runtime_verification:
            session = ort.InferenceSession(
                str(temporary_path), providers=["CPUExecutionProvider"]
            )
            onnx_outputs = [
                session.run(
                    ["crack_probability"], {"images": item.numpy()}
                )[0]
                for item in inputs
            ]
            runtime_equivalence = compare_outputs(original_outputs, onnx_outputs)
            if not runtime_equivalence["gate"]["passed"]:
                raise RuntimeError(
                    "ONNX Runtime output failed PyTorch equivalence: "
                    + json.dumps(runtime_equivalence, ensure_ascii=False)
                )

        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise

    metadata = {
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_name": "BGCrack Steelcrack",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "onnx": str(output_path),
        "onnx_sha256": sha256_file(output_path),
        "onnx_bytes": output_path.stat().st_size,
        "exporter": "torch.onnx legacy tracer",
        "opset": actual_opsets,
        "input": {
            "name": "images",
            "shape": [1, 3, height, width],
            "dtype": "float32",
        },
        "output": {
            "name": "crack_probability",
            "shape": [1, 1, height, width],
            "dtype": "float32",
            "range": [0.0, 1.0],
            "threshold": 0.5,
        },
        "preprocessing": {
            "color": "RGB",
            "resize": [height, width],
            "scale": "uint8 / 255.0",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
            "equivalent_range": [-1.0, 1.0],
        },
        "fixed_shape": True,
        "fp32_export": True,
        "fft_rewrite": "rfft2/irfft2 rewritten as real MatMul operations",
        "contains_onnx_dft": False,
        "operator_types": operator_types,
        "verification_samples": sample_names,
        "fft_rewrite_equivalence": fft_rewrite_equivalence,
        "onnx_runtime_equivalence": runtime_equivalence,
        "deployment_note": (
            "Build the TensorRT engine on the target Jetson using its installed "
            "JetPack/TensorRT version. FP16 TensorRT inference may be enabled there."
        ),
    }
    metadata_path = output_path.with_suffix(".metadata.json")
    json_dump(metadata_path, metadata)
    output_path.with_suffix(".sha256.txt").write_text(
        f"{metadata['onnx_sha256']}  {output_path.name}\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
