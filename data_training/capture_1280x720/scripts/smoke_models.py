from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from common import (
    CAPTURE_ROOT,
    CRACK_INTERNAL_HEIGHT,
    CRACK_INTERNAL_WIDTH,
    EXTERNAL_HEIGHT,
    EXTERNAL_WIDTH,
    CaptureBGCrack,
    build_capture_bgcrack,
    build_corrosion_model,
    load_config,
    project_path,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FP32 capture-model forward smoke test.")
    parser.add_argument("--model", choices=("corrosion", "crack"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the 1280x720 smoke test")
    config = load_config()
    checkpoint = (
        args.checkpoint.resolve()
        if args.checkpoint
        else project_path(config[args.model]["initial_checkpoint"])
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if args.model == "corrosion":
        model = build_corrosion_model(checkpoint).to(device).eval()
        images = torch.zeros(
            (1, 3, EXTERNAL_HEIGHT, EXTERNAL_WIDTH), dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            output = model(images)
        expected = (1, 4, EXTERNAL_HEIGHT, EXTERNAL_WIDTH)
        extra = {"preprocessing": "OpenCV BGR FP32 0..255"}
    else:
        core = build_capture_bgcrack(checkpoint)
        model = CaptureBGCrack(core).to(device).eval()
        images = torch.zeros(
            (1, 3, EXTERNAL_HEIGHT, EXTERNAL_WIDTH), dtype=torch.float32, device=device
        )
        with torch.inference_mode():
            output = model(images)
        expected = (1, 1, EXTERNAL_HEIGHT, EXTERNAL_WIDTH)
        extra = {
            "preprocessing": "RGB FP32 [-1,1]",
            "internal_shape": [1, 3, CRACK_INTERNAL_HEIGHT, CRACK_INTERNAL_WIDTH],
            "padding": {"top": 8, "bottom": 8, "value": 0.0},
            "mobilevit_feature_padding": {
                "MVT3": "none at 46x80",
                "MVT4": "temporary bottom 1 row at 23x40, cropped after block"
            },
            "dct_buffer_shapes": {
                "HFIE1_S": list(core.HFIE1_S.dct_layer.weight.shape),
                "HFIE2_S": list(core.HFIE2_S.dct_layer.weight.shape),
                "HFIE1_C": list(core.HFIE1_C.dct_layer.weight.shape),
                "HFIE2_C": list(core.HFIE2_C.dct_layer.weight.shape),
            },
        }
    torch.cuda.synchronize(device)
    if tuple(output.shape) != expected:
        raise RuntimeError(f"Unexpected output shape {tuple(output.shape)}, expected {expected}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Model output contains NaN or Inf")
    report = {
        "model": args.model,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "precision": "fp32",
        "input_shape": list(images.shape),
        "output_shape": list(output.shape),
        "output_min": float(output.min()),
        "output_max": float(output.max()),
        "finite": True,
        "max_cuda_memory_mib": torch.cuda.max_memory_allocated() / (1024**2),
        **extra,
    }
    report_path = CAPTURE_ROOT / "manifests" / f"smoke_{args.model}.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
