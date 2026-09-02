from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from common import load_config, resolve_path, seed_everything, write_json
from light_dualhead_96 import LightDualHead96, load_encoder_from_rust_checkpoint_strict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-non-5070ti", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    report: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "issues": [],
    }
    issues: list[str] = report["issues"]  # type: ignore[assignment]
    try:
        import onnx
        import onnxruntime as ort
        report["onnx"] = onnx.__version__
        report["onnxruntime"] = ort.__version__
        report["ort_providers"] = ort.get_available_providers()
    except Exception as error:
        issues.append(f"ONNX environment unavailable: {error}")
    cache_root = resolve_path(config["paths"]["teacher_cache"])
    cache_root.mkdir(parents=True, exist_ok=True)
    disk = shutil.disk_usage(cache_root)
    report["cache_disk"] = {"path": str(cache_root), "free_bytes": disk.free, "total_bytes": disk.total}
    if not torch.cuda.is_available():
        issues.append("BLOCKED_RTX5070TI_PREFLIGHT: torch CUDA is unavailable")
    else:
        device_name = torch.cuda.get_device_name(0)
        report["gpu"] = {"index": 0, "name": device_name, "capability": list(torch.cuda.get_device_capability(0))}
        if "5070 Ti" not in device_name and not args.allow_non_5070ti:
            issues.append(f"BLOCKED_RTX5070TI_PREFLIGHT: expected RTX 5070 Ti, got {device_name}")
        seed_everything(17)
        device = torch.device("cuda:0")
        x = torch.randn(1024, 1024, device=device)
        y = x @ x
        torch.cuda.synchronize()
        if not torch.isfinite(y).all():
            issues.append("CUDA matrix smoke produced non-finite values")
        torch.cuda.reset_peak_memory_stats(device)
        model = LightDualHead96(resolve_path(config["paths"]["official_training"]))
        load_encoder_from_rust_checkpoint_strict(
            model, resolve_path(config["paths"]["rust_initialization_checkpoint"])
        )
        model.to(device).train()
        images = torch.randn(1, 3, 240, 1280, device=device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
        optimizer.zero_grad(set_to_none=True)
        output = model(images)
        loss = output.float().square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        report["model_smoke"] = {
            "output_shape": list(output.shape),
            "loss": float(loss.detach()),
            "finite_output": bool(torch.isfinite(output).all()),
            "finite_gradients": bool(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())),
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        if output.shape != (1, 5, 240, 1280) or not torch.isfinite(output).all():
            issues.append("Full-shape model smoke failed")
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
        report["nvidia_smi"] = smi.stdout or smi.stderr
    except OSError as error:
        issues.append(f"nvidia-smi unavailable: {error}")
    report["passed"] = not issues
    output_path = resolve_path(config["paths"]["environment"]) / "preflight.json"
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
