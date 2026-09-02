from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("+")[0].split(".") if part.isdigit())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if version_tuple(ultralytics.__version__) < (8, 4, 0):
        raise RuntimeError(f"Ultralytics >=8.4.0 is required, got {ultralytics.__version__}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; CPU training is intentionally blocked")

    model = YOLO(str(args.model))
    if model.task != "detect":
        raise RuntimeError(f"Expected a detection model, got {model.task}")
    device = torch.device("cuda:0")
    network = model.model.float().eval().to(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        network(torch.zeros(1, 3, 640, 640, device=device))
    torch.cuda.synchronize(device)

    model_path = args.model.resolve()
    report = {
        "passed": True,
        "python_torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_mib": round(torch.cuda.get_device_properties(0).total_memory / 2**20, 2),
        "smoke_peak_cuda_memory_mib": round(torch.cuda.max_memory_allocated(device) / 2**20, 2),
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "input": [1, 3, 640, 640],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
