from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fail-closed CUDA/model/data preflight for hard-negative fine-tuning.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    project = config_path.parents[1]
    root = Path(os.environ.get("RAIL_ROBOT_ROOT", project.parents[1])).expanduser().resolve()
    resolve = lambda value: (root / value).resolve()
    baseline = resolve(config["paths"]["baseline_model"])
    workspace = resolve(config["paths"]["workspace"])
    data_yaml = workspace / "datasets" / "waste_detect_hn_v8_1280_seed42" / "data.yaml"
    issues: list[str] = []
    if not baseline.is_file():
        issues.append(f"baseline best.pt missing: {baseline}")
        actual_hash = None
        actual_size = None
    else:
        actual_hash = sha256(baseline)
        actual_size = baseline.stat().st_size
        if actual_hash != config["baseline_contract"]["sha256"]:
            issues.append(f"baseline SHA mismatch: {actual_hash}")
        if actual_size != int(config["baseline_contract"]["size_bytes"]):
            issues.append(f"baseline size mismatch: {actual_size}")
    if not data_yaml.is_file():
        issues.append(f"prepared data.yaml missing: {data_yaml}")
    if ultralytics.__version__ != str(config["baseline_contract"]["ultralytics"]):
        issues.append(f"Ultralytics {ultralytics.__version__} != expected {config['baseline_contract']['ultralytics']}")
    if not torch.cuda.is_available():
        issues.append("CUDA unavailable")
        gpu_name = "NO CUDA"
    else:
        gpu_name = torch.cuda.get_device_name(0)
        if "RTX 5070 Ti" not in gpu_name:
            issues.append(f"unexpected GPU: {gpu_name}")
    model_contract: dict[str, object] | None = None
    if not issues and baseline.is_file():
        model = YOLO(str(baseline))
        model_contract = {"task": model.task, "names": model.names}
        if model.task != "detect" or list(model.names.values()) != ["obstacle"]:
            issues.append(f"baseline model contract mismatch: {model_contract}")
    report = {
        "passed": not issues,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "ultralytics": ultralytics.__version__,
        "baseline_path": str(baseline),
        "baseline_size": actual_size,
        "baseline_sha256": actual_hash,
        "data_yaml": str(data_yaml),
        "model_contract": model_contract,
        "training_contract": {"imgsz": 1280, "rect": True, "camera_shape_hw": [720, 1280]},
        "issues": issues,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()


