from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

from common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the exact training Python/GPU environment.")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    freeze_result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    )
    if freeze_result.returncode == 0:
        freeze = freeze_result.stdout
    else:
        freeze = "\n".join(
            sorted(
                f"{distribution.metadata['Name']}=={distribution.version}"
                for distribution in importlib.metadata.distributions()
                if distribution.metadata.get("Name")
            )
        ) + "\n"
    (output / "pip-freeze.txt").write_text(freeze, encoding="utf-8")
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    try:
        import torch

        torch_info = {
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
        }
    except ImportError:
        torch_info = None
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": torch_info,
        "gpu": gpu.stdout.strip() if gpu.returncode == 0 else None,
    }
    write_json(output / "environment.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
