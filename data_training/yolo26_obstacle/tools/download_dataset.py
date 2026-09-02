from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the private Roboflow YOLO26 obstacle dataset.")
    parser.add_argument("--location", type=Path, required=True)
    args = parser.parse_args()
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise RuntimeError("ROBOFLOW_API_KEY is not set in this PowerShell session")
    location = args.location.resolve()
    if (location / "data.yaml").exists():
        print(f"Dataset already exists: {location}")
        return
    location.parent.mkdir(parents=True, exist_ok=True)

    from roboflow import Roboflow

    dataset = (
        Roboflow(api_key=key)
        .workspace("-ohs3h")
        .project("2-iemaw")
        .version(1)
        .download("yolo26", location=str(location))
    )
    print(f"Dataset ready: {dataset.location}")


if __name__ == "__main__":
    main()
