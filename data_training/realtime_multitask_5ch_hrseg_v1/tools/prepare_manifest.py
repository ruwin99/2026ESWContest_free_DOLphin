from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from common import IMAGE_SUFFIXES, audit_readiness, load_config, resolve_path, write_json
from model import build_model, load_rust_checkpoint_strict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit candidate assets and user-authored train/val manifests. "
            "This tool never invents provenance groups or locked splits."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--allow-blocked", action="store_true", help="Return exit code 0 while reporting blockers."
    )
    return parser.parse_args()


def scan_images(path: Path) -> dict[str, Any]:
    dimensions: Counter[str] = Counter()
    candidates = 0
    count = 0
    if path.is_dir():
        for item in path.rglob("*"):
            if not item.is_file() or item.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            relative_parts = [part.lower() for part in item.relative_to(path).parts[:-1]]
            if relative_parts and not any(part in {"images", "images_512"} for part in relative_parts):
                continue
            try:
                with Image.open(item) as image:
                    width, height = int(image.width), int(image.height)
                dimensions[f"{width}x{height}"] += 1
                candidates += int(width >= 1280 and height >= 240)
                count += 1
            except OSError:
                continue
    return {
        "path": str(path),
        "files": count,
        "w1280_h240_candidates": candidates,
        "dimensions": dict(dimensions.most_common()),
    }


def git_identity(path: Path) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {
            "is_repository": False,
            "commit": None,
            "note": "git executable is unavailable; Git identity is optional for this non-repository workspace",
        }
    return {
        "is_repository": result.returncode == 0,
        "commit": result.stdout.strip() if result.returncode == 0 else None,
        "note": result.stderr.strip() if result.returncode != 0 else None,
    }


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    report = audit_readiness(config_path)

    try:
        model = build_model(config)
        checkpoint = load_rust_checkpoint_strict(
            model, resolve_path(config["paths"]["rust_checkpoint"])
        )
        strict_rust = {
            "passed": True,
            "model_name": checkpoint.get("model_name"),
            "parameter_tensors": len(checkpoint["model_state_dict"]),
        }
    except Exception as error:
        strict_rust = {"passed": False, "error": str(error)}
        report["issues"].append(f"strict rust initialization failed: {error}")
        report["ready_for_training"] = False

    sources = config["data"]["candidate_sources"]
    candidate_audit: dict[str, Any] = {
        "cssd_original": scan_images(resolve_path(sources["cssd_original"])),
        "steelcrack_512": scan_images(resolve_path(sources["steelcrack_512"])),
        "demo_normal_roots": [scan_images(resolve_path(path)) for path in sources["demo_normal_roots"]],
    }
    candidate_audit["notes"] = [
        "Steelcrack 512x512 is not camera-native W1280 crack-positive GT and cannot authorize training.",
        "Demo-normal folders contain crack-negative images only; they cannot replace crack-positive GT.",
        "CSSD images require user-audited crop/provenance/group metadata before manifest admission.",
    ]
    report["strict_rust_initialization"] = strict_rust
    report["candidate_assets"] = candidate_audit
    report["workspace_git"] = git_identity(resolve_path("."))

    destination = (
        args.report.resolve()
        if args.report
        else resolve_path(config["paths"]["metrics"]) / "readiness_audit.json"
    )
    write_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print(f"\nReadiness report: {destination}")
    if not report["ready_for_training"] and not args.allow_blocked:
        sys.exit(2)


if __name__ == "__main__":
    main()
