from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    RustManifestDataset,
    build_model,
    canonical_evaluation_backend_report,
    evaluate_models,
    load_config,
    read_manifest,
    resolve_path,
    sha256_file,
    configured_validation_gate,
    verify_manifest_lock,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only baseline/candidate comparison.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-frame validation")
    config = load_config(args.config)
    lock = verify_manifest_lock(config)
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Unsupported candidate checkpoint")
    if checkpoint.get("provenance", {}).get("manifest_lock", {}).get("files") != lock.get("files"):
        raise RuntimeError("Checkpoint manifest provenance does not match the current lock")
    initial = resolve_path(config["paths"]["initial_checkpoint"])
    candidate = build_model(config, initial)
    candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    baseline = build_model(config, initial)
    device = torch.device("cuda:0")
    candidate.to(device).eval()
    baseline.to(device).eval()
    rows = read_manifest(resolve_path(config["paths"]["validation_manifest"]), expected_split="validation")
    loader = DataLoader(RustManifestDataset(rows, config, training=False), batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0)
    metrics = evaluate_models(candidate, baseline, loader, device)
    gate = configured_validation_gate(metrics, config)
    report = {
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_manifest": str(resolve_path(config["paths"]["validation_manifest"])),
        "validation_manifest_sha256": sha256_file(resolve_path(config["paths"]["validation_manifest"])),
        "evaluation_backend": canonical_evaluation_backend_report(),
        "metrics": metrics,
        "gate": gate,
        "interpretation": (
            "fixed-epoch positive safety check; normal false-positive performance is not measured here"
            if config.get("selection", {}).get("mode") == "positive_safety_fixed_epoch"
            else "false-positive suppression candidate; sealed accuracy is not final"
        ),
        "status": {"candidate": True, "uart_status": "NOT_FOR_UART", "deployment_status": "NOT_DEPLOYED", "accuracy": "NOT_FINAL"},
    }
    write_json(args.output.resolve(), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
