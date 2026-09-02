from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import audit_manifest, load_config, read_manifest_rows, resolve_path, sha256_file, write_json
from data import MultitaskDataset
from model import build_model
from train_multitask import evaluate_epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an audited checkpoint on validation data only (never the independent locked test)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    manifest = args.manifest.resolve() if args.manifest else resolve_path(config["paths"]["val_manifest"])
    if "locked" in manifest.name.lower() or "test" in manifest.name.lower():
        raise PermissionError("The training agent is not authorized to read locked/test manifests")
    audit = audit_manifest(config, manifest, "val")
    if audit.issues:
        raise ValueError("Validation manifest audit failed:\n - " + "\n - ".join(audit.issues))
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        MultitaskDataset(read_manifest_rows(manifest), config, training=False, use_cache=False),
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    metrics = evaluate_epoch(model, loader, device, config)
    payload = {
        "scope": "validation_only_not_locked_test",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "metrics_at_probability_0_5": metrics,
    }
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
