from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import CAPTURE_ROOT, CaptureDataset, load_config, project_path, sha256_file, write_json
from train_capture import build_model, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained capture checkpoint.")
    parser.add_argument("--model", choices=("corrosion", "crack"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--data-root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--allow-test-rerun", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("task") != args.model:
        raise ValueError("Checkpoint task mismatch")
    marker = checkpoint_path.parent.parent / f"test_evaluation_{args.model}.json"
    if args.split == "test" and marker.exists() and not args.allow_test_rerun:
        raise FileExistsError(f"Locked Test was already evaluated: {marker}")
    config = load_config()
    initial = project_path(config[args.model]["initial_checkpoint"])
    model = build_model(args.model, initial)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device("cuda:0")
    model.to(device).eval()
    task_dir = args.model.replace("corrosion", "rust")
    dataset = CaptureDataset(args.data_root.resolve(), task_dir, args.split, False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    report = {
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "task": args.model,
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics": evaluate(args.model, model, loader, device),
    }
    write_json(args.output.resolve(), report)
    if args.split == "test":
        write_json(marker, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
