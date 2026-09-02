from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OFFICIAL_CODE,
    MetricSums,
    SteelCrackDataset,
    build_bgcrack,
    json_dump,
    normalize_state_dict_keys,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Steelcrack BGCrack checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("Validation", "Test"), required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--official-code", type=Path, default=DEFAULT_OFFICIAL_CODE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-test-rerun", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")

    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output.resolve()
    marker = checkpoint_path.parent.parent / "test_evaluation.json"
    if args.split == "Test" and marker.exists() and not args.allow_test_rerun:
        raise FileExistsError(
            f"Test was already evaluated for this run: {marker}. "
            "Use --allow-test-rerun only if a repeat is intentional."
        )

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    state_dict = normalize_state_dict_keys(state_dict)

    device = torch.device("cuda:0")
    model = build_bgcrack(args.official_code.resolve())
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    dataset = SteelCrackDataset(args.data_root.resolve(), args.split, include_edge=False)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    metrics = MetricSums()
    with torch.inference_mode():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, enabled=args.amp):
                body, _, _ = model(images)
            metrics.update(body, masks)

    report = {
        "evaluated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "threshold": 0.5,
        "metrics": metrics.averages(),
        "warning": (
            "Test is for final reporting only; do not select hyperparameters from this result."
            if args.split == "Test"
            else None
        ),
    }
    json_dump(output_dir / "evaluation.json", report)
    if args.split == "Test":
        json_dump(marker, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
