from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate

from common import assert_ready, load_config, resolve_path, write_json
from dataset import DemoManifestDataset
from light_dualhead_96 import LightDualHead96, freeze_batchnorm_running_stats, load_encoder_from_rust_checkpoint_strict, set_stage
from training_core import compute_losses, finite_losses, move_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="One public-positive + one normal-negative real Phase A batch smoke.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    assert_ready(config_path)
    config = load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dataset = DemoManifestDataset(
        resolve_path(config["paths"]["train_manifest"]),
        resolve_path(config["paths"]["teacher_cache"]) / "_smoke",
        augment=False,
    )
    positive_index = next(i for i, row in enumerate(dataset.rows) if row["scenario"] != "clean")
    normal_index = next(i for i, row in enumerate(dataset.rows) if row["scenario"] == "clean")
    batch = default_collate([dataset[positive_index], dataset[normal_index]])
    device = torch.device("cuda:0")
    batch = move_batch(batch, device)
    model = LightDualHead96(resolve_path(config["paths"]["official_training"]))
    load_encoder_from_rust_checkpoint_strict(model, resolve_path(config["paths"]["rust_initialization_checkpoint"]))
    set_stage(model, 3)
    model.to(device).train()
    freeze_batchnorm_running_stats(model)
    torch.cuda.reset_peak_memory_stats(device)
    output = model(batch["image"])
    losses = compute_losses(output, batch, config)
    if output.shape != (2, 5, 240, 1280) or not finite_losses(losses):
        raise RuntimeError("Phase A real-batch shape/finite smoke failed")
    losses["total"].backward()
    finite_gradients = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    if not finite_gradients:
        raise RuntimeError("Phase A real-batch gradient is non-finite")
    report = {
        "phase": "PHASE_A_DEVELOPMENT_ONLY", "result_labels": ["ACCURACY_NOT_FINAL", "NOT_FOR_UART"],
        "sample_ids": batch["sample_id"], "output_shape": list(output.shape),
        "losses": {key: float(value.detach()) for key, value in losses.items()},
        "finite_gradients": True, "max_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
    }
    output_path = resolve_path(config["paths"]["reports"]) / "phase_a_real_batch_smoke.json"
    write_json(output_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
