from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from common import (
    RustManifestDataset,
    build_model,
    evaluate_models,
    load_config,
    read_manifest,
    resolve_path,
    sha256_file,
    verify_manifest_lock,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-shot evaluation on the locked normal-only sealed set."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-frame sealed evaluation")

    config = load_config(args.config)
    lock = verify_manifest_lock(config)
    checkpoint_path = args.checkpoint.resolve()
    output = args.output.resolve()
    receipt = output.with_suffix(".unseal.json")
    if output.exists() or receipt.exists():
        raise FileExistsError(
            f"Sealed evaluation is one-shot; output or unseal receipt already exists: {output} / {receipt}"
        )

    commitment_path = resolve_path(config["paths"]["sealed_commitment"])
    commitment = yaml.safe_load(commitment_path.read_text(encoding="utf-8"))
    if commitment.get("status") != "APPROVED_LOCKED":
        raise RuntimeError("Sealed commitment is not APPROVED_LOCKED")
    if not bool(commitment.get("unseal_once")):
        raise RuntimeError("Sealed commitment does not authorize one-shot unsealing")
    if bool(commitment.get("prior_model_inference_on_sealed_images")):
        raise RuntimeError("Commitment records prior inference on sealed images")

    sealed_path = resolve_path(config["paths"]["sealed_manifest"])
    sealed_sha = sha256_file(sealed_path)
    if sealed_sha != str(commitment.get("sealed_manifest_sha256", "")).lower():
        raise RuntimeError("Sealed manifest no longer matches its commitment")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Unsupported candidate checkpoint")
    if not checkpoint.get("validation_gate", {}).get("passed", False):
        raise RuntimeError("Only a validation-gated best candidate may unseal the holdout")
    if checkpoint.get("provenance", {}).get("manifest_lock", {}).get("files") != lock.get("files"):
        raise RuntimeError("Checkpoint manifest provenance does not match the current lock")

    initial = resolve_path(config["paths"]["initial_checkpoint"])
    baseline_onnx = resolve_path(config["paths"]["baseline_onnx"])
    frozen = {
        "candidate_checkpoint": str(checkpoint_path),
        "candidate_checkpoint_sha256": sha256_file(checkpoint_path),
        "baseline_state_dict": str(initial),
        "baseline_state_dict_sha256": sha256_file(initial),
        "baseline_onnx": str(baseline_onnx),
        "baseline_onnx_sha256": sha256_file(baseline_onnx),
        "sealed_manifest": str(sealed_path),
        "sealed_manifest_sha256": sealed_sha,
        "sealed_commitment": str(commitment_path),
        "sealed_commitment_sha256": sha256_file(commitment_path),
        "config": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config.resolve()),
        "acceptance_gate": {
            "candidate_total_false_positive_must_not_exceed_baseline": True,
            "candidate_group_false_positive_must_not_exceed_baseline": True,
            "threshold_tuning": False,
            "model_selection": False,
        },
    }

    # Write this before loading any sealed image. If evaluation is interrupted,
    # the receipt remains and deliberately prevents a second unseal.
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        receipt,
        {
            "unsealed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "UNSEALED_EVALUATION_STARTED",
            "authorization": "explicit_user_request_2026-08-20_v8_validation",
            "frozen_before_inference": frozen,
        },
    )

    rows = read_manifest(sealed_path, expected_split="sealed_test")
    if not rows or any(row["source_type"] != "hard_negative" for row in rows):
        raise RuntimeError("This sealed evaluator requires a non-empty normal-only hard-negative set")

    candidate = build_model(config, initial)
    candidate.load_state_dict(checkpoint["model_state_dict"], strict=True)
    baseline = build_model(config, initial)
    device = torch.device("cuda:0")
    candidate.to(device).eval()
    baseline.to(device).eval()
    loader = DataLoader(
        RustManifestDataset(rows, config, training=False),
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    metrics = evaluate_models(candidate, baseline, loader, device)
    hard = metrics["hard_negative"]
    worsened_groups = [name for name, item in hard["groups"].items() if item["worsened"]]
    reasons: list[str] = []
    if int(metrics["positive_samples"]) != 0:
        reasons.append("sealed set unexpectedly contained positive samples")
    if int(metrics["hard_negative_samples"]) != len(rows):
        reasons.append("not all sealed rows were evaluated as hard-negative")
    if int(hard["candidate_fp"]) > int(hard["baseline_fp"]):
        reasons.append("candidate total false positives exceeded baseline")
    if worsened_groups:
        reasons.append("candidate worsened sealed groups: " + ", ".join(worsened_groups))
    gate = {
        "passed": not reasons,
        "reasons": reasons,
        "candidate_fp_must_not_exceed_baseline": True,
        "hard_negative_group_may_worsen": False,
    }
    report = {
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_kind": "ONE_SHOT_LOCKED_SEALED_NORMAL_ONLY",
        "frozen_before_inference": frozen,
        "samples": len(rows),
        "metrics": {"hard_negative": hard},
        "gate": gate,
        "interpretation": "false-positive suppression only; positive sealed set is missing",
        "restrictions": {
            "may_be_used_for_training": False,
            "may_be_used_for_model_selection": False,
            "may_be_used_for_threshold_tuning": False,
            "full_corrosion_accuracy": "NOT_MEASURED",
        },
        "status": {
            "candidate": True,
            "uart_status": "NOT_FOR_UART",
            "deployment_status": "NOT_DEPLOYED",
            "accuracy": "NOT_FINAL",
        },
    }
    write_json(output, report)
    write_json(
        receipt,
        {
            "unsealed_utc": json.loads(receipt.read_text(encoding="utf-8"))["unsealed_utc"],
            "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "UNSEALED_EVALUATION_COMPLETED",
            "authorization": "explicit_user_request_2026-08-20_v8_validation",
            "frozen_before_inference": frozen,
            "report": str(output),
            "report_sha256": sha256_file(output),
            "gate_passed": gate["passed"],
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
