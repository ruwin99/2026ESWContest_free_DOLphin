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
    canonical_evaluation_backend_report,
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
        description="Evaluate one frozen v7-refit candidate on the reused v8 normal confirmation set."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-frame v8 confirmation")
    config = load_config(args.config)
    if config.get("sealed", {}).get("commitment_status") != "APPROVED_REUSED_CONFIRMATION":
        raise RuntimeError("This evaluator only accepts the explicit reused-v8 confirmation profile")
    lock = verify_manifest_lock(config)
    output = args.output.resolve()
    receipt = output.with_suffix(".receipt.json")
    if output.exists() or receipt.exists():
        raise FileExistsError("Refusing to repeat v8 confirmation for this output")

    commitment_path = resolve_path(config["paths"]["sealed_commitment"])
    commitment = yaml.safe_load(commitment_path.read_text(encoding="utf-8"))
    if commitment.get("status") != "APPROVED_REUSED_CONFIRMATION":
        raise RuntimeError("v8 reused-confirmation commitment is not approved")
    manifest_path = resolve_path(config["paths"]["sealed_manifest"])
    if sha256_file(manifest_path) != str(commitment["sealed_manifest_sha256"]).lower():
        raise RuntimeError("v8 confirmation manifest hash mismatch")
    prior_report_path = resolve_path(commitment["prior_report"])
    if sha256_file(prior_report_path) != str(commitment["prior_report_sha256"]).lower():
        raise RuntimeError("Prior v6 v8 report hash mismatch")
    prior_report = json.loads(prior_report_path.read_text(encoding="utf-8"))
    prior_hard = prior_report["metrics"]["hard_negative"]

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Unsupported candidate checkpoint")
    if checkpoint.get("selection_mode") != "positive_safety_fixed_epoch":
        raise RuntimeError("v8 confirmation requires a fixed-epoch v7-refit checkpoint")
    if not checkpoint.get("validation_gate", {}).get("passed", False):
        raise RuntimeError("Candidate did not pass positive safety validation")
    if checkpoint.get("provenance", {}).get("manifest_lock", {}).get("files") != lock.get("files"):
        raise RuntimeError("Checkpoint manifest provenance does not match the current v7-refit lock")
    positive_recheck_path = checkpoint_path.parent.parent / "best.positive-safety.recheck.json"
    if not positive_recheck_path.is_file():
        raise RuntimeError(f"Independent positive safety recheck is missing: {positive_recheck_path}")
    positive_recheck = json.loads(positive_recheck_path.read_text(encoding="utf-8"))
    if positive_recheck.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise RuntimeError("Independent positive safety recheck checkpoint hash mismatch")
    if positive_recheck.get("validation_manifest_sha256") != lock["files"]["validation_manifest"]["sha256"]:
        raise RuntimeError("Independent positive safety recheck manifest hash mismatch")
    if not positive_recheck.get("gate", {}).get("passed", False):
        raise RuntimeError("Independent positive safety recheck failed; v8 confirmation is blocked")

    rows = read_manifest(manifest_path, expected_split="sealed_test")
    if not rows or any(row["source_type"] != "hard_negative" for row in rows):
        raise RuntimeError("v8 confirmation set must be normal-only")
    initial = resolve_path(config["paths"]["initial_checkpoint"])
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
    candidate_by_class = hard["candidate_false_positive_by_class"]

    # The committed v6 report predates the canonical FP32 evaluation backend.
    # Re-evaluate its frozen checkpoint in this same process so tiny argmax
    # boundary differences are not mistaken for a model regression.
    prior_checkpoint_path = Path(prior_report["frozen_before_inference"]["candidate_checkpoint"]).resolve()
    expected_prior_sha = str(commitment["prior_candidate_checkpoint_sha256"]).lower()
    if not prior_checkpoint_path.is_file() or sha256_file(prior_checkpoint_path) != expected_prior_sha:
        raise RuntimeError("Committed prior v6 checkpoint is missing or its hash changed")
    prior_checkpoint = torch.load(prior_checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(prior_checkpoint, dict) or "model_state_dict" not in prior_checkpoint:
        raise ValueError("Unsupported prior v6 checkpoint")
    prior_candidate = build_model(config, initial)
    prior_candidate.load_state_dict(prior_checkpoint["model_state_dict"], strict=True)
    prior_candidate.to(device).eval()
    prior_canonical_metrics = evaluate_models(prior_candidate, baseline, loader, device)
    prior_canonical_hard = prior_canonical_metrics["hard_negative"]
    prior_by_class = prior_canonical_hard["candidate_false_positive_by_class"]
    reasons: list[str] = []
    if int(hard["candidate_fp"]) >= int(prior_canonical_hard["candidate_fp"]):
        reasons.append("candidate did not strictly reduce total v8 false-positive pixels versus canonical-FP32 v6")
    if int(hard["candidate_fp"]) > int(hard["baseline_fp"]):
        reasons.append("candidate exceeded original teacher total false-positive pixels")
    for name in ("Poor", "Severe"):
        if int(candidate_by_class[name]) > int(prior_by_class[name]):
            reasons.append(f"candidate increased v8 {name} false-positive pixels versus canonical-FP32 v6")
    worsened = [name for name, item in hard["groups"].items() if item["worsened"]]
    if worsened:
        reasons.append("candidate worsened groups versus original teacher: " + ", ".join(worsened))
    gate = {
        "passed": not reasons,
        "reasons": reasons,
        "frozen_requirements": {
            "total_fp_strictly_less_than_canonical_prior_v6": int(prior_canonical_hard["candidate_fp"]),
            "total_fp_not_above_original_teacher": int(hard["baseline_fp"]),
            "poor_fp_not_above_canonical_prior_v6": int(prior_by_class["Poor"]),
            "severe_fp_not_above_canonical_prior_v6": int(prior_by_class["Severe"]),
        },
    }
    report = {
        "evaluated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "evaluation_kind": "REUSED_V8_FINAL_NORMAL_CONFIRMATION_NOT_INDEPENDENT",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "prior_v6_report": str(prior_report_path),
        "independent_positive_safety_recheck": str(positive_recheck_path),
        "independent_positive_safety_recheck_sha256": sha256_file(positive_recheck_path),
        "evaluation_backend": canonical_evaluation_backend_report(),
        "historical_prior_v6_candidate_noncanonical": {
            "checkpoint_sha256": commitment["prior_candidate_checkpoint_sha256"],
            "false_positive_pixels": prior_hard["candidate_fp"],
            "false_positive_by_class": prior_hard["candidate_false_positive_by_class"],
        },
        "canonical_prior_v6_candidate": {
            "checkpoint": str(prior_checkpoint_path),
            "checkpoint_sha256": sha256_file(prior_checkpoint_path),
            "false_positive_pixels": prior_canonical_hard["candidate_fp"],
            "false_positive_by_class": prior_by_class,
            "hard_negative": prior_canonical_hard,
        },
        "v7_refit_metrics": {"hard_negative": hard},
        "gate": gate,
        "interpretation": "reused normal-only final regression confirmation; not an unbiased sealed test",
        "status": {
            "candidate": True,
            "uart_status": "NOT_FOR_UART",
            "deployment_status": "NOT_DEPLOYED",
            "accuracy": "NOT_FINAL",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, report)
    write_json(receipt, {
        "completed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "report": str(output),
        "report_sha256": sha256_file(output),
        "gate_passed": gate["passed"],
        "no_training_or_threshold_tuning_authorized": True,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
