from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO


TARGET_PARAMETERS = {
    f"model.23.{branch}.{scale}.2.{kind}"
    for branch in ("cv3", "one2one_cv3")
    for scale in range(3)
    for kind in ("weight", "bias")
}


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def restrict_trainable_parameters(trainer: Any) -> None:
    model = unwrap(trainer.model)
    found: set[str] = set()
    for name, parameter in model.named_parameters():
        trainable = name in TARGET_PARAMETERS
        parameter.requires_grad_(trainable)
        if trainable:
            found.add(name)

    missing = TARGET_PARAMETERS - found
    unexpected = found - TARGET_PARAMETERS
    if missing or unexpected:
        raise RuntimeError(
            f"Safe-head parameter contract failed. missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    print(
        f"SAFE_TRAINABLE_CONTRACT tensors={len(found)} "
        f"parameters={trainable_count}/{total_count}"
    )


def freeze_batchnorm_statistics(trainer: Any) -> None:
    # Ultralytics calls model.train() at every epoch. Re-apply eval only to BN
    # modules before every forward pass, while Detect itself stays in train mode.
    model = unwrap(trainer.model)
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def compare_checkpoints(baseline: Path, candidate: Path) -> dict[str, Any]:
    baseline_state = YOLO(str(baseline)).model.float().state_dict()
    candidate_state = YOLO(str(candidate)).model.float().state_dict()
    if baseline_state.keys() != candidate_state.keys():
        raise RuntimeError("Baseline and candidate state_dict keys differ")

    changed_trainable: list[str] = []
    changed_frozen: list[dict[str, Any]] = []
    for name, baseline_tensor in baseline_state.items():
        candidate_tensor = candidate_state[name]
        if torch.equal(baseline_tensor, candidate_tensor):
            continue
        max_abs_diff = float((baseline_tensor - candidate_tensor).abs().max().item())
        if name in TARGET_PARAMETERS:
            changed_trainable.append(name)
        else:
            changed_frozen.append({"name": name, "max_abs_diff": max_abs_diff})

    return {
        "candidate": str(candidate),
        "target_parameter_names": sorted(TARGET_PARAMETERS),
        "changed_target_tensors": changed_trainable,
        "unchanged_target_tensors": sorted(TARGET_PARAMETERS - set(changed_trainable)),
        "changed_frozen_tensors": changed_frozen,
        "frozen_contract_passed": not changed_frozen,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--epochs", required=True, type=int)
    parser.add_argument("--batch", default=8, type=int)
    parser.add_argument("--workers", default=2, type=int)
    parser.add_argument("--lr0", default=2e-6, type=float)
    parser.add_argument("--audit-report", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.project / args.name
    if run_dir.exists():
        raise FileExistsError(f"Refusing automatic suffix; run exists: {run_dir}")

    model = YOLO(str(args.baseline))
    model.add_callback("on_pretrain_routine_end", restrict_trainable_parameters)
    model.add_callback("on_train_batch_start", freeze_batchnorm_statistics)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=1280,
        rect=True,
        batch=args.batch,
        workers=args.workers,
        device=0,
        amp=True,
        cache="disk",
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=1.0,
        warmup_epochs=0.0,
        warmup_bias_lr=0.0,
        mosaic=0.0,
        close_mosaic=0,
        patience=0,
        save_period=1,
        seed=42,
        deterministic=True,
        plots=True,
        project=str(args.project),
        name=args.name,
        resume=False,
    )

    candidate = run_dir / "weights" / "best.pt"
    if not candidate.is_file():
        raise FileNotFoundError(f"Candidate checkpoint missing: {candidate}")
    audit = compare_checkpoints(args.baseline, candidate)
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    args.audit_report.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if not audit["frozen_contract_passed"]:
        raise RuntimeError(
            f"Frozen tensor contract failed; see {args.audit_report}"
        )


if __name__ == "__main__":
    main()


