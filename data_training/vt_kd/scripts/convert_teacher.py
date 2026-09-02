from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from common import (
    CLASS_NAMES,
    PROJECT_ROOT,
    build_teacher,
    configure_model_imports,
    load_config,
    resolve_project_path,
    sha256_file,
)


OFFICIAL_SOURCE_URL = (
    "https://data.lib.vt.edu/articles/software/"
    "Trained_Model_for_the_Semantic_Segmentation_of_Corrosion_Condition_States/16628668"
)


def parse_args() -> argparse.Namespace:
    default_config = PROJECT_ROOT / "data_training" / "vt_kd" / "configs" / "student_mnv2_os8.yaml"
    parser = argparse.ArgumentParser(
        description="Convert a verified Virginia Tech legacy teacher pickle to a state_dict bundle."
    )
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--architecture",
        choices=("deeplabv3plus_resnet50", "deeplabv3plus_resnet101"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-unsafe-official-pickle",
        action="store_true",
        help=(
            "Required acknowledgement: torch.load(weights_only=False) may execute code. "
            "Use only after the exact official file and SHA-256 were independently verified."
        ),
    )
    return parser.parse_args()


def normalized_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    state = module.state_dict()
    if state and all(key.startswith("module.") for key in state):
        return {key.removeprefix("module."): value for key, value in state.items()}
    return dict(state)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    teacher_cfg = config["teacher"]

    raw_path = (
        args.input.resolve()
        if args.input is not None
        else resolve_project_path(teacher_cfg["raw_checkpoint"])
    )
    output_path = (
        args.output.resolve()
        if args.output is not None
        else resolve_project_path(teacher_cfg["checkpoint"])
    )
    architecture = args.architecture or str(teacher_cfg["architecture"])
    expected_sha256 = (
        args.expected_sha256 or str(teacher_cfg["source_sha256"])
    ).lower()

    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw teacher checkpoint not found: {raw_path}")
    actual_sha256 = sha256_file(raw_path)
    if actual_sha256.lower() != expected_sha256:
        raise RuntimeError(
            "Teacher SHA-256 mismatch; refusing to open the pickle.\n"
            f"expected={expected_sha256}\nactual={actual_sha256}\npath={raw_path}"
        )
    if not args.allow_unsafe_official_pickle:
        raise RuntimeError(
            "Conversion stopped before opening the legacy pickle. "
            "torch.load(weights_only=False) can execute arbitrary code. "
            "After verifying this exact official file and hash, rerun with "
            "--allow-unsafe-official-pickle."
        )

    configure_model_imports(config)
    # Importing the original modules makes their trusted class names available to pickle.
    __import__("model_plus")
    __import__("network._deeplab")
    __import__("network.modeling")

    # This is intentionally the only unsafe load in the pipeline. It is gated by the
    # exact official SHA-256 and an explicit command-line acknowledgement.
    legacy_teacher = torch.load(
        raw_path,
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(legacy_teacher, torch.nn.Module):
        raise TypeError(
            f"Expected a torch.nn.Module in the legacy checkpoint, got {type(legacy_teacher)!r}"
        )
    legacy_teacher.eval()
    legacy_state = normalized_state_dict(legacy_teacher)

    rebuilt_teacher = build_teacher(config, architecture, pretrained_backbone=False)
    rebuilt_teacher.load_state_dict(legacy_state, strict=True)
    rebuilt_teacher.eval()

    generator = torch.Generator(device="cpu").manual_seed(20260803)
    probe = torch.randn((1, 3, 64, 64), generator=generator, dtype=torch.float32)
    with torch.inference_mode():
        legacy_logits = legacy_teacher(probe)
        rebuilt_logits = rebuilt_teacher(probe)
    max_abs_error = float((legacy_logits - rebuilt_logits).abs().max())
    if legacy_logits.shape != (1, 4, 64, 64):
        raise RuntimeError(f"Unexpected teacher output shape: {tuple(legacy_logits.shape)}")
    if max_abs_error > 1e-6:
        raise RuntimeError(
            f"Legacy/rebuilt teacher output mismatch: max_abs_error={max_abs_error}"
        )

    parameter_count = sum(parameter.numel() for parameter in rebuilt_teacher.parameters())
    bundle = {
        "format_version": 1,
        "state_dict": rebuilt_teacher.state_dict(),
        "architecture": architecture,
        "num_classes": 4,
        "output_stride": 8,
        "class_names": list(CLASS_NAMES),
        "teacher_preprocessing": {
            "color": "BGR",
            "dtype": "float32",
            "scale": "0..255",
            "mean": None,
            "std": None,
        },
        "source_file": raw_path.name,
        "source_sha256": actual_sha256,
        "source_url": OFFICIAL_SOURCE_URL,
        "legacy_python_type": f"{type(legacy_teacher).__module__}.{type(legacy_teacher).__qualname__}",
        "legacy_parameter_count": sum(
            parameter.numel() for parameter in legacy_teacher.parameters()
        ),
        "parameter_count": parameter_count,
        "conversion_probe_shape": list(probe.shape),
        "conversion_max_abs_error": max_abs_error,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, output_path)

    safe_bundle = torch.load(output_path, map_location="cpu", weights_only=True)
    safe_rebuilt = build_teacher(
        config, str(safe_bundle["architecture"]), pretrained_backbone=False
    )
    safe_rebuilt.load_state_dict(safe_bundle["state_dict"], strict=True)
    print(f"Converted teacher: {output_path}")
    print(f"Source SHA-256: {actual_sha256}")
    print(f"Architecture: {architecture}")
    print(f"Parameters: {parameter_count}")
    print(f"Legacy/rebuilt max abs error: {max_abs_error:.9g}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
