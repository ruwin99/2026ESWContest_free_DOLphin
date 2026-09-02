from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

from common import (
    assert_training_ready,
    cache_key,
    crop_native_rgb,
    load_config,
    read_manifest_rows,
    resolve_path,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache audited rust logits and HrSegNet crack margins without resampling."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--provider", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_session(path: Path, provider: str):
    # Importing PyTorch first loads the CUDA/cuDNN DLLs bundled with the cu128
    # wheel. On Windows this lets ONNX Runtime resolve the same CUDA 12 runtime
    # without requiring a separate system-wide CUDA Toolkit installation.
    if provider == "cuda":
        import torch  # noqa: F401

    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable; providers={available}")
        providers.insert(0, "CUDAExecutionProvider")
    session = ort.InferenceSession(str(path), providers=providers)
    active = session.get_providers()
    if provider == "cuda" and (
        not active or active[0] != "CUDAExecutionProvider"
    ):
        raise RuntimeError(
            "CUDAExecutionProvider was requested but ONNX Runtime fell back "
            f"to {active}. Install a CUDA-compatible onnxruntime-gpu build or "
            "rerun explicitly with --provider cpu."
        )
    return session


def save_npz_atomic(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as handle:
        np.savez(handle, **arrays)
        temporary = Path(handle.name)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    readiness = assert_training_ready(config_path)
    config = load_config(config_path)
    cache_contract = config.get("teacher_cache", {})
    use_rust = bool(cache_contract.get("rust", True))
    use_crack = bool(cache_contract.get("crack", True))
    if not use_rust and not use_crack:
        raise ValueError("At least one teacher must be enabled for cache generation")
    rust_path = resolve_path(config["paths"]["rust_teacher_onnx"])
    crack_path = resolve_path(config["paths"]["hrseg_teacher_onnx"])
    rust_session = make_session(rust_path, args.provider) if use_rust else None
    crack_session = make_session(crack_path, args.provider) if use_crack else None
    cache_root = resolve_path(config["paths"]["teacher_cache"])
    cache_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    manifest_hashes: dict[str, str] = {}
    for split in ("train", "val"):
        path = resolve_path(config["paths"][f"{split}_manifest"])
        rows.extend(read_manifest_rows(path))
        manifest_hashes[split] = sha256_file(path)

    mean = np.asarray(config["contracts"]["mean"], dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(config["contracts"]["std"], dtype=np.float32).reshape(1, 1, 3)
    items: list[dict[str, str]] = []
    for row in tqdm(rows, desc="teacher cache"):
        destination = cache_root / f"{cache_key(row['sample_id'])}.npz"
        if destination.is_file() and not args.overwrite:
            items.append(
                {"sample_id": row["sample_id"], "file": destination.name, "sha256": sha256_file(destination)}
            )
            continue
        crop = tuple(
            int(row[name]) for name in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")
        )
        rgb = crop_native_rgb(resolve_path(row["image_path"]), crop)
        arrays: dict[str, np.ndarray] = {}
        if rust_session is not None:
            rust_input = ((rgb.astype(np.float32) / 255.0 - mean) / std).transpose(2, 0, 1)[None]
            rust = rust_session.run([config["teachers"]["rust"]["output_name"]], {config["teachers"]["rust"]["input_name"]: rust_input})[0]
            if rust.shape != (1, 4, 240, 1280):
                raise RuntimeError(f"Rust teacher runtime shape mismatch: {rust.shape}")
            arrays["rust_logits"] = rust[0].astype(np.float32)
        if crack_session is not None:
            crack_rgb = rgb[112:240]
            crack_input = (crack_rgb.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)[None]
            crack = crack_session.run([config["teachers"]["crack"]["output_name"]], {config["teachers"]["crack"]["input_name"]: crack_input})[0]
            if crack.shape != (1, 2, 128, 1280):
                raise RuntimeError(f"Crack teacher runtime shape mismatch: {crack.shape}")
            arrays["crack_margin"] = (crack[:, 1:2] - crack[:, 0:1])[0].astype(np.float32)
        save_npz_atomic(destination, **arrays)
        items.append(
            {"sample_id": row["sample_id"], "file": destination.name, "sha256": sha256_file(destination)}
        )

    payload = {
        "schema_version": 1,
        "config_sha256": readiness["config_sha256"],
        "manifest_sha256": manifest_hashes,
        "teacher_sha256": {
            "rust": sha256_file(rust_path),
            "crack": sha256_file(crack_path),
        },
        "dtype": "float32",
        "requested_provider": args.provider,
        "active_providers": {
            "rust": rust_session.get_providers() if rust_session is not None else None,
            "crack": crack_session.get_providers() if crack_session is not None else None,
        },
        "rust_shape": [4, 240, 1280] if use_rust else None,
        "crack_margin_shape": [1, 128, 1280] if use_crack else None,
        "items": items,
    }
    write_json(cache_root / "cache_manifest.json", payload)
    print(json.dumps({"cached": len(items), "manifest": str(cache_root / 'cache_manifest.json')}, ensure_ascii=False))


if __name__ == "__main__":
    main()
