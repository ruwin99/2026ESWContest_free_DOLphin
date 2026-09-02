from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from common import assert_ready, load_config, read_rows, resolve_path, sha256_file, write_json


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def canonical_rgb(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("RGB"))
    if image.shape not in {(720, 1280, 3), (240, 1280, 3)}:
        raise ValueError(f"Expected [720,1280,3] or Phase-A [240,1280,3], got {image.shape}: {path}")
    return np.ascontiguousarray(image[:240])


def rust_input(rgb: np.ndarray) -> np.ndarray:
    value = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray((value - IMAGENET_MEAN) / IMAGENET_STD)


def crack_input(rgb: np.ndarray) -> np.ndarray:
    value = rgb[112:240].transpose(2, 0, 1)[None].astype(np.float32)
    return np.ascontiguousarray(value / 127.5 - 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create immutable FP32 train/validation teacher caches.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--allow-cpu", action="store_true", help="Debug only; production cache requires CUDA ORT")
    parser.add_argument("--smoke-only", action="store_true", help="Cache only two rows under _smoke; never satisfies training")
    args = parser.parse_args()
    config_path = args.config.resolve()
    readiness = assert_ready(config_path)
    config = load_config(config_path)
    manifest = resolve_path(config["paths"][f"{args.split}_manifest"])
    rows = read_rows(manifest)
    if not rows:
        raise RuntimeError(f"Empty {args.split} manifest")
    if args.smoke_only:
        clean = next((row for row in rows if row.get("scenario") == "clean"), rows[0])
        defect = next((row for row in rows if row.get("scenario") != "clean"), rows[0])
        rows = [defect, clean]
    import onnxruntime as ort

    available = ort.get_available_providers()
    provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in available else "CPUExecutionProvider"
    if provider != "CUDAExecutionProvider" and not args.allow_cpu:
        raise RuntimeError("BLOCKED_RTX5070TI_PREFLIGHT: CUDAExecutionProvider is unavailable")
    rust_path = resolve_path(config["paths"]["rust_teacher_onnx"])
    crack_path = resolve_path(config["paths"]["hrseg_teacher_onnx"])
    providers = [provider]
    rust_session = ort.InferenceSession(str(rust_path), providers=providers)
    crack_session = ort.InferenceSession(str(crack_path), providers=providers)
    destination = resolve_path(config["paths"]["teacher_cache"]) / ("_smoke" if args.smoke_only else args.split)
    if args.smoke_only:
        destination = destination / args.split
    destination.mkdir(parents=True, exist_ok=True)
    code_sha = sha256_file(Path(__file__))
    metadata_rows: list[dict[str, object]] = []
    rust_teacher_sha = str(readiness["teachers"]["rust"]["sha256"])
    crack_teacher_sha = str(readiness["teachers"]["crack"]["sha256"])
    repeat_max_abs = 0.0
    for position, row in enumerate(rows):
        output = destination / f"{row['sample_id']}.npz"
        reused = False
        if output.is_file():
            try:
                with np.load(output, allow_pickle=False) as cached:
                    metadata_matches = (
                        str(cached["image_sha256"].item()) == row["image_sha256"]
                        and str(cached["rust_teacher_sha256"].item()) == rust_teacher_sha
                        and str(cached["hrseg_teacher_sha256"].item()) == crack_teacher_sha
                        and str(cached["cache_code_sha256"].item()) == code_sha
                    )
                    if metadata_matches:
                        rust = np.ascontiguousarray(cached["rust_teacher_logits"].astype(np.float32, copy=True))
                        margin = np.ascontiguousarray(cached["hrseg_logit_margin"].astype(np.float32, copy=True))
                        reused = True
            except (KeyError, OSError, ValueError):
                reused = False
        if not reused:
            rgb = canonical_rgb(resolve_path(row["relative_image_path"]))
            rust = rust_session.run(["logits"], {"images": rust_input(rgb)})[0][0].astype(np.float32, copy=False)
            crack2 = crack_session.run(["crack_logits"], {"images": crack_input(rgb)})[0][0].astype(np.float32, copy=False)
            if rust.shape != (4, 240, 1280) or crack2.shape != (2, 128, 1280):
                raise RuntimeError(f"Teacher output shape mismatch for {row['sample_id']}: {rust.shape}, {crack2.shape}")
            margin = np.ascontiguousarray((crack2[1:2] - crack2[0:1]).astype(np.float32))
            rust = np.ascontiguousarray(rust)
        if rust.shape != (4, 240, 1280) or margin.shape != (1, 128, 1280):
            raise RuntimeError(f"Cached teacher shape mismatch for {row['sample_id']}: {rust.shape}, {margin.shape}")
        if not np.isfinite(rust).all() or not np.isfinite(margin).all():
            raise RuntimeError(f"Non-finite teacher output: {row['sample_id']}")
        if position == 0:
            rgb = canonical_rgb(resolve_path(row["relative_image_path"]))
            rust_repeat = rust_session.run(["logits"], {"images": rust_input(rgb)})[0][0]
            crack_repeat = crack_session.run(["crack_logits"], {"images": crack_input(rgb)})[0][0]
            margin_repeat = crack_repeat[1:2] - crack_repeat[0:1]
            repeat_max_abs = max(float(np.max(np.abs(rust_repeat - rust))), float(np.max(np.abs(margin_repeat - margin))))
            if repeat_max_abs > 1e-5:
                raise RuntimeError(f"Teacher repeatability failed, max_abs={repeat_max_abs}")
        if not reused:
            temporary = output.with_suffix(".tmp.npz")
            np.savez(
                temporary, rust_teacher_logits=rust, hrseg_logit_margin=margin,
                image_sha256=np.asarray(row["image_sha256"]),
                rust_teacher_sha256=np.asarray(rust_teacher_sha),
                hrseg_teacher_sha256=np.asarray(crack_teacher_sha),
                cache_code_sha256=np.asarray(code_sha),
            )
            temporary.replace(output)
        metadata_rows.append({
            "sample_id": row["sample_id"],
            "image_sha256": row["image_sha256"],
            "canonical_crop": [0, 0, 1280, 240],
            "rust_teacher_path": str(rust_path),
            "rust_teacher_sha256": rust_teacher_sha,
            "hrseg_teacher_path": str(crack_path),
            "hrseg_teacher_sha256": crack_teacher_sha,
            "rust_preprocess_id": config["teachers"]["rust"]["preprocess_id"],
            "hrseg_preprocess_id": config["teachers"]["crack"]["preprocess_id"],
            "rust_logits_shape": list(rust.shape),
            "rust_logits_sha256": array_sha256(rust),
            "hrseg_margin_shape": list(margin.shape),
            "hrseg_margin_sha256": array_sha256(margin),
            "cache_code_sha256": code_sha,
            "onnxruntime_version": ort.__version__,
            "provider": provider,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cache_path": str(output),
            "cache_bytes": output.stat().st_size,
            "cache_file_sha256": sha256_file(output),
            "cache_reused": reused,
        })
        print(f"[{position + 1}/{len(rows)}] {'reused' if reused else 'created'} {row['sample_id']}")
    report = {
        "split": args.split,
        "smoke_only": args.smoke_only,
        "config_sha256": readiness["config_sha256"],
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "provider": provider,
        "samples": len(metadata_rows),
        "total_bytes": sum(int(row["cache_bytes"]) for row in metadata_rows),
        "repeat_max_abs": repeat_max_abs,
        "entries": metadata_rows,
    }
    write_json(destination / ("cache_smoke_manifest.json" if args.smoke_only else "cache_manifest.json"), report)
    print(json.dumps({key: value for key, value in report.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
