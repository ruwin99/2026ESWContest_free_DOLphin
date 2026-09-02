from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from common import load_config, read_rows, resolve_path, sha256_file, write_json, write_rows


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def relative(path: Path) -> str:
    return path.resolve().relative_to(resolve_path(".")).as_posix()


def public_rows(source_manifest: Path, split: str, fields: list[str]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for row in read_rows(source_manifest):
        image = resolve_path(row["image_path"])
        mask = resolve_path(row["crack_mask_path"])
        if not image.is_file() or not mask.is_file():
            raise FileNotFoundError(f"Public panel pair missing: {image} / {mask}")
        output.append({
            "sample_id": f"phase-a-public-{row['sample_id']}",
            "relative_image_path": relative(image), "image_sha256": sha256_file(image),
            "native_width": row["native_width"], "native_height": row["native_height"],
            "geometry_contract_id": row["geometry_contract_id"], "source": "crackseg9k_v4_public",
            "source_group_ids": row["source_group_ids"], "normal_session_id": "",
            "group_id": row["group_id"], "split": split, "scenario": "crack_only",
            "rust_mask_path": "", "rust_mask_sha256": "",
            "crack_mask_path": relative(mask), "crack_mask_sha256": sha256_file(mask),
            "crack_mask_encoding": "binary_0_255_positive", "rust_label_mode": "teacher_only",
            "crack_label_mode": "gt", "synthetic": "true", "development_only": "true",
            "provenance_status": "official_crackseg9k_v4_split_transformed_to_four_panel",
        })
    return output


def normal_rows(
    roots: list[str], split: str, fields: list[str], rust_mask: Path, crack_mask: Path,
    reserved_hashes: set[str], seen_hashes: set[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    output: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    rust_sha = sha256_file(rust_mask)
    crack_sha = sha256_file(crack_mask)
    for root_value in roots:
        root = resolve_path(root_value)
        if not root.is_dir():
            raise FileNotFoundError(f"Normal-negative root missing: {root}")
        # Only original frames directly inside the folder are accepted; review
        # overlays under review/ are deliberately excluded.
        images = sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        for frame_index, image in enumerate(images):
            with Image.open(image) as opened:
                width, height = opened.size
            image_sha = sha256_file(image)
            reason = None
            if (width, height) != (1280, 720):
                reason = f"geometry_{width}x{height}"
            elif image_sha in reserved_hashes:
                reason = "duplicate_of_validation"
            elif image_sha in seen_hashes:
                reason = "duplicate_within_phase_a"
            if reason:
                excluded.append({"path": str(image), "sha256": image_sha, "reason": reason})
                continue
            seen_hashes.add(image_sha)
            sample_id = f"phase-a-normal-{split}-{root.name}-{frame_index:06d}"
            output.append({
                "sample_id": sample_id, "relative_image_path": relative(image), "image_sha256": image_sha,
                "native_width": "1280", "native_height": "720",
                "geometry_contract_id": "camera-top-crop-w1280-h240-v1", "source": "reviewed_normal_rail_negative",
                "source_group_ids": root.name, "normal_session_id": root.name,
                "group_id": f"normal-session-{root.name}", "split": split, "scenario": "clean",
                "rust_mask_path": relative(rust_mask), "rust_mask_sha256": rust_sha,
                "crack_mask_path": relative(crack_mask), "crack_mask_sha256": crack_sha,
                "crack_mask_encoding": "indexed_0_1_ignore255", "rust_label_mode": "gt",
                "crack_label_mode": "gt", "synthetic": "false", "development_only": "true",
                "provenance_status": "user_approved_existing_normal_rail_negative_session",
            })
    return output, excluded


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase-A-only public crack + normal-negative manifests.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    if config["status"]["phase"] != "PHASE_A_DEVELOPMENT_ONLY":
        raise RuntimeError("prepare_phase_a requires PHASE_A_DEVELOPMENT_ONLY")
    mask_root = resolve_path(config["paths"]["phase_a_normal_mask_root"])
    mask_root.mkdir(parents=True, exist_ok=True)
    rust_mask = mask_root / "rust_class0_w1280_h240.png"
    crack_mask = mask_root / "crack_negative_rows112_240_w1280_h240.png"
    if not rust_mask.is_file():
        Image.fromarray(np.zeros((240, 1280), dtype=np.uint8), mode="L").save(rust_mask)
    if not crack_mask.is_file():
        values = np.zeros((240, 1280), dtype=np.uint8)
        values[:112] = 255
        Image.fromarray(values, mode="L").save(crack_mask)
    fields = list(config["phase_a_manifest"]["required_columns"])
    validation = public_rows(resolve_path(config["paths"]["phase_a_public_validation_manifest"]), "validation", fields)
    validation_normal, validation_excluded = normal_rows(
        list(config["phase_a_manifest"]["normal_validation_roots"]), "validation", fields,
        rust_mask, crack_mask, set(), set(row["image_sha256"] for row in validation),
    )
    validation.extend(validation_normal)
    validation_hashes = {row["image_sha256"] for row in validation}
    train = public_rows(resolve_path(config["paths"]["phase_a_public_train_manifest"]), "train", fields)
    train_seen = {row["image_sha256"] for row in train}
    train_normal, train_excluded = normal_rows(
        list(config["phase_a_manifest"]["normal_train_roots"]), "train", fields,
        rust_mask, crack_mask, validation_hashes, train_seen,
    )
    train.extend(train_normal)
    write_rows(resolve_path(config["paths"]["train_manifest"]), fields, train)
    write_rows(resolve_path(config["paths"]["validation_manifest"]), fields, validation)
    write_rows(resolve_path(config["paths"]["development_calibration_manifest"]), fields, [])
    report = {
        "phase": "PHASE_A_DEVELOPMENT_ONLY", "result_labels": ["ACCURACY_NOT_FINAL", "NOT_FOR_UART"],
        "train_rows": len(train), "validation_rows": len(validation),
        "train_public": sum(row["source"] == "crackseg9k_v4_public" for row in train),
        "train_normal": sum(row["scenario"] == "clean" for row in train),
        "validation_public": sum(row["source"] == "crackseg9k_v4_public" for row in validation),
        "validation_normal": sum(row["scenario"] == "clean" for row in validation),
        "excluded": validation_excluded + train_excluded,
    }
    output = resolve_path(config["paths"]["reports"]) / "phase_a_manifest_build.json"
    write_json(output, report)
    print(report)


if __name__ == "__main__":
    main()
