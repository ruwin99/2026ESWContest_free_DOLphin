from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm


WORK_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORK_ROOT.parents[1]
PANEL_CONTRACT = "crackseg9k-four-native-crops-w1280-h240-v1"
MANIFEST_COLUMNS = (
    "sample_id,image_path,rust_mask_path,crack_mask_path,rust_valid,crack_valid,"
    "source,physical_specimen_id,capture_session_id,encoder_section_id,group_id,"
    "split,image_sha256,native_width,native_height,crop_x0,crop_y0,crop_x1,crop_y1,"
    "geometry_contract_id,camera_native,synthetic,source_group_ids,seam_ignore_mask_path"
).split(",")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_png(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    Image.fromarray(array).save(temporary, format="PNG", compress_level=3)
    temporary.replace(path)


def load_names(path: Path) -> list[str]:
    return [
        line.strip().replace("\\", "/").removeprefix("./")
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def choose_crop(mask: np.ndarray, name: str) -> tuple[int, int, int]:
    binary = mask > 0
    best_score = -1
    best_xy = (0, 0)
    x_candidates = [*range(0, 81, 8), 80]
    y_candidates = [*range(0, 161, 8), 160]
    for y0 in y_candidates:
        for x0 in x_candidates:
            score = int(binary[y0 + 112 : y0 + 240, x0 : x0 + 320].sum())
            if score > best_score:
                best_score = score
                best_xy = (x0, y0)
    if best_score == 0:
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        best_xy = (int.from_bytes(digest[:2], "little") % 81, int.from_bytes(digest[2:4], "little") % 161)
    return best_xy[0], best_xy[1], best_score


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def build_split(
    split: str,
    names: list[str],
    images: dict[str, Path],
    masks: dict[str, Path],
    output_root: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    usable = len(names) - (len(names) % 4)
    dropped = names[usable:]
    rows: list[dict[str, str]] = []
    positive_pixels = 0
    valid_pixels = 0
    positive_panels = 0
    image_dir = output_root / split / "images"
    mask_dir = output_root / split / "masks"

    for panel_index in tqdm(range(usable // 4), desc=f"CrackSeg9k {split} panels"):
        source_names = names[panel_index * 4 : panel_index * 4 + 4]
        rgb_tiles: list[np.ndarray] = []
        mask_tiles: list[np.ndarray] = []
        source_descriptors: list[str] = []
        for name in source_names:
            with Image.open(images[name]) as image:
                rgb = np.asarray(image.convert("RGB"))
            with Image.open(masks[name]) as image:
                mask = np.asarray(image.convert("L"))
            if rgb.shape != (400, 400, 3) or mask.shape != (400, 400):
                raise ValueError(f"Expected 400x400 pair for {name}, got {rgb.shape}/{mask.shape}")
            x0, y0, _ = choose_crop(mask, name)
            rgb_tiles.append(np.ascontiguousarray(rgb[y0 : y0 + 240, x0 : x0 + 320]))
            mask_tiles.append(np.ascontiguousarray(mask[y0 : y0 + 240, x0 : x0 + 320]))
            source_descriptors.append(f"{name}@{x0},{y0},320,240")

        panel_rgb = np.concatenate(rgb_tiles, axis=1)
        panel_mask = np.concatenate(mask_tiles, axis=1)
        if panel_rgb.shape != (240, 1280, 3) or panel_mask.shape != (240, 1280):
            raise RuntimeError("Panel geometry construction failed")
        panel_mask = np.where(panel_mask > 0, 255, 0).astype(np.uint8)
        sample_id = f"crackseg9k-v4-{split}-{panel_index:05d}"
        image_path = image_dir / f"{sample_id}.png"
        mask_path = mask_dir / f"{sample_id}.png"
        if not image_path.is_file():
            atomic_png(panel_rgb, image_path)
        if not mask_path.is_file():
            atomic_png(panel_mask, mask_path)

        roi = panel_mask[112:240] > 0
        pixels = int(roi.sum())
        positive_pixels += pixels
        valid_pixels += int(roi.size)
        positive_panels += int(pixels > 0)
        group_digest = hashlib.sha256("|".join(source_names).encode("utf-8")).hexdigest()[:16]
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": relative(image_path),
                "rust_mask_path": "",
                "crack_mask_path": relative(mask_path),
                "rust_valid": "false",
                "crack_valid": "true",
                "source": "crackseg9k_v4",
                "physical_specimen_id": f"public-unknown-{group_digest}",
                "capture_session_id": f"crackseg9k-v4-official-{split}",
                "encoder_section_id": sample_id,
                "group_id": f"crackseg9k-v4-{split}-{group_digest}",
                "split": split,
                "image_sha256": sha256_file(image_path),
                "native_width": "1280",
                "native_height": "240",
                "crop_x0": "0",
                "crop_y0": "0",
                "crop_x1": "1280",
                "crop_y1": "240",
                "geometry_contract_id": PANEL_CONTRACT,
                "camera_native": "false",
                "synthetic": "true",
                "source_group_ids": "|".join(source_descriptors),
                "seam_ignore_mask_path": "",
            }
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    coverage = positive_pixels / valid_pixels if valid_pixels else 0.0
    return {
        "split": split,
        "source_rows": len(names),
        "panels": len(rows),
        "dropped_source_rows": dropped,
        "positive_panels": positive_panels,
        "positive_pixels": positive_pixels,
        "valid_pixels": valid_pixels,
        "crack_pixel_coverage": coverage,
        "suggested_capped_pos_weight": min(20.0, max(1.0, (valid_pixels - positive_pixels) / max(1, positive_pixels))),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build no-padding CrackSeg9k V4 1280x240 panels.")
    parser.add_argument(
        "--vol1-dir",
        type=Path,
        default=PROJECT_ROOT / "Final-Dataset-Vol1" / "Final-Dataset-Vol1",
    )
    parser.add_argument(
        "--vol2-dir",
        type=Path,
        default=PROJECT_ROOT / "Final-Dataset-Vol2" / "Final-Dataset-Vol2",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORK_ROOT / "prepared" / "crackseg9k_v4_panels",
    )
    args = parser.parse_args()

    vol1 = args.vol1_dir.resolve()
    vol2 = args.vol2_dir.resolve()
    images = {
        **{path.name: path for path in (vol1 / "Images").glob("*.png")},
        **{path.name: path for path in (vol2 / "Images-2").glob("*.png")},
    }
    masks = {path.name: path for path in (vol1 / "Final_Masks" / "Masks").glob("*.png")}
    train_names = load_names(vol1 / "Final_Masks" / "train.txt")
    test_names = load_names(vol1 / "Final_Masks" / "test.txt")
    expected = set(train_names) | set(test_names)
    if set(images) != expected or set(masks) != expected or set(train_names) & set(test_names):
        raise ValueError("CrackSeg9k image/mask/split audit failed before panel generation")

    reports = [
        build_split(
            "train",
            train_names,
            images,
            masks,
            args.output_root.resolve(),
            WORK_ROOT / "manifests" / "crackseg9k_bootstrap_train.csv",
        ),
        build_split(
            "val",
            test_names,
            images,
            masks,
            args.output_root.resolve(),
            WORK_ROOT / "manifests" / "crackseg9k_bootstrap_val.csv",
        ),
    ]
    payload = {
        "schema_version": 1,
        "source": "CrackSeg9k V4",
        "geometry_contract_id": PANEL_CONTRACT,
        "resize": False,
        "stretch": False,
        "padding": False,
        "construction": "four deterministic native-scale 320x240 crops concatenated horizontally",
        "crack_valid_rows": [112, 240],
        "reports": reports,
        "camera_native_accuracy": "unverified",
    }
    report_path = WORK_ROOT / "metrics" / "crackseg9k_v4_panels.json"
    report_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Panel report: {report_path}")


if __name__ == "__main__":
    main()
