from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from PIL import Image


WORK_ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "Final-Dataset-Vol1.zip": "0ee4b33617db30612184a2700116ba4d",
    "Final-Dataset-Vol2.zip": "d52bccf41c081d74fe50c9feec48ca39",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def md5_file(path: Path) -> str:
    digest = hashlib.md5()  # noqa: S324 - required to verify the publisher's checksum
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    destination_resolved = destination.resolve()
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            try:
                target.relative_to(destination_resolved)
            except ValueError as error:
                raise ValueError(f"Unsafe ZIP member: {member.filename}") from error
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file() and target.stat().st_size == member.file_size:
                continue
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=8 * 1024 * 1024)
            extracted += 1
    return extracted


def find_dataset_roots(extract_dir: Path) -> list[Path]:
    roots: list[Path] = []
    for image_dir in extract_dir.rglob("JPEGImages"):
        parent = image_dir.parent
        if (parent / "SegmentationClass").is_dir() and (parent / "ImageSets").is_dir():
            roots.append(parent)
    return sorted(set(roots))


def file_map(directory: Path) -> dict[str, Path]:
    return {
        path.relative_to(directory).as_posix(): path
        for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def read_split_names(image_sets: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for split_file in sorted(image_sets.glob("*.txt")):
        result[split_file.stem] = [
            line.strip() for line in split_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
    return result


def audit_root(root: Path) -> dict[str, object]:
    images = file_map(root / "JPEGImages")
    masks = file_map(root / "SegmentationClass")
    splits = read_split_names(root / "ImageSets")
    shared = sorted(images.keys() & masks.keys())
    split_reports: dict[str, object] = {}
    for name, entries in splits.items():
        normalized = [entry.replace("\\", "/").removeprefix("./") for entry in entries]
        missing_images = [entry for entry in normalized if entry not in images]
        missing_masks = [entry for entry in normalized if entry not in masks]
        split_reports[name] = {
            "rows": len(entries),
            "missing_images": missing_images[:50],
            "missing_masks": missing_masks[:50],
        }

    dimensions: dict[str, int] = {}
    mask_values: set[int] = set()
    for name in shared[:100]:
        with Image.open(images[name]) as image:
            key = f"{image.width}x{image.height}"
            dimensions[key] = dimensions.get(key, 0) + 1
        with Image.open(masks[name]) as mask:
            extrema = mask.convert("L").getextrema()
            mask_values.update(int(value) for value in extrema)

    return {
        "root": str(root),
        "images": len(images),
        "masks": len(masks),
        "paired": len(shared),
        "image_only": sorted(images.keys() - masks.keys())[:50],
        "mask_only": sorted(masks.keys() - images.keys())[:50],
        "splits": split_reports,
        "sampled_dimensions": dimensions,
        "sampled_mask_extrema_values": sorted(mask_values),
    }


def audit_v4_split_layout(vol1: Path, vol2: Path) -> dict[str, object]:
    image_dirs = [vol1 / "Images", vol2 / "Images-2"]
    mask_dir = vol1 / "Final_Masks" / "Masks"
    heads_dir = vol1 / "Final_Masks" / "Heads"
    split_dir = vol1 / "Final_Masks"
    required = [*image_dirs, mask_dir, heads_dir, split_dir / "train.txt", split_dir / "test.txt"]
    missing_paths = [str(path) for path in required if not path.exists()]
    if missing_paths:
        return {
            "layout": "crackseg9k_v4_two_volume",
            "vol1": str(vol1),
            "vol2": str(vol2),
            "missing_paths": missing_paths,
            "ready_for_pairing": False,
        }

    images_a = file_map(image_dirs[0])
    images_b = file_map(image_dirs[1])
    image_overlap = sorted(images_a.keys() & images_b.keys())
    images = {**images_a, **images_b}
    masks = file_map(mask_dir)
    heads = file_map(heads_dir)
    splits = {
        name: [
            line.strip().replace("\\", "/").removeprefix("./")
            for line in (split_dir / f"{name}.txt").read_text(
                encoding="utf-8-sig"
            ).splitlines()
            if line.strip()
        ]
        for name in ("train", "test")
    }
    train_names = set(splits["train"])
    test_names = set(splits["test"])
    split_names = train_names | test_names

    dimensions: dict[str, int] = {}
    mask_modes: dict[str, int] = {}
    mask_extrema: dict[str, int] = {}
    for name in sorted(images)[:200]:
        with Image.open(images[name]) as image:
            key = f"{image.width}x{image.height}"
            dimensions[key] = dimensions.get(key, 0) + 1
        with Image.open(masks[name]) as mask:
            mask_modes[mask.mode] = mask_modes.get(mask.mode, 0) + 1
            extrema = mask.convert("L").getextrema()
            key = f"{extrema[0]}..{extrema[1]}"
            mask_extrema[key] = mask_extrema.get(key, 0) + 1

    issues: list[str] = []
    if image_overlap:
        issues.append(f"image filename overlap between volumes: {image_overlap[:50]}")
    for label, actual in (("masks", set(masks)), ("heads", set(heads)), ("split", split_names)):
        missing = sorted(set(images) - actual)
        extra = sorted(actual - set(images))
        if missing:
            issues.append(f"{label} missing for images: {missing[:50]}")
        if extra:
            issues.append(f"{label} entries without images: {extra[:50]}")
    split_overlap = sorted(train_names & test_names)
    if split_overlap:
        issues.append(f"train/test filename overlap: {split_overlap[:50]}")

    return {
        "layout": "crackseg9k_v4_two_volume",
        "vol1": str(vol1),
        "vol2": str(vol2),
        "images_vol1": len(images_a),
        "images_vol2": len(images_b),
        "images_total": len(images),
        "masks": len(masks),
        "heads": len(heads),
        "train_rows": len(splits["train"]),
        "test_rows": len(splits["test"]),
        "sampled_image_dimensions": dimensions,
        "sampled_mask_modes": mask_modes,
        "sampled_mask_extrema": mask_extrema,
        "issues": issues,
        "ready_for_pairing": not issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and extract the two official CrackSeg9k V4 archives."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=WORK_ROOT / "raw" / "crackseg9k_v4" / "downloads",
    )
    parser.add_argument(
        "--extract-dir",
        type=Path,
        default=WORK_ROOT / "raw" / "crackseg9k_v4" / "extracted",
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--vol1-dir", type=Path)
    parser.add_argument("--vol2-dir", type=Path)
    args = parser.parse_args()

    download_dir = args.download_dir.resolve()
    extract_dir = args.extract_dir.resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    project_root = WORK_ROOT.parents[1]
    vol1 = (args.vol1_dir or project_root / "Final-Dataset-Vol1" / "Final-Dataset-Vol1").resolve()
    vol2 = (args.vol2_dir or project_root / "Final-Dataset-Vol2" / "Final-Dataset-Vol2").resolve()
    extracted_layout_available = (vol1 / "Images").is_dir() and (vol2 / "Images-2").is_dir()
    missing = [name for name in EXPECTED if not (download_dir / name).is_file()]
    if missing and not extracted_layout_available:
        names = "\n - ".join(missing)
        raise FileNotFoundError(
            f"Place both official V4 ZIP files in {download_dir}:\n - {names}\n"
            f"Or provide already-extracted folders with --vol1-dir and --vol2-dir."
        )

    archives: dict[str, object] = {}
    for name, expected_md5 in EXPECTED.items():
        path = download_dir / name
        if not path.is_file():
            archives[name] = {
                "path": str(path),
                "present": False,
                "md5_verification": "unavailable_extracted_files_only",
            }
            continue
        actual_md5 = md5_file(path)
        if actual_md5 != expected_md5:
            raise ValueError(
                f"MD5 mismatch for {name}: expected={expected_md5}, actual={actual_md5}"
            )
        archives[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "md5": actual_md5,
        }
        if args.extract:
            archives[name]["newly_extracted_files"] = safe_extract(path, extract_dir)

    roots = find_dataset_roots(extract_dir)
    root_reports = [audit_root(root) for root in roots]
    v4_layout_report = (
        audit_v4_split_layout(vol1, vol2) if extracted_layout_available else None
    )
    ready_for_pairing = (
        bool(root_reports)
        and all(
            item["paired"] > 0
            and not item["image_only"]
            and not item["mask_only"]
            and all(
                not split["missing_images"] and not split["missing_masks"]
                for split in item["splits"].values()
            )
            for item in root_reports
        )
    ) or bool(v4_layout_report and v4_layout_report["ready_for_pairing"])
    report = {
        "schema_version": 1,
        "source": "CrackSeg9k V4",
        "doi": "https://doi.org/10.7910/DVN/EGIEBY",
        "official_repository": "https://github.com/Dhananjay42/crackseg9k",
        "archives": archives,
        "extract_dir": str(extract_dir),
        "dataset_roots": root_reports,
        "v4_split_layout": v4_layout_report,
        "ready_for_pairing": ready_for_pairing,
        "usage_note": (
            "CrackSeg9k V4 is 400x400 public crack supervision. It is not camera-native "
            "1280x720 validation evidence and must remain source-tagged."
        ),
    }
    report_path = WORK_ROOT / "metrics" / "crackseg9k_v4_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"CrackSeg9k report: {report_path}")


if __name__ == "__main__":
    main()
