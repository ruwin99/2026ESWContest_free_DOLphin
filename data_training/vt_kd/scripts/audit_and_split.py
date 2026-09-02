#!/usr/bin/env python3
"""Audit the Virginia Tech CSSD 512x512 dataset and create a fixed split.

The public ``Test`` partition is audited for integrity and duplicate files, but
is deliberately never written to the train/validation CSV.  Only the official
``Train`` partition is stratified, using the maximum severity present in each
mask.

Runtime dependencies are intentionally limited to Pillow and NumPy; the rest
of the implementation uses Python's standard library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError


EXPECTED_TRAIN_PAIRS = 396
EXPECTED_TEST_PAIRS = 44
EXPECTED_SIZE = (512, 512)

CLASS_NAMES = ("Good", "Fair", "Poor", "Severe")
ALLOWED_MASK_COLORS = (
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
)

IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
MASK_EXTENSIONS = frozenset({".png"})


class AuditError(RuntimeError):
    """Raised when the dataset fails an integrity requirement."""


@dataclass(frozen=True)
class SampleRecord:
    source_partition: str
    stem: str
    image_path: Path
    mask_path: Path
    image_relative_path: str
    mask_relative_path: str
    image_sha256: str
    mask_sha256: str
    max_severity: int
    pixel_counts: tuple[int, int, int, int]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False
        ) as stream:
            temporary_name = stream.name
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


def _index_files(folder: Path, allowed_extensions: frozenset[str]) -> dict[str, Path]:
    if not folder.is_dir():
        raise AuditError(f"Required directory does not exist: {folder}")

    files = sorted(
        (entry for entry in folder.iterdir() if entry.is_file()),
        key=lambda path: (path.name.casefold(), path.name),
    )
    unsupported = [path.name for path in files if path.suffix.casefold() not in allowed_extensions]
    if unsupported:
        preview = ", ".join(unsupported[:10])
        suffix = " ..." if len(unsupported) > 10 else ""
        allowed = ", ".join(sorted(allowed_extensions))
        raise AuditError(
            f"Unsupported file extension in {folder} (allowed: {allowed}): "
            f"{preview}{suffix}"
        )

    indexed: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        if key in indexed:
            raise AuditError(
                "Duplicate file stem in one directory: "
                f"{indexed[key].name!r} and {path.name!r} in {folder}"
            )
        indexed[key] = path
    return indexed


def _matched_pairs(
    images_dir: Path,
    masks_dir: Path,
    expected_count: int,
    partition: str,
) -> list[tuple[Path, Path]]:
    images = _index_files(images_dir, IMAGE_EXTENSIONS)
    masks = _index_files(masks_dir, MASK_EXTENSIONS)

    image_stems = set(images)
    mask_stems = set(masks)
    missing_masks = sorted(image_stems - mask_stems)
    missing_images = sorted(mask_stems - image_stems)
    if missing_masks or missing_images:
        details: list[str] = []
        if missing_masks:
            details.append(
                "images without masks=" + ", ".join(missing_masks[:10])
                + (" ..." if len(missing_masks) > 10 else "")
            )
        if missing_images:
            details.append(
                "masks without images=" + ", ".join(missing_images[:10])
                + (" ..." if len(missing_images) > 10 else "")
            )
        raise AuditError(f"{partition} stem matching failed: {'; '.join(details)}")

    if len(images) != expected_count:
        raise AuditError(
            f"{partition} must contain exactly {expected_count} image/mask pairs; "
            f"found {len(images)}"
        )

    return [(images[key], masks[key]) for key in sorted(images)]


def _verify_image(path: Path, expected_size: tuple[int, int]) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            size = image.size
            image.load()
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
        raise AuditError(f"Corrupt or unreadable image: {path}: {exc}") from exc

    if size != expected_size:
        raise AuditError(
            f"Image has wrong size: {path}: expected {expected_size}, found {size}"
        )
    return size


def _mask_pixel_counts(
    path: Path, expected_size: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    try:
        with Image.open(path) as mask:
            mask.verify()
        with Image.open(path) as mask:
            size = mask.size
            rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    except (OSError, ValueError, SyntaxError, UnidentifiedImageError) as exc:
        raise AuditError(f"Corrupt or unreadable mask: {path}: {exc}") from exc

    if size != expected_size:
        raise AuditError(
            f"Mask has wrong size: {path}: expected {expected_size}, found {size}"
        )

    # Encode RGB triples as exact 24-bit integers.  This avoids a costly
    # np.unique(..., axis=0) over more than one hundred million pixels.
    rgb32 = rgb.astype(np.uint32, copy=False)
    encoded = (
        (rgb32[..., 0] << np.uint32(16))
        | (rgb32[..., 1] << np.uint32(8))
        | rgb32[..., 2]
    )
    observed_codes, observed_counts = np.unique(encoded, return_counts=True)

    allowed_codes = {
        (red << 16) | (green << 8) | blue: class_index
        for class_index, (red, green, blue) in enumerate(ALLOWED_MASK_COLORS)
    }
    invalid_codes = [int(code) for code in observed_codes if int(code) not in allowed_codes]
    if invalid_codes:
        invalid_colors = [
            ((code >> 16) & 255, (code >> 8) & 255, code & 255)
            for code in invalid_codes
        ]
        preview = ", ".join(str(color) for color in invalid_colors[:10])
        suffix = " ..." if len(invalid_colors) > 10 else ""
        raise AuditError(f"Mask contains disallowed RGB color(s): {path}: {preview}{suffix}")

    counts = [0, 0, 0, 0]
    for code, count in zip(observed_codes.tolist(), observed_counts.tolist()):
        counts[allowed_codes[int(code)]] = int(count)
    return size, (counts[0], counts[1], counts[2], counts[3])


def _audit_partition(
    dataset_root: Path,
    partition: str,
    expected_count: int,
    expected_size: tuple[int, int],
) -> list[SampleRecord]:
    partition_root = dataset_root / partition
    pairs = _matched_pairs(
        partition_root / "images_512",
        partition_root / "mask_512",
        expected_count,
        partition,
    )

    records: list[SampleRecord] = []
    for image_path, mask_path in pairs:
        image_size = _verify_image(image_path, expected_size)
        mask_size, pixel_counts = _mask_pixel_counts(mask_path, expected_size)
        if image_size != mask_size:
            raise AuditError(
                f"Image/mask size mismatch for stem {image_path.stem!r}: "
                f"image={image_size}, mask={mask_size}"
            )

        present_classes = [index for index, count in enumerate(pixel_counts) if count > 0]
        if not present_classes:
            raise AuditError(f"Mask has no pixels: {mask_path}")

        records.append(
            SampleRecord(
                source_partition=partition,
                stem=image_path.stem,
                image_path=image_path,
                mask_path=mask_path,
                image_relative_path=image_path.relative_to(dataset_root).as_posix(),
                mask_relative_path=mask_path.relative_to(dataset_root).as_posix(),
                image_sha256=sha256_file(image_path),
                mask_sha256=sha256_file(mask_path),
                max_severity=max(present_classes),
                pixel_counts=pixel_counts,
            )
        )
    return records


def _duplicate_hash_groups(
    records: Sequence[SampleRecord], attribute: str
) -> list[tuple[str, list[str]]]:
    by_hash: dict[str, list[str]] = defaultdict(list)
    for record in records:
        digest = getattr(record, attribute)
        if attribute == "image_sha256":
            relative_path = record.image_relative_path
        else:
            relative_path = record.mask_relative_path
        by_hash[digest].append(relative_path)
    return sorted(
        (digest, sorted(paths)) for digest, paths in by_hash.items() if len(paths) > 1
    )


def _reject_exact_duplicates(records: Sequence[SampleRecord]) -> None:
    image_duplicates = _duplicate_hash_groups(records, "image_sha256")
    mask_duplicates = _duplicate_hash_groups(records, "mask_sha256")
    if not image_duplicates and not mask_duplicates:
        return

    details: list[str] = []
    for label, groups in (("image", image_duplicates), ("mask", mask_duplicates)):
        for digest, paths in groups[:10]:
            details.append(f"{label} {digest}: {', '.join(paths)}")
        if len(groups) > 10:
            details.append(f"{label}: {len(groups) - 10} additional duplicate groups")
    raise AuditError("Exact SHA-256 duplicate(s) found: " + "; ".join(details))


def _allocate_validation_counts(
    strata_sizes: dict[int, int], target_count: int
) -> dict[int, int]:
    total = sum(strata_sizes.values())
    if not 0 < target_count < total:
        raise AuditError(
            f"Validation count must be between 1 and {total - 1}; found {target_count}"
        )

    # When possible, every non-singleton maximum-severity stratum appears in
    # both train and validation.  A singleton is kept in train.
    lower = {
        severity: 1 if size >= 2 else 0 for severity, size in strata_sizes.items()
    }
    upper = {
        severity: size - 1 if size >= 2 else 0
        for severity, size in strata_sizes.items()
    }
    minimum = sum(lower.values())
    maximum = sum(upper.values())
    if not minimum <= target_count <= maximum:
        raise AuditError(
            "Cannot create a stratified split that keeps each non-singleton "
            f"severity in train and validation: target={target_count}, "
            f"feasible={minimum}..{maximum}, strata={strata_sizes}"
        )

    ideals = {
        severity: size * target_count / total
        for severity, size in strata_sizes.items()
    }
    allocation = {
        severity: min(upper[severity], max(lower[severity], math.floor(ideals[severity])))
        for severity in strata_sizes
    }

    while sum(allocation.values()) < target_count:
        candidates = [
            severity
            for severity in sorted(strata_sizes)
            if allocation[severity] < upper[severity]
        ]
        chosen = max(
            candidates,
            key=lambda severity: (
                ideals[severity] - allocation[severity],
                strata_sizes[severity],
                -severity,
            ),
        )
        allocation[chosen] += 1

    while sum(allocation.values()) > target_count:
        candidates = [
            severity
            for severity in sorted(strata_sizes)
            if allocation[severity] > lower[severity]
        ]
        chosen = max(
            candidates,
            key=lambda severity: (
                allocation[severity] - ideals[severity],
                strata_sizes[severity],
                -severity,
            ),
        )
        allocation[chosen] -= 1

    return allocation


def _stratified_split(
    records: Sequence[SampleRecord], seed: int, val_ratio: float
) -> tuple[list[SampleRecord], list[SampleRecord], dict[int, int]]:
    if not 0.0 < val_ratio < 1.0:
        raise AuditError(f"--val-ratio must be between 0 and 1; found {val_ratio}")

    strata: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in records:
        strata[record.max_severity].append(record)
    for samples in strata.values():
        samples.sort(key=lambda record: (record.image_relative_path.casefold(), record.image_relative_path))

    target_count = math.ceil(len(records) * val_ratio)
    allocation = _allocate_validation_counts(
        {severity: len(samples) for severity, samples in sorted(strata.items())},
        target_count,
    )

    generator = random.Random(seed)
    validation_keys: set[str] = set()
    for severity in sorted(strata):
        samples = list(strata[severity])
        generator.shuffle(samples)
        validation_keys.update(
            record.image_relative_path for record in samples[: allocation[severity]]
        )

    train = [record for record in records if record.image_relative_path not in validation_keys]
    validation = [record for record in records if record.image_relative_path in validation_keys]
    sort_key = lambda record: (record.image_relative_path.casefold(), record.image_relative_path)
    train.sort(key=sort_key)
    validation.sort(key=sort_key)

    if len(validation) != target_count or len(train) + len(validation) != len(records):
        raise AuditError("Internal error while constructing the stratified split")
    if any(record.source_partition != "Train" for record in train + validation):
        raise AuditError("Internal error: public Test record entered train/validation")
    if any(record.max_severity == 3 for record in records):
        if not any(record.max_severity == 3 for record in train):
            raise AuditError("Train split contains no Severe sample")
        if not any(record.max_severity == 3 for record in validation):
            raise AuditError("Validation split contains no Severe sample")

    return train, validation, allocation


def _split_statistics(records: Sequence[SampleRecord]) -> dict[str, object]:
    max_severity_counts = {name: 0 for name in CLASS_NAMES}
    class_presence_counts = {name: 0 for name in CLASS_NAMES}
    pixel_counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)

    for record in records:
        max_severity_counts[CLASS_NAMES[record.max_severity]] += 1
        for class_index, count in enumerate(record.pixel_counts):
            pixel_counts[class_index] += count
            if count > 0:
                class_presence_counts[CLASS_NAMES[class_index]] += 1

    total_pixels = int(pixel_counts.sum())
    return {
        "pair_count": len(records),
        "max_severity_image_counts": max_severity_counts,
        "class_presence_image_counts": class_presence_counts,
        "pixel_counts": {
            name: int(pixel_counts[index]) for index, name in enumerate(CLASS_NAMES)
        },
        "pixel_ratios": {
            name: (float(pixel_counts[index]) / total_pixels if total_pixels else 0.0)
            for index, name in enumerate(CLASS_NAMES)
        },
    }


def _manifest_sha256(records: Iterable[SampleRecord]) -> str:
    digest = hashlib.sha256()
    for record in sorted(records, key=lambda item: item.image_relative_path):
        line = (
            f"{record.image_relative_path}\t{record.image_sha256}\t"
            f"{record.mask_relative_path}\t{record.mask_sha256}\n"
        )
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def _csv_bytes(
    train: Sequence[SampleRecord], validation: Sequence[SampleRecord]
) -> bytes:
    columns = (
        "image_path",
        "mask_path",
        "split",
        "max_severity",
        "group_id",
        "image_sha256",
        "mask_sha256",
    )
    text = io.StringIO(newline="")
    writer = csv.DictWriter(text, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for split_name, records in (("train", train), ("val", validation)):
        for record in records:
            writer.writerow(
                {
                    "image_path": record.image_relative_path,
                    "mask_path": record.mask_relative_path,
                    "split": split_name,
                    "max_severity": CLASS_NAMES[record.max_severity],
                    # The public files do not provide a reliable structure/member
                    # group identifier.  Keep this explicitly empty rather than
                    # inventing a grouping key from the numeric filename.
                    "group_id": "",
                    "image_sha256": record.image_sha256,
                    "mask_sha256": record.mask_sha256,
                }
            )
    return text.getvalue().encode("utf-8")


def audit_and_split(
    dataset_root: Path,
    output_csv: Path,
    *,
    seed: int = 42,
    val_ratio: float = 0.20,
    stats_output: Path | None = None,
    sha256_output: Path | None = None,
    expected_train_pairs: int = EXPECTED_TRAIN_PAIRS,
    expected_test_pairs: int = EXPECTED_TEST_PAIRS,
    expected_size: tuple[int, int] = EXPECTED_SIZE,
) -> dict[str, object]:
    """Run the audit and write CSV, JSON statistics, and a SHA-256 sidecar.

    The ``expected_*`` arguments exist to permit small synthetic unit fixtures.
    The command-line interface intentionally does not expose them and therefore
    always enforces the official 396/44 counts and 512x512 size.
    """

    dataset_root = dataset_root.expanduser().resolve()
    output_csv = output_csv.expanduser().resolve()
    if not dataset_root.is_dir():
        raise AuditError(f"Dataset root does not exist: {dataset_root}")

    if stats_output is None:
        stats_output = output_csv.with_name(f"{output_csv.stem}.stats.json")
    else:
        stats_output = stats_output.expanduser().resolve()
    if sha256_output is None:
        sha256_output = Path(f"{output_csv}.sha256")
    else:
        sha256_output = sha256_output.expanduser().resolve()

    artifact_paths = {output_csv, stats_output, sha256_output}
    if len(artifact_paths) != 3:
        raise AuditError("CSV, statistics JSON, and SHA-256 sidecar paths must be distinct")

    source_train = _audit_partition(
        dataset_root, "Train", expected_train_pairs, expected_size
    )
    test = _audit_partition(dataset_root, "Test", expected_test_pairs, expected_size)
    all_records = source_train + test
    _reject_exact_duplicates(all_records)

    train, validation, allocation = _stratified_split(source_train, seed, val_ratio)
    csv_data = _csv_bytes(train, validation)
    csv_sha256 = hashlib.sha256(csv_data).hexdigest()

    statistics: dict[str, object] = {
        "schema_version": 1,
        "dataset_root": str(dataset_root),
        "csv_paths_relative_to": "dataset_root",
        "seed": seed,
        "val_ratio": val_ratio,
        "expected_image_size": list(expected_size),
        "expected_source_counts": {
            "Train": expected_train_pairs,
            "Test": expected_test_pairs,
        },
        "class_names": list(CLASS_NAMES),
        "allowed_mask_colors_rgb": [list(color) for color in ALLOWED_MASK_COLORS],
        "allowed_extensions": {
            "images": sorted(IMAGE_EXTENSIONS),
            "masks": sorted(MASK_EXTENSIONS),
        },
        "audit": {
            "stem_pairing_complete": True,
            "all_files_readable": True,
            "all_sizes_match": True,
            "all_masks_use_only_allowed_rgb_colors": True,
            "image_sha256_duplicate_groups": 0,
            "mask_sha256_duplicate_groups": 0,
        },
        "split_policy": {
            "source": "Train only",
            "stratify_by": "maximum mask severity (Good < Fair < Poor < Severe)",
            "rounding": "ceil(source_count * val_ratio)",
            "grouping_applied": False,
            "grouping_note": (
                "No reliable site/member/camera-run group ID is present in the public "
                "numeric filenames; group_id is intentionally blank."
            ),
            "validation_allocations_by_max_severity": {
                CLASS_NAMES[severity]: count for severity, count in sorted(allocation.items())
            },
            "test_policy": "audited for integrity only; excluded from split CSV",
        },
        "splits": {
            "source_train": _split_statistics(source_train),
            "train": _split_statistics(train),
            "val": _split_statistics(validation),
            "test": _split_statistics(test),
        },
        "manifest_sha256": {
            "source_train": _manifest_sha256(source_train),
            "test": _manifest_sha256(test),
        },
        "artifacts": {
            "split_csv": output_csv.name,
            "split_csv_sha256": csv_sha256,
            "split_csv_sha256_sidecar": sha256_output.name,
            "statistics_json": stats_output.name,
        },
    }

    stats_data = (
        json.dumps(statistics, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    sidecar_data = f"{csv_sha256}  {output_csv.name}\n".encode("ascii")

    _atomic_write(output_csv, csv_data)
    _atomic_write(stats_output, stats_data)
    _atomic_write(sha256_output, sidecar_data)

    # Verify the on-disk bytes, not only the in-memory content used above.
    written_sha256 = sha256_file(output_csv)
    if written_sha256 != csv_sha256:
        raise AuditError(
            f"Written split CSV hash mismatch: expected {csv_sha256}, found {written_sha256}"
        )
    return statistics


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit the Virginia Tech CSSD 512x512 dataset and stratify only "
            "the 396-pair Train partition into train/validation CSV rows."
        )
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        type=Path,
        help="Root containing Train/images_512, Train/mask_512, Test/images_512, and Test/mask_512",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed (default: 42)")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Fraction of official Train assigned to validation (default: 0.20)",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output train/val CSV path")
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help="Statistics JSON path (default: <output stem>.stats.json)",
    )
    parser.add_argument(
        "--sha256-output",
        type=Path,
        default=None,
        help="CSV SHA-256 sidecar path (default: <output>.sha256)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    try:
        statistics = audit_and_split(
            args.dataset_root,
            args.output,
            seed=args.seed,
            val_ratio=args.val_ratio,
            stats_output=args.stats_output,
            sha256_output=args.sha256_output,
        )
    except AuditError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    split_stats = statistics["splits"]
    artifacts = statistics["artifacts"]
    print("Dataset audit passed.")
    print(
        "pairs: "
        f"source Train={split_stats['source_train']['pair_count']}, "
        f"train={split_stats['train']['pair_count']}, "
        f"val={split_stats['val']['pair_count']}, "
        f"locked Test={split_stats['test']['pair_count']}"
    )
    print(f"split CSV: {args.output.expanduser().resolve()}")
    print(f"split CSV SHA-256: {artifacts['split_csv_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
