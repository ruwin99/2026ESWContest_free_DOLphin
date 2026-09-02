from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from PIL import Image

from common import (
    DEFAULT_DATA_ROOT,
    EXPECTED_COUNTS,
    EXPECTED_IMAGE_SIZE,
    TRAINING_ROOT,
    json_dump,
    paired_items,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the official Steelcrack dataset.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=TRAINING_ROOT / "audit" / "steelcrack_audit.json",
    )
    return parser.parse_args()


def audit_split(data_root: Path, split: str) -> tuple[dict, dict[str, str], list[str]]:
    items = paired_items(data_root, split)
    errors: list[str] = []
    image_modes: collections.Counter[str] = collections.Counter()
    mask_values: set[int] = set()
    edge_values: set[int] = set()
    mask_positive_pixels = 0
    edge_positive_pixels = 0
    total_pixels = 0
    empty_masks = 0
    empty_edges = 0
    image_hashes: dict[str, str] = {}

    for item in items:
        try:
            with Image.open(item.image) as source:
                source.verify()
            with Image.open(item.image) as source:
                image_modes[source.mode] += 1
                if source.size != EXPECTED_IMAGE_SIZE:
                    errors.append(f"{split}/{item.name}: image size={source.size}")
        except Exception as exc:
            errors.append(f"{split}/{item.name}: corrupt image: {exc}")
            continue

        image_hashes[item.name] = sha256_file(item.image)
        for kind, path, values in (
            ("mask", item.mask, mask_values),
            ("edge", item.edge, edge_values),
        ):
            try:
                with Image.open(path) as source:
                    source.verify()
                with Image.open(path) as source:
                    if source.size != EXPECTED_IMAGE_SIZE:
                        errors.append(f"{split}/{item.name}: {kind} size={source.size}")
                    array = np.asarray(source.convert("L"), dtype=np.uint8)
                unique_values = np.unique(array)
                values.update(int(value) for value in unique_values)
                positive = int(np.count_nonzero(array))
                if kind == "mask":
                    mask_positive_pixels += positive
                    if positive == 0:
                        empty_masks += 1
                else:
                    edge_positive_pixels += positive
                    if positive == 0:
                        empty_edges += 1
            except Exception as exc:
                errors.append(f"{split}/{item.name}: corrupt {kind}: {exc}")
        total_pixels += EXPECTED_IMAGE_SIZE[0] * EXPECTED_IMAGE_SIZE[1]

    expected_count = EXPECTED_COUNTS[split]
    if len(items) != expected_count:
        errors.append(f"{split}: expected {expected_count} triplets, found {len(items)}")
    if not mask_values.issubset({0, 255}):
        errors.append(f"{split}: non-binary mask values={sorted(mask_values)}")
    if not edge_values.issubset({0, 255}):
        errors.append(f"{split}: non-binary edge values={sorted(edge_values)}")

    report = {
        "triplets": len(items),
        "expected_triplets": expected_count,
        "image_modes": dict(sorted(image_modes.items())),
        "mask_values": sorted(mask_values),
        "edge_values": sorted(edge_values),
        "empty_masks": empty_masks,
        "empty_edges": empty_edges,
        "mask_positive_fraction": mask_positive_pixels / total_pixels if total_pixels else 0.0,
        "edge_positive_fraction": edge_positive_pixels / total_pixels if total_pixels else 0.0,
        "errors": errors,
    }
    return report, image_hashes, errors


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    all_errors: list[str] = []
    split_reports: dict[str, dict] = {}
    split_hashes: dict[str, dict[str, str]] = {}

    for split in EXPECTED_COUNTS:
        report, hashes, errors = audit_split(data_root, split)
        split_reports[split] = report
        split_hashes[split] = hashes
        all_errors.extend(errors)

    cross_split_duplicates: list[dict[str, str]] = []
    splits = list(EXPECTED_COUNTS)
    for left_index, left_split in enumerate(splits):
        left_by_hash: dict[str, list[str]] = collections.defaultdict(list)
        for name, digest in split_hashes[left_split].items():
            left_by_hash[digest].append(name)
        for right_split in splits[left_index + 1 :]:
            for right_name, digest in split_hashes[right_split].items():
                for left_name in left_by_hash.get(digest, []):
                    cross_split_duplicates.append(
                        {
                            "left_split": left_split,
                            "left_name": left_name,
                            "right_split": right_split,
                            "right_name": right_name,
                            "sha256": digest,
                        }
                    )

    if cross_split_duplicates:
        all_errors.append(
            f"exact image duplicates across official splits: {len(cross_split_duplicates)}"
        )

    payload = {
        "dataset": "Steelcrack",
        "data_root": str(data_root),
        "expected_image_size": list(EXPECTED_IMAGE_SIZE),
        "official_split_preserved": True,
        "splits": split_reports,
        "cross_split_exact_image_duplicates": cross_split_duplicates,
        "ok": not all_errors,
        "errors": all_errors,
    }
    json_dump(args.output.resolve(), payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if all_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
