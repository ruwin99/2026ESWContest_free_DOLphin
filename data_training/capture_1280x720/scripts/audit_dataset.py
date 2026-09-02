from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from common import (
    CAPTURE_ROOT,
    EXTERNAL_HEIGHT,
    EXTERNAL_WIDTH,
    IMAGE_EXTENSIONS,
    MASK_COLORS_RGB,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit capture 1280x720 datasets.")
    parser.add_argument("--root", type=Path, default=CAPTURE_ROOT)
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def file_map(directory: Path) -> dict[str, Path]:
    return {
        path.stem: path
        for path in sorted(directory.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def rust_values(array: np.ndarray) -> set[Any]:
    if array.ndim == 2:
        return set(np.unique(array).tolist())
    rgb = array[..., :3]
    colors = np.unique(rgb.reshape(-1, 3), axis=0)
    return {tuple(int(value) for value in color) for color in colors}


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    summaries: dict[str, Any] = {}
    image_hash_splits: dict[str, set[str]] = defaultdict(set)

    for task in ("rust", "crack"):
        summaries[task] = {}
        for split in ("train", "validation", "test"):
            base = root / task / split
            kinds = ("images", "masks") if task == "rust" else ("images", "masks", "edges")
            maps = {kind: file_map(base / kind) for kind in kinds}
            expected = set(maps["images"])
            counts = {kind: len(files) for kind, files in maps.items()}
            if any(set(files) != expected for files in maps.values()):
                errors.append(f"stem mismatch: {task}/{split}: {counts}")
            summaries[task][split] = {"counts": counts, "paired": len(expected)}

            for stem in sorted(set().union(*(set(files) for files in maps.values()))):
                for kind, files in maps.items():
                    path = files.get(stem)
                    if path is None:
                        continue
                    try:
                        with Image.open(path) as source:
                            source.load()
                            width, height = source.size
                            mode = source.mode
                            array = np.asarray(source).copy()
                        if (width, height) != (EXTERNAL_WIDTH, EXTERNAL_HEIGHT):
                            errors.append(
                                f"wrong size {width}x{height}: {path.relative_to(root)}"
                            )
                        label_values: Any = None
                        if kind != "images":
                            if task == "rust" and kind == "masks":
                                values = rust_values(array)
                                valid_index = values.issubset({0, 1, 2, 3, 255})
                                valid_palette = values.issubset(set(MASK_COLORS_RGB))
                                if not (valid_index or valid_palette):
                                    errors.append(
                                        f"invalid rust label values: {path.relative_to(root)}"
                                    )
                                label_values = sorted(str(value) for value in values)
                            else:
                                values = set(np.unique(array).tolist())
                                if not values.issubset({0, 1, 255}):
                                    errors.append(
                                        f"invalid binary values {sorted(values)}: "
                                        f"{path.relative_to(root)}"
                                    )
                                label_values = sorted(values)
                        digest = sha256_file(path)
                        if kind == "images":
                            image_hash_splits[digest].add(f"{task}/{split}")
                        rows.append(
                            {
                                "task": task,
                                "split": split,
                                "stem": stem,
                                "kind": kind,
                                "relative_path": path.relative_to(root).as_posix(),
                                "width": width,
                                "height": height,
                                "mode": mode,
                                "sha256": digest,
                                "label_values": json.dumps(label_values, ensure_ascii=False),
                            }
                        )
                    except Exception as exc:
                        errors.append(f"unreadable {path.relative_to(root)}: {exc}")

    duplicates = {
        digest: sorted(splits)
        for digest, splits in image_hash_splits.items()
        if len(splits) > 1
    }
    if duplicates:
        errors.append(f"cross-split exact image duplicates: {len(duplicates)}")

    ready = not errors and all(
        summaries[task][split]["paired"] > 0
        for task in ("rust", "crack")
        for split in ("train", "validation", "test")
    )
    report = {
        "audited_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "root": str(root),
        "required_size": [EXTERNAL_WIDTH, EXTERNAL_HEIGHT],
        "ready_for_training_and_locked_evaluation": ready,
        "summaries": summaries,
        "cross_split_duplicate_groups": duplicates,
        "errors": errors,
        "note": (
            "Empty folders mean the real 1280x720 capture dataset has not been supplied yet."
            if not rows
            else None
        ),
    }
    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    with (manifest_dir / "dataset_files.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "task",
                "split",
                "stem",
                "kind",
                "relative_path",
                "width",
                "height",
                "mode",
                "sha256",
                "label_values",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    write_json(manifest_dir / "dataset_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.require_ready and not ready:
        raise RuntimeError("Capture dataset audit is not ready; inspect dataset_audit.json")


if __name__ == "__main__":
    main()
