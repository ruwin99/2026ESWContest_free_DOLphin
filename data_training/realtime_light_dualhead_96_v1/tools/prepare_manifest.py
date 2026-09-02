from __future__ import annotations

import argparse
from pathlib import Path

from common import load_config, read_rows, resolve_path, write_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Materialize reviewed train/calibration/validation rows; never reads sealed-test content."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config.resolve())
    source = resolve_path(config["paths"]["source_inventory"])
    if not source.is_file():
        raise FileNotFoundError(
            f"Reviewed source inventory is missing: {source}. Fill the provided CSV template first."
        )
    rows = read_rows(source)
    if not rows:
        raise RuntimeError("Source inventory template has no reviewed rows")
    required = list(config["manifest"]["required_columns"])
    if set(required) - set(rows[0]):
        raise ValueError(f"Source inventory columns are incomplete: {sorted(set(required)-set(rows[0]))}")
    if any(row.get("split") == "sealed_test" for row in rows):
        raise PermissionError("Sealed-test rows must be held by the independent evaluator")
    for split in ("train", "development_calibration", "validation"):
        destination = resolve_path(config["paths"][f"{split}_manifest"])
        selected = [row for row in rows if row.get("split") == split]
        write_rows(destination, required, selected)
        print(f"{split}: {len(selected)} rows -> {destination}")


if __name__ == "__main__":
    main()
