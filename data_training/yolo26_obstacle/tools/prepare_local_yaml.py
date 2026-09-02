from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a local-path-safe copy of a Roboflow data.yaml.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Roboflow data.yaml missing: {source}")
    config = yaml.safe_load(source.read_text(encoding="utf-8-sig"))
    root = source.parent
    expected = {"train": "train/images", "val": "valid/images", "test": "test/images"}
    missing = [relative for relative in expected.values() if not (root / relative).is_dir()]
    if missing:
        raise FileNotFoundError(f"Downloaded split directories missing under {root}: {missing}")
    local = dict(config)
    local["path"] = root.as_posix()
    local.update(expected)
    output.write_text(yaml.safe_dump(local, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"Local training YAML ready: {output}")


if __name__ == "__main__":
    main()
