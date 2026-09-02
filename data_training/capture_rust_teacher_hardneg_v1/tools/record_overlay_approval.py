from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from common import load_config, read_manifest, resolve_path, write_json


FIELDS = ("sample_id", "group_id", "labeler", "reviewer", "decision", "reviewed_utc", "notes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the user's explicit representative overlay approval by hard-negative group.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = resolve_path(config["paths"]["manifest_lock"])
    if lock.exists():
        raise RuntimeError(f"Manifest is already locked: {lock}")
    rows = []
    for split in ("train", "validation"):
        key = "train_manifest" if split == "train" else "validation_manifest"
        rows.extend(read_manifest(resolve_path(config["paths"][key]), expected_split=split))
    representatives = {}
    for row in rows:
        if row["source_type"] == "hard_negative":
            representatives.setdefault(row["group_id"], row)
    if not representatives:
        raise RuntimeError("No hard-negative groups found")
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    qa_rows = [
        {
            "sample_id": row["sample_id"],
            "group_id": group,
            "labeler": args.approved_by,
            "reviewer": args.approved_by,
            "decision": "approved",
            "reviewed_utc": timestamp,
            "notes": "Explicit single-user representative overlay approval; no independent second reviewer. Same demo-domain cross-session visual similarity accepted by user.",
        }
        for group, row in sorted(representatives.items())
    ]
    output = resolve_path(config["paths"]["two_person_overlay_qa"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(qa_rows)
    report = {
        "recorded_utc": timestamp,
        "approved_by": args.approved_by,
        "groups": len(qa_rows),
        "independent_second_reviewer": False,
        "scope": "representative overlay approval and cross-session visual similarity acceptance",
    }
    write_json(output.with_suffix(".approval.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
