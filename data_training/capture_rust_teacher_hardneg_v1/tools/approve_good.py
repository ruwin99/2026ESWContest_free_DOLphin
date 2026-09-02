from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
from pathlib import Path

from common import load_config, read_manifest, resolve_path, write_manifest, write_json


VALIDATION_SOURCE = "for model test v7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record an explicit user bulk approval that all inventoried hard negatives are rust-free Good.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    lock = resolve_path(config["paths"]["manifest_lock"])
    if lock.exists():
        raise RuntimeError(f"Manifest is locked and cannot be approved again: {lock}")
    review_path = resolve_path(config["paths"]["review_manifest"])
    rows = read_manifest(review_path)
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    counts = {"train": 0, "validation": 0}
    for row in rows:
        source = row["source_dataset"]
        split = "validation" if source == VALIDATION_SOURCE else "train"
        row.update({
            "split": split,
            "camera_id": "unrecorded-user-camera",
            "session_id": source,
            "source_print_id": "unrecorded-demo-source",
            "placement_id": "unrecorded",
            "lighting_id": source,
            "group_id": f"unrecorded-demo-source|unrecorded|{source}",
            "label_status": "approved_good",
            "labeler": args.approved_by,
            "reviewer": "independent-overlay-qa-not-yet-performed",
            "notes": (row["notes"] + "; " if row["notes"] else "") + f"explicit bulk Good approval by {args.approved_by} at {timestamp}; second-person sampled overlay QA pending",
        })
        counts[split] += 1
    write_manifest(review_path, rows)
    report = {
        "approved_utc": timestamp,
        "approved_by": args.approved_by,
        "label_status": "approved_good",
        "rows": len(rows),
        "split_policy": {"validation_source": VALIDATION_SOURCE, "all_other_sources": "train"},
        "counts": counts,
        "second_person_sampled_overlay_qa": "PENDING",
        "sealed_test": "PENDING",
    }
    output = review_path.with_suffix(".approval.json")
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

