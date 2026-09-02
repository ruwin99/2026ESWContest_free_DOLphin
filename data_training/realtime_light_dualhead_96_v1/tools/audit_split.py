from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import readiness, resolve_path, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit all pre-training contracts without opening sealed data.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--allow-blocked", action="store_true")
    args = parser.parse_args()
    report = readiness(args.config.resolve())
    output = resolve_path("data_training/realtime_light_dualhead_96_v1/reports/readiness_audit.json")
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Readiness report: {output}")
    if not report["ready_for_training"] and not args.allow_blocked:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
