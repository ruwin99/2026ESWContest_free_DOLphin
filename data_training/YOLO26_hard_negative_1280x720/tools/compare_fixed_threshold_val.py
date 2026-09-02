from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def evaluate(model_path: Path, data: Path, project: Path, name: str) -> dict[str, float]:
    metrics = YOLO(str(model_path)).val(
        data=str(data),
        imgsz=1280,
        rect=True,
        conf=0.30,
        iou=0.70,
        batch=16,
        workers=2,
        device=0,
        plots=False,
        project=str(project),
        name=name,
        exist_ok=False,
        verbose=False,
    )
    return {key: float(value) for key, value in metrics.results_dict.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "imgsz": 1280,
        "rect": True,
        "confidence": 0.30,
        "nms_iou": 0.70,
        "baseline": evaluate(args.baseline, args.data, args.project, "all1410_baseline_val_conf030_v1"),
        "candidate": evaluate(args.candidate, args.data, args.project, "all1410_candidate_val_conf030_v1"),
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


