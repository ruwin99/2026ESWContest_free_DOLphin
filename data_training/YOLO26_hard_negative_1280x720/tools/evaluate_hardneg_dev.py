from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


THRESHOLDS = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


def evaluate(model_path: Path, image_dir: Path) -> dict:
    model = YOLO(str(model_path))
    results = model.predict(
        source=str(image_dir),
        imgsz=(720, 1280),
        conf=min(THRESHOLDS),
        iou=0.7,
        device=0,
        batch=8,
        workers=2,
        stream=False,
        save=False,
        verbose=False,
    )
    confidences = [
        [float(value) for value in result.boxes.conf.detach().cpu().tolist()]
        for result in results
    ]
    by_threshold = {}
    for threshold in THRESHOLDS:
        detections = [sum(value >= threshold for value in values) for values in confidences]
        by_threshold[f"{threshold:.2f}"] = {
            "false_positive_frames": sum(count > 0 for count in detections),
            "false_positive_boxes": sum(detections),
            "false_positive_frame_rate": sum(count > 0 for count in detections) / len(detections),
        }
    return {
        "model": str(model_path),
        "images": len(confidences),
        "max_confidence": max((max(values) for values in confidences if values), default=0.0),
        "thresholds": by_threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    report = {
        "camera_shape_hw": [720, 1280],
        "development_set_is_empty_label_only": True,
        "baseline": evaluate(args.baseline, args.images),
        "candidate": evaluate(args.candidate, args.images),
    }
    report["delta"] = {
        threshold: {
            "false_positive_frames": (
                report["candidate"]["thresholds"][threshold]["false_positive_frames"]
                - report["baseline"]["thresholds"][threshold]["false_positive_frames"]
            ),
            "false_positive_boxes": (
                report["candidate"]["thresholds"][threshold]["false_positive_boxes"]
                - report["baseline"]["thresholds"][threshold]["false_positive_boxes"]
            ),
        }
        for threshold in report["baseline"]["thresholds"]
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


