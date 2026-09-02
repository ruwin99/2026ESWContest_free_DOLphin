from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

from PIL import Image
from ultralytics import YOLO


CONFIDENCE = 0.30
NMS_IOU = 0.70


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def predict(model_path: Path, images: list[Path], device: str, batch: int) -> dict:
    results = YOLO(str(model_path)).predict(
        source=[str(path) for path in images],
        imgsz=(720, 1280),
        conf=CONFIDENCE,
        iou=NMS_IOU,
        device=device,
        batch=batch,
        save=False,
        verbose=False,
    )
    confidences = [
        [float(value) for value in result.boxes.conf.detach().cpu().tolist()]
        for result in results
    ]
    return {
        "model": str(model_path),
        "model_sha256": sha256(model_path),
        "negative_images": len(confidences),
        "false_positive_images": sum(bool(values) for values in confidences),
        "false_positive_boxes": sum(len(values) for values in confidences),
        "negative_frame_fpr": sum(bool(values) for values in confidences) / len(confidences),
        "boxes_per_frame": sum(len(values) for values in confidences) / len(confidences),
        "max_confidence": max((max(values) for values in confidences if values), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the sealed negative test exactly once with a fixed candidate."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(os.environ["RAIL_ROBOT_ROOT"])
        if os.environ.get("RAIL_ROBOT_ROOT")
        else None,
        help="rail_robot project root; alternatively set RAIL_ROBOT_ROOT",
    )
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.root is None:
        parser.error("--root or the RAIL_ROBOT_ROOT environment variable is required")
    if args.batch < 1:
        parser.error("--batch must be at least 1")
    root = args.root.expanduser().resolve()
    project = root / "data_training/yolo26_obstacle_hardneg_v1"
    final_source = root / "for model test"
    baseline = project / "models/baseline/best.pt"
    candidate = root / "outputs/training/yolo26_obstacle_hardneg_v1/yolo26n_hn_all1410_1280rect_clsout_lr2e5_seed42_b8_v2/weights/best.pt"
    report_path = project / "reports/locked_final_test_once.json"
    if report_path.exists():
        raise FileExistsError(f"Locked final test was already evaluated: {report_path}")

    expected_candidate_sha = "5f08e1fd963627aa81522d00b75b72b4c633016b162e7de529f9849786881d56"
    if sha256(candidate) != expected_candidate_sha:
        raise RuntimeError("Fixed candidate SHA changed")

    images = sorted(final_source.glob("*.jpg"))
    review_rows = list(
        csv.DictReader((final_source / "review/image_review.csv").open("r", encoding="utf-8-sig", newline=""))
    )
    reviewed = {row["filename"]: row for row in review_rows}
    if len(images) != 66 or len(review_rows) != 66:
        raise RuntimeError(f"Locked final count mismatch: images={len(images)}, review={len(review_rows)}")

    training_rows = list(
        csv.DictReader((project / "manifests/split_manifest_all1410_v2.csv").open("r", encoding="utf-8-sig", newline=""))
    )
    training_hashes = {row["source_sha256"] for row in training_rows}
    final_hashes: set[str] = set()
    for image in images:
        row = reviewed.get(image.name)
        if row is None:
            raise RuntimeError(f"Final image missing review row: {image}")
        digest = sha256(image)
        if digest != row["sha256"]:
            raise RuntimeError(f"Final image SHA changed: {image}")
        with Image.open(image) as decoded:
            decoded.verify()
        with Image.open(image) as decoded:
            if decoded.size != (1280, 720):
                raise RuntimeError(f"Final image shape mismatch {decoded.size}: {image}")
        if digest in training_hashes:
            raise RuntimeError(f"Locked final image leaked into train/val/test manifest: {image}")
        if digest in final_hashes:
            raise RuntimeError(f"Duplicate inside locked final set: {image}")
        final_hashes.add(digest)

    report = {
        "status": "LOCKED_FINAL_EVALUATED_ONCE",
        "source": str(final_source),
        "source_images": len(images),
        "review_png_files_excluded": 3,
        "exact_overlap_with_training_manifest": 0,
        "camera_shape_hw": [720, 1280],
        "stride_aligned_inference_shape_hw": [736, 1280],
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
        "baseline": predict(baseline, images, args.device, args.batch),
        "candidate": predict(candidate, images, args.device, args.batch),
    }
    report["delta_candidate_minus_baseline"] = {
        "false_positive_images": report["candidate"]["false_positive_images"] - report["baseline"]["false_positive_images"],
        "false_positive_boxes": report["candidate"]["false_positive_boxes"] - report["baseline"]["false_positive_boxes"],
        "negative_frame_fpr": report["candidate"]["negative_frame_fpr"] - report["baseline"]["negative_frame_fpr"],
        "max_confidence": report["candidate"]["max_confidence"] - report["baseline"]["max_confidence"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

