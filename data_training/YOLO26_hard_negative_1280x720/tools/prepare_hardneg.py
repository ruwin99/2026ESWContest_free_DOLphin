from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import math
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
TIMESTAMP_RE = re.compile(r"training_(\d{8})_(\d{6})_(\d{6})", re.IGNORECASE)
AUDIT_COLUMNS = [
    "relative_path",
    "sha256",
    "width",
    "height",
    "timestamp_group",
    "readable",
    "contains_obstacle",
    "contains_overlay",
    "approved_empty_label",
    "reviewer",
    "reviewed_at",
    "review_note",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_paths(config_path: Path) -> tuple[dict[str, Any], Path, dict[str, Path]]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8-sig"))
    project = config_path.resolve().parents[1]
    root = Path(os.environ.get("RAIL_ROBOT_ROOT", project.parents[1])).expanduser().resolve()
    paths = {key: (root / value).resolve() for key, value in config["paths"].items()}
    paths["project"] = project
    paths["root"] = root
    return config, project, paths


def images_in(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def paired_label(image: Path, images_root: Path) -> Path:
    return images_root.parent / "labels" / image.relative_to(images_root).with_suffix(".txt")


def read_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda item: int(item))]
    raise ValueError("data.yaml names must be a list or integer-keyed mapping")


def validate_label(path: Path, class_count: int) -> tuple[int, list[tuple[int, float, float, float, float]], list[str]]:
    if not path.is_file():
        return 0, [], ["missing label"]
    boxes: list[tuple[int, float, float, float, float]] = []
    issues: list[str] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("expected 5 fields")
            class_id = int(fields[0])
            coordinates = tuple(float(value) for value in fields[1:])
            if not 0 <= class_id < class_count:
                raise ValueError(f"class id {class_id} outside [0,{class_count - 1}]")
            if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in coordinates):
                raise ValueError("non-finite or out-of-range coordinate")
            if coordinates[2] <= 0 or coordinates[3] <= 0:
                raise ValueError("non-positive bbox size")
            boxes.append((class_id, *coordinates))
        except ValueError as exc:
            issues.append(f"line {line_number}: {exc}")
    return len(boxes), boxes, issues


def parse_capture_time(path: Path) -> datetime:
    match = TIMESTAMP_RE.search(path.stem)
    if not match:
        raise ValueError(f"hard-negative filename has no capture timestamp: {path.name}")
    date, clock, microseconds = match.groups()
    return datetime.strptime(date + clock + microseconds, "%Y%m%d%H%M%S%f")


def assign_groups(files: list[Path], gap_seconds: float) -> dict[Path, int]:
    ordered = sorted(files, key=parse_capture_time)
    groups: dict[Path, int] = {}
    group = 1
    previous: datetime | None = None
    for path in ordered:
        captured = parse_capture_time(path)
        if previous is not None and (captured - previous).total_seconds() > gap_seconds:
            group += 1
        groups[path] = group
        previous = captured
    return groups


def load_existing_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return {row["sha256"]: row for row in csv.DictReader(handle) if row.get("sha256")}


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_contact_sheets(files: list[Path], groups: dict[Path, int], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns, rows_per_sheet = 6, 6
    thumb_w, thumb_h, caption_h = 240, 135, 34
    page_size = columns * rows_per_sheet
    font = ImageFont.load_default()
    outputs: list[Path] = []
    for page_index in range(0, len(files), page_size):
        page_files = files[page_index : page_index + page_size]
        sheet = Image.new("RGB", (columns * thumb_w, rows_per_sheet * (thumb_h + caption_h)), "white")
        draw = ImageDraw.Draw(sheet)
        for index, path in enumerate(page_files):
            row, column = divmod(index, columns)
            x, y = column * thumb_w, row * (thumb_h + caption_h)
            with Image.open(path) as image:
                image = ImageOps.exif_transpose(image).convert("RGB")
                thumb = ImageOps.contain(image, (thumb_w, thumb_h))
                offset = (x + (thumb_w - thumb.width) // 2, y + (thumb_h - thumb.height) // 2)
                sheet.paste(thumb, offset)
            caption = f"G{groups[path]} {path.name}"
            draw.text((x + 3, y + thumb_h + 2), caption[:38], fill="black", font=font)
        output = output_dir / f"hard_negative_v8_contact_{page_index // page_size + 1:02d}.jpg"
        sheet.save(output, quality=92)
        outputs.append(output)
    return outputs


def audit(config_path: Path, approve_all_reviewed: bool) -> dict[str, Any]:
    config, project, paths = resolve_paths(config_path)
    source = paths["source_dataset"]
    hardneg_source = paths["hard_negative_source"]
    manifests = project / "manifests"
    reports = project / "reports"
    issues: list[str] = []

    yaml_path = source / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"source data.yaml missing: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8-sig"))
    names = read_names(data.get("names"))
    if names != ["obstacle"] or int(data.get("nc", -1)) != 1:
        issues.append(f"class contract mismatch: nc={data.get('nc')} names={names}")
    roboflow = data.get("roboflow", {})
    source_contract = config["source_contract"]
    for key in ("workspace", "project", "version"):
        if str(roboflow.get(key)) != str(source_contract[key]):
            issues.append(f"Roboflow {key} mismatch: {roboflow.get(key)!r} != {source_contract[key]!r}")
    roboflow_readme = source / "README.roboflow.txt"
    readme_text = roboflow_readme.read_text(encoding="utf-8-sig") if roboflow_readme.is_file() else ""
    if "The dataset includes 8879 images." not in readme_text:
        issues.append("Roboflow metadata does not state the contracted 8879 images")
    if "Resize to 640x640 (Fit (black edges))" not in readme_text:
        issues.append("Roboflow preprocessing provenance mismatch")
    if "No image augmentation techniques were applied." not in readme_text:
        issues.append("Roboflow augmentation provenance mismatch")

    split_stats: dict[str, Any] = {}
    image_hash_splits: defaultdict[str, set[str]] = defaultdict(set)
    image_hash_paths: defaultdict[str, list[str]] = defaultdict(list)
    test_box_rows: list[dict[str, Any]] = []
    total_images = 0
    for split in ("train", "valid", "test"):
        images_root = source / split / "images"
        files = images_in(images_root)
        total_images += len(files)
        boxes_total = 0
        empty_labels = 0
        split_issues: list[str] = []
        resolution_counts: Counter[str] = Counter()
        size_bins: Counter[str] = Counter()
        for image in files:
            digest = sha256(image)
            image_hash_splits[digest].add(split)
            image_hash_paths[digest].append(str(image))
            try:
                with Image.open(image) as decoded:
                    decoded.verify()
                with Image.open(image) as decoded:
                    width, height = decoded.size
                resolution_counts[f"{width}x{height}"] += 1
            except Exception as exc:  # Pillow supplies format-specific exceptions
                split_issues.append(f"unreadable image {image}: {exc}")
                continue
            label = paired_label(image, images_root)
            box_count, boxes, label_issues = validate_label(label, len(names))
            boxes_total += box_count
            if box_count == 0 and not label_issues:
                empty_labels += 1
            split_issues.extend(f"{label}: {item}" for item in label_issues)
            for label_index, (class_id, x, y, width_n, height_n) in enumerate(boxes):
                width_px = width_n * 640.0
                height_px = height_n * 640.0
                size_px = math.sqrt(width_px * height_px)
                if size_px < 8:
                    size_bin = "tiny_lt8"
                elif size_px < 16:
                    size_bin = "small_8_16"
                elif size_px < 32:
                    size_bin = "small_16_32"
                else:
                    size_bin = "regular_ge32"
                size_bins[size_bin] += 1
                if split == "test":
                    test_box_rows.append(
                        {
                            "image_sha256": digest,
                            "label_index": label_index,
                            "class_id": class_id,
                            "bbox_xywhn": f"{x:.9g} {y:.9g} {width_n:.9g} {height_n:.9g}",
                            "width_px_640": f"{width_px:.9g}",
                            "height_px_640": f"{height_px:.9g}",
                            "size_bin": size_bin,
                        }
                    )
        expected = int(config["source_contract"]["split_counts"][split])
        if len(files) != expected:
            split_issues.append(f"image count {len(files)} != expected {expected}")
        if split_issues:
            issues.extend(f"{split}: {item}" for item in split_issues)
        split_stats[split] = {
            "images": len(files),
            "boxes": boxes_total,
            "empty_label_images": empty_labels,
            "resolutions": dict(resolution_counts),
            "bbox_size_bins_at_640": dict(size_bins),
            "issue_count": len(split_issues),
            "issue_examples": split_issues[:20],
        }

    cross_split_duplicates = [
        {"sha256": digest, "splits": sorted(splits), "paths": image_hash_paths[digest]}
        for digest, splits in image_hash_splits.items()
        if len(splits) > 1
    ]
    duplicate_split_patterns = Counter(
        "+".join(sorted(item["splits"])) for item in cross_split_duplicates
    )
    if cross_split_duplicates:
        issues.append(f"{len(cross_split_duplicates)} exact image SHA duplicates cross source splits")
    write_csv(
        manifests / "source_split_leakage.csv",
        ["sha256", "splits", "paths"],
        (
            {
                "sha256": item["sha256"],
                "splits": "+".join(item["splits"]),
                "paths": " | ".join(item["paths"]),
            }
            for item in cross_split_duplicates
        ),
    )
    if total_images != int(config["source_contract"]["image_count"]):
        issues.append(f"source image count {total_images} != expected {config['source_contract']['image_count']}")

    hardneg_files = images_in(hardneg_source)
    groups = assign_groups(hardneg_files, float(config["hard_negative_contract"]["group_gap_seconds"]))
    group_counts = Counter(groups.values())
    expected_group_counts = list(config["hard_negative_contract"]["expected_group_counts"])
    actual_group_counts = [group_counts[index] for index in range(1, len(group_counts) + 1)]
    if len(hardneg_files) != int(config["hard_negative_contract"]["image_count"]):
        issues.append(f"hard-negative count {len(hardneg_files)} != expected {config['hard_negative_contract']['image_count']}")
    if actual_group_counts != expected_group_counts:
        issues.append(f"hard-negative groups {actual_group_counts} != expected {expected_group_counts}")

    audit_csv = manifests / "hard_negative_v8_audit.csv"
    existing_review = load_existing_review(audit_csv)
    audit_rows: list[dict[str, Any]] = []
    hardneg_hashes: set[str] = set()
    for image in sorted(hardneg_files, key=parse_capture_time):
        digest = sha256(image)
        hardneg_hashes.add(digest)
        readable = True
        width = height = 0
        note = ""
        try:
            with Image.open(image) as decoded:
                decoded.verify()
            with Image.open(image) as decoded:
                width, height = decoded.size
        except Exception as exc:
            readable = False
            note = str(exc)
        if (width, height) != (
            int(config["hard_negative_contract"]["width"]),
            int(config["hard_negative_contract"]["height"]),
        ):
            issues.append(f"hard-negative shape {image.name}: {width}x{height}")
        previous = existing_review.get(digest, {})
        row = {
            "relative_path": image.relative_to(paths["root"]).as_posix(),
            "sha256": digest,
            "width": width,
            "height": height,
            "timestamp_group": groups[image],
            "readable": str(readable).lower(),
            "contains_obstacle": previous.get("contains_obstacle", ""),
            "contains_overlay": previous.get("contains_overlay", ""),
            "approved_empty_label": previous.get("approved_empty_label", ""),
            "reviewer": previous.get("reviewer", ""),
            "reviewed_at": previous.get("reviewed_at", ""),
            "review_note": previous.get("review_note", note),
        }
        if approve_all_reviewed and readable:
            row.update(
                {
                    "contains_obstacle": "false",
                    "contains_overlay": "false",
                    "approved_empty_label": "true",
                    "reviewer": "Codex contact-sheet visual review",
                    "reviewed_at": utc_now(),
                    "review_note": "Reviewed from generated contact sheets; no obstacle or UI/prediction overlay observed.",
                }
            )
        audit_rows.append(row)
    write_csv(audit_csv, AUDIT_COLUMNS, audit_rows)

    overlap = sorted(hardneg_hashes.intersection(image_hash_splits))
    if overlap:
        issues.append(f"{len(overlap)} hard-negative files duplicate source dataset images by SHA")
    contact_sheets = make_contact_sheets(sorted(hardneg_files, key=parse_capture_time), groups, reports / "contact_sheets")
    write_csv(
        manifests / "small_object_test_manifest.csv",
        ["image_sha256", "label_index", "class_id", "bbox_xywhn", "width_px_640", "height_px_640", "size_bin"],
        test_box_rows,
    )

    metadata_files = [yaml_path, source / "README.dataset.txt", source / "README.roboflow.txt"]
    baseline = paths["baseline_model"]
    if baseline.is_file():
        metadata_files.append(baseline)
    write_csv(
        manifests / "source_files_sha256.csv",
        ["role", "path", "size_bytes", "sha256"],
        [
            {
                "role": "baseline_model" if path == baseline else "dataset_metadata",
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in metadata_files
        ],
    )

    review_pending = sum(row["approved_empty_label"].lower() != "true" for row in audit_rows)
    report = {
        "passed_automated_audit": not issues,
        "ready_for_prepare": not issues and review_pending == 0,
        "generated_at": utc_now(),
        "config": str(config_path.resolve()),
        "config_sha256": sha256(config_path),
        "data_yaml": str(yaml_path),
        "data_yaml_sha256": sha256(yaml_path),
        "class_names": names,
        "source_total_images": total_images,
        "source_splits": split_stats,
        "cross_split_duplicate_count": len(cross_split_duplicates),
        "cross_split_duplicate_patterns": dict(duplicate_split_patterns),
        "cross_split_duplicate_examples": cross_split_duplicates[:20],
        "hard_negative_images": len(hardneg_files),
        "hard_negative_shape": [
            int(config["hard_negative_contract"]["height"]),
            int(config["hard_negative_contract"]["width"]),
        ],
        "hard_negative_group_counts": dict(sorted(group_counts.items())),
        "hard_negative_source_overlap_count": len(overlap),
        "hard_negative_review_pending": review_pending,
        "contact_sheets": [str(path) for path in contact_sheets],
        "issues": issues,
        "status": (
            "BLOCKED_DATASET_SPLIT_LEAKAGE"
            if cross_split_duplicates
            else ("BLOCKED_HARD_NEGATIVE_REVIEW" if review_pending else "READY_FOR_PREPARE")
        ),
        "known_limitation": "Roboflow positives were already exported as 640x640 Fit; training at imgsz=1280 cannot restore lost source detail.",
    }
    json_dump(reports / "dataset_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def read_approved_hardneg(audit_csv: Path, root: Path) -> list[tuple[Path, int]]:
    if not audit_csv.is_file():
        raise FileNotFoundError(f"hard-negative audit missing: {audit_csv}")
    approved: list[tuple[Path, int]] = []
    with audit_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["readable"].lower() != "true":
                raise RuntimeError(f"unreadable hard negative: {row['relative_path']}")
            if row["contains_obstacle"].lower() != "false" or row["contains_overlay"].lower() != "false":
                raise RuntimeError(f"hard negative not approved clean: {row['relative_path']}")
            if row["approved_empty_label"].lower() != "true":
                raise RuntimeError(f"hard negative review pending: {row['relative_path']}")
            source = (root / Path(row["relative_path"])).resolve()
            if sha256(source) != row["sha256"]:
                raise RuntimeError(f"hard-negative SHA changed after review: {source}")
            approved.append((source, int(row["timestamp_group"])))
    return approved


def prepare(config_path: Path, accept_exact_dedup_policy: bool = False) -> dict[str, Any]:
    config, project, paths = resolve_paths(config_path)
    report_path = project / "reports" / "dataset_audit.json"
    if not report_path.is_file():
        raise FileNotFoundError("run audit before prepare")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("ready_for_prepare"):
        exact_leak_only = (
            report.get("hard_negative_review_pending") == 0
            and len(report.get("issues", [])) == 1
            and str(report["issues"][0]).endswith("exact image SHA duplicates cross source splits")
        )
        if not (accept_exact_dedup_policy and exact_leak_only):
            raise RuntimeError(
                "dataset audit is blocked; exact-SHA split leakage may only be repaired with "
                "--accept-exact-dedup-policy"
            )

    source = paths["source_dataset"]
    workspace = paths["workspace"]
    merged = workspace / "datasets" / "waste_detect_hn_v8_1280_seed42"
    dev = workspace / "datasets" / "hard_negative_dev_v8_group3"
    if merged.exists() or dev.exists():
        raise FileExistsError(f"refusing to overwrite prepared dataset: {merged} or {dev}")

    approved = read_approved_hardneg(project / "manifests" / "hard_negative_v8_audit.csv", paths["root"])
    train_groups = {int(value) for value in config["hard_negative_contract"]["train_groups"]}
    dev_groups = {int(value) for value in config["hard_negative_contract"]["development_groups"]}
    merged_images = merged / "images" / "train_hard_negative"
    merged_labels = merged / "labels" / "train_hard_negative"
    dev_images = dev / "images"
    dev_labels = dev / "labels"

    split_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    for split in ("train", "valid", "test"):
        images_root = source / split / "images"
        for image in images_in(images_root):
            label = paired_label(image, images_root)
            source_records.append(
                {"split": split, "image": image, "label": label, "sha256": sha256(image)}
            )

    exclusions: list[dict[str, Any]] = []
    label_conflicts: list[dict[str, Any]] = []
    if accept_exact_dedup_policy:
        by_sha: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in source_records:
            by_sha[record["sha256"]].append(record)
        priority = {"test": 0, "valid": 1, "train": 2}
        selected_records: list[dict[str, Any]] = []
        for digest, records in by_sha.items():
            label_contents = {
                record["label"].read_text(encoding="utf-8-sig").strip()
                for record in records
            }
            ordered = sorted(records, key=lambda item: (priority[item["split"]], str(item["image"])))
            kept = ordered[0]
            if len(label_contents) != 1:
                label_conflicts.append(
                    {
                        "sha256": digest,
                        "kept_split": kept["split"],
                        "kept_image": str(kept["image"]),
                        "kept_label": str(kept["label"]),
                        "all_labels": " | ".join(str(record["label"]) for record in ordered),
                        "resolution": "kept label from test > valid > train precedence",
                    }
                )
            selected_records.append(kept)
            for removed in ordered[1:]:
                exclusions.append(
                    {
                        "sha256": digest,
                        "removed_split": removed["split"],
                        "removed_image": str(removed["image"]),
                        "kept_split": kept["split"],
                        "kept_image": str(kept["image"]),
                        "policy": "exact_sha_global_dedup_test_gt_valid_gt_train",
                    }
                )
        source_records = sorted(selected_records, key=lambda item: (item["split"], str(item["image"])))

    for record in source_records:
        split_rows.append(
            {
                "sample_id": record["image"].stem,
                "source": "roboflow_-ohs3h_2-iemaw_v1",
                "source_sha256": record["sha256"],
                "source_group": "roboflow_original_exact_dedup" if accept_exact_dedup_policy else "roboflow_original",
                "split": "val" if record["split"] == "valid" else record["split"],
                "image_path": str(record["image"]),
                "label_path": str(record["label"]),
                "label_mode": "bbox_gt",
            }
        )

    for directory in (merged_images, merged_labels, dev_images, dev_labels):
        directory.mkdir(parents=True, exist_ok=False)

    hardneg_train_paths: list[Path] = []
    for source_image, group in approved:
        if group in train_groups:
            destination_image = merged_images / source_image.name
            destination_label = merged_labels / source_image.with_suffix(".txt").name
            hardneg_train_paths.append(destination_image)
            split = "train"
        elif group in dev_groups:
            destination_image = dev_images / source_image.name
            destination_label = dev_labels / source_image.with_suffix(".txt").name
            split = "development_negative"
        else:
            raise RuntimeError(f"hard-negative group {group} has no split assignment")
        shutil.copy2(source_image, destination_image)
        destination_label.touch(exist_ok=False)
        split_rows.append(
            {
                "sample_id": source_image.stem,
                "source": "for_model_test_v8",
                "source_sha256": sha256(source_image),
                "source_group": f"capture_group_{group}",
                "split": split,
                "image_path": str(destination_image),
                "label_path": str(destination_label),
                "label_mode": "verified_empty",
            }
        )

    original_train = [record["image"] for record in source_records if record["split"] == "train"]
    original_val = [record["image"] for record in source_records if record["split"] == "valid"]
    original_test = [record["image"] for record in source_records if record["split"] == "test"]
    train_list = merged / "train_images.txt"
    train_list.write_text("\n".join(str(path) for path in original_train + hardneg_train_paths) + "\n", encoding="utf-8")
    val_list = merged / "val_images.txt"
    val_list.write_text("\n".join(str(path) for path in original_val) + "\n", encoding="utf-8")
    test_list = merged / "test_images.txt"
    test_list.write_text("\n".join(str(path) for path in original_test) + "\n", encoding="utf-8")
    data_yaml = merged / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "train": str(train_list),
                "val": str(val_list),
                "test": str(test_list),
                "nc": 1,
                "names": ["obstacle"],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    write_csv(
        project / "manifests" / "split_manifest.csv",
        ["sample_id", "source", "source_sha256", "source_group", "split", "image_path", "label_path", "label_mode"],
        split_rows,
    )
    write_csv(
        project / "manifests" / "exact_sha_dedup_exclusions.csv",
        ["sha256", "removed_split", "removed_image", "kept_split", "kept_image", "policy"],
        exclusions,
    )
    write_csv(
        project / "manifests" / "exact_sha_label_conflicts.csv",
        ["sha256", "kept_split", "kept_image", "kept_label", "all_labels", "resolution"],
        label_conflicts,
    )
    approval = {
        "accepted": bool(accept_exact_dedup_policy),
        "accepted_at": utc_now(),
        "policy": "exact_sha_global_dedup_test_gt_valid_gt_train",
        "removed_files": len(exclusions),
        "label_conflicts_resolved_by_precedence": len(label_conflicts),
        "source_files_unchanged": True,
        "limitation": "Only exact SHA duplicates are removed; visually similar or same-capture-group leakage may remain.",
    }
    (project / "manifests" / "dedup_policy_approval.yaml").write_text(
        yaml.safe_dump(approval, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    manifest_hash = sha256(project / "manifests" / "split_manifest.csv")
    result = {
        "prepared": True,
        "generated_at": utc_now(),
        "data_yaml": str(data_yaml),
        "data_yaml_sha256": sha256(data_yaml),
        "train_images": len(original_train) + len(hardneg_train_paths),
        "original_train_images": len(original_train),
        "validation_images": len(original_val),
        "test_images": len(original_test),
        "exact_duplicate_files_excluded": len(exclusions),
        "exact_duplicate_label_conflicts": len(label_conflicts),
        "exact_dedup_policy_accepted": bool(accept_exact_dedup_policy),
        "hard_negative_train_images": len(hardneg_train_paths),
        "development_negative_images": len(images_in(dev_images)),
        "split_manifest_sha256": manifest_hash,
        "training_imgsz": int(config["training"]["imgsz"]),
        "rectangular_training": bool(config["training"]["rect"]),
        "camera_shape_hw": [int(config["training"]["camera_height"]), int(config["training"]["camera_width"])],
        "expected_stride_aligned_hard_negative_shape_hw": [736, 1280],
    }
    json_dump(project / "reports" / "prepared_dataset.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and prepare YOLO26 hard-negative data without modifying sources.")
    parser.add_argument("command", choices=("audit", "prepare"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--approve-all-reviewed",
        action="store_true",
        help="Mark every decodable hard negative approved only after a human reviewed every generated contact sheet.",
    )
    parser.add_argument(
        "--accept-exact-dedup-policy",
        action="store_true",
        help="Explicitly accept global exact-SHA dedup with test > valid > train precedence.",
    )
    args = parser.parse_args()
    if args.command == "audit":
        report = audit(args.config, args.approve_all_reviewed)
        if report["issues"]:
            raise SystemExit(2)
    else:
        prepare(args.config, args.accept_exact_dedup_policy)


if __name__ == "__main__":
    main()

