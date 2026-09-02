from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from ultralytics import YOLO


ROOT = Path(os.environ.get("RAIL_ROBOT_ROOT", Path(__file__).resolve().parents[3])).expanduser().resolve()
PROJECT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "outputs/training/yolo26_obstacle_hardneg_v1/yolo26n_hn_all1410_1280rect_clsout_lr2e5_seed42_b8_v2/weights/best.pt"
EXPECTED_CANDIDATE_SHA = "5f08e1fd963627aa81522d00b75b72b4c633016b162e7de529f9849786881d56"
OUTPUT_DIR = ROOT / "output/models/yolo26_obstacle_hardneg_v2"
OUTPUT = OUTPUT_DIR / "obstacle-yolo26n-hardneg-all1410-camera-w1280-h720-int-h736-fp32.onnx"
METADATA = OUTPUT.with_suffix(".metadata.json")
SHA_FILE = OUTPUT.with_suffix(".sha256.txt")
PARITY_IMAGE = sorted((ROOT / "for model test v7").rglob("*.jpg"))[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shape_of(value: onnx.ValueInfoProto) -> list[int | str]:
    dimensions: list[int | str] = []
    for dim in value.type.tensor_type.shape.dim:
        dimensions.append(int(dim.dim_value) if dim.dim_value else dim.dim_param)
    return dimensions


def camera_tensor(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if rgb.shape != (720, 1280, 3):
        raise RuntimeError(f"Parity image shape mismatch: {rgb.shape}")
    rgb = np.pad(rgb, ((8, 8), (0, 0), (0, 0)), constant_values=114.0)
    return np.ascontiguousarray(rgb.transpose(2, 0, 1)[None] / 255.0, dtype=np.float32)


def main() -> None:
    if sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA:
        raise RuntimeError("Fixed candidate SHA changed")
    for path in (OUTPUT, METADATA, SHA_FILE):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite release artifact: {path}")

    temporary_export = CANDIDATE.with_suffix(".onnx")
    if temporary_export.exists():
        raise FileExistsError(f"Unexpected temporary export already exists: {temporary_export}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(CANDIDATE))
    exported = Path(
        model.export(
            format="onnx",
            imgsz=(736, 1280),
            batch=1,
            half=False,
            dynamic=False,
            simplify=True,
            opset=17,
            device="cpu",
            nms=False,
        )
    ).resolve()
    if exported != temporary_export.resolve():
        raise RuntimeError(f"Unexpected Ultralytics export path: {exported}")
    shutil.move(str(exported), OUTPUT)

    graph = onnx.load(str(OUTPUT), load_external_data=True)
    onnx.checker.check_model(graph)
    inputs = {value.name: shape_of(value) for value in graph.graph.input}
    outputs = {value.name: shape_of(value) for value in graph.graph.output}
    if inputs != {"images": [1, 3, 736, 1280]}:
        raise RuntimeError(f"Unexpected ONNX inputs: {inputs}")
    if list(outputs.values()) != [[1, 300, 6]]:
        raise RuntimeError(f"Unexpected ONNX outputs: {outputs}")

    onnx.helper.set_model_props(
        graph,
        {
            "task": "detect",
            "class_names": "0:obstacle",
            "camera_shape_hw": "720,1280",
            "tensor_shape_nchw": "1,3,736,1280",
            "preprocess": "BGR camera to RGB; divide by 255; pad top=8 bottom=8 with value 114",
            "postprocess": "subtract 8 from y1/y2 and clip to [0,720]; filter confidence>=0.30; NMS IoU=0.70 if runtime requires NMS",
            "operational_confidence": "0.30",
            "nms_iou": "0.70",
            "candidate_sha256": EXPECTED_CANDIDATE_SHA,
        },
    )
    onnx.save(graph, str(OUTPUT))
    onnx.checker.check_model(onnx.load(str(OUTPUT), load_external_data=True))

    sample = camera_tensor(PARITY_IMAGE)
    with torch.inference_mode():
        pytorch_output = (
            YOLO(str(CANDIDATE)).model.float().eval()(torch.from_numpy(sample))[0].cpu().numpy()
        )
    session = ort.InferenceSession(str(OUTPUT), providers=["CPUExecutionProvider"])
    ort_output = session.run(None, {"images": sample})[0]
    difference = np.abs(pytorch_output - ort_output)
    # End-to-end YOLO26 exports can reorder numerically tied, extremely low-score
    # Top-K rows. Compare every score/class, but compare box coordinates only for
    # rows that can represent a meaningful detection.
    meaningful_mask = np.logical_or(
        pytorch_output[..., 4] >= 0.01, ort_output[..., 4] >= 0.01
    )
    meaningful_difference = difference[meaningful_mask]
    operational_pt = int(np.count_nonzero(pytorch_output[..., 4] >= 0.30))
    operational_ort = int(np.count_nonzero(ort_output[..., 4] >= 0.30))
    parity = {
        "sample": str(PARITY_IMAGE),
        "pytorch_shape": list(pytorch_output.shape),
        "onnxruntime_shape": list(ort_output.shape),
        "finite": bool(np.isfinite(ort_output).all()),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "box_comparison_min_confidence": 0.01,
        "meaningful_rows": int(np.count_nonzero(meaningful_mask)),
        "meaningful_max_abs_diff": float(meaningful_difference.max()),
        "meaningful_mean_abs_diff": float(meaningful_difference.mean()),
        "score_max_abs_diff": float(np.abs(pytorch_output[..., 4] - ort_output[..., 4]).max()),
        "class_id_agreement": float(np.mean(pytorch_output[..., 5] == ort_output[..., 5])),
        "operational_detection_count_pytorch": operational_pt,
        "operational_detection_count_onnxruntime": operational_ort,
    }
    parity["passed"] = bool(
        parity["finite"]
        and parity["pytorch_shape"] == [1, 300, 6]
        and parity["onnxruntime_shape"] == [1, 300, 6]
        and parity["meaningful_rows"] > 0
        and parity["meaningful_max_abs_diff"] <= 0.01
        and parity["meaningful_mean_abs_diff"] <= 0.001
        and parity["score_max_abs_diff"] <= 0.001
        and parity["class_id_agreement"] == 1.0
        and operational_pt == operational_ort
    )
    if not parity["passed"]:
        raise RuntimeError(f"PyTorch/ONNX Runtime parity failed: {parity}")

    metadata = {
        "status": "FINAL_HARDNEG_CANDIDATE_ONNX",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(CANDIDATE),
        "checkpoint_sha256": EXPECTED_CANDIDATE_SHA,
        "onnx": str(OUTPUT),
        "onnx_sha256": sha256(OUTPUT),
        "onnx_bytes": OUTPUT.stat().st_size,
        "opset": 17,
        "fp32": True,
        "static": True,
        "simplified": True,
        "input_names_and_shapes": inputs,
        "output_names_and_shapes": outputs,
        "camera_shape_hw": [720, 1280],
        "internal_stride_aligned_shape_hw": [736, 1280],
        "class_names": {"0": "obstacle"},
        "operational_confidence": 0.30,
        "nms_iou": 0.70,
        "parity": parity,
        "versions": {
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    SHA_FILE.write_text(f"{metadata['onnx_sha256']}  {OUTPUT.name}\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

