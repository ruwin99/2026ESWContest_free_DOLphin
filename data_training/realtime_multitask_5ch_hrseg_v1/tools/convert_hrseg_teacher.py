from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


WORK_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = WORK_ROOT / "official_hrsegnet"
DEFAULT_WEIGHTS = (
    WORK_ROOT
    / "official_assets"
    / "hrsegnetb32"
    / "hrsegnetb32"
    / "best_model"
    / "model.pdparams"
)
DEFAULT_OUTPUT = (
    WORK_ROOT
    / "official_assets"
    / "hrsegnet-b32-crackseg9k-realtime-w1280-h128-fp32.onnx"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def load_model_class(repo: Path):
    source = repo / "models" / "hrsegnet_b32.py"
    if not source.is_file():
        raise FileNotFoundError(f"Official B32 model source missing: {source}")
    specification = importlib.util.spec_from_file_location("audited_hrsegnet_b32", source)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Unable to load model source: {source}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.HrSegNetB32


def rename_graph_value(model: Any, old: str, new: str) -> None:
    for node in model.graph.node:
        for index, value in enumerate(node.input):
            if value == old:
                node.input[index] = new
        for index, value in enumerate(node.output):
            if value == old:
                node.output[index] = new
    for collection in (model.graph.input, model.graph.output, model.graph.value_info):
        for value in collection:
            if value.name == old:
                value.name = new


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the audited official Paddle HrSegNet-B32 checkpoint to fixed-shape ONNX."
    )
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--opset", type=int, default=11)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    weights = args.weights.resolve()
    output = args.output.resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Official checkpoint missing: {weights}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing ONNX: {output}")

    import onnx
    import onnxruntime as ort
    import paddle
    import paddle2onnx

    paddle.disable_static()
    model_class = load_model_class(repo)
    model = model_class(in_channels=3, base=32, num_classes=2, pretrained=None)
    checkpoint = paddle.load(str(weights))
    if not isinstance(checkpoint, dict):
        raise TypeError("Official .pdparams root must be a state dictionary")
    model_state = model.state_dict()
    expected_keys = set(model_state)
    actual_keys = set(checkpoint)
    if actual_keys != expected_keys:
        raise ValueError(
            "Official checkpoint/model key mismatch: "
            f"missing={sorted(expected_keys - actual_keys)[:20]}, "
            f"unexpected={sorted(actual_keys - expected_keys)[:20]}"
        )
    shape_mismatches = {
        key: {"model": list(model_state[key].shape), "checkpoint": list(checkpoint[key].shape)}
        for key in expected_keys
        if list(model_state[key].shape) != list(checkpoint[key].shape)
    }
    if shape_mismatches:
        raise ValueError(f"Official checkpoint shape mismatch: {shape_mismatches}")
    model.set_state_dict(checkpoint)
    model.eval()

    rng = np.random.default_rng(20260816)
    sample = rng.normal(0.0, 1.0, size=(1, 3, 128, 1280)).astype(np.float32)
    with paddle.no_grad():
        paddle_output = model(paddle.to_tensor(sample))[0].numpy()
    if paddle_output.shape != (1, 2, 128, 1280):
        raise RuntimeError(f"Paddle output shape mismatch: {paddle_output.shape}")
    if not np.isfinite(paddle_output).all():
        raise RuntimeError("Paddle output contains non-finite values")

    output.parent.mkdir(parents=True, exist_ok=True)
    input_spec = [paddle.static.InputSpec(shape=[1, 3, 128, 1280], dtype="float32", name="images")]
    paddle.onnx.export(
        model,
        str(output.with_suffix("")),
        input_spec=input_spec,
        opset_version=int(args.opset),
        enable_onnx_checker=True,
    )
    if not output.is_file():
        raise RuntimeError(f"Paddle exporter did not create expected file: {output}")

    graph = onnx.load(str(output))
    if len(graph.graph.input) != 1 or len(graph.graph.output) != 1:
        raise RuntimeError(
            f"Expected one ONNX input/output, got {len(graph.graph.input)}/{len(graph.graph.output)}"
        )
    rename_graph_value(graph, graph.graph.input[0].name, "images")
    rename_graph_value(graph, graph.graph.output[0].name, "crack_logits")
    for dimension, value in zip(
        graph.graph.input[0].type.tensor_type.shape.dim, (1, 3, 128, 1280)
    ):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    for dimension, value in zip(
        graph.graph.output[0].type.tensor_type.shape.dim, (1, 2, 128, 1280)
    ):
        dimension.ClearField("dim_param")
        dimension.dim_value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(output))

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    onnx_output = session.run(["crack_logits"], {"images": sample})[0]
    if onnx_output.shape != paddle_output.shape or not np.isfinite(onnx_output).all():
        raise RuntimeError(f"ONNX output contract failed: shape={onnx_output.shape}")
    absolute = np.abs(onnx_output - paddle_output)
    max_abs = float(absolute.max())
    mean_abs = float(absolute.mean())
    if max_abs > 2e-4:
        raise RuntimeError(f"Paddle/ONNX parity failed: max_abs={max_abs}")

    commit = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    metadata = {
        "schema_version": 1,
        "purpose": "training_teacher_only",
        "official_repository": "https://github.com/CHDyshli/HrSegNet4CrackSegmentation",
        "official_checkpoint_url": "https://chdeducn-my.sharepoint.com/:u:/g/personal/2018024008_chd_edu_cn/EVaZjUC9tVNMoMkbNOdmemEBh6xPEBUzo2-0ddjGl3bfRQ?e=MWs6Z9",
        "repository_commit": commit,
        "checkpoint": str(weights),
        "checkpoint_sha256": sha256_file(weights),
        "strict_key_count": len(expected_keys),
        "strict_key_match": True,
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "opset": int(args.opset),
        "input": {"name": "images", "shape": [1, 3, 128, 1280], "dtype": "float32"},
        "output": {"name": "crack_logits", "shape": [1, 2, 128, 1280], "dtype": "float32"},
        "preprocessing": {
            "color": "RGB",
            "formula": "pixel / 127.5 - 1.0",
            "source": "PaddleSeg Normalize defaults in the official hrsegnetb32.yml",
        },
        "distillation_signal": "crack_logits[:,1:2]-crack_logits[:,0:1]",
        "parity": {"seed": 20260816, "max_abs": max_abs, "mean_abs": mean_abs},
        "environment": {
            "python": platform.python_version(),
            "paddle": paddle.__version__,
            "paddleseg": __import__("paddleseg").__version__,
            "paddle2onnx": paddle2onnx.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
        },
    }
    metadata_path = output.with_suffix(".metadata.json")
    write_json(metadata_path, metadata)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
