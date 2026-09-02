from __future__ import annotations

import numpy as np
import onnxruntime as ort
import torch

from light_dualhead_96 import LightDualHead96


def test_small_static_fp32_export_matches_pytorch(tmp_path, official_training) -> None:
    model = LightDualHead96(official_training).float().eval()
    sample = torch.randn(1, 3, 64, 128)
    output = tmp_path / "small.onnx"
    torch.onnx.export(
        model, sample, output, input_names=["images"], output_names=["multitask_logits"],
        opset_version=17, dynamic_axes=None, dynamo=False,
    )
    with torch.inference_mode():
        expected = model(sample).numpy()
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(
        ["multitask_logits"], {"images": sample.numpy()}
    )[0]
    assert np.max(np.abs(expected - actual)) <= 1e-4
    assert np.mean(expected[:, :4].argmax(1) == actual[:, :4].argmax(1)) >= 0.99999
