from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common import sha256_file, write_json


class CandidatePostprocess(nn.Module):
    """GPU-friendly prefilter only; deliberately excludes CCL and control."""

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rust_class_map = logits[:, :4].argmax(dim=1)
        crack_candidate_count = (logits[:, 4, 112:240] >= 0.0).sum(dim=(1, 2))
        return rust_class_map, crack_candidate_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a weights-free GPU argmax/threshold candidate wrapper.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = CandidatePostprocess().eval()
    sample = torch.zeros(1, 5, 240, 1280, dtype=torch.float32)
    torch.onnx.export(
        model, sample, output, input_names=["multitask_logits"],
        output_names=["rust_class_map", "crack_candidate_count"], opset_version=17,
        do_constant_folding=True, dynamic_axes=None, dynamo=False,
    )
    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(onnx.load(str(output)))
    rng = np.random.default_rng(96)
    value = rng.standard_normal((1, 5, 240, 1280), dtype=np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(value))
    actual = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"]).run(None, {"multitask_logits": value})
    parity = bool(np.array_equal(expected[0].numpy(), actual[0]) and np.array_equal(expected[1].numpy(), actual[1]))
    report = {
        "role": "GPU_PREFILTER_ONLY_NO_CCL_NO_CONTROL", "onnx": str(output), "onnx_sha256": sha256_file(output),
        "input": {"name": "multitask_logits", "dtype": "float32", "shape": [1, 5, 240, 1280]},
        "outputs": {"rust_class_map": [1, 240, 1280], "crack_candidate_count": [1]},
        "parity": parity, "uart_authorization": "PROHIBITED",
    }
    write_json(output.with_suffix(output.suffix + ".json"), report)
    print(json.dumps(report, indent=2))
    if not parity:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
