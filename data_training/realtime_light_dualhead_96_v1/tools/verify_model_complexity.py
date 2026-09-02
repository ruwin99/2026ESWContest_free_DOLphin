from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common import sha256_file, write_json


def dimensions(value) -> list[int | None]:
    return [int(dim.dim_value) if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]


def main() -> None:
    parser = argparse.ArgumentParser(description="Static ONNX parameter, Conv MAC, and activation report.")
    parser.add_argument("--onnx", type=Path, required=True)
    args = parser.parse_args()
    import onnx
    from onnx import numpy_helper, shape_inference

    path = args.onnx.resolve()
    model = shape_inference.infer_shapes(onnx.load(str(path), load_external_data=True))
    initializers = {item.name: numpy_helper.to_array(item) for item in model.graph.initializer}
    shapes = {}
    for item in [*model.graph.input, *model.graph.value_info, *model.graph.output]:
        shapes[item.name] = dimensions(item)
    parameters = int(sum(value.size for value in initializers.values()))
    parameter_bytes = int(sum(value.nbytes for value in initializers.values()))
    conv_macs = 0
    conv_rows = []
    forbidden_304_256 = []
    for node in model.graph.node:
        if node.op_type != "Conv" or len(node.input) < 2 or node.input[1] not in initializers:
            continue
        weight = initializers[node.input[1]]
        output_shape = shapes.get(node.output[0])
        if not output_shape or any(value is None for value in output_shape):
            continue
        n, c_out, h_out, w_out = (int(value) for value in output_shape)
        group = next((int(attr.i) for attr in node.attribute if attr.name == "group"), 1)
        kernel_h, kernel_w = int(weight.shape[2]), int(weight.shape[3])
        c_in_per_group = int(weight.shape[1])
        macs = n * c_out * h_out * w_out * c_in_per_group * kernel_h * kernel_w
        conv_macs += macs
        row = {"name": node.name, "weight": node.input[1], "weight_shape": list(weight.shape), "output_shape": output_shape, "group": group, "macs": macs}
        conv_rows.append(row)
        if list(weight.shape) == [256, 304, 3, 3]:
            forbidden_304_256.append(row)
    activation_elements = [int(np.prod(shape)) for name, shape in shapes.items() if shape and all(v is not None for v in shape)]
    report = {
        "onnx": str(path), "onnx_sha256": sha256_file(path), "parameter_count": parameters,
        "parameter_bytes": parameter_bytes, "conv_macs": conv_macs,
        "largest_single_activation_elements": max(activation_elements, default=0),
        "largest_single_activation_fp32_bytes": 4 * max(activation_elements, default=0),
        "forbidden_304_to_256_3x3": forbidden_304_256,
        "conv_layers": conv_rows,
        "notes": ["Conv MAC is exact for statically inferred Conv nodes.", "Runtime peak workspace/activation reuse must be measured on TensorRT; the single-tensor value is not total peak memory."],
        "passed": not forbidden_304_256,
    }
    output = path.with_suffix(path.suffix + ".complexity.json")
    write_json(output, report)
    print(json.dumps({key: value for key, value in report.items() if key != "conv_layers"}, indent=2))
    if forbidden_304_256:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
