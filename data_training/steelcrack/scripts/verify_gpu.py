from __future__ import annotations

import json

import cv2
import einops
import numpy as np
import torch
import torchvision


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Steelcrack environment")

    device = torch.device("cuda:0")
    value = torch.randn((32, 32), device=device, requires_grad=True)
    loss = value.square().mean()
    loss.backward()
    capability = list(torch.cuda.get_device_capability(device))
    arch_list = torch.cuda.get_arch_list()
    report = {
        "python_torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(device),
        "capability": capability,
        "compiled_arch_list": arch_list,
        "sm_120_compiled": "sm_120" in arch_list,
        "forward_backward_loss": float(loss.detach().cpu()),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "einops": getattr(einops, "__version__", "unknown"),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if capability != [12, 0] or "sm_120" not in arch_list:
        raise RuntimeError("Installed PyTorch build does not fully support RTX 5070 Ti sm_120")


if __name__ == "__main__":
    main()
