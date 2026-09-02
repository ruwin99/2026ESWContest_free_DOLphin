from __future__ import annotations

import json

import torch
import torchvision


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch가 CUDA GPU를 찾지 못했습니다.")

    capability = torch.cuda.get_device_capability(0)
    arch_list = torch.cuda.get_arch_list()
    device_name = torch.cuda.get_device_name(0)

    x = torch.randn(8, 64, device="cuda", requires_grad=True)
    weight = torch.randn(64, 4, device="cuda", requires_grad=True)
    loss = (x @ weight).square().mean()
    loss.backward()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("CUDA forward/backward loss is not finite")

    report = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": device_name,
        "capability": list(capability),
        "compiled_arch_list": arch_list,
        "sm_120_compiled": "sm_120" in arch_list,
        "forward_backward_loss": float(loss.detach().cpu()),
        "guide_environment_match": {
            "torch_2_12_1": torch.__version__.startswith("2.12.1"),
            "torchvision_0_27_1": torchvision.__version__.startswith("0.27.1"),
            "cuda_13_0": torch.version.cuda == "13.0",
            "rtx_5070_ti": "RTX 5070 Ti" in device_name,
            "capability_12_0": capability == (12, 0),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if capability != (12, 0) or "sm_120" not in arch_list:
        raise RuntimeError(
            "RTX 5070 Ti용 sm_120 지원이 확인되지 않았습니다. cu130 환경을 확인하세요."
        )


if __name__ == "__main__":
    main()
