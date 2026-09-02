from __future__ import annotations

import ctypes
from typing import Any

import numpy as np


THREADS_PER_BLOCK = 256
KERNEL_NAME = b"rust_argmax_finite"


def _kernel_source(class_count: int, pixel_count: int) -> str:
    return f"""
__device__ __forceinline__ bool finite_float(float value) {{
    return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}}

extern "C" __global__ void rust_argmax_finite(
    const float* logits,
    unsigned char* class_map,
    unsigned int* invalid_flag) {{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= {pixel_count}) {{
        return;
    }}

    float best_value = logits[pixel];
    unsigned char best_class = 0;
    bool finite = finite_float(best_value);

#pragma unroll
    for (int class_id = 1; class_id < {class_count}; ++class_id) {{
        const float candidate = logits[class_id * {pixel_count} + pixel];
        finite = finite && finite_float(candidate);
        if (candidate > best_value) {{
            best_value = candidate;
            best_class = (unsigned char)class_id;
        }}
    }}

    class_map[pixel] = best_class;
    if (!finite) {{
        atomicExch(invalid_flag, 1u);
    }}
}}
"""


class CudaArgmaxPostprocessor:
    """Run finite validation and first-tie argmax on TensorRT device logits."""

    def __init__(
        self,
        cudart: Any,
        output_shape: tuple[int, int, int, int],
    ) -> None:
        if len(output_shape) != 4 or output_shape[0] != 1:
            raise ValueError(
                "CUDA argmax requires a static NCHW output with batch size one."
            )
        _, class_count, height, width = output_shape
        if class_count <= 1 or height <= 0 or width <= 0:
            raise ValueError(f"Invalid CUDA argmax output shape: {output_shape}")

        self._cudart = cudart
        self.class_count = int(class_count)
        self.height = int(height)
        self.width = int(width)
        self.pixel_count = self.height * self.width
        self.d2h_nbytes = self.pixel_count + np.dtype(np.uint32).itemsize
        self._cuda: Any = None
        self._nvrtc: Any = None
        self._module: Any = None
        self._function: Any = None
        self._device_class_map: Any = None
        self._device_invalid_flag: Any = None
        self._host_class_map_pointer: Any = None
        self._host_invalid_flag_pointer: Any = None
        self._host_class_map: np.ndarray | None = None
        self._host_invalid_flag: np.ndarray | None = None

        try:
            from cuda import cuda, nvrtc
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Python with cuda.cuda and cuda.nvrtc is required for "
                "GPU rust argmax."
            ) from exc

        self._cuda = cuda
        self._nvrtc = nvrtc
        try:
            self._driver_call(cuda.cuInit(0), "cuInit")
            current_context = self._driver_call(
                cuda.cuCtxGetCurrent(),
                "cuCtxGetCurrent",
            )
            if int(current_context) == 0:
                raise RuntimeError(
                    "CUDA argmax requires the active TensorRT CUDA context."
                )
            device = self._driver_call(cuda.cuCtxGetDevice(), "cuCtxGetDevice")
            major = self._driver_call(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ),
                "cuDeviceGetAttribute(compute major)",
            )
            minor = self._driver_call(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                ),
                "cuDeviceGetAttribute(compute minor)",
            )
            cubin = self._compile_cubin(int(major), int(minor))
            self._module = self._driver_call(
                cuda.cuModuleLoadData(np.char.array(cubin)),
                "cuModuleLoadData",
            )
            self._function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, KERNEL_NAME),
                "cuModuleGetFunction(rust_argmax_finite)",
            )

            map_nbytes = self.pixel_count * np.dtype(np.uint8).itemsize
            flag_nbytes = np.dtype(np.uint32).itemsize
            self._host_class_map_pointer = self._runtime_call(
                cudart.cudaHostAlloc(map_nbytes, 0),
                "cudaHostAlloc(class map)",
            )
            self._host_invalid_flag_pointer = self._runtime_call(
                cudart.cudaHostAlloc(flag_nbytes, 0),
                "cudaHostAlloc(invalid flag)",
            )
            self._host_class_map = self._host_array(
                self._host_class_map_pointer,
                (self.height, self.width),
                np.uint8,
            )
            self._host_invalid_flag = self._host_array(
                self._host_invalid_flag_pointer,
                (1,),
                np.uint32,
            )
            self._device_class_map = self._runtime_call(
                cudart.cudaMalloc(map_nbytes),
                "cudaMalloc(class map)",
            )
            self._device_invalid_flag = self._runtime_call(
                cudart.cudaMalloc(flag_nbytes),
                "cudaMalloc(invalid flag)",
            )
        except Exception:
            self.close()
            raise

    def _compile_cubin(self, major: int, minor: int) -> bytes:
        nvrtc = self._nvrtc
        program = self._nvrtc_call(
            nvrtc.nvrtcCreateProgram(
                _kernel_source(self.class_count, self.pixel_count).encode("utf-8"),
                b"rust_argmax_finite.cu",
                0,
                [],
                [],
            ),
            "nvrtcCreateProgram",
        )
        try:
            options = [
                f"--gpu-architecture=sm_{major}{minor}".encode("ascii"),
                b"--std=c++11",
            ]
            compile_result = nvrtc.nvrtcCompileProgram(
                program,
                len(options),
                options,
            )
            log = self._program_log(program)
            try:
                self._nvrtc_call(compile_result, "nvrtcCompileProgram")
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}{': ' + log if log else ''}") from exc
            cubin_size = self._nvrtc_call(
                nvrtc.nvrtcGetCUBINSize(program),
                "nvrtcGetCUBINSize",
            )
            cubin = b" " * int(cubin_size)
            self._nvrtc_call(
                nvrtc.nvrtcGetCUBIN(program, cubin),
                "nvrtcGetCUBIN",
            )
            return cubin
        finally:
            try:
                self._nvrtc_call(
                    nvrtc.nvrtcDestroyProgram(program),
                    "nvrtcDestroyProgram",
                )
            except Exception:
                pass

    def _program_log(self, program: Any) -> str:
        nvrtc = self._nvrtc
        try:
            log_size = self._nvrtc_call(
                nvrtc.nvrtcGetProgramLogSize(program),
                "nvrtcGetProgramLogSize",
            )
            if int(log_size) <= 1:
                return ""
            log = b" " * int(log_size)
            self._nvrtc_call(
                nvrtc.nvrtcGetProgramLog(program, log),
                "nvrtcGetProgramLog",
            )
            return log.decode("utf-8", errors="replace").strip("\x00 \n")
        except Exception:
            return ""

    @staticmethod
    def _host_array(pointer: Any, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        dtype = np.dtype(dtype)
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)),
            ctypes.POINTER(ctypes.c_ubyte),
        )
        byte_array = np.ctypeslib.as_array(byte_pointer, shape=(byte_count,))
        return byte_array.view(dtype).reshape(shape)

    def _driver_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA driver status.")
        error = result[0]
        if error != self._cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed with CUDA driver error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    def _nvrtc_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no NVRTC status.")
        error = result[0]
        if error != self._nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{operation} failed with NVRTC error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    def _runtime_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA runtime status.")
        error = result[0]
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed with CUDA runtime error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    def process(self, logits_pointer: Any, stream: Any) -> np.ndarray:
        if self._function is None or self._host_class_map is None:
            raise RuntimeError("CUDA argmax postprocessor is closed.")
        cudart = self._cudart
        self._runtime_call(
            cudart.cudaMemsetAsync(
                self._device_invalid_flag,
                0,
                np.dtype(np.uint32).itemsize,
                stream,
            ),
            "cudaMemsetAsync(invalid flag)",
        )
        arguments = (
            self._cuda.CUdeviceptr(int(logits_pointer)),
            self._cuda.CUdeviceptr(int(self._device_class_map)),
            self._cuda.CUdeviceptr(int(self._device_invalid_flag)),
        )
        grid_x = (self.pixel_count + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self._driver_call(
            self._cuda.cuLaunchKernel(
                self._function,
                grid_x,
                1,
                1,
                THREADS_PER_BLOCK,
                1,
                1,
                0,
                self._cuda.CUstream(int(stream)),
                (arguments, (None, None, None)),
                0,
            ),
            "cuLaunchKernel(rust_argmax_finite)",
        )
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._host_class_map.ctypes.data,
                self._device_class_map,
                self._host_class_map.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(class map)",
        )
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._host_invalid_flag.ctypes.data,
                self._device_invalid_flag,
                self._host_invalid_flag.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(invalid flag)",
        )
        self._runtime_call(
            cudart.cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(GPU argmax)",
        )
        if int(self._host_invalid_flag[0]) != 0:
            raise ValueError("Segmentation logits contain NaN or infinity.")
        return self._host_class_map.copy()

    def close(self) -> None:
        cudart = self._cudart
        for attribute in ("_device_class_map", "_device_invalid_flag"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    self._runtime_call(
                        cudart.cudaFree(allocation),
                        f"cudaFree({attribute})",
                    )
                except Exception:
                    pass
                setattr(self, attribute, None)
        for attribute in (
            "_host_class_map_pointer",
            "_host_invalid_flag_pointer",
        ):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    self._runtime_call(
                        cudart.cudaFreeHost(allocation),
                        f"cudaFreeHost({attribute})",
                    )
                except Exception:
                    pass
                setattr(self, attribute, None)
        if self._module is not None and self._cuda is not None:
            try:
                self._driver_call(
                    self._cuda.cuModuleUnload(self._module),
                    "cuModuleUnload",
                )
            except Exception:
                pass
        self._module = None
        self._function = None
        self._host_class_map = None
        self._host_invalid_flag = None
