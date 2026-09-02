from __future__ import annotations

import ctypes
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np


CAMERA_WIDTH = 1280
CAMERA_CHANNELS = 3
ROI_HEIGHT = 240
MODEL_HEIGHT = 256
MODEL_WIDTH = 1280
MODEL_CHANNELS = 3
PAD_TOP = 8
PAD_BOTTOM = 8
PAD_VALUE = 114
OUTPUT_ROWS = 300
OUTPUT_COLUMNS = 6
CONTROL_ROI_Y_END = 240
THREADS_PER_BLOCK = 256
PREPROCESS_KERNEL_NAME = b"obstacle_bgr_to_rgb_nchw"
POSTPROCESS_KERNEL_NAME = b"obstacle_validate_compact"
RAW_INPUT_SHAPE = (ROI_HEIGHT, CAMERA_WIDTH, CAMERA_CHANNELS)
MODEL_INPUT_SHAPE = (1, MODEL_CHANNELS, MODEL_HEIGHT, MODEL_WIDTH)
COMPACT_SHAPE = (OUTPUT_ROWS, OUTPUT_COLUMNS)
SUMMARY_SHAPE = (3,)
RAW_INPUT_NBYTES = int(np.prod(RAW_INPUT_SHAPE)) * np.dtype(np.uint8).itemsize
MODEL_INPUT_NBYTES = (
    int(np.prod(MODEL_INPUT_SHAPE)) * np.dtype(np.float32).itemsize
)
COMPACT_NBYTES = int(np.prod(COMPACT_SHAPE)) * np.dtype(np.float32).itemsize
SUMMARY_NBYTES = int(np.prod(SUMMARY_SHAPE)) * np.dtype(np.uint32).itemsize


@dataclass(frozen=True)
class CompactObstacleOutput:
    records: np.ndarray
    control_roi_detected: bool


def _float32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", np.float32(value)))[0]


def _kernel_source(confidence_threshold: float) -> str:
    threshold_bits = _float32_bits(confidence_threshold)
    return f"""
__device__ __forceinline__ bool finite_float(float value) {{
    return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}}

__device__ __forceinline__ float clamp_float(
    float value,
    float lower,
    float upper) {{
    return value < lower ? lower : (value > upper ? upper : value);
}}

extern "C" __global__ void obstacle_bgr_to_rgb_nchw(
    const unsigned char* bgr,
    float* images) {{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    const int plane = {MODEL_HEIGHT * MODEL_WIDTH};
    if (pixel >= plane) {{
        return;
    }}

    const int y = pixel / {MODEL_WIDTH};
    const int x = pixel - y * {MODEL_WIDTH};
    if (y < {PAD_TOP} || y >= {PAD_TOP + ROI_HEIGHT}) {{
        const float pad = {PAD_VALUE}.0f / 255.0f;
        images[pixel] = pad;
        images[plane + pixel] = pad;
        images[2 * plane + pixel] = pad;
        return;
    }}

    const int source = ((y - {PAD_TOP}) * {CAMERA_WIDTH} + x) * 3;
    images[pixel] = ((float)bgr[source + 2]) / 255.0f;
    images[plane + pixel] = ((float)bgr[source + 1]) / 255.0f;
    images[2 * plane + pixel] = ((float)bgr[source]) / 255.0f;
}}

extern "C" __global__ void obstacle_validate_compact(
    const float* output,
    float* compact,
    unsigned int* summary) {{
    if (blockIdx.x != 0 || threadIdx.x != 0) {{
        return;
    }}

    const float threshold = __uint_as_float(0x{threshold_bits:08x}u);
    unsigned int count = 0;
    unsigned int control = 0;
    unsigned int invalid = 0;
    for (int row = 0; row < {OUTPUT_ROWS}; ++row) {{
        const int offset = row * {OUTPUT_COLUMNS};
        const float raw_x1 = output[offset];
        const float raw_y1 = output[offset + 1];
        const float raw_x2 = output[offset + 2];
        const float raw_y2 = output[offset + 3];
        const float score = output[offset + 4];
        const float class_value = output[offset + 5];

        const bool finite = finite_float(raw_x1) && finite_float(raw_y1) &&
            finite_float(raw_x2) && finite_float(raw_y2) &&
            finite_float(score) && finite_float(class_value);
        const bool valid = finite && score >= 0.0f && score <= 1.0f &&
            raw_x2 >= raw_x1 && raw_y2 >= raw_y1 && class_value == 0.0f;
        if (!valid) {{
            invalid = 1;
            continue;
        }}
        if (!(score >= threshold)) {{
            continue;
        }}

        const float x1 = clamp_float(raw_x1, 0.0f, {CAMERA_WIDTH}.0f);
        const float x2 = clamp_float(raw_x2, 0.0f, {CAMERA_WIDTH}.0f);
        const float y1 = clamp_float(
            raw_y1 - {PAD_TOP}.0f, 0.0f, {ROI_HEIGHT}.0f);
        const float y2 = clamp_float(
            raw_y2 - {PAD_TOP}.0f, 0.0f, {ROI_HEIGHT}.0f);
        if (x2 <= x1 || y2 <= y1) {{
            continue;
        }}

        const int compact_offset = ((int)count) * {OUTPUT_COLUMNS};
        compact[compact_offset] = x1;
        compact[compact_offset + 1] = y1;
        compact[compact_offset + 2] = x2;
        compact[compact_offset + 3] = y2;
        compact[compact_offset + 4] = score;
        compact[compact_offset + 5] = 0.0f;
        ++count;
        if (y1 < {CONTROL_ROI_Y_END}.0f && y2 > 0.0f) {{
            control = 1;
        }}
    }}

    summary[0] = count;
    summary[1] = control;
    summary[2] = invalid;
}}
"""


class CudaObstaclePipeline:
    """GPU preprocessing and compact postprocessing on the TensorRT stream."""

    def __init__(self, cudart: Any, confidence_threshold: float) -> None:
        self._cudart = cudart
        self.confidence_threshold = float(np.float32(confidence_threshold))
        self.h2d_nbytes = RAW_INPUT_NBYTES
        self._cuda: Any = None
        self._nvrtc: Any = None
        self._module: Any = None
        self._preprocess_function: Any = None
        self._postprocess_function: Any = None
        self._host_raw_pointer: Any = None
        self._host_summary_pointer: Any = None
        self._host_compact_pointer: Any = None
        self._host_raw: np.ndarray | None = None
        self._host_summary: np.ndarray | None = None
        self._host_compact: np.ndarray | None = None
        self._device_raw: Any = None
        self._device_summary: Any = None
        self._device_compact: Any = None

        try:
            from cuda import cuda, nvrtc
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Python with cuda.cuda and cuda.nvrtc is required for "
                "the GPU obstacle pipeline."
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
                    "The GPU obstacle pipeline requires the active TensorRT "
                    "CUDA context."
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
            self._preprocess_function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, PREPROCESS_KERNEL_NAME),
                "cuModuleGetFunction(obstacle_bgr_to_rgb_nchw)",
            )
            self._postprocess_function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, POSTPROCESS_KERNEL_NAME),
                "cuModuleGetFunction(obstacle_validate_compact)",
            )

            self._host_raw_pointer = self._runtime_call(
                cudart.cudaHostAlloc(RAW_INPUT_NBYTES, 0),
                "cudaHostAlloc(obstacle raw input)",
            )
            self._host_summary_pointer = self._runtime_call(
                cudart.cudaHostAlloc(SUMMARY_NBYTES, 0),
                "cudaHostAlloc(obstacle summary)",
            )
            self._host_compact_pointer = self._runtime_call(
                cudart.cudaHostAlloc(COMPACT_NBYTES, 0),
                "cudaHostAlloc(obstacle compact output)",
            )
            self._host_raw = self._host_array(
                self._host_raw_pointer,
                RAW_INPUT_SHAPE,
                np.uint8,
            )
            self._host_summary = self._host_array(
                self._host_summary_pointer,
                SUMMARY_SHAPE,
                np.uint32,
            )
            self._host_compact = self._host_array(
                self._host_compact_pointer,
                COMPACT_SHAPE,
                np.float32,
            )
            self._device_raw = self._runtime_call(
                cudart.cudaMalloc(RAW_INPUT_NBYTES),
                "cudaMalloc(obstacle raw input)",
            )
            self._device_summary = self._runtime_call(
                cudart.cudaMalloc(SUMMARY_NBYTES),
                "cudaMalloc(obstacle summary)",
            )
            self._device_compact = self._runtime_call(
                cudart.cudaMalloc(COMPACT_NBYTES),
                "cudaMalloc(obstacle compact output)",
            )
        except Exception:
            self.close()
            raise

    def _compile_cubin(self, major: int, minor: int) -> bytes:
        nvrtc = self._nvrtc
        program = self._nvrtc_call(
            nvrtc.nvrtcCreateProgram(
                _kernel_source(self.confidence_threshold).encode("utf-8"),
                b"obstacle_gpu_pipeline.cu",
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

    def upload_and_preprocess(
        self,
        frame: np.ndarray,
        tensor_pointer: Any,
        stream: Any,
    ) -> None:
        if self._preprocess_function is None or self._host_raw is None:
            raise RuntimeError("CUDA obstacle pipeline is closed.")
        if not isinstance(frame, np.ndarray) or frame.shape != RAW_INPUT_SHAPE:
            shape = getattr(frame, "shape", None)
            raise ValueError(
                "Obstacle TensorRT input must be the fixed 1280x240 HxWx3 "
                f"top-camera y=0:240 ROI; got {shape}. The detector does not "
                "resize the ROI."
            )
        if frame.dtype != np.uint8:
            raise ValueError("Obstacle camera input must contain uint8 BGR pixels.")

        np.copyto(self._host_raw, frame, casting="no")
        cudart = self._cudart
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._device_raw,
                self._host_raw.ctypes.data,
                RAW_INPUT_NBYTES,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync(obstacle raw input)",
        )
        arguments = (
            self._cuda.CUdeviceptr(int(self._device_raw)),
            self._cuda.CUdeviceptr(int(tensor_pointer)),
        )
        plane = MODEL_HEIGHT * MODEL_WIDTH
        grid_x = (plane + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self._driver_call(
            self._cuda.cuLaunchKernel(
                self._preprocess_function,
                grid_x,
                1,
                1,
                THREADS_PER_BLOCK,
                1,
                1,
                0,
                self._cuda.CUstream(int(stream)),
                (arguments, (None, None)),
                0,
            ),
            "cuLaunchKernel(obstacle_bgr_to_rgb_nchw)",
        )

    def postprocess(
        self,
        output_pointer: Any,
        stream: Any,
    ) -> CompactObstacleOutput:
        if self._postprocess_function is None or self._host_summary is None:
            raise RuntimeError("CUDA obstacle pipeline is closed.")
        cudart = self._cudart
        arguments = (
            self._cuda.CUdeviceptr(int(output_pointer)),
            self._cuda.CUdeviceptr(int(self._device_compact)),
            self._cuda.CUdeviceptr(int(self._device_summary)),
        )
        self._driver_call(
            self._cuda.cuLaunchKernel(
                self._postprocess_function,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                self._cuda.CUstream(int(stream)),
                (arguments, (None, None, None)),
                0,
            ),
            "cuLaunchKernel(obstacle_validate_compact)",
        )
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._host_summary.ctypes.data,
                self._device_summary,
                SUMMARY_NBYTES,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(obstacle summary)",
        )
        self._runtime_call(
            cudart.cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(obstacle summary)",
        )

        count, control, invalid = (int(value) for value in self._host_summary)
        if invalid != 0:
            raise ValueError("Obstacle output failed GPU validation.")
        if count < 0 or count > OUTPUT_ROWS:
            raise ValueError(f"Obstacle GPU compact count is invalid: {count}.")
        if control not in (0, 1):
            raise ValueError(f"Obstacle GPU control flag is invalid: {control}.")
        if count == 0:
            return CompactObstacleOutput(
                records=np.empty((0, OUTPUT_COLUMNS), dtype=np.float32),
                control_roi_detected=bool(control),
            )

        compact_nbytes = count * OUTPUT_COLUMNS * np.dtype(np.float32).itemsize
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._host_compact.ctypes.data,
                self._device_compact,
                compact_nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(obstacle compact output)",
        )
        self._runtime_call(
            cudart.cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(obstacle compact output)",
        )
        return CompactObstacleOutput(
            records=self._host_compact[:count].copy(),
            control_roi_detected=bool(control),
        )

    def close(self) -> None:
        cudart = self._cudart
        for attribute in ("_device_raw", "_device_summary", "_device_compact"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._runtime_call(
                            cudart.cudaFree(allocation),
                            f"cudaFree({attribute})",
                        )
                except Exception:
                    pass
                setattr(self, attribute, None)
        for attribute in (
            "_host_raw_pointer",
            "_host_summary_pointer",
            "_host_compact_pointer",
        ):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
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
                    "cuModuleUnload(obstacle GPU pipeline)",
                )
            except Exception:
                pass
        self._module = None
        self._preprocess_function = None
        self._postprocess_function = None
        self._host_raw = None
        self._host_summary = None
        self._host_compact = None
