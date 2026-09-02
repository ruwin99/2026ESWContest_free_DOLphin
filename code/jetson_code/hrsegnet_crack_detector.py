from __future__ import annotations

import ctypes
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from crack_detector import CrackDetectionResult


INPUT_NAME = "images"
OUTPUT_NAME = "crack_logits"
INPUT_SHAPE = (1, 3, 128, 1280)
OUTPUT_SHAPE = (1, 2, 128, 1280)
CAPTURE_INPUT_SHAPE = (1, 3, 720, 1280)
CAPTURE_OUTPUT_SHAPE = (1, 2, 720, 1280)
TENSORRT_VERSION_PREFIX = "10.3."
DETECTOR_PREFIX = "hrsegnet-b32-tensorrt/realtime/crack/"
CAPTURE_DETECTOR_PREFIX = "hrsegnet-b32-tensorrt/capture/crack/"
DEFAULT_PROBABILITY_THRESHOLD = 0.55
DEFAULT_MIN_COMPONENT_PIXELS = 20

ROLE_CONTRACTS = {
    "realtime": (INPUT_SHAPE, OUTPUT_SHAPE, DETECTOR_PREFIX),
    "capture": (
        CAPTURE_INPUT_SHAPE,
        CAPTURE_OUTPUT_SHAPE,
        CAPTURE_DETECTOR_PREFIX,
    ),
}


@dataclass
class HrSegNetCrackDetectionResult(CrackDetectionResult):
    """Control result plus read-only diagnostics from the same raw logits."""

    max_crack_probability: float = 0.0
    candidate_pixels: int = 0
    filtered_pixels: int = 0
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS


def _stable_sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def probability_threshold_to_logit_margin(probability_threshold: float) -> float:
    probability_threshold = float(probability_threshold)
    if not math.isfinite(probability_threshold) or not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be finite and between zero and one.")
    return math.log(probability_threshold) - math.log1p(-probability_threshold)


def preprocess_frame(
    frame: np.ndarray,
    input_shape: tuple[int, int, int, int] = INPUT_SHAPE,
) -> np.ndarray:
    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be an HxWx3 NumPy array.")
    if frame.dtype != np.uint8:
        raise ValueError("Camera frame must contain uint8 BGR pixels.")
    expected_height, expected_width = input_shape[2:]
    if frame.shape[:2] != (expected_height, expected_width):
        raise ValueError(
            "HrSegNet TensorRT input frame must be "
            f"{expected_width}x{expected_height}; got {frame.shape[1]}x{frame.shape[0]}. "
            "The role-specific engine does not resize or pad."
        )

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    normalized = rgb / 127.5 - 1.0
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def result_from_logits(
    logits: np.ndarray,
    method: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
    output_shape: tuple[int, int, int, int] = OUTPUT_SHAPE,
) -> HrSegNetCrackDetectionResult:
    margin_threshold = probability_threshold_to_logit_margin(probability_threshold)
    if (
        not isinstance(min_component_pixels, int)
        or isinstance(min_component_pixels, bool)
        or min_component_pixels <= 0
    ):
        raise ValueError("min_component_pixels must be a positive integer.")

    logits = np.asarray(logits)
    if logits.dtype != np.float32:
        raise ValueError(f"HrSegNet logits must have dtype float32, got {logits.dtype}.")
    if logits.shape != output_shape:
        raise ValueError(
            f"HrSegNet logits must have shape {output_shape}, got {logits.shape}."
        )
    if not np.all(np.isfinite(logits)):
        raise ValueError("HrSegNet logits contain NaN or infinity.")

    margin = logits[0, 1] - logits[0, 0]
    if not np.all(np.isfinite(margin)):
        raise ValueError("HrSegNet logit margin contains NaN or infinity.")
    candidate = (margin >= margin_threshold).astype(np.uint8)
    candidate_pixels = int(np.count_nonzero(candidate))
    max_crack_probability = _stable_sigmoid(float(np.max(margin)))
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    filtered = np.zeros(candidate.shape, dtype=np.uint8)
    boxes: list[tuple[int, int, int, int]] = []
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_component_pixels:
            continue
        filtered[labels == component_id] = 255
        boxes.append(
            (
                int(stats[component_id, cv2.CC_STAT_LEFT]),
                int(stats[component_id, cv2.CC_STAT_TOP]),
                int(stats[component_id, cv2.CC_STAT_WIDTH]),
                int(stats[component_id, cv2.CC_STAT_HEIGHT]),
            )
        )

    crack_pixels = int(np.count_nonzero(filtered))
    inspected_pixels = int(filtered.size)
    return HrSegNetCrackDetectionResult(
        mask=filtered,
        boxes=boxes,
        crack_pixels=crack_pixels,
        inspected_pixels=inspected_pixels,
        crack_ratio=crack_pixels / inspected_pixels,
        detected=crack_pixels > 0,
        method=str(method),
        probability_threshold=float(probability_threshold),
        max_crack_probability=max_crack_probability,
        candidate_pixels=candidate_pixels,
        filtered_pixels=crack_pixels,
        min_component_pixels=min_component_pixels,
    )


class HrSegNetCrackDetector:
    """Synchronous TensorRT runtime for the approved HrSegNet-B32 baseline."""

    def __init__(
        self,
        engine_path: Path,
        expected_sha256: str,
        probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
        min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
        *,
        role: str,
    ) -> None:
        probability_threshold_to_logit_margin(probability_threshold)
        if (
            not isinstance(min_component_pixels, int)
            or isinstance(min_component_pixels, bool)
            or min_component_pixels <= 0
        ):
            raise ValueError("min_component_pixels must be a positive integer.")
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError("expected_sha256 must be a 64-character hexadecimal digest.")
        if role not in ROLE_CONTRACTS:
            raise ValueError("role must be exactly 'capture' or 'realtime'.")

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(f"HrSegNet TensorRT engine was not found: {self.engine_path}")
        self.probability_threshold = float(probability_threshold)
        self.min_component_pixels = min_component_pixels
        self.role = role
        self.input_shape, self.output_shape, detector_prefix = ROLE_CONTRACTS[role]
        self.engine_sha256: str | None = None
        self.method = f"{detector_prefix}cuda:0/logit-margin"

        self._trt: Any = None
        self._cudart: Any = None
        self._logger: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._host_input_pointer: Any = None
        self._host_output_pointer: Any = None
        self._host_input: np.ndarray | None = None
        self._host_output: np.ndarray | None = None
        self._device_input: Any = None
        self._device_output: Any = None

        try:
            import tensorrt as trt
            from cuda import cudart
        except ImportError as exc:
            raise RuntimeError(
                "TensorRT and NVIDIA CUDA Python are required. "
                "Run scripts/requirement.sh first."
            ) from exc

        self._trt = trt
        self._cudart = cudart
        try:
            version = str(getattr(trt, "__version__", ""))
            if not version.startswith(TENSORRT_VERSION_PREFIX):
                raise RuntimeError(
                    "This HrSegNet plan requires TensorRT 10.3.x, got "
                    f"{version or 'an unknown version'}."
                )
            self.method = f"{detector_prefix}trt-{version}/cuda:0/logit-margin"
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            try:
                serialized_engine = self.engine_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read HrSegNet TensorRT engine: {self.engine_path}"
                ) from exc
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if self.engine_sha256 != expected_sha256:
                raise RuntimeError(
                    "HrSegNet TensorRT engine changed after preflight SHA-256 "
                    f"validation: expected {expected_sha256}, got {self.engine_sha256}."
                )
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError("TensorRT could not deserialize the HrSegNet engine.")
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an HrSegNet execution context.")

            input_nbytes = int(np.prod(self.input_shape)) * np.dtype(np.float32).itemsize
            output_nbytes = int(np.prod(self.output_shape)) * np.dtype(np.float32).itemsize
            self._host_input_pointer = self._cuda_call(
                cudart.cudaHostAlloc(input_nbytes, 0), "cudaHostAlloc(HrSegNet input)"
            )
            self._host_output_pointer = self._cuda_call(
                cudart.cudaHostAlloc(output_nbytes, 0), "cudaHostAlloc(HrSegNet output)"
            )
            self._host_input = self._host_array(
                self._host_input_pointer, self.input_shape, np.dtype(np.float32)
            )
            self._host_output = self._host_array(
                self._host_output_pointer, self.output_shape, np.dtype(np.float32)
            )
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes), "cudaMalloc(HrSegNet input)"
            )
            self._device_output = self._cuda_call(
                cudart.cudaMalloc(output_nbytes), "cudaMalloc(HrSegNet output)"
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(), "cudaStreamCreate(HrSegNet)"
            )
            if not self._context.set_tensor_address(INPUT_NAME, int(self._device_input)):
                raise RuntimeError("TensorRT rejected the HrSegNet input buffer address.")
            if not self._context.set_tensor_address(OUTPUT_NAME, int(self._device_output)):
                raise RuntimeError("TensorRT rejected the HrSegNet output buffer address.")
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> None:
        trt = self._trt
        engine = self._engine
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        if len(names) != 2 or set(names) != {INPUT_NAME, OUTPUT_NAME}:
            raise RuntimeError(
                "HrSegNet TensorRT engine I/O names must be exactly "
                f"{INPUT_NAME!r} and {OUTPUT_NAME!r}, got {sorted(names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Tensor {INPUT_NAME!r} is not an engine input.")
        if engine.get_tensor_mode(OUTPUT_NAME) != trt.TensorIOMode.OUTPUT:
            raise RuntimeError(f"Tensor {OUTPUT_NAME!r} is not an engine output.")
        for name, shape in (
            (INPUT_NAME, self.input_shape),
            (OUTPUT_NAME, self.output_shape),
        ):
            actual_shape = tuple(engine.get_tensor_shape(name))
            if actual_shape != shape:
                raise RuntimeError(
                    f"HrSegNet TensorRT tensor {name!r} shape must be {shape}, "
                    f"got {actual_shape}."
                )
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"HrSegNet TensorRT tensor {name!r} dtype must be float32, got {dtype}."
                )
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f"Tensor {name!r} must use LINEAR/CHW format.")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(f"Tensor {name!r} must be located on the GPU device.")

    def _cuda_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA status.")
        error = result[0]
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed with CUDA error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    @staticmethod
    def _host_array(
        pointer: Any,
        shape: tuple[int, ...],
        dtype: np.dtype[Any],
    ) -> np.ndarray:
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)), ctypes.POINTER(ctypes.c_ubyte)
        )
        byte_array = np.ctypeslib.as_array(byte_pointer, shape=(byte_count,))
        return byte_array.view(dtype).reshape(shape)

    def detect(self, frame: np.ndarray) -> CrackDetectionResult:
        if self._context is None or self._stream is None:
            raise RuntimeError("HrSegNet TensorRT detector is closed.")
        tensor = preprocess_frame(frame, self.input_shape)
        try:
            np.copyto(self._host_input, tensor, casting="no")
            self._cuda_call(
                self._cudart.cudaMemcpyAsync(
                    self._device_input,
                    self._host_input.ctypes.data,
                    self._host_input.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self._stream,
                ),
                "cudaMemcpyAsync(HrSegNet input)",
            )
            if not self._context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
            self._cuda_call(
                self._cudart.cudaMemcpyAsync(
                    self._host_output.ctypes.data,
                    self._device_output,
                    self._host_output.nbytes,
                    self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self._stream,
                ),
                "cudaMemcpyAsync(HrSegNet output)",
            )
            self._cuda_call(
                self._cudart.cudaStreamSynchronize(self._stream),
                "cudaStreamSynchronize(HrSegNet)",
            )
        except Exception as exc:
            raise RuntimeError(f"HrSegNet TensorRT inference failed: {exc}") from exc

        try:
            return result_from_logits(
                self._host_output,
                self.method,
                self.probability_threshold,
                self.min_component_pixels,
                self.output_shape,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"HrSegNet TensorRT output validation failed: {exc}"
            ) from exc

    def close(self) -> None:
        cudart = self._cudart
        stream = self._stream
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(HrSegNet close)",
                )
            except Exception:
                pass
        for attribute in ("_device_input", "_device_output"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(cudart.cudaFree(allocation), f"cudaFree({attribute})")
                except Exception:
                    pass
                setattr(self, attribute, None)
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(cudart.cudaStreamDestroy(stream), "cudaStreamDestroy(HrSegNet)")
            except Exception:
                pass
        for attribute in ("_host_input_pointer", "_host_output_pointer"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(cudart.cudaFreeHost(allocation), f"cudaFreeHost({attribute})")
                except Exception:
                    pass
                setattr(self, attribute, None)
        self._stream = None
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None
        self._host_input = None
        self._host_output = None
        self._cudart = None
