from __future__ import annotations

import ctypes
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np


INPUT_NAME = "images"
OUTPUT_NAME = "logits"
CLASS_NAMES = ("Good", "Fair", "Poor", "Severe")
TENSORRT_VERSION_PREFIX = "10.3."
STUDENT_PROFILE = "student"
TEACHER_PROFILE = "teacher"
MODEL_PROFILES = (STUDENT_PROFILE, TEACHER_PROFILE)
STUDENT_INPUT_SHAPE = (1, 3, 240, 1280)
STUDENT_OUTPUT_SHAPE = (1, 4, 240, 1280)
TEACHER_INPUT_SHAPE = (1, 3, 720, 1280)
TEACHER_OUTPUT_SHAPE = (1, 4, 720, 1280)
PROFILE_SHAPES = {
    STUDENT_PROFILE: (STUDENT_INPUT_SHAPE, STUDENT_OUTPUT_SHAPE),
    TEACHER_PROFILE: (TEACHER_INPUT_SHAPE, TEACHER_OUTPUT_SHAPE),
}
# Backwards-compatible names identify the realtime/student contract.
INPUT_SHAPE = STUDENT_INPUT_SHAPE
OUTPUT_SHAPE = STUDENT_OUTPUT_SHAPE
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)
CLASS_COLORS_BGR = {
    1: (0, 215, 255),
    2: (0, 128, 255),
    3: (0, 0, 255),
}
MIN_BOX_AREA = 250.0


@dataclass(frozen=True)
class LetterboxTransform:
    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    top: int
    left: int


@dataclass
class DetectionResult:
    mask: np.ndarray
    boxes: list[tuple[int, int, int, int]]
    rust_ratio: float
    method: str
    status: str = "ready"
    class_map: np.ndarray | None = None
    class_ratios: dict[str, float] = field(default_factory=dict)


def preprocess_frame(
    frame: np.ndarray,
    profile: str = STUDENT_PROFILE,
) -> tuple[np.ndarray, LetterboxTransform]:
    """Apply a native static-shape teacher or student input contract."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Camera frame must be an HxWx3 NumPy array.")
    if frame.dtype != np.uint8 or not np.all(np.isfinite(frame)):
        raise ValueError("Camera frame must contain finite uint8 BGR pixels.")
    if profile not in MODEL_PROFILES:
        raise ValueError(f"Unknown TensorRT preprocessing profile: {profile}")
    original_height, original_width = frame.shape[:2]
    if original_height <= 0 or original_width <= 0:
        raise ValueError("Camera frame must not be empty.")

    input_shape, _ = PROFILE_SHAPES[profile]
    target_height, target_width = input_shape[2:]
    if (original_height, original_width) != (target_height, target_width):
        raise ValueError(
            f"TensorRT {profile} input frame must be "
            f"{target_width}x{target_height}; got "
            f"{original_width}x{original_height}. Native engines do not resize, "
            "letterbox, or pad frames."
        )
    if profile == STUDENT_PROFILE:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        model_input = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    else:
        model_input = frame.astype(np.float32)

    tensor = np.ascontiguousarray(model_input.transpose(2, 0, 1)[None])
    transform = LetterboxTransform(
        original_height=original_height,
        original_width=original_width,
        resized_height=original_height,
        resized_width=original_width,
        top=0,
        left=0,
    )
    return tensor, transform


def restore_class_map(
    logits: np.ndarray,
    transform: LetterboxTransform,
) -> np.ndarray:
    """Map native logits one-to-one to the input frame."""

    logits = np.asarray(logits)
    if logits.dtype != np.float32:
        raise ValueError(
            f"TensorRT logits must have dtype float32, got {logits.dtype}."
        )
    expected_shape = (
        1,
        len(CLASS_NAMES),
        transform.original_height,
        transform.original_width,
    )
    if logits.shape != expected_shape:
        raise ValueError(
            f"TensorRT output must have shape {expected_shape}, got {logits.shape}."
        )
    return np.argmax(logits[0], axis=0).astype(np.uint8)


def result_from_logits(
    logits: np.ndarray,
    transform: LetterboxTransform,
    method: str,
) -> DetectionResult:
    if not np.all(np.isfinite(logits)):
        raise ValueError("Segmentation logits contain NaN or infinity.")
    class_map = restore_class_map(logits, transform)
    return result_from_class_map(class_map, transform, method)


def result_from_class_map(
    class_map: np.ndarray,
    transform: LetterboxTransform,
    method: str,
) -> DetectionResult:
    class_map = np.asarray(class_map)
    expected_shape = (transform.original_height, transform.original_width)
    if class_map.dtype != np.uint8:
        raise ValueError(
            f"Segmentation class map must have dtype uint8, got {class_map.dtype}."
        )
    if class_map.shape != expected_shape:
        raise ValueError(
            f"Segmentation class map must have shape {expected_shape}, "
            f"got {class_map.shape}."
        )
    if np.any(class_map >= len(CLASS_NAMES)):
        raise ValueError("Segmentation class map contains an invalid class ID.")
    mask = np.where(class_map > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [
        cv2.boundingRect(contour)
        for contour in contours
        if cv2.contourArea(contour) >= MIN_BOX_AREA
    ]
    inspected_pixels = max(class_map.size, 1)
    class_ratios = {
        class_name: float(np.count_nonzero(class_map == class_id) / inspected_pixels)
        for class_id, class_name in enumerate(CLASS_NAMES)
    }
    rust_ratio = float(np.count_nonzero(mask) / inspected_pixels)
    return DetectionResult(
        mask=mask,
        boxes=boxes,
        rust_ratio=rust_ratio,
        method=method,
        class_map=class_map,
        class_ratios=class_ratios,
    )


class RustDetector:
    """Synchronous TensorRT 10 runtime for a DeepLabV3+ engine."""

    def __init__(
        self,
        engine_path: Path,
        profile: str = STUDENT_PROFILE,
        expected_sha256: str | None = None,
        *,
        gpu_argmax: bool = False,
    ) -> None:
        if profile not in MODEL_PROFILES:
            raise ValueError(f"Unknown TensorRT preprocessing profile: {profile}")
        if expected_sha256 is not None:
            expected_sha256 = str(expected_sha256).lower()
            if len(expected_sha256) != 64 or any(
                character not in "0123456789abcdef"
                for character in expected_sha256
            ):
                raise ValueError(
                    "expected_sha256 must be a 64-character hexadecimal digest."
                )
        self.profile = profile
        self.input_shape, self.output_shape = PROFILE_SHAPES[profile]
        self.engine_path = engine_path.expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(f"TensorRT engine was not found: {self.engine_path}")

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
        self._gpu_postprocessor: Any = None
        self.engine_sha256: str | None = None
        self.method = f"deeplabv3plus-tensorrt/{profile}/cuda:0"
        if gpu_argmax and profile != STUDENT_PROFILE:
            raise ValueError("GPU rust argmax is supported only for the student profile.")

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
            tensorrt_version = str(getattr(trt, "__version__", ""))
            if not tensorrt_version.startswith(TENSORRT_VERSION_PREFIX):
                raise RuntimeError(
                    "This plan requires TensorRT 10.3.x, got "
                    f"{tensorrt_version or 'an unknown version'}. Rebuild the plan "
                    "after changing TensorRT."
                )
            self.method = (
                f"deeplabv3plus-tensorrt/{profile}/"
                f"trt-{tensorrt_version}/cuda:0"
            )
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            try:
                serialized_engine = self.engine_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read TensorRT engine: {self.engine_path}"
                ) from exc
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if (
                expected_sha256 is not None
                and self.engine_sha256 != expected_sha256
            ):
                raise RuntimeError(
                    "TensorRT engine changed after preflight SHA-256 validation: "
                    f"expected {expected_sha256}, got {self.engine_sha256}."
                )
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT could not deserialize the engine. Verify the Jetson, "
                    "TensorRT version, and engine SHA-256."
                )
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError("TensorRT could not create an execution context.")

            input_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(INPUT_NAME)))
            output_dtype = np.dtype(trt.nptype(self._engine.get_tensor_dtype(OUTPUT_NAME)))
            if input_dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"TensorRT input dtype must be float32, got {input_dtype}."
                )
            if output_dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"TensorRT output dtype must be float32, got {output_dtype}."
                )

            input_nbytes = int(np.prod(self.input_shape)) * input_dtype.itemsize
            output_nbytes = int(np.prod(self.output_shape)) * output_dtype.itemsize
            self._host_input_pointer = self._cuda_call(
                cudart.cudaHostAlloc(input_nbytes, 0),
                "cudaHostAlloc(input)",
            )
            self._host_input = self._host_array(
                self._host_input_pointer,
                self.input_shape,
                input_dtype,
            )
            if not gpu_argmax:
                self._host_output_pointer = self._cuda_call(
                    cudart.cudaHostAlloc(output_nbytes, 0),
                    "cudaHostAlloc(output)",
                )
                self._host_output = self._host_array(
                    self._host_output_pointer,
                    self.output_shape,
                    output_dtype,
                )
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes),
                "cudaMalloc(input)",
            )
            self._device_output = self._cuda_call(
                cudart.cudaMalloc(output_nbytes),
                "cudaMalloc(output)",
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(),
                "cudaStreamCreate",
            )
            if not self._context.set_tensor_address(
                INPUT_NAME, int(self._device_input)
            ):
                raise RuntimeError("TensorRT rejected the input buffer address.")
            if not self._context.set_tensor_address(
                OUTPUT_NAME, int(self._device_output)
            ):
                raise RuntimeError("TensorRT rejected the output buffer address.")
            if gpu_argmax:
                from cuda_argmax import CudaArgmaxPostprocessor

                self._gpu_postprocessor = CudaArgmaxPostprocessor(
                    cudart,
                    self.output_shape,
                )
                self.method += "/gpu-argmax"
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> None:
        trt = self._trt
        engine = self._engine
        tensor_names = {
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        }
        if tensor_names != {INPUT_NAME, OUTPUT_NAME}:
            raise RuntimeError(
                "TensorRT engine I/O names must be exactly "
                f"{INPUT_NAME!r} and {OUTPUT_NAME!r}, got {sorted(tensor_names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Tensor {INPUT_NAME!r} is not an engine input.")
        if engine.get_tensor_mode(OUTPUT_NAME) != trt.TensorIOMode.OUTPUT:
            raise RuntimeError(f"Tensor {OUTPUT_NAME!r} is not an engine output.")
        input_shape = tuple(engine.get_tensor_shape(INPUT_NAME))
        output_shape = tuple(engine.get_tensor_shape(OUTPUT_NAME))
        if input_shape != self.input_shape:
            raise RuntimeError(
                f"TensorRT {self.profile} input shape must be "
                f"{self.input_shape}, got {input_shape}."
            )
        if output_shape != self.output_shape:
            raise RuntimeError(
                f"TensorRT {self.profile} output shape must be "
                f"{self.output_shape}, got {output_shape}."
            )
        input_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(INPUT_NAME)))
        output_dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(OUTPUT_NAME)))
        if input_dtype != np.dtype(np.float32):
            raise RuntimeError(
                f"TensorRT input dtype must be float32, got {input_dtype}."
            )
        if output_dtype != np.dtype(np.float32):
            raise RuntimeError(
                f"TensorRT output dtype must be float32, got {output_dtype}."
            )
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            if engine.get_tensor_format(tensor_name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(
                    f"Tensor {tensor_name!r} must use LINEAR/CHW format."
                )
            if engine.get_tensor_location(tensor_name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(
                    f"Tensor {tensor_name!r} must be located on the GPU device."
                )

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
        dtype = np.dtype(dtype)
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)),
            ctypes.POINTER(ctypes.c_ubyte),
        )
        byte_array = np.ctypeslib.as_array(byte_pointer, shape=(byte_count,))
        return byte_array.view(dtype).reshape(shape)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        if self._context is None or self._stream is None:
            raise RuntimeError("TensorRT detector is closed.")
        tensor, transform = preprocess_frame(frame, self.profile)
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
                "cudaMemcpyAsync(input)",
            )
            executed = self._context.execute_async_v3(
                stream_handle=int(self._stream)
            )
            if not executed:
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
            if self._gpu_postprocessor is not None:
                class_map = self._gpu_postprocessor.process(
                    self._device_output,
                    self._stream,
                )
            else:
                self._cuda_call(
                    self._cudart.cudaMemcpyAsync(
                        self._host_output.ctypes.data,
                        self._device_output,
                        self._host_output.nbytes,
                        self._cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                        self._stream,
                    ),
                    "cudaMemcpyAsync(output)",
                )
                self._cuda_call(
                    self._cudart.cudaStreamSynchronize(self._stream),
                    "cudaStreamSynchronize",
                )
        except Exception as exc:
            raise RuntimeError(f"TensorRT inference failed: {exc}") from exc

        try:
            if self._gpu_postprocessor is not None:
                return result_from_class_map(class_map, transform, self.method)
            logits = self._host_output.reshape(self.output_shape)
            return result_from_logits(logits, transform, self.method)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"TensorRT output validation failed: {exc}") from exc

    def close(self) -> None:
        cudart = self._cudart
        stream = self._stream
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(close)",
                )
            except Exception:
                pass
        if self._gpu_postprocessor is not None:
            try:
                self._gpu_postprocessor.close()
            except Exception:
                pass
            self._gpu_postprocessor = None
        for attribute in ("_device_input", "_device_output"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(
                            cudart.cudaFree(allocation),
                            f"cudaFree({attribute})",
                        )
                except Exception:
                    pass
                setattr(self, attribute, None)
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamDestroy(stream),
                    "cudaStreamDestroy",
                )
            except Exception:
                pass
        for attribute in ("_host_input_pointer", "_host_output_pointer"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    if cudart is not None:
                        self._cuda_call(
                            cudart.cudaFreeHost(allocation),
                            f"cudaFreeHost({attribute})",
                        )
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


def annotate(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
    output = frame.copy()
    overlay = output.copy()
    if result.class_map is not None and result.class_map.shape == output.shape[:2]:
        for class_id, color in CLASS_COLORS_BGR.items():
            overlay[result.class_map == class_id] = color
    else:
        overlay[result.mask > 0] = (0, 95, 255)
    output = cv2.addWeighted(overlay, 0.35, output, 0.65, 0)

    fair_ratio = result.class_ratios.get("Fair", 0.0)
    poor_ratio = result.class_ratios.get("Poor", 0.0)
    severe_ratio = result.class_ratios.get("Severe", 0.0)
    if severe_ratio > 0.0:
        color = CLASS_COLORS_BGR[3]
    elif poor_ratio > 0.0:
        color = CLASS_COLORS_BGR[2]
    elif fair_ratio > 0.0:
        color = CLASS_COLORS_BGR[1]
    else:
        color = (60, 180, 75)

    for x, y, width, height in result.boxes:
        cv2.rectangle(output, (x, y), (x + width, y + height), color, 2)

    ratio_label = (
        f"rust={result.rust_ratio * 100:.2f}%  "
        f"Fair={fair_ratio * 100:.2f}%  Poor={poor_ratio * 100:.2f}%  "
        f"Severe={severe_ratio * 100:.2f}%"
    )
    method_label = result.method
    if result.status != "ready":
        method_label += f"  ({result.status})"
    cv2.rectangle(
        output,
        (12, 12),
        (min(output.shape[1] - 12, 920), 76),
        (20, 35, 50),
        -1,
    )
    cv2.putText(
        output,
        ratio_label,
        (24, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        method_label,
        (24, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        color,
        2,
        cv2.LINE_AA,
    )
    return output
