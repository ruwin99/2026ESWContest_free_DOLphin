from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cuda_obstacle_pipeline import (
    CONTROL_ROI_Y_END,
    MODEL_INPUT_NBYTES,
    OUTPUT_ROWS,
    CudaObstaclePipeline,
)


INPUT_NAME = "images"
OUTPUT_NAME = "output0"
CAMERA_FRAME_SHAPE = (720, 1280, 3)
OBSTACLE_ROI_SHAPE = (240, 1280, 3)
INPUT_SHAPE = (1, 3, 256, 1280)
OUTPUT_SHAPE = (1, 300, 6)
PAD_TOP = 8
PAD_BOTTOM = 8
PAD_VALUE = 114
DEFAULT_CONFIDENCE_THRESHOLD = 0.30
METADATA_NMS_IOU = 0.70
CLASS_NAMES = ("obstacle",)
SOURCE_ONNX_SHA256 = (
    "76d64f7f0ccc3acea12df95eb8268ab73a69c04ec09bf37ca4fe09d221085bd9"
)
TENSORRT_VERSION_PREFIX = "10.3."
DETECTOR_PREFIX = "yolo26n-tensorrt/realtime/obstacle/"


@dataclass(frozen=True)
class ObstacleDetection:
    box_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.class_id]


@dataclass(frozen=True)
class ObstacleDetectionResult:
    detections: tuple[ObstacleDetection, ...]
    method: str
    confidence_threshold: float
    control_roi_detected: bool

    @property
    def detected(self) -> bool:
        return bool(self.detections)

    @property
    def boxes(self) -> tuple[tuple[float, float, float, float], ...]:
        return tuple(detection.box_xyxy for detection in self.detections)

    @property
    def scores(self) -> tuple[float, ...]:
        return tuple(detection.confidence for detection in self.detections)

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(detection.class_id for detection in self.detections)

    @property
    def status(self) -> str:
        return "ready"


def _validated_confidence_threshold(confidence_threshold: float) -> float:
    confidence_threshold = float(confidence_threshold)
    if not math.isfinite(confidence_threshold) or not 0.0 < confidence_threshold < 1.0:
        raise ValueError(
            "confidence_threshold must be finite and between zero and one."
        )
    return confidence_threshold


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """Convert the fixed top-camera control ROI to padded RGB FP32."""

    if not isinstance(frame, np.ndarray) or frame.shape != OBSTACLE_ROI_SHAPE:
        shape = getattr(frame, "shape", None)
        raise ValueError(
            "Obstacle TensorRT input must be the fixed 1280x240 HxWx3 top-camera "
            f"y=0:240 ROI; got {shape}. The detector does not resize the ROI."
        )
    if frame.dtype != np.uint8:
        raise ValueError("Obstacle camera input must contain uint8 BGR pixels.")

    tensor = np.full(
        INPUT_SHAPE,
        np.float32(PAD_VALUE / 255.0),
        dtype=np.float32,
    )
    rgb_chw = frame[:, :, ::-1].transpose(2, 0, 1).astype(np.float32)
    tensor[0, :, PAD_TOP : PAD_TOP + OBSTACLE_ROI_SHAPE[0], :] = rgb_chw / 255.0
    return np.ascontiguousarray(tensor)


def result_from_output(
    output: np.ndarray,
    method: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ObstacleDetectionResult:
    """Decode the exported one-to-one TopK rows without applying extra NMS."""

    confidence_threshold = _validated_confidence_threshold(confidence_threshold)
    output = np.asarray(output)
    if output.dtype != np.float32:
        raise ValueError(
            f"Obstacle output must have dtype float32, got {output.dtype}."
        )
    if output.shape != OUTPUT_SHAPE:
        raise ValueError(
            f"Obstacle output must have shape {OUTPUT_SHAPE}, got {output.shape}."
        )
    if not np.all(np.isfinite(output)):
        raise ValueError("Obstacle output contains NaN or infinity.")

    rows = output[0]
    boxes = rows[:, :4]
    scores = rows[:, 4]
    class_values = rows[:, 5]
    if np.any(scores < 0.0) or np.any(scores > 1.0):
        raise ValueError("Obstacle confidence values must be from zero through one.")
    if np.any(boxes[:, 2] < boxes[:, 0]) or np.any(boxes[:, 3] < boxes[:, 1]):
        raise ValueError("Obstacle output contains a reversed xyxy box.")

    rounded_classes = np.rint(class_values)
    if not np.array_equal(class_values, rounded_classes):
        raise ValueError("Obstacle class IDs must be exact integer-valued floats.")
    class_ids = rounded_classes.astype(np.int64)
    if np.any(class_ids != 0):
        raise ValueError("Obstacle output contains a class ID other than 0.")

    roi_height, roi_width = OBSTACLE_ROI_SHAPE[:2]
    detections: list[ObstacleDetection] = []
    for index in np.flatnonzero(scores >= confidence_threshold):
        x1, y1, x2, y2 = (float(value) for value in boxes[index])
        x1 = float(np.clip(x1, 0.0, float(roi_width)))
        x2 = float(np.clip(x2, 0.0, float(roi_width)))
        y1 = float(np.clip(y1 - PAD_TOP, 0.0, float(roi_height)))
        y2 = float(np.clip(y2 - PAD_TOP, 0.0, float(roi_height)))
        if x2 <= x1 or y2 <= y1:
            continue
        detections.append(
            ObstacleDetection(
                box_xyxy=(x1, y1, x2, y2),
                confidence=float(scores[index]),
                class_id=int(class_ids[index]),
            )
        )

    return ObstacleDetectionResult(
        detections=tuple(detections),
        method=str(method),
        confidence_threshold=confidence_threshold,
        control_roi_detected=any(
            detection.box_xyxy[1] < CONTROL_ROI_Y_END
            and detection.box_xyxy[3] > 0.0
            for detection in detections
        ),
    )


def result_from_compact_output(
    compact: np.ndarray,
    control_roi_detected: bool,
    method: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
) -> ObstacleDetectionResult:
    """Validate GPU-compacted rows without re-decoding raw TensorRT output."""

    confidence_threshold = _validated_confidence_threshold(confidence_threshold)
    compact = np.asarray(compact)
    if compact.dtype != np.float32:
        raise ValueError(
            f"Obstacle compact output must have dtype float32, got {compact.dtype}."
        )
    if (
        compact.ndim != 2
        or compact.shape[1:] != (OUTPUT_SHAPE[2],)
        or compact.shape[0] > OUTPUT_ROWS
    ):
        raise ValueError(
            "Obstacle compact output must have shape (N, 6), where 0 <= N <= "
            f"{OUTPUT_ROWS}; got {compact.shape}."
        )
    if not isinstance(control_roi_detected, (bool, np.bool_)):
        raise ValueError("Obstacle GPU control ROI flag must be boolean.")
    if not np.all(np.isfinite(compact)):
        raise ValueError("Obstacle compact output contains NaN or infinity.")

    boxes = compact[:, :4]
    scores = compact[:, 4]
    class_values = compact[:, 5]
    roi_height, roi_width = OBSTACLE_ROI_SHAPE[:2]
    if np.any(scores < confidence_threshold) or np.any(scores > 1.0):
        raise ValueError(
            "Obstacle compact confidence values violate the locked threshold."
        )
    if np.any(class_values != 0.0):
        raise ValueError("Obstacle compact output contains a class other than 0.")
    if (
        np.any(boxes[:, 0] < 0.0)
        or np.any(boxes[:, 2] > roi_width)
        or np.any(boxes[:, 1] < 0.0)
        or np.any(boxes[:, 3] > roi_height)
        or np.any(boxes[:, 2] <= boxes[:, 0])
        or np.any(boxes[:, 3] <= boxes[:, 1])
    ):
        raise ValueError("Obstacle compact output contains an invalid xyxy box.")

    expected_control = bool(
        np.any((boxes[:, 1] < CONTROL_ROI_Y_END) & (boxes[:, 3] > 0.0))
    )
    if bool(control_roi_detected) != expected_control:
        raise ValueError(
            "Obstacle GPU control ROI flag disagrees with compact detections."
        )
    detections = tuple(
        ObstacleDetection(
            box_xyxy=tuple(float(value) for value in row[:4]),
            confidence=float(row[4]),
            class_id=0,
        )
        for row in compact
    )
    return ObstacleDetectionResult(
        detections=detections,
        method=str(method),
        confidence_threshold=confidence_threshold,
        control_roi_detected=bool(control_roi_detected),
    )


class ObstacleDetector:
    """Synchronous TensorRT 10.3 runtime for the pinned YOLO26n export."""

    def __init__(
        self,
        engine_path: Path,
        expected_sha256: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    ) -> None:
        confidence_threshold = _validated_confidence_threshold(confidence_threshold)
        expected_sha256 = str(expected_sha256).lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(
                "expected_sha256 must be a 64-character hexadecimal digest."
            )

        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise ValueError(
                f"Obstacle TensorRT engine was not found: {self.engine_path}"
            )
        self.confidence_threshold = confidence_threshold
        self.engine_sha256: str | None = None
        self.method = (
            f"{DETECTOR_PREFIX}cuda:0/end2end-topk/no-nms/"
            "roi-y0-240/gpu-preprocess/gpu-compact"
        )

        self._trt: Any = None
        self._cudart: Any = None
        self._logger: Any = None
        self._runtime: Any = None
        self._engine: Any = None
        self._context: Any = None
        self._stream: Any = None
        self._device_input: Any = None
        self._device_output: Any = None
        self._gpu_pipeline: CudaObstaclePipeline | None = None

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
                    "This obstacle plan requires TensorRT 10.3.x, got "
                    f"{version or 'an unknown version'}."
                )
            self.method = (
                f"{DETECTOR_PREFIX}trt-{version}/cuda:0/end2end-topk/no-nms/"
                "roi-y0-240/gpu-preprocess/gpu-compact"
            )
            self._logger = trt.Logger(trt.Logger.WARNING)
            trt.init_libnvinfer_plugins(self._logger, "")
            self._runtime = trt.Runtime(self._logger)
            try:
                serialized_engine = self.engine_path.read_bytes()
            except OSError as exc:
                raise RuntimeError(
                    f"Could not read obstacle TensorRT engine: {self.engine_path}"
                ) from exc
            self.engine_sha256 = hashlib.sha256(serialized_engine).hexdigest()
            if self.engine_sha256 != expected_sha256:
                raise RuntimeError(
                    "Obstacle TensorRT engine changed after preflight SHA-256 "
                    f"validation: expected {expected_sha256}, got "
                    f"{self.engine_sha256}."
                )
            self._engine = self._runtime.deserialize_cuda_engine(serialized_engine)
            if self._engine is None:
                raise RuntimeError(
                    "TensorRT could not deserialize the obstacle engine."
                )
            self._validate_engine_contract()
            self._context = self._engine.create_execution_context()
            if self._context is None:
                raise RuntimeError(
                    "TensorRT could not create an obstacle execution context."
                )

            input_nbytes = int(np.prod(INPUT_SHAPE)) * np.dtype(np.float32).itemsize
            output_nbytes = int(np.prod(OUTPUT_SHAPE)) * np.dtype(np.float32).itemsize
            if input_nbytes != MODEL_INPUT_NBYTES:
                raise RuntimeError("Obstacle CUDA input byte contract changed.")
            self._device_input = self._cuda_call(
                cudart.cudaMalloc(input_nbytes),
                "cudaMalloc(obstacle input)",
            )
            self._device_output = self._cuda_call(
                cudart.cudaMalloc(output_nbytes),
                "cudaMalloc(obstacle output)",
            )
            self._stream = self._cuda_call(
                cudart.cudaStreamCreate(),
                "cudaStreamCreate(obstacle)",
            )
            if not self._context.set_tensor_address(
                INPUT_NAME, int(self._device_input)
            ):
                raise RuntimeError(
                    "TensorRT rejected the obstacle input buffer address."
                )
            if not self._context.set_tensor_address(
                OUTPUT_NAME, int(self._device_output)
            ):
                raise RuntimeError(
                    "TensorRT rejected the obstacle output buffer address."
                )
            self._gpu_pipeline = CudaObstaclePipeline(
                cudart,
                self.confidence_threshold,
            )
        except Exception:
            self.close()
            raise

    def _validate_engine_contract(self) -> None:
        trt = self._trt
        engine = self._engine
        names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
        if len(names) != 2 or set(names) != {INPUT_NAME, OUTPUT_NAME}:
            raise RuntimeError(
                "Obstacle TensorRT engine I/O names must be exactly "
                f"{INPUT_NAME!r} and {OUTPUT_NAME!r}, got {sorted(names)}."
            )
        if engine.get_tensor_mode(INPUT_NAME) != trt.TensorIOMode.INPUT:
            raise RuntimeError(f"Tensor {INPUT_NAME!r} is not an engine input.")
        if engine.get_tensor_mode(OUTPUT_NAME) != trt.TensorIOMode.OUTPUT:
            raise RuntimeError(f"Tensor {OUTPUT_NAME!r} is not an engine output.")
        for name, expected_shape in (
            (INPUT_NAME, INPUT_SHAPE),
            (OUTPUT_NAME, OUTPUT_SHAPE),
        ):
            actual_shape = tuple(engine.get_tensor_shape(name))
            if actual_shape != expected_shape:
                raise RuntimeError(
                    f"Obstacle TensorRT tensor {name!r} shape must be "
                    f"{expected_shape}, got {actual_shape}."
                )
            dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(name)))
            if dtype != np.dtype(np.float32):
                raise RuntimeError(
                    f"Obstacle TensorRT tensor {name!r} dtype must be float32, "
                    f"got {dtype}."
                )
            if engine.get_tensor_format(name) != trt.TensorFormat.LINEAR:
                raise RuntimeError(f"Tensor {name!r} must use LINEAR/CHW format.")
            if engine.get_tensor_location(name) != trt.TensorLocation.DEVICE:
                raise RuntimeError(
                    f"Tensor {name!r} must be located on the GPU device."
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

    def detect(self, frame: np.ndarray) -> ObstacleDetectionResult:
        if (
            self._context is None
            or self._stream is None
            or self._gpu_pipeline is None
        ):
            raise RuntimeError("Obstacle TensorRT detector is closed.")
        try:
            self._gpu_pipeline.upload_and_preprocess(
                frame,
                self._device_input,
                self._stream,
            )
            if not self._context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError("TensorRT execute_async_v3 returned false.")
            compact_output = self._gpu_pipeline.postprocess(
                self._device_output,
                self._stream,
            )
        except Exception as exc:
            raise RuntimeError(f"Obstacle TensorRT inference failed: {exc}") from exc

        try:
            return result_from_compact_output(
                compact_output.records,
                compact_output.control_roi_detected,
                self.method,
                self.confidence_threshold,
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Obstacle TensorRT output validation failed: {exc}"
            ) from exc

    def close(self) -> None:
        cudart = self._cudart
        stream = self._stream
        if cudart is not None and stream is not None:
            try:
                self._cuda_call(
                    cudart.cudaStreamSynchronize(stream),
                    "cudaStreamSynchronize(obstacle close)",
                )
            except Exception:
                pass
        gpu_pipeline = self._gpu_pipeline
        if gpu_pipeline is not None:
            try:
                gpu_pipeline.close()
            except Exception:
                pass
        self._gpu_pipeline = None
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
                    "cudaStreamDestroy(obstacle)",
                )
            except Exception:
                pass
        self._stream = None
        self._context = None
        self._engine = None
        self._runtime = None
        self._logger = None
        self._cudart = None
