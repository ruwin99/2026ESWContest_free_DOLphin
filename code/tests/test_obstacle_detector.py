from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from obstacle_detector import (  # noqa: E402
    CAMERA_FRAME_SHAPE,
    CLASS_NAMES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    INPUT_NAME,
    INPUT_SHAPE,
    OBSTACLE_ROI_SHAPE,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    PAD_BOTTOM,
    PAD_TOP,
    PAD_VALUE,
    SOURCE_ONNX_SHA256,
    ObstacleDetector,
    preprocess_frame,
    result_from_compact_output,
    result_from_output,
)


class ObstaclePreprocessTests(unittest.TestCase):
    def test_converts_fixed_roi_bgr_to_padded_normalized_rgb(self) -> None:
        frame = np.zeros(OBSTACLE_ROI_SHAPE, dtype=np.uint8)
        frame[0, 0] = (0, 127, 255)

        tensor = preprocess_frame(frame)

        self.assertEqual(tensor.shape, INPUT_SHAPE)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(
            tensor[0, :, PAD_TOP, 0],
            np.array([1.0, 127.0 / 255.0, 0.0], dtype=np.float32),
            atol=1e-7,
        )
        np.testing.assert_allclose(
            tensor[:, :, :PAD_TOP, :],
            np.float32(PAD_VALUE / 255.0),
            atol=0.0,
        )
        np.testing.assert_allclose(
            tensor[:, :, -PAD_BOTTOM:, :],
            np.float32(PAD_VALUE / 255.0),
            atol=0.0,
        )

    def test_rejects_resize_or_non_uint8_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "y=0:240 ROI"):
            preprocess_frame(np.zeros(CAMERA_FRAME_SHAPE, dtype=np.uint8))
        with self.assertRaisesRegex(ValueError, "uint8"):
            preprocess_frame(np.zeros(OBSTACLE_ROI_SHAPE, dtype=np.float32))


class ObstaclePostprocessTests(unittest.TestCase):
    @staticmethod
    def _output() -> np.ndarray:
        output = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
        output[0, :, 2] = 1.0
        output[0, :, 3] = 1.0
        return output

    def test_filters_at_metadata_threshold_and_restores_camera_coordinates(self) -> None:
        output = self._output()
        output[0, 0] = (-5.0, 4.0, 1300.0, 260.0, 0.80, 0.0)
        output[0, 1] = (10.0, 20.0, 30.0, 40.0, 0.30, 0.0)
        output[0, 2] = (50.0, 50.0, 60.0, 60.0, 0.299, 0.0)
        output[0, 3] = (10.0, 0.0, 20.0, 7.0, 0.90, 0.0)

        result = result_from_output(output, "test")

        self.assertEqual(DEFAULT_CONFIDENCE_THRESHOLD, 0.30)
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.detected)
        self.assertEqual(
            result.boxes,
            (
                (0.0, 0.0, 1280.0, 240.0),
                (10.0, 12.0, 30.0, 32.0),
            ),
        )
        self.assertAlmostEqual(result.scores[0], 0.80, places=6)
        self.assertAlmostEqual(result.scores[1], 0.30, places=6)
        self.assertEqual(result.class_ids, (0, 0))
        self.assertTrue(result.control_roi_detected)
        self.assertEqual(result.detections[0].class_name, "obstacle")
        self.assertEqual(CLASS_NAMES, ("obstacle",))

    def test_gpu_compact_result_matches_cpu_reference(self) -> None:
        output = self._output()
        output[0, 0] = (-5.0, 4.0, 1300.0, 260.0, 0.80, 0.0)
        output[0, 1] = (10.0, 247.0, 30.0, 260.0, 0.30, 0.0)
        output[0, 2] = (10.0, 0.0, 20.0, 7.0, 0.90, 0.0)
        output[0, 3] = (50.0, 50.0, 60.0, 60.0, 0.299, 0.0)
        reference = result_from_output(output, "cpu")
        compact = np.array(
            [
                (*detection.box_xyxy, detection.confidence, detection.class_id)
                for detection in reference.detections
            ],
            dtype=np.float32,
        )

        gpu = result_from_compact_output(
            compact,
            reference.control_roi_detected,
            "gpu",
        )

        self.assertEqual(gpu.detections, reference.detections)
        self.assertEqual(
            gpu.control_roi_detected,
            reference.control_roi_detected,
        )

    def test_gpu_compact_result_fails_closed_on_contract_mismatch(self) -> None:
        valid = np.array(
            [[10.0, 20.0, 30.0, 40.0, 0.30, 0.0]],
            dtype=np.float32,
        )
        with self.assertRaisesRegex(ValueError, "disagrees"):
            result_from_compact_output(valid, False, "gpu")
        with self.assertRaisesRegex(ValueError, "class other than 0"):
            invalid_class = valid.copy()
            invalid_class[0, 5] = 1.0
            result_from_compact_output(invalid_class, True, "gpu")
        with self.assertRaisesRegex(ValueError, "locked threshold"):
            below_threshold = valid.copy()
            below_threshold[0, 4] = 0.299
            result_from_compact_output(below_threshold, True, "gpu")

    def test_padding_only_box_is_dropped_before_roi_control(self) -> None:
        output = self._output()
        output[0, 0] = (10.0, 247.0, 30.0, 260.0, 0.80, 0.0)
        self.assertTrue(result_from_output(output, "cpu").control_roi_detected)

        output[0, 0] = (10.0, 248.0, 30.0, 260.0, 0.80, 0.0)
        self.assertFalse(result_from_output(output, "cpu").control_roi_detected)

    def test_keeps_overlapping_one_to_one_topk_rows_without_extra_nms(self) -> None:
        output = self._output()
        output[0, 0] = (10.0, 18.0, 100.0, 108.0, 0.90, 0.0)
        output[0, 1] = (11.0, 19.0, 99.0, 107.0, 0.80, 0.0)

        result = result_from_output(output, "test")

        self.assertEqual(len(result.detections), 2)

    def test_rejects_invalid_output_before_control_can_use_it(self) -> None:
        invalid_outputs = []

        wrong_dtype = self._output().astype(np.float64)
        invalid_outputs.append(wrong_dtype)
        invalid_outputs.append(np.zeros((1, 299, 6), dtype=np.float32))

        nan_output = self._output()
        nan_output[0, 0, 4] = np.nan
        invalid_outputs.append(nan_output)

        invalid_score = self._output()
        invalid_score[0, 0, 4] = 1.01
        invalid_outputs.append(invalid_score)

        fractional_class = self._output()
        fractional_class[0, 0, 5] = 0.5
        invalid_outputs.append(fractional_class)

        unknown_class = self._output()
        unknown_class[0, 0, 5] = 1.0
        invalid_outputs.append(unknown_class)

        reversed_box = self._output()
        reversed_box[0, 0, :4] = (10.0, 10.0, 9.0, 20.0)
        invalid_outputs.append(reversed_box)

        for output in invalid_outputs:
            with self.subTest(shape=output.shape, dtype=output.dtype):
                with self.assertRaises(ValueError):
                    result_from_output(output, "test")

    def test_rejects_invalid_confidence_threshold(self) -> None:
        output = self._output()
        for threshold in (0.0, 1.0, float("nan"), float("inf")):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                result_from_output(output, "test", threshold)


class _FakeTensorIOMode:
    INPUT = "input"
    OUTPUT = "output"


class _FakeTensorFormat:
    LINEAR = "linear"


class _FakeTensorLocation:
    DEVICE = "device"


class _FakeTensorRT:
    TensorIOMode = _FakeTensorIOMode
    TensorFormat = _FakeTensorFormat
    TensorLocation = _FakeTensorLocation

    @staticmethod
    def nptype(dtype):
        return dtype


class _FakeEngine:
    def __init__(self) -> None:
        self.names = [OUTPUT_NAME, INPUT_NAME]
        self.num_io_tensors = len(self.names)
        self.modes = {
            INPUT_NAME: _FakeTensorIOMode.INPUT,
            OUTPUT_NAME: _FakeTensorIOMode.OUTPUT,
        }
        self.shapes = {INPUT_NAME: INPUT_SHAPE, OUTPUT_NAME: OUTPUT_SHAPE}
        self.dtypes = {INPUT_NAME: np.float32, OUTPUT_NAME: np.float32}
        self.formats = {
            INPUT_NAME: _FakeTensorFormat.LINEAR,
            OUTPUT_NAME: _FakeTensorFormat.LINEAR,
        }
        self.locations = {
            INPUT_NAME: _FakeTensorLocation.DEVICE,
            OUTPUT_NAME: _FakeTensorLocation.DEVICE,
        }

    def get_tensor_name(self, index: int) -> str:
        return self.names[index]

    def get_tensor_mode(self, name: str) -> str:
        return self.modes[name]

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str):
        return self.dtypes[name]

    def get_tensor_format(self, name: str) -> str:
        return self.formats[name]

    def get_tensor_location(self, name: str) -> str:
        return self.locations[name]


class ObstacleEngineContractTests(unittest.TestCase):
    @staticmethod
    def _detector() -> ObstacleDetector:
        detector = ObstacleDetector.__new__(ObstacleDetector)
        detector._trt = _FakeTensorRT()
        detector._engine = _FakeEngine()
        return detector

    def test_accepts_exact_static_fp32_linear_device_contract(self) -> None:
        self._detector()._validate_engine_contract()

    def test_rejects_any_engine_contract_change(self) -> None:
        mutations = (
            lambda engine: engine.names.append("extra"),
            lambda engine: engine.modes.__setitem__(INPUT_NAME, "output"),
            lambda engine: engine.modes.__setitem__(OUTPUT_NAME, "input"),
            lambda engine: engine.shapes.__setitem__(INPUT_NAME, (1, 3, 720, 1280)),
            lambda engine: engine.shapes.__setitem__(OUTPUT_NAME, (1, 300, 7)),
            lambda engine: engine.dtypes.__setitem__(INPUT_NAME, np.float16),
            lambda engine: engine.dtypes.__setitem__(OUTPUT_NAME, np.float16),
            lambda engine: engine.formats.__setitem__(INPUT_NAME, "chw32"),
            lambda engine: engine.locations.__setitem__(OUTPUT_NAME, "host"),
        )
        for mutate in mutations:
            detector = self._detector()
            mutate(detector._engine)
            detector._engine.num_io_tensors = len(detector._engine.names)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                detector._validate_engine_contract()

    def test_rejects_invalid_expected_digest_before_engine_or_cuda_loading(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected_sha256"):
            ObstacleDetector(Path("missing.plan"), "invalid")

    def test_records_the_inspected_source_onnx_digest(self) -> None:
        self.assertEqual(
            SOURCE_ONNX_SHA256,
            "76d64f7f0ccc3acea12df95eb8268ab73a69c04ec09bf37ca4fe09d221085bd9",
        )


if __name__ == "__main__":
    unittest.main()
    OBSTACLE_ROI_SHAPE,
