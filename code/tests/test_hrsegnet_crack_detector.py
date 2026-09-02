from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np


JETSON_CODE = Path(__file__).resolve().parents[1] / "jetson_code"
sys.path.insert(0, str(JETSON_CODE))

from hrsegnet_crack_detector import (  # noqa: E402
    CAPTURE_DETECTOR_PREFIX,
    CAPTURE_INPUT_SHAPE,
    CAPTURE_OUTPUT_SHAPE,
    DEFAULT_PROBABILITY_THRESHOLD,
    DETECTOR_PREFIX,
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUT_NAME,
    OUTPUT_SHAPE,
    HrSegNetCrackDetector,
    preprocess_frame,
    probability_threshold_to_logit_margin,
    result_from_logits,
)


class HrSegNetPostprocessTests(unittest.TestCase):
    def test_preprocess_uses_rgb_and_half_range_normalization_without_resize(self) -> None:
        frame = np.zeros((128, 1280, 3), dtype=np.uint8)
        frame[0, 0] = (0, 127, 255)

        tensor = preprocess_frame(frame)

        self.assertEqual(tensor.shape, INPUT_SHAPE)
        self.assertEqual(tensor.dtype, np.float32)
        self.assertTrue(tensor.flags.c_contiguous)
        np.testing.assert_allclose(
            tensor[0, :, 0, 0],
            np.array([1.0, 127.0 / 127.5 - 1.0, -1.0], dtype=np.float32),
            atol=1e-7,
        )
        with self.assertRaisesRegex(ValueError, "does not resize"):
            preprocess_frame(np.zeros((127, 1280, 3), dtype=np.uint8))

    def test_capture_preprocess_uses_native_full_frame_without_resize_or_padding(self) -> None:
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[0, 0] = (0, 127, 255)

        tensor = preprocess_frame(frame, CAPTURE_INPUT_SHAPE)

        self.assertEqual(tensor.shape, CAPTURE_INPUT_SHAPE)
        np.testing.assert_allclose(
            tensor[0, :, 0, 0],
            np.array([1.0, 127.0 / 127.5 - 1.0, -1.0], dtype=np.float32),
            atol=1e-7,
        )
        with self.assertRaisesRegex(ValueError, "does not resize or pad"):
            preprocess_frame(np.zeros((719, 1280, 3), dtype=np.uint8), CAPTURE_INPUT_SHAPE)

    def test_logit_margin_is_equivalent_to_two_class_softmax_probability(self) -> None:
        self.assertEqual(DEFAULT_PROBABILITY_THRESHOLD, 0.55)
        for probability_threshold in (0.1, 0.5, 0.55, 0.73, 0.85, 0.99):
            with self.subTest(probability_threshold=probability_threshold):
                margin = probability_threshold_to_logit_margin(probability_threshold)
                probability = 1.0 / (1.0 + math.exp(-margin))
                self.assertAlmostEqual(probability, probability_threshold, places=12)

    def test_tie_behavior_matches_probability_threshold(self) -> None:
        logits = np.zeros(OUTPUT_SHAPE, dtype=np.float32)

        at_half = result_from_logits(logits, "test", 0.5, 1)
        above_half = result_from_logits(logits, "test", 0.5001, 1)

        self.assertEqual(at_half.crack_pixels, 128 * 1280)
        self.assertEqual(above_half.crack_pixels, 0)

    def test_rejects_nan_infinity_and_non_float32_logits(self) -> None:
        invalid_logits = (
            np.zeros(OUTPUT_SHAPE, dtype=np.float16),
            np.full(OUTPUT_SHAPE, np.nan, dtype=np.float32),
            np.full(OUTPUT_SHAPE, np.inf, dtype=np.float32),
        )
        for logits in invalid_logits:
            with self.subTest(dtype=logits.dtype, value=logits.flat[0]):
                with self.assertRaises(ValueError):
                    result_from_logits(logits, "test")

    def test_filters_8_connected_components_and_returns_control_result(self) -> None:
        logits = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
        logits[:, 0] = 1.0
        logits[0, 1, 10, 10] = 2.0
        logits[0, 1, 11, 11] = 2.0
        logits[0, 1, 40, 40:43] = 2.0
        logits[0, 1, 80, 80] = 2.0

        result = result_from_logits(
            logits,
            f"{DETECTOR_PREFIX}test",
            probability_threshold=0.5,
            min_component_pixels=2,
        )

        self.assertEqual(result.crack_pixels, 5)
        self.assertEqual(result.boxes, [(10, 10, 2, 2), (40, 40, 3, 1)])
        self.assertEqual(result.mask.dtype, np.uint8)
        self.assertEqual(result.mask.shape, (128, 1280))
        self.assertEqual(result.status, "ready")
        self.assertTrue(result.detected)
        self.assertAlmostEqual(result.max_crack_probability, 0.7310585786)
        self.assertEqual(result.candidate_pixels, 6)
        self.assertEqual(result.filtered_pixels, 5)
        self.assertEqual(result.min_component_pixels, 2)

    def test_diagnostic_max_probability_uses_overflow_safe_sigmoid(self) -> None:
        logits = np.zeros(OUTPUT_SHAPE, dtype=np.float32)
        logits[:, 0] = 100.0
        logits[:, 1] = -100.0
        result = result_from_logits(logits, "test", 0.5, 1)
        self.assertAlmostEqual(
            result.max_crack_probability,
            math.exp(-200.0) / (1.0 + math.exp(-200.0)),
        )
        self.assertEqual(result.candidate_pixels, 0)

        logits[:, 0] = -100.0
        logits[:, 1] = 100.0
        result = result_from_logits(logits, "test", 0.5, 1)
        self.assertEqual(result.max_crack_probability, 1.0)
        self.assertEqual(result.candidate_pixels, 128 * 1280)


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
        return _FakeTensorIOMode.INPUT if name == INPUT_NAME else _FakeTensorIOMode.OUTPUT

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        return self.shapes[name]

    def get_tensor_dtype(self, name: str):
        return self.dtypes[name]

    def get_tensor_format(self, name: str) -> str:
        return self.formats[name]

    def get_tensor_location(self, name: str) -> str:
        return self.locations[name]


class HrSegNetEngineContractTests(unittest.TestCase):
    def _detector(self) -> HrSegNetCrackDetector:
        detector = HrSegNetCrackDetector.__new__(HrSegNetCrackDetector)
        detector._trt = _FakeTensorRT()
        detector._engine = _FakeEngine()
        detector.input_shape = INPUT_SHAPE
        detector.output_shape = OUTPUT_SHAPE
        return detector

    def test_accepts_exact_static_fp32_linear_device_contract(self) -> None:
        self._detector()._validate_engine_contract()

    def test_accepts_exact_capture_contract_and_uses_capture_method_prefix(self) -> None:
        detector = self._detector()
        detector.input_shape = CAPTURE_INPUT_SHAPE
        detector.output_shape = CAPTURE_OUTPUT_SHAPE
        detector._engine.shapes = {
            INPUT_NAME: CAPTURE_INPUT_SHAPE,
            OUTPUT_NAME: CAPTURE_OUTPUT_SHAPE,
        }

        detector._validate_engine_contract()
        capture_logits = np.zeros(CAPTURE_OUTPUT_SHAPE, dtype=np.float32)
        result = result_from_logits(
            capture_logits,
            f"{CAPTURE_DETECTOR_PREFIX}test",
            output_shape=CAPTURE_OUTPUT_SHAPE,
        )
        self.assertEqual(result.mask.shape, (720, 1280))
        self.assertTrue(result.method.startswith(CAPTURE_DETECTOR_PREFIX))

    def test_rejects_contract_changes(self) -> None:
        mutations = (
            lambda engine: engine.names.append("extra"),
            lambda engine: engine.shapes.__setitem__(OUTPUT_NAME, (1, 1, 128, 1280)),
            lambda engine: engine.dtypes.__setitem__(INPUT_NAME, np.float16),
            lambda engine: engine.formats.__setitem__(OUTPUT_NAME, "chw32"),
            lambda engine: engine.locations.__setitem__(INPUT_NAME, "host"),
        )
        for mutate in mutations:
            detector = self._detector()
            mutate(detector._engine)
            detector._engine.num_io_tensors = len(detector._engine.names)
            with self.subTest(mutate=mutate), self.assertRaises(RuntimeError):
                detector._validate_engine_contract()

    def test_rejects_wrong_input_or_output_tensor_mode(self) -> None:
        for tensor_name in (INPUT_NAME, OUTPUT_NAME):
            detector = self._detector()
            original = detector._engine.get_tensor_mode
            detector._engine.get_tensor_mode = (
                lambda name, target=tensor_name: (
                    _FakeTensorIOMode.OUTPUT
                    if name == target and target == INPUT_NAME
                    else (
                        _FakeTensorIOMode.INPUT
                        if name == target
                        else original(name)
                    )
                )
            )
            with self.subTest(tensor_name=tensor_name), self.assertRaises(RuntimeError):
                detector._validate_engine_contract()


if __name__ == "__main__":
    unittest.main()
