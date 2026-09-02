from __future__ import annotations

import argparse
import hashlib
import os
import queue
import select
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import serial

from dashboard_export import (
    export_capture_record,
    export_top_crack_record,
    finalize_dashboard_run,
)
from crack_detector import (
    CRACK_COLOR_BGR,
    DEFAULT_MIN_COMPONENT_PIXELS,
    DEFAULT_PROBABILITY_THRESHOLD,
    REALTIME_DETECTOR_PREFIX as REALTIME_CRACK_DETECTOR_PREFIX,
    REALTIME_PROFILE as CRACK_REALTIME_PROFILE,
    CrackDetector,
    annotate_cracks,
)
from inspection_report import (
    INITIAL_PHASE,
    MANUAL_PHASE,
    RAIL_SECTION_TARGET,
    RESCAN_PHASE,
    InspectionWorkbook,
)
from hrsegnet_crack_detector import (
    CAPTURE_DETECTOR_PREFIX as HRSEGNET_CAPTURE_CRACK_DETECTOR_PREFIX,
    DEFAULT_MIN_COMPONENT_PIXELS as HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS,
    DEFAULT_PROBABILITY_THRESHOLD as HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD,
    DETECTOR_PREFIX as HRSEGNET_CRACK_DETECTOR_PREFIX,
    HrSegNetCrackDetector,
)
from multitask_detector import (
    CRACK_METHOD_PREFIX as MULTITASK_CRACK_METHOD_PREFIX,
    OptimizedMultitaskDetector,
)
from optimized_rust_detector import OptimizedRustDetector
from obstacle_detector import (
    DEFAULT_CONFIDENCE_THRESHOLD as OBSTACLE_DEFAULT_CONFIDENCE_THRESHOLD,
    DETECTOR_PREFIX as OBSTACLE_DETECTOR_PREFIX,
    ObstacleDetector,
    ObstacleDetectionResult,
)
from rust_detector import (
    CLASS_COLORS_BGR,
    LetterboxTransform,
    STUDENT_PROFILE,
    TEACHER_PROFILE,
    RustDetector,
    annotate,
    result_from_class_map,
)


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_RATE = 30
LATEST_FRAME_TIMEOUT_SECONDS = 0.5
LATEST_FRAME_CLOSE_TIMEOUT_SECONDS = 2.0
LATEST_DISPLAY_CLOSE_TIMEOUT_SECONDS = 2.0
DISPLAY_EVENT_POLL_SECONDS = 0.01
REALTIME_INPUT_MAX_AGE_SECONDS = 0.2
REALTIME_RESULT_MAX_AGE_SECONDS = 0.8
REALTIME_FRAME_TIMEOUT_SECONDS = 0.2
DUAL_CAMERA_READ_POLL_SECONDS = 0.001
REALTIME_TEST_ACTUATOR_LOG_INTERVAL_SECONDS = 1.0
DUAL_TIMING_WARMUP_SECONDS = 10.0
DUAL_TIMING_REPORT_INTERVAL_SECONDS = 5.0
DUAL_GUI_FRAME_RATE = 7.0
REALTIME_ACTUATOR_HEARTBEAT_SECONDS = 0.2
REALTIME_LAST_VALID_COMMAND_HOLD_SECONDS = 1.2
REALTIME_ACTUATOR_WORKER_CLOSE_TIMEOUT_SECONDS = 2.0
UART_WRITE_TIMEOUT_SECONDS = 0.2
WINDOW_NAME = "B0441 Rail Surface Inspection Preview"
DEFAULT_SERIAL_PORT = "/dev/ttyACM0"
DEFAULT_BAUD_RATE = 115200
START_COMMAND = b"START\r\n"
CAPTURE_OK_COMMAND = b"CAPTURE_OK\r\n"
ACTUATOR_TEST_START_COMMAND = b"ACTUATOR_TEST_START\r\n"
ACTUATOR_TEST_STOP_COMMAND = b"ACTUATOR_TEST_STOP\r\n"
CLEANER_PWM_33_3_COMMAND = b"CLEANER_PWM_33_3\r\n"
CLEANER_PWM_55_6_COMMAND = b"CLEANER_PWM_55_6\r\n"
# Backward-compatible Python name; the transmitted protocol is now explicit.
CLEANER_ON_COMMAND = CLEANER_PWM_33_3_COMMAND
CLEANER_OFF_COMMAND = b"CLEANER_OFF\r\n"
PUMP_ON_COMMAND = b"PUMP_ON\r\n"
PUMP_OFF_COMMAND = b"PUMP_OFF\r\n"
FRONT_CLEANER_PWM_33_3_COMMAND = b"FRONT_CLEANER_PWM_33_3\r\n"
FRONT_CLEANER_PWM_55_6_COMMAND = b"FRONT_CLEANER_PWM_55_6\r\n"
FRONT_CLEANER_OFF_COMMAND = b"FRONT_CLEANER_OFF\r\n"
FRONT_PUMP_ON_COMMAND = b"FRONT_PUMP_ON\r\n"
FRONT_PUMP_OFF_COMMAND = b"FRONT_PUMP_OFF\r\n"
SIDE_CLEANER_PWM_33_3_COMMAND = b"SIDE_CLEANER_PWM_33_3\r\n"
SIDE_CLEANER_PWM_55_6_COMMAND = b"SIDE_CLEANER_PWM_55_6\r\n"
SIDE_CLEANER_OFF_COMMAND = b"SIDE_CLEANER_OFF\r\n"
SIDE_PUMP_ON_COMMAND = b"SIDE_PUMP_ON\r\n"
SIDE_PUMP_OFF_COMMAND = b"SIDE_PUMP_OFF\r\n"
ACTUATOR_TEST_READY_TRIGGER = "ACTUATOR_TEST_READY"
ACTUATOR_TEST_READY_TIMEOUT_SECONDS = 2.0
ACTUATOR_TEST_READY_POLL_SECONDS = 0.01
CLEANER_HEARTBEAT_SECONDS = 0.2
PUMP_HEARTBEAT_SECONDS = 0.2
REALTIME_CONTROL_WINDOW_SIZE = 4
CRACK_CONTROL_STOP_RATIO = 0.0005
CLEANER_BASE_DUTY_PERCENT = 33.3
CLEANER_STAGE_ONE_DUTY_PERCENT = 55.6
RUST_CLASS_IDS = (0, 1, 2, 3)
REALTIME_RUST_ROI_TOP = 0
REALTIME_RUST_ROI_BOTTOM = 240
REALTIME_CRACK_ROI_TOP = 112
REALTIME_CRACK_ROI_BOTTOM = 240
STARTED_TRIGGER = "STARTED"
CAPTURE_TRIGGER = "CAMERA_CAPTURE"
RETURN_START_TRIGGER = "RETURN_START"
REALTIME_START_TRIGGER = "REALTIME_START"
RESCAN_RETURN_START_TRIGGER = "RESCAN_RETURN_START"
RESCAN_START_TRIGGER = "RESCAN_START"
RESCAN_DONE_TRIGGER = "RESCAN_DONE"
DONE_TRIGGER = "DONE"
CAPTURE_SCAN_MODE = "CAPTURE_SCAN"
RETURN_MODE = "RETURN"
REALTIME_MODE = "REALTIME"
RESCAN_RETURN_MODE = "RESCAN_RETURN"
RESCAN_MODE = "RESCAN"
RESCAN_DONE_MODE = "RESCAN_DONE"
COMPLETE_MODE = "COMPLETE"
AUTOMATIC_RAIL_SECTION_TARGET = RAIL_SECTION_TARGET
CAPTURE_NOTICE_SECONDS = 0.5
CAPTURE_SETTLE_SECONDS = 0.05
MOTION_TRIGGERS = (
    CAPTURE_TRIGGER,
    RETURN_START_TRIGGER,
    REALTIME_START_TRIGGER,
    RESCAN_RETURN_START_TRIGGER,
    RESCAN_START_TRIGGER,
    RESCAN_DONE_TRIGGER,
    DONE_TRIGGER,
)
CAPTURE_DIRECTORY = Path(__file__).resolve().parent.parent / "outputs" / "captures"
DEFAULT_REPORT_DIRECTORY = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_STUDENT_ENGINE = (
    Path.home()
    / "models"
    / "plans_new"
    / "realtime-rust-mnv2-os8-w1280-h240-fp16.plan"
)
DEFAULT_TEACHER_ENGINE = (
    Path.home()
    / "models"
    / "plans_new"
    / "corrosion-capture-r101-os8-w1280-h720-fp32.plan"
)
DEFAULT_CAPTURE_HRSEGNET_CRACK_ENGINE = (
    Path.home()
    / "models"
    / "hrsegnet_b32_20260816"
    / "plans"
    / "hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan"
)
DEFAULT_REALTIME_CRACK_ENGINE = (
    Path.home()
    / "models"
    / "plans_new"
    / "realtime-crack-bgcrack-w1280-h128-fp32.plan"
)
DEFAULT_REALTIME_MULTITASK_ENGINE = (
    Path.home()
    / "models"
    / "plans_new"
    / "realtime-rust-crack-multitask-optimized-w1280-h240-p099-fp32-nansafe-v2.plan"
)
DEFAULT_REALTIME_HRSEGNET_CRACK_ENGINE = (
    Path.home()
    / "models"
    / "hrsegnet_b32_20260816"
    / "plans"
    / "hrsegnet-b32-crackseg9k-realtime-w1280-h128-fp32-notf32.plan"
)
DEFAULT_OBSTACLE_ENGINE = (
    Path.home()
    / "models"
    / "obstacle_yolo26n_20260821"
    / "plans"
    / "obstacle-yolo26n-hardneg-all1410-roi-y0-240-w1280-h240-int-h256-fp32-notf32.plan"
)
DEFAULT_SIDE_CAMERA_DEVICE = (
    "/dev/v4l/by-path/"
    "platform-3610000.usb-usb-0:2.1:1.0-video-index0"
)
DEFAULT_TOP_CAMERA_DEVICE = (
    "/dev/v4l/by-path/"
    "platform-3610000.usb-usb-0:2.3:1.0-video-index0"
)
SIDE_CAMERA_ROLE = "side"
TOP_CAMERA_ROLE = "top"
TEACHER_DETECTOR_PREFIX = "deeplabv3plus-tensorrt/teacher/"
CAPTURE_ANALYSIS_QUEUE_CAPACITY = AUTOMATIC_RAIL_SECTION_TARGET * 2
CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class CaptureAnalysisTask:
    raw_capture_path: Path
    phase: str
    phase_sequence: int | None
    trigger: str
    captured_at: datetime
    camera_role: str = SIDE_CAMERA_ROLE
    frame_read_completed_at: float | None = None


@dataclass(frozen=True)
class CameraFrame:
    frame: np.ndarray
    sequence: int
    read_completed_at: float


@dataclass(frozen=True)
class DualRealtimeFrameRead:
    side_frame: CameraFrame | None
    top_frame: CameraFrame | None
    side_cache_expired: bool
    top_cache_expired: bool
    side_error: str | None = None
    top_error: str | None = None
    side_waiting: bool = False
    top_waiting: bool = False


@dataclass(frozen=True)
class ActuatorCommandSet:
    cleaner_pwm_33_3: bytes
    cleaner_pwm_55_6: bytes
    cleaner_off: bytes
    pump_on: bytes
    pump_off: bytes


LEGACY_ACTUATOR_COMMANDS = ActuatorCommandSet(
    CLEANER_PWM_33_3_COMMAND,
    CLEANER_PWM_55_6_COMMAND,
    CLEANER_OFF_COMMAND,
    PUMP_ON_COMMAND,
    PUMP_OFF_COMMAND,
)
FRONT_ACTUATOR_COMMANDS = ActuatorCommandSet(
    FRONT_CLEANER_PWM_33_3_COMMAND,
    FRONT_CLEANER_PWM_55_6_COMMAND,
    FRONT_CLEANER_OFF_COMMAND,
    FRONT_PUMP_ON_COMMAND,
    FRONT_PUMP_OFF_COMMAND,
)
SIDE_ACTUATOR_COMMANDS = ActuatorCommandSet(
    SIDE_CLEANER_PWM_33_3_COMMAND,
    SIDE_CLEANER_PWM_55_6_COMMAND,
    SIDE_CLEANER_OFF_COMMAND,
    SIDE_PUMP_ON_COMMAND,
    SIDE_PUMP_OFF_COMMAND,
)


class LatestFrameCamera:
    """Drain one VideoCapture in a thread and expose only its newest frame."""

    def __init__(self, capture) -> None:
        self._capture = capture
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._latest: CameraFrame | None = None
        self._error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._reader_loop,
            name="latest-camera-frame-reader",
            daemon=True,
        )
        self._thread.start()

    @property
    def latest_sequence(self) -> int:
        with self._condition:
            return 0 if self._latest is None else self._latest.sequence

    def read_latest(
        self,
        *,
        after_sequence: int | None = None,
        timeout: float = LATEST_FRAME_TIMEOUT_SECONDS,
    ) -> CameraFrame:
        """Return a new-enough snapshot without replaying queued old frames."""

        deadline = time.monotonic() + timeout
        minimum_sequence = 0 if after_sequence is None else after_sequence + 1
        while True:
            with self._condition:
                latest = self._latest
                if (
                    latest is not None
                    and latest.sequence >= minimum_sequence
                ):
                    return latest
                if self._error is not None:
                    raise RuntimeError(f"Camera reader failed: {self._error}") from self._error
                if self._closed:
                    raise RuntimeError("Camera reader is closed.")
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"No fresh camera frame arrived within {timeout:.3f} seconds."
                    )
                self._condition.wait(remaining)

    def close(self) -> None:
        if self._closed and not self._thread.is_alive():
            if self._error is not None:
                raise RuntimeError(f"Camera reader failed: {self._error}") from self._error
            return
        self._stop_requested.set()
        self._thread.join(LATEST_FRAME_CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("Camera reader thread did not stop cleanly.")
        if self._error is not None:
            raise RuntimeError(f"Camera reader failed: {self._error}") from self._error

    def release(self) -> None:
        self.close()

    def _reader_loop(self) -> None:
        sequence = 0
        try:
            while not self._stop_requested.is_set():
                frame_ok, frame = self._capture.read()
                read_completed_at = time.monotonic()
                if not frame_ok or frame is None:
                    if not self._stop_requested.is_set():
                        raise RuntimeError("Could not read a camera frame.")
                    break
                sequence += 1
                with self._condition:
                    self._latest = CameraFrame(frame, sequence, read_completed_at)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            try:
                self._capture.release()
            except Exception as exc:
                with self._condition:
                    if self._error is None:
                        self._error = exc
            with self._condition:
                self._closed = True
                self._condition.notify_all()


class LatestFrameDisplay:
    """Present only the newest submitted image from one HighGUI owner thread."""

    def __init__(self, window_name: str = WINDOW_NAME) -> None:
        self._window_name = window_name
        self._condition = threading.Condition()
        self._stop_requested = threading.Event()
        self._latest: np.ndarray | None = None
        self._keys: queue.SimpleQueue[int] = queue.SimpleQueue()
        self._error: Exception | None = None
        self._exit_requested = False
        self._closed = False
        self._dropped_frames = 0
        self._presented_frames = 0
        self._first_presented_at: float | None = None
        self._last_presented_at: float | None = None
        self._thread = threading.Thread(
            target=self._display_loop,
            name="latest-frame-highgui",
            daemon=True,
        )
        self._thread.start()

    @property
    def dropped_frames(self) -> int:
        with self._condition:
            return self._dropped_frames

    def submit(self, frame: np.ndarray) -> bool:
        """Overwrite any not-yet-presented image without blocking control."""

        if not isinstance(frame, np.ndarray):
            raise TypeError("Display frame must be a NumPy array.")
        with self._condition:
            self._raise_if_failed_locked()
            if self._exit_requested or self._closed:
                return False
            if self._latest is not None:
                self._dropped_frames += 1
            self._latest = frame
            self._condition.notify_all()
            return True

    def check_status(self) -> bool:
        """Raise a UI failure or report an operator/window close request."""

        with self._condition:
            self._raise_if_failed_locked()
            return self._exit_requested

    def poll_key(self) -> int | None:
        try:
            return self._keys.get_nowait()
        except queue.Empty:
            return None

    def completed_statistics(self) -> tuple[int, float]:
        """Return completed display intervals and their wall-clock span."""

        with self._condition:
            if (
                self._presented_frames <= 1
                or self._first_presented_at is None
                or self._last_presented_at is None
            ):
                return 0, 0.0
            return (
                self._presented_frames - 1,
                max(0.0, self._last_presented_at - self._first_presented_at),
            )

    def close(self) -> None:
        self._stop_requested.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(LATEST_DISPLAY_CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("HighGUI display thread did not stop cleanly.")
        with self._condition:
            self._raise_if_failed_locked()

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError(f"HighGUI display failed: {self._error}") from self._error

    def _display_loop(self) -> None:
        shown_once = False
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            while not self._stop_requested.is_set():
                with self._condition:
                    if self._latest is None:
                        self._condition.wait(DISPLAY_EVENT_POLL_SECONDS)
                    frame = self._latest
                    self._latest = None
                if frame is not None:
                    cv2.imshow(self._window_name, frame)
                    shown_once = True
                key = cv2.waitKey(1) & 0xFF
                completed_at = time.perf_counter()
                if frame is not None:
                    with self._condition:
                        self._presented_frames += 1
                        if self._first_presented_at is None:
                            self._first_presented_at = completed_at
                        self._last_presented_at = completed_at
                if key in (ord("q"), 27):
                    with self._condition:
                        self._exit_requested = True
                        self._condition.notify_all()
                    break
                if key not in (0, 255):
                    self._keys.put(key)
                if shown_once and cv2.getWindowProperty(
                    self._window_name,
                    cv2.WND_PROP_VISIBLE,
                ) == 0.0:
                    with self._condition:
                        self._exit_requested = True
                        self._condition.notify_all()
                    break
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception as exc:
                with self._condition:
                    if self._error is None:
                        self._error = exc
            with self._condition:
                self._closed = True
                self._condition.notify_all()


class DisplayRateLimiter:
    """Allow expensive display rendering no faster than a fixed frame rate."""

    def __init__(self, frame_rate: float) -> None:
        if frame_rate <= 0.0:
            raise ValueError("Display frame rate must be positive.")
        self._interval_seconds = 1.0 / frame_rate
        self._next_render_at: float | None = None

    def should_render(self, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        if self._next_render_at is not None and current < self._next_render_at:
            return False
        self._next_render_at = current + self._interval_seconds
        return True


def open_latest_frame_camera(camera_source: int | str) -> LatestFrameCamera:
    """Configure VideoCapture, then transfer all reads/releases to one thread."""

    capture = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    try:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        capture.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        capture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not capture.isOpened():
            raise RuntimeError(f"Could not open camera source {camera_source!r}.")
    except Exception:
        capture.release()
        raise
    return LatestFrameCamera(capture)


def validate_distinct_camera_devices(side_device: str, top_device: str) -> None:
    """Reject missing or aliased dual-camera device paths before UART is opened."""

    side = Path(side_device).expanduser()
    top = Path(top_device).expanduser()
    if not side.exists():
        raise ValueError(f"Side camera device was not found: {side}")
    if not top.exists():
        raise ValueError(f"Top camera device was not found: {top}")
    try:
        side_resolved = side.resolve(strict=True)
        top_resolved = top.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Could not resolve dual-camera device paths: {exc}") from exc
    if side_resolved == top_resolved:
        raise ValueError(
            "Side and top camera device paths resolve to the same video node: "
            f"{side_resolved}"
        )


def open_latest_frame_display() -> LatestFrameDisplay:
    return LatestFrameDisplay()


def graphical_display_environment_available(
    environment: dict[str, str] | None = None,
) -> bool:
    """Return whether this process has an X11 or Wayland display target."""

    values = os.environ if environment is None else environment
    return any(
        str(values.get(name, "")).strip()
        for name in ("DISPLAY", "WAYLAND_DISPLAY")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B0441 camera rust-detection preview")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument(
        "--side-camera-device",
        help=(
            "stable V4L2 by-path for the existing side camera; when provided, "
            "enables explicit dual-camera role binding"
        ),
    )
    parser.add_argument(
        "--top-camera-device",
        help="stable V4L2 by-path for the new top camera",
    )
    parser.add_argument(
        "--obstacle-engine",
        type=Path,
        help=(
            "FP32 TensorRT 10.3 YOLO obstacle plan for the fixed 1280x240 "
            f"top-camera y=0:240 ROI (recommended: {DEFAULT_OBSTACLE_ENGINE})"
        ),
    )
    parser.add_argument(
        "--obstacle-engine-sha256",
        help="required approved SHA-256 for the top-camera obstacle plan",
    )
    parser.add_argument(
        "--obstacle-confidence-threshold",
        type=float,
        default=OBSTACLE_DEFAULT_CONFIDENCE_THRESHOLD,
        help="top-camera obstacle confidence threshold (default: 0.30)",
    )
    parser.add_argument(
        "--student-engine",
        "--engine",
        dest="student_engine",
        type=Path,
        default=None,
        help=(
            "realtime student TensorRT plan "
            f"(default: {DEFAULT_STUDENT_ENGINE})"
        ),
    )
    parser.add_argument(
        "--student-engine-sha256",
        "--engine-sha256",
        dest="student_engine_sha256",
        help=(
            "approved student engine digest; required whenever UART is enabled; "
            "with --realtime-multitask-engine, selects the separate student "
            "for realtime rust while multitask supplies crack"
        ),
    )
    parser.add_argument(
        "--optimized-student-engine",
        type=Path,
        help=(
            "realtime student TensorRT plan with GPU rust argmax/count/finite "
            "outputs; mutually exclusive with --student-engine"
        ),
    )
    parser.add_argument(
        "--optimized-student-engine-sha256",
        help="approved digest for the GPU-postprocessed student rust plan",
    )
    parser.add_argument(
        "--teacher-engine",
        type=Path,
        default=DEFAULT_TEACHER_ENGINE,
        help=(
            "capture teacher TensorRT plan "
            f"(default: {DEFAULT_TEACHER_ENGINE})"
        ),
    )
    parser.add_argument(
        "--teacher-engine-sha256",
        help="approved teacher engine digest; required whenever UART is enabled",
    )
    parser.add_argument(
        "--capture-crack-engine",
        type=Path,
        help=(
            "LEGACY/UNSUPPORTED: capture BGCrack was replaced by "
            "--capture-hrsegnet-crack-engine"
        ),
    )
    parser.add_argument(
        "--capture-crack-engine-sha256",
        help="LEGACY/UNSUPPORTED capture BGCrack digest",
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-engine",
        type=Path,
        help=(
            "original HrSegNet-B32 TensorRT 10.3 plan for the native "
            "1280x720 capture frame; no resize or padding; approved path: "
            f"{DEFAULT_CAPTURE_HRSEGNET_CRACK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-engine-sha256",
        help="required approved SHA-256 for the capture HrSegNet-B32 plan",
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-probability-threshold",
        type=float,
        help=(
            "capture crack probability threshold converted to the equivalent "
            "class-1 minus class-0 logit margin (default: 0.55)"
        ),
    )
    parser.add_argument(
        "--capture-hrsegnet-crack-min-component-pixels",
        type=int,
        help="capture HrSegNet 8-connected component minimum (default: 20 pixels)",
    )
    parser.add_argument(
        "--realtime-crack-engine",
        type=Path,
        help=(
            "realtime BGCrack TensorRT plan for the fixed 1280x128 ROI; "
            f"recommended deployment path: {DEFAULT_REALTIME_CRACK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--realtime-crack-engine-sha256",
        help=(
            "approved realtime crack engine digest; required whenever UART is enabled"
        ),
    )
    parser.add_argument(
        "--realtime-multitask-engine",
        type=Path,
        help=(
            "optional optimized realtime rust+crack TensorRT plan with GPU "
            "postprocessed uint8 maps/scalars; a raw 5-channel plan is rejected; "
            f"recommended path: {DEFAULT_REALTIME_MULTITASK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--realtime-multitask-engine-sha256",
        help="required approved digest for the optimized realtime multitask plan",
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-engine",
        type=Path,
        help=(
            "HrSegNet-B32 TensorRT 10.3 plan for the fixed 1280x128 "
            "realtime crack ROI; this never replaces the capture crack model; "
            f"approved deployment path: {DEFAULT_REALTIME_HRSEGNET_CRACK_ENGINE}"
        ),
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-engine-sha256",
        help="required approved SHA-256 for the realtime HrSegNet-B32 plan",
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-probability-threshold",
        type=float,
        help=(
            "approved realtime crack probability threshold; converted to an "
            "equivalent class-1 minus class-0 logit margin (default: 0.55)"
        ),
    )
    parser.add_argument(
        "--realtime-hrsegnet-crack-min-component-pixels",
        type=int,
        help="approved realtime 8-connected component minimum (default: 20 pixels)",
    )
    parser.add_argument(
        "--no-crack",
        action="store_true",
        help=(
            "TEST ONLY: run realtime cleaning without the crack detector or "
            "fixed y=112:240 crack safety interlock"
        ),
    )
    parser.add_argument(
        "--capture-crack-threshold",
        type=float,
        help="LEGACY/UNSUPPORTED capture BGCrack threshold",
    )
    parser.add_argument(
        "--realtime-crack-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help="realtime crack threshold calibrated on Validation (default: 0.5)",
    )
    parser.add_argument(
        "--capture-crack-min-component-pixels",
        type=int,
        help="LEGACY/UNSUPPORTED capture BGCrack component minimum",
    )
    parser.add_argument(
        "--realtime-crack-min-component-pixels",
        type=int,
        default=DEFAULT_MIN_COMPONENT_PIXELS,
        help="discard smaller realtime crack candidates (default: 20 pixels)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="new XLSX path; defaults to a unique timestamped report",
    )
    parser.add_argument(
        "--serial-port",
        default=DEFAULT_SERIAL_PORT,
        help="UART device (default: /dev/ttyACM0 for ST-LINK USB VCP)",
    )
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument(
        "--no-uart",
        action="store_true",
        help="disable the UART trigger for manual capture testing",
    )
    parser.add_argument(
        "--realtime-test",
        action="store_true",
        help=(
            "TEST ONLY: run the realtime student and crack models from the "
            "camera with simulated actuator output by default; captures and "
            "reports remain disabled"
        ),
    )
    parser.add_argument(
        "--realtime-test-uart",
        action="store_true",
        help=(
            "TEST ONLY: with --realtime-test and two cameras, enter the STM "
            "actuator-test state and apply FRONT/SIDE cleaner and pump commands"
        ),
    )
    parser.add_argument(
        "--dual-timing",
        action="store_true",
        help=(
            "TEST ONLY: after a 10-second warmup, print 5-second dual-camera "
            "loop, inference, frame-age, stale, and ready-to-off statistics"
        ),
    )
    parser.add_argument(
        "--capture-test",
        action="store_true",
        help=(
            "TEST ONLY: preview the full camera frame and run only the "
            "capture rust and crack models when S is pressed"
        ),
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help=(
            "disable the camera preview and display-only overlays while keeping "
            "camera inference, UART/FSM control, automatic rail sections, and reports"
        ),
    )
    args = parser.parse_args()
    explicit_headless = args.headless
    args.headless_auto = (
        not explicit_headless and not graphical_display_environment_available()
    )
    args.headless = explicit_headless or args.headless_auto
    dual_camera_device_configured = (
        args.side_camera_device is not None or args.top_camera_device is not None
    )
    if dual_camera_device_configured and (
        args.side_camera_device is None or args.top_camera_device is None
    ):
        parser.error(
            "dual-camera operation requires both --side-camera-device and "
            "--top-camera-device"
        )
    args.dual_camera = bool(
        args.side_camera_device is not None and args.top_camera_device is not None
    )
    if args.realtime_test_uart and not args.realtime_test:
        parser.error("--realtime-test-uart requires --realtime-test")
    if args.realtime_test_uart and args.no_uart:
        parser.error("--realtime-test-uart cannot be combined with --no-uart")
    if args.realtime_test_uart and not args.dual_camera:
        parser.error(
            "--realtime-test-uart requires explicit side and top camera devices"
        )
    if args.dual_timing and (not args.realtime_test or not args.dual_camera):
        parser.error("--dual-timing requires --realtime-test with two cameras")
    obstacle_configured = (
        args.obstacle_engine is not None or args.obstacle_engine_sha256 is not None
    )
    if obstacle_configured and not args.dual_camera:
        parser.error(
            "--obstacle-engine options require explicit side and top camera devices"
        )
    if args.obstacle_engine is None and args.obstacle_engine_sha256 is not None:
        parser.error("--obstacle-engine-sha256 requires --obstacle-engine")
    if args.obstacle_engine is not None and args.obstacle_engine_sha256 is None:
        parser.error("--obstacle-engine requires --obstacle-engine-sha256")
    if args.dual_camera and not args.capture_test and args.obstacle_engine is None:
        parser.error(
            "dual-camera realtime operation requires --obstacle-engine and its SHA-256"
        )
    if args.dual_camera and args.no_crack:
        parser.error("dual-camera operation cannot bypass either crack detector")
    if args.capture_test and args.obstacle_engine is not None:
        parser.error(
            "--capture-test does not run the obstacle model; omit --obstacle-engine"
        )
    if not 0.0 < args.obstacle_confidence_threshold < 1.0:
        parser.error("--obstacle-confidence-threshold must be between zero and one")
    raw_student_configured = (
        args.student_engine is not None or args.student_engine_sha256 is not None
    )
    optimized_student_configured = (
        args.optimized_student_engine is not None
        or args.optimized_student_engine_sha256 is not None
    )
    if raw_student_configured and optimized_student_configured:
        parser.error(
            "raw --student-engine options cannot be combined with optimized "
            "student engine options"
        )
    if (
        args.optimized_student_engine is None
        and args.optimized_student_engine_sha256 is not None
    ):
        parser.error(
            "--optimized-student-engine-sha256 requires "
            "--optimized-student-engine"
        )
    if args.student_engine is None and args.student_engine_sha256 is not None:
        args.student_engine = DEFAULT_STUDENT_ENGINE
    if args.capture_test and args.realtime_test:
        parser.error("--capture-test and --realtime-test are mutually exclusive")
    if args.capture_test and args.headless:
        parser.error(
            "--capture-test requires a graphical display because S/Q/Esc are "
            "interactive; normal operation and --realtime-test support --headless"
        )
    if args.capture_test and args.no_crack:
        parser.error("--capture-test cannot be combined with --no-crack")
    legacy_capture_options = (
        args.capture_crack_engine,
        args.capture_crack_engine_sha256,
        args.capture_crack_threshold,
        args.capture_crack_min_component_pixels,
    )
    if any(value is not None for value in legacy_capture_options):
        parser.error(
            "capture BGCrack options are no longer supported; use the explicit "
            "--capture-hrsegnet-crack-* options"
        )
    capture_hrsegnet_options = (
        args.capture_hrsegnet_crack_engine,
        args.capture_hrsegnet_crack_engine_sha256,
        args.capture_hrsegnet_crack_probability_threshold,
        args.capture_hrsegnet_crack_min_component_pixels,
    )
    capture_hrsegnet_configured = any(
        value is not None for value in capture_hrsegnet_options
    )
    capture_hrsegnet_enabled = args.capture_hrsegnet_crack_engine is not None
    if capture_hrsegnet_configured and not capture_hrsegnet_enabled:
        parser.error(
            "capture HrSegNet threshold, component, and SHA options require "
            "--capture-hrsegnet-crack-engine"
        )
    if capture_hrsegnet_enabled and args.capture_hrsegnet_crack_engine_sha256 is None:
        parser.error(
            "--capture-hrsegnet-crack-engine requires its approved "
            "--capture-hrsegnet-crack-engine-sha256"
        )
    if args.capture_test and not capture_hrsegnet_enabled:
        parser.error("--capture-test requires --capture-hrsegnet-crack-engine")
    if capture_hrsegnet_enabled:
        if args.capture_hrsegnet_crack_probability_threshold is None:
            args.capture_hrsegnet_crack_probability_threshold = (
                HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD
            )
        if args.capture_hrsegnet_crack_min_component_pixels is None:
            args.capture_hrsegnet_crack_min_component_pixels = (
                HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS
            )
    else:
        args.capture_hrsegnet_crack_probability_threshold = (
            HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD
        )
        args.capture_hrsegnet_crack_min_component_pixels = (
            HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS
        )
    if args.realtime_test and args.no_crack:
        parser.error("--realtime-test cannot be combined with --no-crack")
    if args.no_crack and not args.no_uart:
        parser.error("--no-crack is test-only and requires --no-uart")
    multitask_enabled = args.realtime_multitask_engine is not None
    hrsegnet_options = (
        args.realtime_hrsegnet_crack_engine,
        args.realtime_hrsegnet_crack_engine_sha256,
        args.realtime_hrsegnet_crack_probability_threshold,
        args.realtime_hrsegnet_crack_min_component_pixels,
    )
    hrsegnet_configured = any(value is not None for value in hrsegnet_options)
    hrsegnet_enabled = args.realtime_hrsegnet_crack_engine is not None
    if hrsegnet_configured and not hrsegnet_enabled:
        parser.error(
            "HrSegNet threshold, component, and SHA options require "
            "--realtime-hrsegnet-crack-engine"
        )
    if hrsegnet_enabled and args.realtime_hrsegnet_crack_engine_sha256 is None:
        parser.error(
            "--realtime-hrsegnet-crack-engine requires its approved "
            "--realtime-hrsegnet-crack-engine-sha256"
        )
    if hrsegnet_enabled and (
        multitask_enabled
        or args.no_crack
        or args.realtime_crack_engine is not None
        or args.realtime_crack_engine_sha256 is not None
    ):
        parser.error(
            "--realtime-hrsegnet-crack-engine is mutually exclusive with "
            "--no-crack, BGCrack, and multitask realtime "
            "crack engine options"
        )
    if hrsegnet_enabled:
        if args.realtime_hrsegnet_crack_probability_threshold is None:
            args.realtime_hrsegnet_crack_probability_threshold = (
                HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD
            )
        if args.realtime_hrsegnet_crack_min_component_pixels is None:
            args.realtime_hrsegnet_crack_min_component_pixels = (
                HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS
            )
    if (
        not args.capture_test
        and not multitask_enabled
        and args.optimized_student_engine is None
        and args.student_engine is None
    ):
        args.student_engine = DEFAULT_STUDENT_ENGINE
    if (
        args.realtime_multitask_engine is None
        and args.realtime_multitask_engine_sha256 is not None
    ):
        parser.error(
            "--realtime-multitask-engine-sha256 requires "
            "--realtime-multitask-engine"
        )
    if (
        multitask_enabled
        and not args.capture_test
        and args.realtime_multitask_engine_sha256 is None
    ):
        parser.error(
            "--realtime-multitask-engine requires its approved "
            "--realtime-multitask-engine-sha256"
        )
    if (
        args.realtime_test
        and not multitask_enabled
        and not hrsegnet_enabled
        and args.realtime_crack_engine is None
    ):
        parser.error(
            "--realtime-test requires --realtime-crack-engine or "
            "--realtime-multitask-engine or --realtime-hrsegnet-crack-engine"
        )
    if not args.capture_test and multitask_enabled and (
        args.no_crack
        or args.realtime_crack_engine is not None
        or args.realtime_crack_engine_sha256 is not None
    ):
        parser.error(
            "--realtime-multitask-engine is mutually exclusive with --no-crack "
            "and separate realtime crack engine options"
        )
    if args.no_crack and (
        args.capture_hrsegnet_crack_engine is not None
        or args.capture_hrsegnet_crack_engine_sha256 is not None
        or args.realtime_crack_engine is not None
        or args.realtime_crack_engine_sha256 is not None
    ):
        parser.error(
            "--no-crack cannot be combined with capture/realtime crack engines"
        )
    if args.realtime_crack_engine is None and args.realtime_crack_engine_sha256 is not None:
        parser.error(
            "--realtime-crack-engine-sha256 requires --realtime-crack-engine"
        )
    if (
        not args.capture_test
        and multitask_enabled
        and not args.realtime_test
        and not capture_hrsegnet_enabled
    ):
        parser.error(
            "normal multitask operation still requires "
            "--capture-hrsegnet-crack-engine "
            "for the independent capture/report path"
        )
    if (
        not args.capture_test
        and not multitask_enabled
        and not args.realtime_test
        and not args.no_crack
        and (
            not capture_hrsegnet_enabled
            or (args.realtime_crack_engine is None) == (not hrsegnet_enabled)
        )
    ):
        parser.error(
            "normal operation requires the independent "
            "--capture-hrsegnet-crack-engine "
            "and exactly one realtime crack engine: BGCrack or HrSegNet"
        )
    if not 0.0 < args.realtime_crack_threshold < 1.0:
        parser.error("--realtime-crack-threshold must be between zero and one")
    if args.realtime_crack_min_component_pixels <= 0:
        parser.error("--realtime-crack-min-component-pixels must be positive")
    if capture_hrsegnet_enabled:
        if not 0.0 < args.capture_hrsegnet_crack_probability_threshold < 1.0:
            parser.error(
                "--capture-hrsegnet-crack-probability-threshold must be "
                "between zero and one"
            )
        if args.capture_hrsegnet_crack_min_component_pixels <= 0:
            parser.error(
                "--capture-hrsegnet-crack-min-component-pixels must be positive"
            )
    if hrsegnet_enabled:
        if not 0.0 < args.realtime_hrsegnet_crack_probability_threshold < 1.0:
            parser.error(
                "--realtime-hrsegnet-crack-probability-threshold must be "
                "between zero and one"
            )
        if args.realtime_hrsegnet_crack_min_component_pixels <= 0:
            parser.error(
                "--realtime-hrsegnet-crack-min-component-pixels must be positive"
            )
    if args.report is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        args.report = DEFAULT_REPORT_DIRECTORY / f"rust_inspection_{timestamp}.xlsx"
    return args


def read_uart_messages(
    uart: serial.Serial, pending: bytearray
) -> list[str]:
    waiting = min(uart.in_waiting, 1024)
    if waiting > 0:
        pending.extend(uart.read(waiting))

    messages: list[str] = []
    while b"\n" in pending:
        raw_line, _, remainder = pending.partition(b"\n")
        pending[:] = remainder
        message = raw_line.decode("ascii", errors="replace").strip()
        if message:
            messages.append(message)
    return messages


def send_uart_test_command(uart, command: bytes) -> None:
    """Write one complete actuator-test protocol line or fail closed."""

    command_name = command.decode("ascii").strip()
    try:
        written = uart.write(command)
    except (serial.SerialException, OSError) as exc:
        raise RuntimeError(
            f"Could not send {command_name} to the STM: {exc}"
        ) from exc
    if written != len(command):
        raise RuntimeError(
            f"Could not send the complete {command_name} command "
            f"({written}/{len(command)} bytes)."
        )
    print(f"UART TX: {command_name}")


def enter_uart_actuator_test(
    uart,
    timeout_seconds: float = ACTUATOR_TEST_READY_TIMEOUT_SECONDS,
) -> None:
    """Enter the STM test state before any actuator ON/PWM command is allowed."""

    if timeout_seconds <= 0.0:
        raise ValueError("Actuator-test READY timeout must be positive.")
    try:
        uart.reset_input_buffer()
    except (AttributeError, serial.SerialException, OSError) as exc:
        raise RuntimeError(f"Could not clear stale STM UART input: {exc}") from exc

    send_uart_test_command(uart, ACTUATOR_TEST_START_COMMAND)
    pending = bytearray()
    deadline = time.monotonic() + timeout_seconds
    while True:
        try:
            messages = read_uart_messages(uart, pending)
        except (serial.SerialException, OSError) as exc:
            raise RuntimeError(
                f"Could not read {ACTUATOR_TEST_READY_TRIGGER} from the STM: {exc}"
            ) from exc
        for message in messages:
            print(f"UART RX: {message}")
            if message == ACTUATOR_TEST_READY_TRIGGER:
                return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"STM did not send {ACTUATOR_TEST_READY_TRIGGER} within "
                f"{timeout_seconds:g} seconds; actuator commands were not enabled."
            )
        time.sleep(ACTUATOR_TEST_READY_POLL_SECONDS)


class CleanerController:
    """Command one of the two approved cleaner PWM setpoints or safe OFF."""

    def __init__(
        self,
        uart,
        commands: ActuatorCommandSet = LEGACY_ACTUATOR_COMMANDS,
    ) -> None:
        self.uart = uart
        self.commands = commands
        self.is_on = False
        self.duty_percent = 0.0
        self._last_on_sent: float | None = None

    def _send(self, command: bytes) -> None:
        if self.uart is None:
            return
        try:
            written = self.uart.write(command)
        except (serial.SerialException, OSError) as exc:
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send {command_name} to the STM: {exc}"
            ) from exc
        if written != len(command):
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send the complete {command_name} command "
                f"({written}/{len(command)} bytes)."
            )

    def force_off(self) -> None:
        """Reset debounce state and immediately request a safe cleaner stop."""

        self.is_on = False
        self.duty_percent = 0.0
        self._last_on_sent = None
        self._send(self.commands.cleaner_off)

    def force_on(self) -> None:
        """Compatibility helper: start at the approved 33.3% baseline."""

        self.set_duty_percent(CLEANER_BASE_DUTY_PERCENT)

    def set_duty_percent(self, duty_percent: float) -> None:
        """Apply OFF, 33.3%, or 55.6%; active commands also act as heartbeats."""

        duty_percent = float(duty_percent)
        if duty_percent == 0.0:
            if self.is_on:
                self.force_off()
            return
        if duty_percent == CLEANER_BASE_DUTY_PERCENT:
            command = self.commands.cleaner_pwm_33_3
        elif duty_percent == CLEANER_STAGE_ONE_DUTY_PERCENT:
            command = self.commands.cleaner_pwm_55_6
        else:
            raise ValueError(
                "Cleaner duty must be exactly 0, 33.3, or 55.6 percent."
            )

        now = time.monotonic()
        if not self.is_on or duty_percent != self.duty_percent:
            self._send(command)
            self.is_on = True
            self.duty_percent = duty_percent
            self._last_on_sent = now
        elif (
            self._last_on_sent is not None
            and now - self._last_on_sent >= CLEANER_HEARTBEAT_SECONDS
        ):
            self._send(command)
            self._last_on_sent = now


class PumpController:
    """Control water immediately; temporal stability is handled upstream."""

    def __init__(
        self,
        uart,
        commands: ActuatorCommandSet = LEGACY_ACTUATOR_COMMANDS,
    ) -> None:
        self.uart = uart
        self.commands = commands
        self.is_on = False
        self._last_on_sent: float | None = None

    def _send(self, command: bytes) -> None:
        if self.uart is None:
            return
        try:
            written = self.uart.write(command)
        except (serial.SerialException, OSError) as exc:
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send {command_name} to the STM: {exc}"
            ) from exc
        if written != len(command):
            command_name = command.decode("ascii").strip()
            raise RuntimeError(
                f"Could not send the complete {command_name} command "
                f"({written}/{len(command)} bytes)."
            )

    def force_off(self) -> None:
        """Stop water immediately and arm an immediate first clean observation."""

        self.is_on = False
        self._last_on_sent = None
        self._send(self.commands.pump_off)

    def update(self, water_allowed: bool) -> None:
        now = time.monotonic()
        requested_state = bool(water_allowed)

        if requested_state != self.is_on:
            command = (
                self.commands.pump_on
                if requested_state
                else self.commands.pump_off
            )
            self._send(command)
            self.is_on = requested_state
            self._last_on_sent = now if requested_state else None

        if (
            self.is_on
            and self._last_on_sent is not None
            and now - self._last_on_sent >= PUMP_HEARTBEAT_SECONDS
        ):
            self._send(self.commands.pump_on)
            self._last_on_sent = now


@dataclass(frozen=True)
class _ScheduledActuatorDecision:
    decision: RealtimeControlDecision | FrontControlDecision
    valid_until: float


@dataclass(frozen=True)
class RealtimeActuatorHoldSnapshot:
    """Read-only hold state captured before a waiting-path update."""

    captured_at: float
    has_decision: bool
    loss_deadline_was_set: bool
    valid_until: float | None
    watchdog_expired: bool
    loss_started_at: float | None


class RealtimeActuatorDecisionHold:
    """Hold the last validated command after control first loses readiness."""

    def __init__(
        self,
        hold_seconds: float = REALTIME_LAST_VALID_COMMAND_HOLD_SECONDS,
    ) -> None:
        if hold_seconds <= 0.0:
            raise ValueError("Realtime actuator hold must be positive.")
        self.hold_seconds = float(hold_seconds)
        self.clear()

    def clear(self) -> None:
        self._decision = None
        self._watchdog_deadline: float | None = None
        self._loss_deadline: float | None = None

    def accept(
        self,
        decision: RealtimeControlDecision | FrontControlDecision,
        *,
        now: float | None = None,
    ) -> None:
        if now is None:
            now = time.monotonic()
        self._decision = decision
        self._watchdog_deadline = float(now) + self.hold_seconds
        self._loss_deadline = None

    @property
    def valid_until(self) -> float | None:
        if self._decision is None:
            return None
        return (
            self._loss_deadline
            if self._loss_deadline is not None
            else self._watchdog_deadline
        )

    @property
    def has_decision(self) -> bool:
        return self._decision is not None

    @property
    def loss_started_at(self) -> float | None:
        if self._loss_deadline is None:
            return None
        return self._loss_deadline - self.hold_seconds

    def diagnostic_snapshot(
        self,
        *,
        now: float | None = None,
    ) -> RealtimeActuatorHoldSnapshot:
        if now is None:
            now = time.monotonic()
        captured_at = float(now)
        loss_deadline_was_set = self._loss_deadline is not None
        valid_until = self.valid_until
        watchdog_expired = bool(
            self._decision is not None
            and not loss_deadline_was_set
            and self._watchdog_deadline is not None
            and captured_at > self._watchdog_deadline
        )
        return RealtimeActuatorHoldSnapshot(
            captured_at=captured_at,
            has_decision=self._decision is not None,
            loss_deadline_was_set=loss_deadline_was_set,
            valid_until=valid_until,
            watchdog_expired=watchdog_expired,
            loss_started_at=self.loss_started_at,
        )

    def begin_loss(
        self,
        *,
        now: float | None = None,
    ) -> RealtimeControlDecision | FrontControlDecision | None:
        if self._decision is None:
            return None
        if now is None:
            now = time.monotonic()
        if self._loss_deadline is None:
            if (
                self._watchdog_deadline is None
                or float(now) > self._watchdog_deadline
            ):
                self.clear()
                return None
            self._loss_deadline = float(now) + self.hold_seconds
        return self.current(now=now)

    def current(
        self,
        *,
        now: float | None = None,
    ) -> RealtimeControlDecision | FrontControlDecision | None:
        if self._decision is None:
            return None
        if now is None:
            now = time.monotonic()
        valid_until = self.valid_until
        if valid_until is None or float(now) > valid_until:
            self.clear()
            return None
        return self._decision


class RealtimeActuatorArbiter:
    """Own all realtime UART writes and heartbeat FRONT/SIDE independently."""

    _ROLES = ("front", "side")

    def __init__(
        self,
        uart,
        *,
        heartbeat_seconds: float = REALTIME_ACTUATOR_HEARTBEAT_SECONDS,
    ) -> None:
        if uart is None:
            raise ValueError("Realtime actuator arbiter requires an open UART.")
        if heartbeat_seconds <= 0.0:
            raise ValueError("Realtime actuator heartbeat must be positive.")
        self.heartbeat_seconds = float(heartbeat_seconds)
        self._controllers = {
            "front": (
                CleanerController(uart, FRONT_ACTUATOR_COMMANDS),
                PumpController(uart, FRONT_ACTUATOR_COMMANDS),
            ),
            "side": (
                CleanerController(uart, SIDE_ACTUATOR_COMMANDS),
                PumpController(uart, SIDE_ACTUATOR_COMMANDS),
            ),
        }
        self._targets: dict[str, _ScheduledActuatorDecision | None] = {
            role: None for role in self._ROLES
        }
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failure: str | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Realtime actuator arbiter was already started.")
        self._thread = threading.Thread(
            target=self._run,
            name="realtime-uart-arbiter",
            daemon=True,
        )
        self._thread.start()

    def publish(
        self,
        role: str,
        decision: RealtimeControlDecision | FrontControlDecision,
        *,
        valid_until: float,
    ) -> None:
        if role not in self._targets:
            raise ValueError(f"Unknown realtime actuator role: {role}")
        if not isinstance(valid_until, (int, float)):
            raise ValueError("Realtime actuator valid_until must be numeric.")
        with self._lock:
            self._targets[role] = _ScheduledActuatorDecision(
                decision,
                float(valid_until),
            )
        self._wake.set()

    def clear(self, role: str) -> None:
        if role not in self._targets:
            raise ValueError(f"Unknown realtime actuator role: {role}")
        with self._lock:
            self._targets[role] = None
        self._wake.set()

    def clear_all(self) -> None:
        with self._lock:
            for role in self._ROLES:
                self._targets[role] = None
        self._wake.set()

    def desired_state(self) -> tuple[float, bool, float, bool]:
        with self._lock:
            front = self._targets["front"]
            side = self._targets["side"]
        now = time.monotonic()

        def state(target: _ScheduledActuatorDecision | None) -> tuple[float, bool]:
            if target is None or now > target.valid_until:
                return 0.0, False
            return (
                target.decision.cleaner_duty_percent,
                target.decision.pump_on,
            )

        front_state = state(front)
        side_state = state(side)
        return (*front_state, *side_state)

    def failure_message(self) -> str | None:
        with self._lock:
            return self._failure

    def raise_if_failed(self) -> None:
        failure = self.failure_message()
        if failure is not None:
            raise RuntimeError(failure)

    def worker_is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        if self._thread is None:
            return
        self.clear_all()
        self._stop.set()
        self._wake.set()
        self._thread.join(REALTIME_ACTUATOR_WORKER_CLOSE_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            raise RuntimeError("Realtime actuator arbiter did not stop safely.")
        self._thread = None
        self.raise_if_failed()

    def _safe_off_all(self) -> None:
        force_cleaning_pairs_safe_off(
            self._controllers["front"],
            self._controllers["side"],
        )

    def _run(self) -> None:
        try:
            self._safe_off_all()
            while not self._stop.is_set():
                now = time.monotonic()
                with self._lock:
                    targets = dict(self._targets)
                for role in self._ROLES:
                    cleaner, pump = self._controllers[role]
                    target = targets[role]
                    if target is None or now > target.valid_until:
                        force_cleaning_safe_off(cleaner, pump)
                    else:
                        update_cleaning_actuators(
                            cleaner,
                            pump,
                            decision=target.decision,
                        )
                self._wake.wait(self.heartbeat_seconds)
                self._wake.clear()
        except Exception as exc:
            with self._lock:
                self._failure = f"Realtime UART arbiter failed: {exc}"
            self._stop.set()
        finally:
            try:
                self._safe_off_all()
            except Exception as exc:
                with self._lock:
                    if self._failure is None:
                        self._failure = (
                            f"Realtime UART arbiter safe-OFF failed: {exc}"
                        )


def read_terminal_command() -> tuple[str | None, bool]:
    try:
        readable, _, _ = select.select([sys.stdin], [], [], 0)
    except (OSError, ValueError):
        return None, False
    if not readable:
        return None, True

    line = sys.stdin.readline()
    if not line:
        return None, False
    command = line.strip().lower()
    return command or None, True


def water_control_roi_bounds(image: np.ndarray) -> tuple[int, int]:
    """Return the top full-width strip used for rust inference and water control."""

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Water control requires a two- or three-dimensional image.")
    if image.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            "Realtime control requires the original "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} camera frame; received "
            f"{image.shape[1]}x{image.shape[0]}."
        )
    return REALTIME_RUST_ROI_TOP, REALTIME_RUST_ROI_BOTTOM


def highest_rust_grade(class_map) -> int:
    """Return the highest valid rust class present in the realtime ROI."""

    if not isinstance(class_map, np.ndarray) or class_map.ndim != 2:
        raise ValueError("Realtime rust class map must be a two-dimensional array.")
    expected_shape = (
        REALTIME_RUST_ROI_BOTTOM - REALTIME_RUST_ROI_TOP,
        FRAME_WIDTH,
    )
    if class_map.shape != expected_shape:
        raise ValueError("Realtime rust class map shape does not match the control ROI.")
    if not np.isin(class_map, RUST_CLASS_IDS).all():
        raise ValueError("Realtime rust class map contains an invalid class ID.")
    return int(np.max(class_map))


def water_control_blocked_by_rust(class_map) -> bool:
    """Fail closed; every non-zero rust grade blocks the water pump."""

    try:
        return highest_rust_grade(class_map) >= 1
    except ValueError:
        return True


def crack_stop_roi_bounds(image: np.ndarray) -> tuple[int, int]:
    """Return the full-width strip ending at the rust ROI's lower boundary."""

    if not isinstance(image, np.ndarray) or image.ndim not in (2, 3):
        raise ValueError("Crack stop control requires a two- or three-dimensional image.")
    if image.shape[:2] != (FRAME_HEIGHT, FRAME_WIDTH):
        raise ValueError(
            "Realtime control requires the original "
            f"{FRAME_WIDTH}x{FRAME_HEIGHT} camera frame; received "
            f"{image.shape[1]}x{image.shape[0]}."
        )
    return REALTIME_CRACK_ROI_TOP, REALTIME_CRACK_ROI_BOTTOM


def validated_crack_ratio(crack_result) -> float:
    """Validate a realtime crack result and return its filtered ROI ratio."""

    if crack_result is None or getattr(crack_result, "status", None) != "ready":
        raise ValueError("Realtime crack result is missing or not ready.")
    crack_method = str(getattr(crack_result, "method", ""))
    if not crack_method.startswith(
        (
            REALTIME_CRACK_DETECTOR_PREFIX,
            MULTITASK_CRACK_METHOD_PREFIX,
            HRSEGNET_CRACK_DETECTOR_PREFIX,
        )
    ):
        raise ValueError("Realtime crack result method is not approved.")
    mask = getattr(crack_result, "mask", None)
    expected_shape = (
        REALTIME_CRACK_ROI_BOTTOM - REALTIME_CRACK_ROI_TOP,
        FRAME_WIDTH,
    )
    if (
        not isinstance(mask, np.ndarray)
        or mask.ndim != 2
        or mask.shape != expected_shape
    ):
        raise ValueError("Realtime crack mask shape does not match the control ROI.")
    if not np.isin(mask, (0, 255)).all():
        raise ValueError("Realtime crack mask must contain only zero and 255.")
    ratio = float(getattr(crack_result, "crack_ratio", float("nan")))
    measured_ratio = float(np.count_nonzero(mask)) / float(mask.size)
    if not np.isfinite(ratio) or abs(ratio - measured_ratio) > 1e-9:
        raise ValueError("Realtime crack ratio does not match its filtered mask.")
    return ratio


def cleaning_blocked_by_crack(crack_result) -> bool:
    """Return whether the validated crack ROI exceeds the 0.05-percent limit."""

    try:
        return validated_crack_ratio(crack_result) > CRACK_CONTROL_STOP_RATIO
    except ValueError:
        return True


def obstacle_detected_in_control_roi(result: ObstacleDetectionResult) -> bool:
    """Return the GPU-validated top-camera y=0:240 obstacle decision."""

    if not isinstance(result, ObstacleDetectionResult) or result.status != "ready":
        raise ValueError("Realtime obstacle result is missing or not ready.")
    if not str(result.method).startswith(OBSTACLE_DETECTOR_PREFIX):
        raise ValueError("Realtime obstacle result method is not approved.")
    if type(result.control_roi_detected) is not bool:
        raise ValueError("Realtime obstacle control ROI flag must be boolean.")
    return result.control_roi_detected


def extract_realtime_control_rois(frame: np.ndarray):
    """Crop fixed full-width rust and crack inputs from the original frame."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("Realtime camera frame must be an HxWx3 image.")
    water_top, water_bottom = water_control_roi_bounds(frame)
    crack_top, crack_bottom = crack_stop_roi_bounds(frame)
    return (
        frame[water_top:water_bottom, :],
        frame[crack_top:crack_bottom, :],
    )


def annotate_realtime_control_results(
    frame: np.ndarray,
    rust_result,
    crack_result=None,
    *,
    copy_frame: bool = True,
) -> np.ndarray:
    """Map cropped realtime results back onto original-frame coordinates."""

    water_top, water_bottom = water_control_roi_bounds(frame)
    crack_top, crack_bottom = crack_stop_roi_bounds(frame)
    full_shape = frame.shape[:2]
    water_shape = (water_bottom - water_top, full_shape[1])
    crack_shape = (crack_bottom - crack_top, full_shape[1])

    output = frame.copy() if copy_frame else frame
    if rust_result is not None:
        rust_mask = getattr(rust_result, "mask", None)
        if not isinstance(rust_mask, np.ndarray) or rust_mask.shape != water_shape:
            raise ValueError(
                "Realtime rust mask shape does not match the water-control ROI."
            )
        rust_class_map = getattr(rust_result, "class_map", None)
        rust_overlay = output.copy()
        rust_overlay_roi = rust_overlay[water_top:water_bottom, :]
        if rust_class_map is not None:
            if (
                not isinstance(rust_class_map, np.ndarray)
                or rust_class_map.shape != water_shape
            ):
                raise ValueError(
                    "Realtime rust class-map shape does not match the "
                    "water-control ROI."
                )
            for class_id, color in CLASS_COLORS_BGR.items():
                rust_overlay_roi[rust_class_map == class_id] = color
        else:
            rust_overlay_roi[rust_mask > 0] = (0, 95, 255)
        output = cv2.addWeighted(rust_overlay, 0.35, output, 0.65, 0)
        rust_box_color = (60, 180, 75)
        for class_name, class_id in (("Fair", 1), ("Poor", 2), ("Severe", 3)):
            if rust_result.class_ratios.get(class_name, 0.0) > 0.0:
                rust_box_color = CLASS_COLORS_BGR[class_id]
        for x, y, width, height in rust_result.boxes:
            cv2.rectangle(
                output,
                (x, y + water_top),
                (x + width, y + water_top + height),
                rust_box_color,
                2,
            )

    if crack_result is None:
        return output
    crack_mask = getattr(crack_result, "mask", None)
    if not isinstance(crack_mask, np.ndarray) or crack_mask.shape != crack_shape:
        raise ValueError(
            "Realtime crack mask shape does not match the crack-stop ROI."
        )
    crack_overlay = output.copy()
    crack_overlay[crack_top:crack_bottom, :][crack_mask > 0] = CRACK_COLOR_BGR
    output = cv2.addWeighted(crack_overlay, 0.4, output, 0.6, 0)
    crack_box_color = (0, 0, 255) if crack_result.detected else (60, 180, 75)
    for x, y, width, height in crack_result.boxes:
        cv2.rectangle(
            output,
            (x, y + crack_top),
            (x + width, y + crack_top + height),
            crack_box_color,
            2,
        )
    return output


def annotate_top_realtime_results(
    frame: np.ndarray,
    obstacle_result: ObstacleDetectionResult | None,
    crack_result,
) -> np.ndarray:
    """Draw obstacle boxes first so an overlapping crack remains visible."""

    output = frame.copy()
    if obstacle_result is not None:
        for detection in obstacle_result.detections:
            x1, y1, x2, y2 = (
                int(round(value)) for value in detection.box_xyxy
            )
            cv2.rectangle(
                output,
                (x1, y1),
                (x2, y2),
                (0, 220, 255),
                2,
            )
    return annotate_realtime_control_results(
        output,
        None,
        crack_result,
        copy_frame=False,
    )


def update_cleaning_actuators(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    decision: RealtimeControlDecision | FrontControlDecision,
) -> None:
    """Apply an already validated four-result temporal control decision."""

    cleaner_controller.set_duty_percent(decision.cleaner_duty_percent)
    pump_controller.update(decision.pump_on)


def publish_realtime_actuator_decision(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    role: str,
    history: RealtimeControlHistory | FrontControlHistory,
    decision: RealtimeControlDecision | FrontControlDecision,
    hold: RealtimeActuatorDecisionHold | None = None,
    arbiter: RealtimeActuatorArbiter | None = None,
) -> None:
    """Publish one role atomically, or preserve simulated/direct behavior."""

    if hold is not None:
        hold.accept(decision)
    if arbiter is None:
        update_cleaning_actuators(
            cleaner_controller,
            pump_controller,
            decision=decision,
        )
        return
    valid_until = hold.valid_until if hold is not None else history.valid_until
    if valid_until is None:
        arbiter.clear(role)
        return
    arbiter.publish(role, decision, valid_until=valid_until)


def clear_realtime_actuator_role(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    role: str,
    arbiter: RealtimeActuatorArbiter | None = None,
) -> None:
    if arbiter is None:
        force_cleaning_safe_off(cleaner_controller, pump_controller)
    else:
        arbiter.clear(role)


def maintain_waiting_role_control(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    *,
    history: RealtimeControlHistory | FrontControlHistory,
    inference: AlternatingRealtimeInference | AlternatingTopRealtimeInference,
    role: str = "side",
    hold: RealtimeActuatorDecisionHold | None = None,
    arbiter: RealtimeActuatorArbiter | None = None,
) -> bool:
    """Heartbeat a still-fresh cached decision without mutating its history."""

    if hold is not None:
        history.prune()
        decision = hold.current() if history.ready else hold.begin_loss()
        valid_until = hold.valid_until
        if decision is not None and valid_until is not None:
            if arbiter is None:
                update_cleaning_actuators(
                    cleaner_controller,
                    pump_controller,
                    decision=decision,
                )
            else:
                arbiter.publish(role, decision, valid_until=valid_until)
            return True
        clear_realtime_actuator_role(
            cleaner_controller,
            pump_controller,
            role=role,
            arbiter=arbiter,
        )
        return False

    cache_ttl = inference.remaining_fresh_seconds()
    history.prune()
    if history.ready and cache_ttl is not None and cache_ttl > 0.0:
        publish_realtime_actuator_decision(
            cleaner_controller,
            pump_controller,
            role=role,
            history=history,
            decision=history.decision(),
            arbiter=arbiter,
        )
        return True
    clear_realtime_actuator_role(
        cleaner_controller,
        pump_controller,
        role=role,
        arbiter=arbiter,
    )
    return False


def draw_roi_guide(frame, roi_display=None):
    """Show that capture inference uses the complete native camera frame."""

    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        raise ValueError("Capture guide requires the native 1280x720 BGR frame.")
    display = frame.copy() if roi_display is None else roi_display.copy()
    if display.shape != frame.shape:
        raise ValueError(
            "Capture result shape must match the full camera frame: "
            f"expected {frame.shape}, got {display.shape}."
        )

    guide_color = (0, 255, 255)
    cv2.rectangle(
        display,
        (1, 1),
        (FRAME_WIDTH - 2, FRAME_HEIGHT - 2),
        guide_color,
        2,
    )
    cv2.putText(
        display,
        "CAPTURE ROI FULL FRAME 1280x720",
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        guide_color,
        2,
        cv2.LINE_AA,
    )

    return display


def draw_realtime_roi_guide(
    frame,
    annotated_frame=None,
    *,
    primary_roi_label: str = "RUST",
):
    """Show full-frame realtime results and fixed ROI boundaries."""

    if (
        not isinstance(frame, np.ndarray)
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise ValueError("Realtime camera frame must be an HxWx3 image.")
    water_top, water_bottom = water_control_roi_bounds(frame)
    crack_top, crack_bottom = crack_stop_roi_bounds(frame)
    display = frame.copy()
    if annotated_frame is not None:
        if annotated_frame.shape != frame.shape:
            raise ValueError(
                "Realtime annotated frame shape does not match the camera frame: "
                f"expected {frame.shape}, got {annotated_frame.shape}."
            )
        display = annotated_frame.copy()

    frame_width = display.shape[1]
    water_guide_color = (255, 220, 0)
    crack_guide_color = (255, 0, 255)
    cv2.line(
        display,
        (0, water_bottom),
        (frame_width - 1, water_bottom),
        water_guide_color,
        2,
    )
    cv2.putText(
        display,
        f"{primary_roi_label} ROI FULL WIDTH (Y={water_top}:{water_bottom})",
        (12, water_bottom - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        water_guide_color,
        2,
        cv2.LINE_AA,
    )
    cv2.line(
        display,
        (0, crack_top),
        (frame_width - 1, crack_top),
        crack_guide_color,
        2,
    )
    cv2.putText(
        display,
        f"CRACK ROI FULL WIDTH (Y={crack_top}:{crack_bottom})",
        (12, crack_top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        crack_guide_color,
        2,
        cv2.LINE_AA,
    )
    return display


def stack_dual_camera_displays(
    side_display: np.ndarray,
    top_display: np.ndarray,
) -> np.ndarray:
    """Place native SIDE and TOP camera displays side by side."""

    expected_shape = (FRAME_HEIGHT, FRAME_WIDTH, 3)
    if side_display.shape != expected_shape or top_display.shape != expected_shape:
        raise ValueError(
            "Dual-camera displays must both be native 1280x720 BGR frames."
        )
    return np.ascontiguousarray(
        np.hstack(
            (
                cv2.resize(side_display, (640, 360)),
                cv2.resize(top_display, (640, 360)),
            )
        )
    )


def draw_realtime_result_summary(
    frame: np.ndarray,
    rust_result,
    crack_result,
    *,
    crack_bypassed: bool = False,
) -> np.ndarray:
    """Draw the persistent UI-only result summary in the bottom-right corner."""

    output = frame.copy()
    if rust_result is None:
        rust_label = "RUST WAITING"
    else:
        fair = rust_result.class_ratios.get("Fair", 0.0) * 100.0
        poor = rust_result.class_ratios.get("Poor", 0.0) * 100.0
        severe = rust_result.class_ratios.get("Severe", 0.0) * 100.0
        rust_label = (
            f"RUST {rust_result.rust_ratio * 100.0:.2f}%  "
            f"F {fair:.2f}  P {poor:.2f}  S {severe:.2f}"
        )
    if crack_bypassed:
        crack_label = "CRACK BYPASS(TEST)"
    elif crack_result is None:
        crack_label = "CRACK WAITING"
    else:
        crack_state = "DETECTED" if crack_result.detected else "CLEAR"
        crack_label = (
            f"CRACK {crack_state}  {crack_result.crack_ratio * 100.0:.3f}%"
        )

    labels = (rust_label, crack_label)
    font_scale = 0.56
    thickness = 2
    sizes = [
        cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        for label in labels
    ]
    max_width = max(size[0][0] for size in sizes)
    max_height = max(size[0][1] for size in sizes)
    max_baseline = max(size[1] for size in sizes)
    line_step = max_height + max_baseline + 8
    margin = 12
    padding = 8
    label_x = max(margin + padding, output.shape[1] - margin - max_width)
    second_baseline_y = output.shape[0] - margin - padding
    first_baseline_y = second_baseline_y - line_step
    box_left = max(0, label_x - padding)
    box_top = max(0, first_baseline_y - max_height - padding)
    box_right = min(output.shape[1] - 1, output.shape[1] - margin + padding)
    box_bottom = min(
        output.shape[0] - 1,
        second_baseline_y + max_baseline + padding,
    )
    cv2.rectangle(
        output,
        (box_left, box_top),
        (box_right, box_bottom),
        (20, 35, 50),
        -1,
    )
    for label, baseline_y, color in (
        (rust_label, first_baseline_y, (0, 180, 255)),
        (crack_label, second_baseline_y, (0, 0, 255)),
    ):
        cv2.putText(
            output,
            label,
            (label_x, baseline_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return output


def draw_realtime_test_statistics(frame, statistics_label: str):
    """Overlay control and completed display-loop timing on a preview frame."""

    output = frame.copy()
    label_lines = statistics_label.split(" | ")
    label_sizes = [
        cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        for line in label_lines
    ]
    label_width = max(size[0][0] for size in label_sizes)
    label_height = max(size[0][1] for size in label_sizes)
    label_baseline = max(size[1] for size in label_sizes)
    line_step = label_height + label_baseline + 8
    label_x = 12
    label_y = max(
        label_height + 12,
        output.shape[0] - 20 - line_step * (len(label_lines) - 1),
    )
    label_right = min(output.shape[1] - 1, label_x + label_width + 16)
    cv2.rectangle(
        output,
        (label_x - 8, label_y - label_height - 8),
        (
            label_right,
            label_y
            + line_step * (len(label_lines) - 1)
            + label_baseline
            + 8,
        ),
        (20, 35, 50),
        -1,
    )
    for line_index, line in enumerate(label_lines):
        cv2.putText(
            output,
            line,
            (label_x, label_y + line_step * line_index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (60, 220, 90),
            2,
            cv2.LINE_AA,
        )
    return output


def format_realtime_test_statistics(
    frame_count: int,
    decision_span_seconds: float,
    total_control_latency_seconds: float,
    total_rust_seconds: float,
    total_crack_seconds: float,
    *,
    app_frame_count: int = 0,
    total_app_seconds: float = 0.0,
    multitask: bool = False,
) -> str:
    """Return truthful control-path and completed display-loop statistics."""

    if frame_count <= 0:
        raise ValueError("Realtime test statistics require at least one frame.")
    control_ms = total_control_latency_seconds * 1000.0 / frame_count
    rust_ms = total_rust_seconds * 1000.0 / frame_count
    crack_ms = total_crack_seconds * 1000.0 / frame_count
    control_fps = (
        (frame_count - 1) / decision_span_seconds
        if frame_count > 1 and decision_span_seconds > 0.0
        else None
    )
    if app_frame_count > 0:
        app_ms = total_app_seconds * 1000.0 / app_frame_count
        app_fps = (
            app_frame_count / total_app_seconds if total_app_seconds > 0.0 else 0.0
        )
        app_label = f"APP={app_fps:.1f} FPS {app_ms:.1f} ms"
    else:
        app_label = "APP=warming up"
    control_rate = (
        "CONTROL=warming up"
        if control_fps is None
        else f"CONTROL={control_fps:.1f} FPS"
    )
    control_label = f"{control_rate} latency={control_ms:.1f} ms"
    if multitask:
        return (
            f"{control_label} | {app_label} | "
            f"multitask={rust_ms:.1f} ms frames={frame_count}"
        )
    return (
        f"{control_label} | {app_label} | "
        f"rust={rust_ms:.1f} ms crack={crack_ms:.1f} ms frames={frame_count}"
    )


def detect_realtime_control(
    water_roi: np.ndarray,
    crack_roi: np.ndarray,
    student_detector,
    realtime_crack_detector,
    *,
    multitask_enabled: bool,
    hybrid_enabled: bool,
):
    """Select rust and crack results without mixing detector roles."""

    if hybrid_enabled:
        rust_result = student_detector.detect(water_roi)
        _unused_multitask_rust, crack_result = realtime_crack_detector.detect(
            water_roi
        )
        return rust_result, crack_result
    if multitask_enabled:
        return student_detector.detect(water_roi)
    rust_result = student_detector.detect(water_roi)
    crack_result = (
        None
        if realtime_crack_detector is None
        else realtime_crack_detector.detect(crack_roi)
    )
    return rust_result, crack_result


@dataclass(frozen=True)
class TimedRealtimeResult:
    result: object
    frame_read_completed_at: float


@dataclass(frozen=True)
class RealtimeInferenceOutcome:
    rust_result: object | None
    crack_result: object | None
    display_rust_result: object | None
    display_crack_result: object | None
    rust_seconds: float
    crack_seconds: float
    ready: bool
    frame_read_completed_at: float | None = None


@dataclass(frozen=True)
class TopRealtimeInferenceOutcome:
    obstacle_result: ObstacleDetectionResult | None
    crack_result: object | None
    display_obstacle_result: ObstacleDetectionResult | None
    display_crack_result: object | None
    obstacle_seconds: float
    crack_seconds: float
    ready: bool
    frame_read_completed_at: float | None = None


class DualTimingLogger:
    """Aggregate dual-camera test timings without printing on every loop."""

    _METRIC_NAMES = (
        "loop",
        "side_update",
        "top_update",
        "side_rust",
        "side_crack",
        "top_obstacle",
        "top_crack",
        "side_frame_age",
        "top_frame_age",
    )

    def __init__(
        self,
        *,
        warmup_seconds: float = DUAL_TIMING_WARMUP_SECONDS,
        report_interval_seconds: float = DUAL_TIMING_REPORT_INTERVAL_SECONDS,
        started_at: float | None = None,
    ) -> None:
        if warmup_seconds < 0.0:
            raise ValueError("Dual timing warmup must not be negative.")
        if report_interval_seconds <= 0.0:
            raise ValueError("Dual timing report interval must be positive.")
        if started_at is None:
            started_at = time.monotonic()
        self.warmup_seconds = float(warmup_seconds)
        self.report_interval_seconds = float(report_interval_seconds)
        self.measurement_started_at = float(started_at) + self.warmup_seconds
        self.next_report_at = (
            self.measurement_started_at + self.report_interval_seconds
        )
        self._samples = {name: [] for name in self._METRIC_NAMES}
        self._stale_counts = {"side": 0, "top": 0}
        self._ready_to_off_counts = {"side": 0, "front": 0}
        self._previous_ready = {"side": None, "front": None}

    @staticmethod
    def _append_positive(samples: list[float], seconds: float | None) -> None:
        if seconds is not None and seconds > 0.0:
            samples.append(float(seconds) * 1000.0)

    @staticmethod
    def _format_samples(name: str, samples: list[float]) -> str:
        if not samples:
            return f"{name}_ms(n=0)"
        values = np.asarray(samples, dtype=np.float64)
        p50, p95, p99 = np.percentile(values, (50, 95, 99))
        return (
            f"{name}_ms(n={len(samples)},mean={values.mean():.1f},"
            f"p50={p50:.1f},p95={p95:.1f},p99={p99:.1f},"
            f"max={values.max():.1f})"
        )

    def record(
        self,
        *,
        loop_seconds: float,
        side_update_seconds: float | None,
        top_update_seconds: float | None,
        side_outcome: RealtimeInferenceOutcome | None,
        top_outcome: TopRealtimeInferenceOutcome | None,
        side_frame_age_seconds: float | None,
        top_frame_age_seconds: float | None,
        side_stale: bool,
        top_stale: bool,
        side_ready: bool,
        front_ready: bool,
        now: float | None = None,
    ) -> str | None:
        """Record one loop and return a summary when the 5-second window closes."""

        if now is None:
            now = time.monotonic()
        current_ready = {"side": bool(side_ready), "front": bool(front_ready)}
        if now < self.measurement_started_at:
            self._previous_ready = current_ready
            return None

        self._append_positive(self._samples["loop"], loop_seconds)
        self._append_positive(
            self._samples["side_update"], side_update_seconds
        )
        self._append_positive(self._samples["top_update"], top_update_seconds)
        self._append_positive(
            self._samples["side_frame_age"], side_frame_age_seconds
        )
        self._append_positive(
            self._samples["top_frame_age"], top_frame_age_seconds
        )
        if side_outcome is not None:
            self._append_positive(
                self._samples["side_rust"], side_outcome.rust_seconds
            )
            self._append_positive(
                self._samples["side_crack"], side_outcome.crack_seconds
            )
        if top_outcome is not None:
            self._append_positive(
                self._samples["top_obstacle"], top_outcome.obstacle_seconds
            )
            self._append_positive(
                self._samples["top_crack"], top_outcome.crack_seconds
            )
        self._stale_counts["side"] += int(side_stale)
        self._stale_counts["top"] += int(top_stale)
        for role, ready in current_ready.items():
            if self._previous_ready[role] is True and not ready:
                self._ready_to_off_counts[role] += 1
        self._previous_ready = current_ready

        if now < self.next_report_at:
            return None
        window_seconds = max(
            self.report_interval_seconds,
            now - (self.next_report_at - self.report_interval_seconds),
        )
        parts = [
            f"[DUAL_TIMING] window={window_seconds:.1f}s",
            *(
                self._format_samples(name, self._samples[name])
                for name in self._METRIC_NAMES
            ),
            (
                "stale(side="
                f"{self._stale_counts['side']},top={self._stale_counts['top']})"
            ),
            (
                "ready_to_off(side="
                f"{self._ready_to_off_counts['side']},"
                f"front={self._ready_to_off_counts['front']})"
            ),
        ]
        summary = " ".join(parts)
        self._samples = {name: [] for name in self._METRIC_NAMES}
        self._stale_counts = {"side": 0, "top": 0}
        self._ready_to_off_counts = {"side": 0, "front": 0}
        while self.next_report_at <= now:
            self.next_report_at += self.report_interval_seconds
        return summary


@dataclass(frozen=True)
class ControlOffDiagnostic:
    reason: str
    monotonic_seconds: float
    history_ready: bool
    detail: str | None = None
    frame_sequence: int | None = None
    frame_age_ms: float | None = None
    new_crack_pct: float | None = None
    new_crack_blocked: bool = False
    new_rust_grade: int | None = None
    history_crack_pct: float | None = None
    history_rust_grade: int | None = None
    loss_age_ms: float | None = None
    hold_remaining_ms: float | None = None
    pre_loss_deadline_set: bool | None = None
    pre_valid_until: float | None = None
    watchdog_expired_at_off: bool | None = None


def select_control_off_reason(
    *,
    camera_error: bool,
    new_crack_blocked: bool,
    new_rust_grade: int | None,
    new_rust_forced_off: bool,
    rolling_history_hazard: bool,
    had_valid_decision: bool | None,
    loss_deadline_was_set: bool = False,
    watchdog_expired: bool = False,
) -> str:
    """Select one stable OFF reason using the documented safety priority."""

    if camera_error:
        return "camera_error"
    if new_crack_blocked:
        return "new_crack"
    if new_rust_forced_off and new_rust_grade is not None:
        return "new_rust_grade"
    if rolling_history_hazard:
        return "rolling_hazard"
    if watchdog_expired:
        return "decision_watchdog_expired"
    if had_valid_decision is True and loss_deadline_was_set:
        return "history_hold_expired"
    if had_valid_decision is False:
        return "startup_or_no_valid_decision"
    return "unknown"


class DualControlOffTransitionLogger:
    """Emit one diagnostic for each independent role ON-to-OFF transition."""

    _ROLE_STATE_INDICES = {"front": (0, 1), "side": (2, 3)}

    def __init__(self) -> None:
        self._previous_active = {"front": None, "side": None}

    @classmethod
    def _role_active(
        cls,
        role: str,
        actuator_state: tuple[float, bool, float, bool],
    ) -> bool:
        cleaner_index, pump_index = cls._ROLE_STATE_INDICES[role]
        return bool(
            actuator_state[cleaner_index] > 0.0
            or actuator_state[pump_index]
        )

    def is_turning_off(
        self,
        role: str,
        actuator_state: tuple[float, bool, float, bool],
    ) -> bool:
        return bool(
            self._previous_active[role] is True
            and not self._role_active(role, actuator_state)
        )

    @staticmethod
    def _format_optional(value, *, decimals: int = 1) -> str:
        if value is None:
            return "na"
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, float):
            return f"{value:.{decimals}f}"
        return str(value)

    @staticmethod
    def _format_detail(value: str | None) -> str:
        if value is None:
            return "na"
        return repr(" ".join(str(value).split()))

    def observe(
        self,
        actuator_state: tuple[float, bool, float, bool],
        *,
        front: ControlOffDiagnostic | None = None,
        side: ControlOffDiagnostic | None = None,
    ) -> tuple[str, ...]:
        diagnostics = {"front": front, "side": side}
        lines: list[str] = []
        for role in self._ROLE_STATE_INDICES:
            active = self._role_active(role, actuator_state)
            if self._previous_active[role] is True and not active:
                diagnostic = diagnostics[role]
                if diagnostic is None:
                    raise ValueError(f"Missing {role} CONTROL_OFF diagnostic.")
                lines.append(
                    "[CONTROL_OFF] "
                    f"mono={diagnostic.monotonic_seconds:.6f} "
                    f"role={role} reason={diagnostic.reason} "
                    f"detail={self._format_detail(diagnostic.detail)} "
                    f"frame_seq={self._format_optional(diagnostic.frame_sequence)} "
                    "frame_age_ms="
                    f"{self._format_optional(diagnostic.frame_age_ms)} "
                    "new_crack_pct="
                    f"{self._format_optional(diagnostic.new_crack_pct, decimals=4)} "
                    f"new_crack_blocked={str(diagnostic.new_crack_blocked).lower()} "
                    "new_rust_grade="
                    f"{self._format_optional(diagnostic.new_rust_grade)} "
                    f"hist_ready={str(diagnostic.history_ready).lower()} "
                    "hist_crack_pct="
                    f"{self._format_optional(diagnostic.history_crack_pct, decimals=4)} "
                    "hist_rust_grade="
                    f"{self._format_optional(diagnostic.history_rust_grade)} "
                    "loss_age_ms="
                    f"{self._format_optional(diagnostic.loss_age_ms)} "
                    "hold_remaining_ms="
                    f"{self._format_optional(diagnostic.hold_remaining_ms)} "
                    "pre_loss_deadline_set="
                    f"{self._format_optional(diagnostic.pre_loss_deadline_set)} "
                    "pre_valid_until="
                    f"{self._format_optional(diagnostic.pre_valid_until, decimals=6)} "
                    "watchdog_expired_at_off="
                    f"{self._format_optional(diagnostic.watchdog_expired_at_off)}"
                )
            self._previous_active[role] = active
        return tuple(lines)

    def observe_forced_off(
        self,
        actuator_state_before_off: tuple[float, bool, float, bool],
        *,
        reason: str,
        detail: str,
        now: float,
    ) -> tuple[str, ...]:
        diagnostics = {}
        for role in self._ROLE_STATE_INDICES:
            active = self._role_active(role, actuator_state_before_off)
            self._previous_active[role] = active
            if active:
                diagnostics[role] = ControlOffDiagnostic(
                    reason=reason,
                    monotonic_seconds=now,
                    history_ready=False,
                    detail=detail,
                )
        return self.observe(
            (0.0, False, 0.0, False),
            front=diagnostics.get("front"),
            side=diagnostics.get("side"),
        )


def classify_runtime_control_off(exc: Exception) -> tuple[str, str]:
    """Return a stable safety-OFF reason and single-line exception detail."""

    detail = " ".join(str(exc).split()) or type(exc).__name__
    lower_detail = detail.lower()
    reason = (
        "uart_error"
        if any(token in lower_detail for token in ("uart", "serial", " stm"))
        else "runtime_error"
    )
    return reason, detail


def current_dual_actuator_state(
    front_cleaner: CleanerController,
    front_pump: PumpController,
    side_cleaner: CleanerController,
    side_pump: PumpController,
    *,
    arbiter: RealtimeActuatorArbiter | None,
) -> tuple[float, bool, float, bool]:
    if arbiter is not None:
        return arbiter.desired_state()
    return (
        front_cleaner.duty_percent,
        front_pump.is_on,
        side_cleaner.duty_percent,
        side_pump.is_on,
    )


@dataclass(frozen=True)
class RealtimeControlDecision:
    """Worst-case command derived from each detector's last four inferences."""

    cleaner_duty_percent: float
    pump_on: bool
    rust_grade: int
    crack_ratio: float


@dataclass(frozen=True)
class FrontControlDecision:
    """Front-pair command derived from top-camera four-result windows."""

    cleaner_duty_percent: float
    pump_on: bool
    foreign_seen_recently: bool
    crack_ratio: float


class FrontControlHistory:
    """Hold independent top obstacle and crack four-result windows."""

    def __init__(
        self,
        *,
        window_size: int = REALTIME_CONTROL_WINDOW_SIZE,
        result_ttl_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        if window_size <= 0:
            raise ValueError("Front control window size must be positive.")
        if result_ttl_seconds <= 0.0:
            raise ValueError("Front control result TTL must be positive.")
        self.window_size = int(window_size)
        self.result_ttl_seconds = float(result_ttl_seconds)
        self.reset()

    def reset(self) -> None:
        self._crack_ratios: deque[tuple[float, float]] = deque(
            maxlen=self.window_size
        )
        self._foreign_detected: deque[tuple[float, bool]] = deque(
            maxlen=self.window_size
        )

    def prune(self, *, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self.result_ttl_seconds
        while self._crack_ratios and self._crack_ratios[0][0] < cutoff:
            self._crack_ratios.popleft()
        while self._foreign_detected and self._foreign_detected[0][0] < cutoff:
            self._foreign_detected.popleft()

    @property
    def ready(self) -> bool:
        self.prune()
        return self.ready_without_prune

    @property
    def ready_without_prune(self) -> bool:
        return (
            len(self._crack_ratios) == self.window_size
            and len(self._foreign_detected) == self.window_size
        )

    @property
    def valid_until(self) -> float | None:
        self.prune()
        if not self.ready:
            return None
        oldest = min(self._crack_ratios[0][0], self._foreign_detected[0][0])
        return oldest + self.result_ttl_seconds

    def update_crack(
        self,
        crack_result,
        *,
        observed_at: float | None = None,
    ) -> None:
        try:
            ratio = validated_crack_ratio(crack_result)
        except (TypeError, ValueError) as exc:
            self.reset()
            if isinstance(exc, ValueError):
                raise
            raise ValueError("Realtime crack ratio is not numeric.") from exc
        if observed_at is None:
            observed_at = time.monotonic()
        self._crack_ratios.append((float(observed_at), ratio))
        self.prune(now=float(observed_at))

    def update_foreign(
        self,
        detected: bool,
        *,
        observed_at: float | None = None,
    ) -> None:
        if type(detected) is not bool:
            self.reset()
            raise ValueError("Foreign-object control result must be a boolean.")
        if observed_at is None:
            observed_at = time.monotonic()
        self._foreign_detected.append((float(observed_at), detected))
        self.prune(now=float(observed_at))

    def update(self, outcome: TopRealtimeInferenceOutcome) -> None:
        """Append only the top-camera result produced on this frame."""

        if (
            not outcome.ready
            and outcome.display_crack_result is None
            and outcome.display_obstacle_result is None
        ):
            self.reset()
            return
        observed_at = outcome.frame_read_completed_at
        if outcome.display_crack_result is not None:
            self.update_crack(
                outcome.display_crack_result,
                observed_at=observed_at,
            )
        if outcome.display_obstacle_result is not None:
            self.update_foreign(
                obstacle_detected_in_control_roi(
                    outcome.display_obstacle_result
                ),
                observed_at=observed_at,
            )

    def decision(self) -> FrontControlDecision:
        if not self.ready:
            raise RuntimeError(
                "Four fresh top-crack and foreign-object results are required."
            )

        crack_ratio = max(value for _, value in self._crack_ratios)
        foreign_seen_recently = any(value for _, value in self._foreign_detected)
        crack_blocked = crack_ratio > CRACK_CONTROL_STOP_RATIO
        return FrontControlDecision(
            cleaner_duty_percent=(
                0.0
                if crack_blocked
                else (
                    CLEANER_STAGE_ONE_DUTY_PERCENT
                    if foreign_seen_recently
                    else CLEANER_BASE_DUTY_PERCENT
                )
            ),
            pump_on=not crack_blocked,
            foreign_seen_recently=foreign_seen_recently,
            crack_ratio=crack_ratio,
        )


class RealtimeControlHistory:
    """Hold independent four-inference windows for alternating detectors."""

    def __init__(
        self,
        *,
        no_crack: bool = False,
        window_size: int = REALTIME_CONTROL_WINDOW_SIZE,
        result_ttl_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        if window_size <= 0:
            raise ValueError("Realtime control window size must be positive.")
        if result_ttl_seconds <= 0.0:
            raise ValueError("Realtime control result TTL must be positive.")
        self.no_crack = bool(no_crack)
        self.window_size = int(window_size)
        self.result_ttl_seconds = float(result_ttl_seconds)
        self.reset()

    def reset(self) -> None:
        self._rust_grades: deque[tuple[float, int]] = deque(
            maxlen=self.window_size
        )
        self._crack_ratios: deque[tuple[float, float]] = deque(
            maxlen=self.window_size
        )

    def prune(self, *, now: float | None = None) -> None:
        if now is None:
            now = time.monotonic()
        cutoff = now - self.result_ttl_seconds
        while self._rust_grades and self._rust_grades[0][0] < cutoff:
            self._rust_grades.popleft()
        while self._crack_ratios and self._crack_ratios[0][0] < cutoff:
            self._crack_ratios.popleft()

    @property
    def ready(self) -> bool:
        self.prune()
        return self.ready_without_prune

    @property
    def ready_without_prune(self) -> bool:
        rust_ready = len(self._rust_grades) == self.window_size
        crack_ready = self.no_crack or len(self._crack_ratios) == self.window_size
        return rust_ready and crack_ready

    @property
    def valid_until(self) -> float | None:
        self.prune()
        if not self.ready:
            return None
        oldest = self._rust_grades[0][0]
        if not self.no_crack:
            oldest = min(oldest, self._crack_ratios[0][0])
        return oldest + self.result_ttl_seconds

    def update(self, outcome: RealtimeInferenceOutcome) -> None:
        """Append only the detector result produced on this camera frame."""

        if (
            not outcome.ready
            and outcome.display_rust_result is None
            and outcome.display_crack_result is None
        ):
            self.reset()
            return
        observed_at = outcome.frame_read_completed_at
        if observed_at is None:
            observed_at = time.monotonic()
        if outcome.display_rust_result is not None:
            rust_result = outcome.display_rust_result
            if getattr(rust_result, "status", None) != "ready":
                raise ValueError("Realtime rust result is not ready.")
            self._rust_grades.append(
                (
                    float(observed_at),
                    highest_rust_grade(getattr(rust_result, "class_map", None)),
                )
            )
        if not self.no_crack and outcome.display_crack_result is not None:
            self._crack_ratios.append(
                (
                    float(observed_at),
                    validated_crack_ratio(outcome.display_crack_result),
                )
            )
        self.prune(now=float(observed_at))

    def decision(self) -> RealtimeControlDecision:
        if not self.ready:
            raise RuntimeError(
                "Four fresh inference results from each enabled model are required."
            )

        rust_grade = max(value for _, value in self._rust_grades)
        crack_ratio = (
            0.0
            if self.no_crack
            else max(value for _, value in self._crack_ratios)
        )
        crack_blocked = crack_ratio > CRACK_CONTROL_STOP_RATIO

        if crack_blocked or rust_grade >= 2:
            cleaner_duty_percent = 0.0
        elif rust_grade == 1:
            cleaner_duty_percent = CLEANER_STAGE_ONE_DUTY_PERCENT
        else:
            cleaner_duty_percent = CLEANER_BASE_DUTY_PERCENT

        pump_on = not crack_blocked and rust_grade == 0
        return RealtimeControlDecision(
            cleaner_duty_percent=cleaner_duty_percent,
            pump_on=pump_on,
            rust_grade=rust_grade,
            crack_ratio=crack_ratio,
        )


def side_outcome_requires_immediate_off(
    outcome: RealtimeInferenceOutcome | None,
    *,
    history_ready: bool,
) -> bool:
    """Reject newly observed SIDE hazards before a four-result window rebuilds."""

    if outcome is None:
        return False
    if (
        outcome.display_crack_result is not None
        and cleaning_blocked_by_crack(outcome.display_crack_result)
    ):
        return True
    if outcome.display_rust_result is None:
        return False
    rust_result = outcome.display_rust_result
    if getattr(rust_result, "status", None) != "ready":
        return True
    rust_grade = highest_rust_grade(getattr(rust_result, "class_map", None))
    # Grade 1 must at least stop water. Until a complete window can produce
    # that restricted command, conservatively stop the whole SIDE pair.
    return rust_grade >= 2 or (rust_grade == 1 and not history_ready)


def front_outcome_requires_immediate_off(
    outcome: TopRealtimeInferenceOutcome | None,
) -> bool:
    """Reject a newly observed FRONT crack without waiting for history."""

    return bool(
        outcome is not None
        and outcome.display_crack_result is not None
        and cleaning_blocked_by_crack(outcome.display_crack_result)
    )


def build_control_off_diagnostic(
    *,
    role: str,
    outcome: RealtimeInferenceOutcome | TopRealtimeInferenceOutcome | None,
    history_decision: RealtimeControlDecision | FrontControlDecision | None,
    history_ready: bool,
    camera_error: bool,
    camera_error_detail: str | None,
    hold_snapshot: RealtimeActuatorHoldSnapshot,
    hold: RealtimeActuatorDecisionHold,
    frame: CameraFrame | None,
    now: float,
) -> ControlOffDiagnostic:
    """Build diagnostics without consulting display caches or changing control."""

    new_crack_result = (
        None if outcome is None else outcome.display_crack_result
    )
    new_crack_blocked = False
    new_crack_pct = None
    if new_crack_result is not None:
        try:
            new_crack_ratio = validated_crack_ratio(new_crack_result)
            new_crack_pct = new_crack_ratio * 100.0
            new_crack_blocked = new_crack_ratio > CRACK_CONTROL_STOP_RATIO
        except (TypeError, ValueError):
            pass

    new_rust_grade = None
    if role == "side" and outcome is not None:
        new_rust_result = outcome.display_rust_result
        if new_rust_result is not None:
            try:
                if getattr(new_rust_result, "status", None) == "ready":
                    new_rust_grade = highest_rust_grade(
                        getattr(new_rust_result, "class_map", None)
                    )
            except (TypeError, ValueError):
                pass
    new_rust_forced_off = bool(
        new_rust_grade is not None
        and (
            new_rust_grade >= 2
            or (new_rust_grade == 1 and not history_ready)
        )
    )

    rolling_history_hazard = bool(
        history_decision is not None
        and history_decision.cleaner_duty_percent == 0.0
        and not history_decision.pump_on
    )
    history_crack_pct = (
        None
        if history_decision is None
        else history_decision.crack_ratio * 100.0
    )
    history_rust_grade = (
        history_decision.rust_grade
        if isinstance(history_decision, RealtimeControlDecision)
        else None
    )

    loss_started_at = hold.loss_started_at
    if loss_started_at is None:
        loss_started_at = hold_snapshot.loss_started_at
    valid_until = hold.valid_until
    if valid_until is None:
        valid_until = hold_snapshot.valid_until
    loss_age_ms = (
        None
        if loss_started_at is None
        else max(0.0, now - loss_started_at) * 1000.0
    )
    hold_remaining_ms = (
        None
        if valid_until is None
        else max(0.0, valid_until - now) * 1000.0
    )
    frame_age_ms = (
        None
        if frame is None
        else max(0.0, now - frame.read_completed_at) * 1000.0
    )
    watchdog_expired_at_off = bool(
        hold_snapshot.watchdog_expired
        or (
            hold_snapshot.has_decision
            and not hold_snapshot.loss_deadline_was_set
            and hold_snapshot.valid_until is not None
            and now > hold_snapshot.valid_until
        )
    )
    reason = select_control_off_reason(
        camera_error=camera_error,
        new_crack_blocked=new_crack_blocked,
        new_rust_grade=new_rust_grade,
        new_rust_forced_off=new_rust_forced_off,
        rolling_history_hazard=rolling_history_hazard,
        had_valid_decision=hold_snapshot.has_decision,
        loss_deadline_was_set=hold_snapshot.loss_deadline_was_set,
        watchdog_expired=watchdog_expired_at_off,
    )
    return ControlOffDiagnostic(
        reason=reason,
        monotonic_seconds=now,
        history_ready=history_ready,
        detail=(
            " ".join(str(camera_error_detail).split())
            if camera_error_detail is not None
            else None
        ),
        frame_sequence=None if frame is None else frame.sequence,
        frame_age_ms=frame_age_ms,
        new_crack_pct=new_crack_pct,
        new_crack_blocked=new_crack_blocked,
        new_rust_grade=new_rust_grade,
        history_crack_pct=history_crack_pct,
        history_rust_grade=history_rust_grade,
        loss_age_ms=loss_age_ms,
        hold_remaining_ms=hold_remaining_ms,
        pre_loss_deadline_set=hold_snapshot.loss_deadline_was_set,
        pre_valid_until=hold_snapshot.valid_until,
        watchdog_expired_at_off=watchdog_expired_at_off,
    )


class RealtimeDisplayCache:
    """Persist numeric UI-summary results; never use them for control/overlays."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.rust_result = None
        self.crack_result = None

    def update(self, outcome: RealtimeInferenceOutcome | None) -> None:
        if outcome is None:
            return
        if outcome.display_rust_result is not None:
            self.rust_result = outcome.display_rust_result
        if outcome.display_crack_result is not None:
            self.crack_result = outcome.display_crack_result


class AlternatingTopRealtimeInference:
    """Alternate obstacle and crack ROI inference for the top camera."""

    def __init__(
        self,
        obstacle_detector: ObstacleDetector,
        crack_detector: HrSegNetCrackDetector,
        *,
        max_result_age_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        self.obstacle_detector = obstacle_detector
        self.crack_detector = crack_detector
        self.max_result_age_seconds = max_result_age_seconds
        self.crack_inference_count = 0
        self.reset()

    def reset(self) -> None:
        # Offset the shared heavy crack model from the side camera: side starts
        # with rust while top starts with crack.
        self._next_role = "crack"
        self._obstacle: TimedRealtimeResult | None = None
        self._crack: TimedRealtimeResult | None = None

    def expire_cache(self, *, now: float | None = None) -> None:
        """Drop only expired results while preserving the next model phase."""

        if now is None:
            now = time.monotonic()
        if self._obstacle is not None and not self._is_fresh(self._obstacle, now):
            self._obstacle = None
        if self._crack is not None and not self._is_fresh(self._crack, now):
            self._crack = None

    def process(
        self,
        camera_frame: CameraFrame,
        obstacle_roi: np.ndarray,
        crack_roi: np.ndarray,
    ) -> TopRealtimeInferenceOutcome:
        obstacle_seconds = 0.0
        crack_seconds = 0.0
        display_obstacle_result = None
        display_crack_result = None
        if self._next_role == "obstacle":
            started = time.perf_counter()
            obstacle_result = self.obstacle_detector.detect(obstacle_roi)
            obstacle_seconds = time.perf_counter() - started
            self._obstacle = TimedRealtimeResult(
                obstacle_result,
                camera_frame.read_completed_at,
            )
            display_obstacle_result = obstacle_result
            self._next_role = "crack"
        else:
            started = time.perf_counter()
            crack_result = self.crack_detector.detect(crack_roi)
            crack_seconds = time.perf_counter() - started
            self.crack_inference_count += 1
            self._crack = TimedRealtimeResult(
                crack_result,
                camera_frame.read_completed_at,
            )
            display_crack_result = crack_result
            self._next_role = "obstacle"

        now = time.monotonic()
        ready = self._is_fresh(self._obstacle, now) and self._is_fresh(
            self._crack, now
        )
        return TopRealtimeInferenceOutcome(
            None if self._obstacle is None else self._obstacle.result,
            None if self._crack is None else self._crack.result,
            display_obstacle_result,
            display_crack_result,
            obstacle_seconds,
            crack_seconds,
            ready,
            camera_frame.read_completed_at,
        )

    def _is_fresh(self, timed_result: TimedRealtimeResult | None, now: float) -> bool:
        return (
            timed_result is not None
            and 0.0
            <= now - timed_result.frame_read_completed_at
            <= self.max_result_age_seconds
        )

    def remaining_fresh_seconds(self, *, now: float | None = None) -> float | None:
        required = (self._obstacle, self._crack)
        if any(result is None for result in required):
            return None
        if now is None:
            now = time.monotonic()
        oldest = min(
            result.frame_read_completed_at for result in required if result is not None
        )
        return max(0.0, oldest + self.max_result_age_seconds - now)


class AlternatingRealtimeInference:
    """Alternate separate models while rejecting missing or stale cached results."""

    def __init__(
        self,
        student_detector,
        crack_detector,
        *,
        multitask_enabled: bool,
        hybrid_enabled: bool,
        no_crack: bool = False,
        max_result_age_seconds: float = REALTIME_RESULT_MAX_AGE_SECONDS,
    ) -> None:
        self.student_detector = student_detector
        self.crack_detector = crack_detector
        self.multitask_enabled = multitask_enabled
        self.hybrid_enabled = hybrid_enabled
        self.no_crack = no_crack
        self.max_result_age_seconds = max_result_age_seconds
        self.crack_inference_count = 0
        self.reset()

    def reset(self) -> None:
        self._next_role = "rust"
        self._rust: TimedRealtimeResult | None = None
        self._crack: TimedRealtimeResult | None = None

    def expire_cache(self, *, now: float | None = None) -> None:
        """Drop only expired results while preserving the next model phase."""

        if now is None:
            now = time.monotonic()
        if self._rust is not None and not self._is_fresh(self._rust, now):
            self._rust = None
        if self._crack is not None and not self._is_fresh(self._crack, now):
            self._crack = None

    def process(
        self,
        camera_frame: CameraFrame,
        water_roi: np.ndarray,
        crack_roi: np.ndarray,
    ) -> RealtimeInferenceOutcome:
        rust_seconds = 0.0
        crack_seconds = 0.0
        display_rust_result = None
        display_crack_result = None

        if self.multitask_enabled and not self.hybrid_enabled:
            started = time.perf_counter()
            rust_result, crack_result = self.student_detector.detect(water_roi)
            rust_seconds = time.perf_counter() - started
            self.crack_inference_count += 1
            self._rust = TimedRealtimeResult(
                rust_result,
                camera_frame.read_completed_at,
            )
            self._crack = TimedRealtimeResult(
                crack_result,
                camera_frame.read_completed_at,
            )
            display_rust_result = rust_result
            display_crack_result = crack_result
        elif self.crack_detector is None:
            started = time.perf_counter()
            rust_result = self.student_detector.detect(water_roi)
            rust_seconds = time.perf_counter() - started
            self._rust = TimedRealtimeResult(
                rust_result,
                camera_frame.read_completed_at,
            )
            display_rust_result = rust_result
        elif self._next_role == "rust":
            started = time.perf_counter()
            rust_result = self.student_detector.detect(water_roi)
            rust_seconds = time.perf_counter() - started
            self._rust = TimedRealtimeResult(
                rust_result,
                camera_frame.read_completed_at,
            )
            display_rust_result = rust_result
            self._next_role = "crack"
        else:
            started = time.perf_counter()
            if self.hybrid_enabled:
                _unused_rust, crack_result = self.crack_detector.detect(water_roi)
            else:
                crack_result = self.crack_detector.detect(crack_roi)
            crack_seconds = time.perf_counter() - started
            self.crack_inference_count += 1
            self._crack = TimedRealtimeResult(
                crack_result,
                camera_frame.read_completed_at,
            )
            display_crack_result = crack_result
            self._next_role = "rust"

        now = time.monotonic()
        ready = self._is_fresh(self._rust, now) and (
            self.no_crack or self._is_fresh(self._crack, now)
        )
        return RealtimeInferenceOutcome(
            None if self._rust is None else self._rust.result,
            None if self._crack is None else self._crack.result,
            display_rust_result,
            display_crack_result,
            rust_seconds,
            crack_seconds,
            ready,
            camera_frame.read_completed_at,
        )

    def _is_fresh(self, timed_result: TimedRealtimeResult | None, now: float) -> bool:
        return (
            timed_result is not None
            and 0.0
            <= now - timed_result.frame_read_completed_at
            <= self.max_result_age_seconds
        )

    def remaining_fresh_seconds(self, *, now: float | None = None) -> float | None:
        """Return the oldest required cache TTL, or None until caches exist."""

        required = [self._rust]
        if not self.no_crack:
            required.append(self._crack)
        if any(result is None for result in required):
            return None
        if now is None:
            now = time.monotonic()
        oldest_read_completed_at = min(
            result.frame_read_completed_at
            for result in required
            if result is not None
        )
        return max(
            0.0,
            oldest_read_completed_at + self.max_result_age_seconds - now,
        )


def realtime_camera_frame_is_fresh(
    camera_frame: CameraFrame,
    *,
    now: float | None = None,
) -> bool:
    if now is None:
        now = time.monotonic()
    age = now - camera_frame.read_completed_at
    return 0.0 <= age <= REALTIME_INPUT_MAX_AGE_SECONDS


def dual_realtime_inputs_are_fresh(
    side_frame: CameraFrame,
    top_frame: CameraFrame,
    side_inference: AlternatingRealtimeInference,
    top_inference: AlternatingTopRealtimeInference,
    *,
    now: float | None = None,
) -> bool:
    """Reject stale inputs while allowing initial alternating-cache warm-up."""

    if now is None:
        now = time.monotonic()
    return (
        realtime_role_inputs_are_fresh(side_frame, side_inference, now=now)
        and realtime_role_inputs_are_fresh(top_frame, top_inference, now=now)
    )


def realtime_role_inputs_are_fresh(
    camera_frame: CameraFrame | None,
    inference: AlternatingRealtimeInference | AlternatingTopRealtimeInference,
    *,
    now: float | None = None,
) -> bool:
    """Validate one camera/inference pair without coupling the other role."""

    if camera_frame is None:
        return False
    if now is None:
        now = time.monotonic()
    inference.expire_cache(now=now)
    return realtime_camera_frame_is_fresh(camera_frame, now=now)


def force_cleaning_safe_off(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
) -> None:
    """Fail closed once without spamming repeated OFF commands."""

    if cleaner_controller.is_on:
        cleaner_controller.force_off()
    if pump_controller.is_on:
        pump_controller.force_off()


def force_cleaning_pairs_safe_off(
    *pairs: tuple[CleanerController, PumpController],
) -> None:
    """Fail every supplied pair closed even if one shared-UART write fails."""

    first_error = None
    for cleaner_controller, pump_controller in pairs:
        for controller in (cleaner_controller, pump_controller):
            try:
                controller.force_off()
            except RuntimeError as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error


def read_realtime_camera_frame(
    camera: LatestFrameCamera,
    *,
    after_sequence: int,
    inference: AlternatingRealtimeInference,
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
) -> tuple[CameraFrame, bool]:
    """Wait for a new frame, stopping outputs exactly when cached results expire."""

    started_at = time.monotonic()
    camera_deadline = started_at + REALTIME_FRAME_TIMEOUT_SECONDS
    cache_ttl = inference.remaining_fresh_seconds(now=started_at)
    cache_expired = cache_ttl == 0.0
    if cache_expired:
        force_cleaning_safe_off(cleaner_controller, pump_controller)

    first_wait = REALTIME_FRAME_TIMEOUT_SECONDS
    if cache_ttl is not None:
        first_wait = min(first_wait, cache_ttl)
    if first_wait > 0.0:
        try:
            return (
                camera.read_latest(
                    after_sequence=after_sequence,
                    timeout=first_wait,
                ),
                cache_expired,
            )
        except TimeoutError:
            now = time.monotonic()
            remaining_cache_ttl = inference.remaining_fresh_seconds(now=now)
            if remaining_cache_ttl is None or remaining_cache_ttl > 1e-9:
                raise
            force_cleaning_safe_off(cleaner_controller, pump_controller)
            cache_expired = True

    remaining_camera_wait = camera_deadline - time.monotonic()
    if remaining_camera_wait <= 0.0:
        raise TimeoutError(
            "No fresh realtime camera frame arrived within "
            f"{REALTIME_FRAME_TIMEOUT_SECONDS:.3f} seconds."
        )
    return (
        camera.read_latest(
            after_sequence=after_sequence,
            timeout=remaining_camera_wait,
        ),
        cache_expired,
    )


def read_dual_realtime_camera_frames(
    side_camera: LatestFrameCamera,
    top_camera: LatestFrameCamera,
    *,
    after_side_sequence: int,
    after_top_sequence: int,
    side_inference: AlternatingRealtimeInference,
    top_inference: AlternatingTopRealtimeInference,
    side_cleaner_controller: CleanerController,
    side_pump_controller: PumpController,
    front_cleaner_controller: CleanerController,
    front_pump_controller: PumpController,
    actuator_arbiter: RealtimeActuatorArbiter | None = None,
) -> DualRealtimeFrameRead:
    """Poll both cameras under one deadline without one role blocking the other."""

    started_at = time.monotonic()
    camera_deadline = started_at + REALTIME_FRAME_TIMEOUT_SECONDS
    roles = {
        "side": {
            "actuator_role": "side",
            "camera": side_camera,
            "after_sequence": after_side_sequence,
            "inference": side_inference,
            "cleaner": side_cleaner_controller,
            "pump": side_pump_controller,
            "frame": None,
            "error": None,
            "expired": False,
        },
        "top": {
            "actuator_role": "front",
            "camera": top_camera,
            "after_sequence": after_top_sequence,
            "inference": top_inference,
            "cleaner": front_cleaner_controller,
            "pump": front_pump_controller,
            "frame": None,
            "error": None,
            "expired": False,
        },
    }
    for role_name, role in roles.items():
        cache_ttl = role["inference"].remaining_fresh_seconds(now=started_at)
        role["cache_deadline"] = (
            None if cache_ttl is None else started_at + cache_ttl
        )
        if cache_ttl == 0.0:
            role["expired"] = True

    while time.monotonic() < camera_deadline:
        ordered_roles = sorted(
            roles.items(),
            key=lambda item: (
                item[1]["cache_deadline"] is None,
                item[1]["cache_deadline"] or 0.0,
            ),
        )
        for role_name, role in ordered_roles:
            if role["frame"] is not None or role["error"] is not None:
                continue
            now = time.monotonic()
            remaining = camera_deadline - now
            if remaining <= 0.0:
                break
            cache_deadline = role["cache_deadline"]
            if not role["expired"] and cache_deadline is not None:
                remaining = min(remaining, max(0.0, cache_deadline - now))
            timeout = min(DUAL_CAMERA_READ_POLL_SECONDS, remaining)
            if timeout <= 0.0:
                role["expired"] = True
                continue
            try:
                role["frame"] = role["camera"].read_latest(
                    after_sequence=role["after_sequence"],
                    timeout=timeout,
                )
            except TimeoutError:
                now = time.monotonic()
                if (
                    not role["expired"]
                    and cache_deadline is not None
                    and now >= cache_deadline - 1e-9
                ):
                    role["expired"] = True
            except RuntimeError as exc:
                clear_realtime_actuator_role(
                    role["cleaner"],
                    role["pump"],
                    role=role["actuator_role"],
                    arbiter=actuator_arbiter,
                )
                role["expired"] = True
                role["error"] = f"{role_name}: {exc}"
        if any(role["frame"] is not None for role in roles.values()):
            break
        if all(role["error"] is not None for role in roles.values()):
            break

    deadline_reached = time.monotonic() >= camera_deadline
    for role_name, role in roles.items():
        if (
            deadline_reached
            and role["frame"] is None
            and role["error"] is None
        ):
            role["error"] = (
                f"{role_name}: no fresh frame within the shared "
                f"{REALTIME_FRAME_TIMEOUT_SECONDS:.3f}s deadline"
            )

    side_frame = roles["side"]["frame"]
    top_frame = roles["top"]["frame"]
    side_cache_expired = bool(roles["side"]["expired"])
    top_cache_expired = bool(roles["top"]["expired"])
    side_error = roles["side"]["error"]
    top_error = roles["top"]["error"]
    side_waiting = (
        side_frame is None
        and side_error is None
        and not side_cache_expired
    )
    top_waiting = (
        top_frame is None
        and top_error is None
        and not top_cache_expired
    )
    if side_frame is None and top_frame is None:
        raise RuntimeError(
            "Both realtime camera streams failed: "
            f"side={side_error}; top={top_error}"
        )
    return DualRealtimeFrameRead(
        side_frame=side_frame,
        top_frame=top_frame,
        side_cache_expired=side_cache_expired,
        top_cache_expired=top_cache_expired,
        side_error=side_error,
        top_error=top_error,
        side_waiting=side_waiting,
        top_waiting=top_waiting,
    )


def read_stopped_capture_frame(
    camera: LatestFrameCamera,
    *,
    after_sequence: int | None = None,
) -> CameraFrame:
    """Wait for a frame completed after the post-stop settling deadline."""

    time.sleep(CAPTURE_SETTLE_SECONDS)
    baseline_sequence = camera.latest_sequence
    if after_sequence is not None:
        baseline_sequence = max(baseline_sequence, after_sequence)
    return camera.read_latest(
        after_sequence=baseline_sequence,
        timeout=LATEST_FRAME_TIMEOUT_SECONDS + CAPTURE_SETTLE_SECONDS,
    )


def read_stopped_capture_pair(
    side_camera: LatestFrameCamera,
    top_camera: LatestFrameCamera,
) -> tuple[CameraFrame, CameraFrame]:
    """Settle once, then obtain one new side/top frame under one deadline."""

    time.sleep(CAPTURE_SETTLE_SECONDS)
    side_baseline = side_camera.latest_sequence
    top_baseline = top_camera.latest_sequence
    deadline = time.monotonic() + LATEST_FRAME_TIMEOUT_SECONDS
    side_frame = side_camera.read_latest(
        after_sequence=side_baseline,
        timeout=max(0.0, deadline - time.monotonic()),
    )
    top_frame = top_camera.read_latest(
        after_sequence=top_baseline,
        timeout=max(0.0, deadline - time.monotonic()),
    )
    return side_frame, top_frame


def validate_engine(
    engine_path: Path,
    expected_sha256: str | None = None,
    sha_option: str = "--engine-sha256",
) -> tuple[Path, str]:
    engine_path = engine_path.expanduser().resolve()
    digest = hashlib.sha256()
    try:
        with engine_path.open("rb") as engine_file:
            for chunk in iter(lambda: engine_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise RuntimeError(f"Could not read TensorRT engine: {engine_path}") from exc
    if engine_path.stat().st_size == 0:
        raise ValueError(f"TensorRT engine is empty: {engine_path}")
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None:
        expected_sha256 = str(expected_sha256).strip().lower()
        if len(expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in expected_sha256
        ):
            raise ValueError(
                f"{sha_option} must be a 64-character hexadecimal digest."
            )
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"TensorRT engine SHA-256 for {sha_option} does not match the "
                "approved digest: "
                f"expected {expected_sha256}, got {actual_sha256}."
            )
    return engine_path, actual_sha256


def validate_tf32_runtime_environment() -> None:
    """Require the process-wide math setting used to build the crack plans."""

    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError(
            "NVIDIA_TF32_OVERRIDE must be exactly 0 before starting Python; "
            "the setting applies to all four TensorRT plans in this process."
        )


def configured_student_engine(
    args: argparse.Namespace,
) -> tuple[Path | None, str | None, bool]:
    if getattr(args, "optimized_student_engine", None) is not None:
        return (
            args.optimized_student_engine,
            args.optimized_student_engine_sha256,
            True,
        )
    return args.student_engine, args.student_engine_sha256, False


def save_capture(
    frame,
    phase: str,
    phase_sequence: int | None = None,
    output_directory: Path = CAPTURE_DIRECTORY,
    captured_at: datetime | None = None,
) -> Path:
    prefixes = {
        INITIAL_PHASE: "initial",
        RESCAN_PHASE: "rescan",
        MANUAL_PHASE: "manual",
    }
    if phase not in prefixes:
        raise ValueError(f"Unknown capture phase: {phase}")
    if phase in (INITIAL_PHASE, RESCAN_PHASE):
        if (
            not isinstance(phase_sequence, int)
            or isinstance(phase_sequence, bool)
            or not 1 <= phase_sequence <= AUTOMATIC_RAIL_SECTION_TARGET
        ):
            raise ValueError(
                "Automatic rail-section sequence must be from 1 through "
                f"{AUTOMATIC_RAIL_SECTION_TARGET}."
            )
        capture_name = f"{prefixes[phase]}_{phase_sequence:02d}"
    else:
        capture_name = prefixes[phase]

    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        encoded_ok, encoded = cv2.imencode(".jpg", frame)
    except cv2.error as exc:
        raise RuntimeError("Could not encode the camera frame as JPEG.") from exc
    if not encoded_ok:
        raise RuntimeError("Could not encode the camera frame as JPEG.")

    captured_at = captured_at or datetime.now()
    timestamp = captured_at.strftime("%Y%m%d_%H%M%S")
    sequence = 0
    while True:
        suffix = "" if sequence == 0 else f"_{sequence:03d}"
        capture_path = output_directory / f"{capture_name}_{timestamp}{suffix}.jpg"
        try:
            with capture_path.open("xb") as capture_file:
                encoded_bytes = encoded.tobytes()
                written = capture_file.write(encoded_bytes)
                if written != len(encoded_bytes):
                    raise OSError(
                        "Could not write the complete JPEG capture "
                        f"({written}/{len(encoded_bytes)} bytes)."
                    )
            return capture_path
        except FileExistsError:
            sequence += 1


def queue_capture_for_analysis(
    frame,
    phase: str,
    phase_sequence: int | None,
    trigger: str,
    analysis_worker,
    output_directory: Path = CAPTURE_DIRECTORY,
    *,
    camera_role: str | None = None,
    captured_at: datetime | None = None,
    frame_read_completed_at: float | None = None,
) -> CaptureAnalysisTask:
    """Persist an original frame, then enqueue its deferred analysis."""

    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        shape = getattr(frame, "shape", None)
        raise ValueError(
            "Capture frame must be the native 1280x720 BGR camera frame; "
            f"got {shape}."
        )
    if camera_role is not None and camera_role not in (
        SIDE_CAMERA_ROLE,
        TOP_CAMERA_ROLE,
    ):
        raise ValueError(f"Unknown capture camera role: {camera_role}")
    task_camera_role = SIDE_CAMERA_ROLE if camera_role is None else camera_role
    captured_at = captured_at or datetime.now()
    raw_directory = output_directory / "raw"
    if camera_role is not None:
        raw_directory /= task_camera_role
    raw_capture_path = save_capture(
        frame,
        phase,
        phase_sequence,
        raw_directory,
        captured_at,
    )
    task = CaptureAnalysisTask(
        raw_capture_path=raw_capture_path,
        phase=phase,
        phase_sequence=phase_sequence,
        trigger=trigger,
        captured_at=captured_at,
        camera_role=task_camera_role,
        frame_read_completed_at=frame_read_completed_at,
    )
    try:
        analysis_worker.submit(task)
    except Exception as exc:
        raise RuntimeError(
            "Capture was saved but could not be queued for analysis; "
            f"no acknowledgement will be sent. Raw capture: {raw_capture_path}. "
            f"Queue error: {exc}"
        ) from exc
    print(f"Raw capture saved and queued: {raw_capture_path}")
    return task


def send_capture_ok(uart) -> None:
    """Acknowledge only a raw capture already accepted by the worker."""

    try:
        written = uart.write(CAPTURE_OK_COMMAND)
    except (serial.SerialException, OSError) as exc:
        raise RuntimeError(f"Could not send CAPTURE_OK to the STM: {exc}") from exc
    if written != len(CAPTURE_OK_COMMAND):
        raise RuntimeError(
            "Could not send the complete CAPTURE_OK command "
            f"({written}/{len(CAPTURE_OK_COMMAND)} bytes)."
        )
    print("UART TX: CAPTURE_OK")


def prioritize_capture_crack_over_rust(result, crack_result):
    """Return a capture-only rust result with crack-overlap pixels set to Good."""

    class_map = result.class_map
    if class_map is None:
        raise ValueError("Capture rust result does not contain a class map.")
    if class_map.shape != crack_result.mask.shape:
        raise ValueError(
            "Capture rust and crack masks must have the same shape; "
            f"got {class_map.shape} and {crack_result.mask.shape}."
        )

    corrected_class_map = class_map.copy()
    corrected_class_map[crack_result.mask != 0] = 0
    height, width = corrected_class_map.shape
    corrected = result_from_class_map(
        corrected_class_map,
        LetterboxTransform(
            original_height=height,
            original_width=width,
            resized_height=height,
            resized_width=width,
            top=0,
            left=0,
        ),
        result.method,
    )
    corrected.status = result.status
    return corrected


def capture_and_analyze(
    frame,
    phase: str,
    phase_sequence: int | None,
    trigger: str,
    detector: RustDetector,
    workbook: InspectionWorkbook,
    output_directory: Path = CAPTURE_DIRECTORY,
    *,
    crack_detector: HrSegNetCrackDetector | None = None,
    crack_zones: set[int] | None = None,
    raw_capture_path: Path | None = None,
    captured_at: datetime | None = None,
):
    captured_at = captured_at or datetime.now()
    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        print(
            "Capture frame must be the native 1280x720 BGR camera frame.",
            file=sys.stderr,
        )
        return None
    try:
        result = detector.detect(frame)
    except RuntimeError as exc:
        print(f"Capture inference failed: {exc}", file=sys.stderr)
        return None

    if result.status != "ready":
        print(
            f"Capture inference did not complete synchronously: {result.status}",
            file=sys.stderr,
        )
        return None

    if not result.method.startswith(TEACHER_DETECTOR_PREFIX):
        print(
            "Capture result was not produced by the TensorRT teacher: "
            f"{result.method}",
            file=sys.stderr,
        )
        return None

    zone_number = (
        phase_sequence if phase in (INITIAL_PHASE, RESCAN_PHASE) else None
    )
    crack_status = "disabled"
    crack_method = None
    crack_detected = None
    crack_pixels = None
    crack_inspected_pixels = None
    crack_ratio = None
    crack_result = None
    if crack_detector is not None:
        try:
            crack_result = crack_detector.detect(frame)
        except RuntimeError as exc:
            print(f"Crack capture inference failed: {exc}", file=sys.stderr)
            return None
        if crack_result.status != "ready":
            print(
                "Crack capture inference did not complete synchronously: "
                f"{crack_result.status}",
                file=sys.stderr,
            )
            return None
        if not crack_result.method.startswith(HRSEGNET_CAPTURE_CRACK_DETECTOR_PREFIX):
            print(
                "Crack result was not produced by the capture HrSegNet TensorRT "
                f"detector: {crack_result.method}",
                file=sys.stderr,
            )
            return None
        if crack_result.mask.shape != frame.shape[:2]:
            print(
                "Crack inference mask shape does not match the capture frame.",
                file=sys.stderr,
            )
            return None
        crack_inspected_pixels = int(crack_result.mask.size)
        if crack_inspected_pixels <= 0:
            print("Crack inference returned an empty mask.", file=sys.stderr)
            return None
        crack_pixels = int(np.count_nonzero(crack_result.mask))
        crack_ratio = crack_pixels / crack_inspected_pixels
        crack_detected = crack_pixels > 0
        crack_status = "ready"
        crack_method = crack_result.method
        try:
            result = prioritize_capture_crack_over_rust(result, crack_result)
        except ValueError as exc:
            print(f"Could not apply capture crack priority: {exc}", file=sys.stderr)
            return None
        if crack_detected and zone_number is not None and crack_zones is not None:
            crack_zones.add(zone_number)

    inspected_pixels = int(result.mask.size)
    if inspected_pixels <= 0:
        print("Capture inference returned an empty mask.", file=sys.stderr)
        return None
    rust_pixels = int((result.mask != 0).sum())
    rust_ratio = rust_pixels / inspected_pixels
    display = annotate(frame, result)
    if crack_detector is not None:
        display = annotate_cracks(display, crack_result, zone_number)

    try:
        capture_path = save_capture(
            display,
            phase,
            phase_sequence,
            output_directory,
            captured_at,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not save segmented capture: {exc}", file=sys.stderr)
        return None

    try:
        overall_ratio = workbook.append_capture(
            capture_path=capture_path,
            phase=phase,
            trigger=trigger,
            detector=result.method,
            rust_pixels=rust_pixels,
            inspected_pixels=inspected_pixels,
            crack_status=crack_status,
            crack_detector=crack_method,
            crack_detected=crack_detected,
            crack_pixels=crack_pixels,
            crack_inspected_pixels=crack_inspected_pixels,
            captured_at=captured_at,
        )
    except Exception as exc:
        print(f"Could not record capture in {workbook.path}: {exc}", file=sys.stderr)
        return None

    if raw_capture_path is not None:
        try:
            dashboard_manifest = export_capture_record(
                frame=frame,
                rust_result=result,
                crack_result=crack_result,
                workbook=workbook,
                capture_path=capture_path,
                raw_capture_path=raw_capture_path,
                phase=phase,
                phase_sequence=phase_sequence,
                trigger=trigger,
                captured_at=captured_at,
                output_directory=output_directory,
            )
            print(f"Dashboard export updated: {dashboard_manifest}")
        except Exception as exc:
            print(
                "Dashboard export failed; XLSX and captured images remain valid: "
                f"{exc}",
                file=sys.stderr,
            )

    notice = (
        f"captured rust={rust_ratio * 100:.2f}% "
        f"phase_overall={overall_ratio * 100:.2f}%"
    )
    if crack_status == "ready":
        crack_state = "yes" if crack_detected else "no"
        notice += f" crack={crack_state} ({crack_ratio * 100:.3f}%)"
        if zone_number is not None:
            notice += f" rail_section={zone_number}"
    cv2.putText(
        display,
        notice,
        (24, max(36, display.shape[0] - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (60, 220, 90),
        2,
        cv2.LINE_AA,
    )
    print(f"Segmented capture saved: {capture_path}")
    print(notice)
    if crack_detected and zone_number is not None:
        print(
            f"크랙 후보 감지: {zone_number}번 레일 구간 "
            f"(Rail Section 인덱스 {zone_number - 1})"
        )
    return display


def capture_top_crack_only(
    frame: np.ndarray,
    task: CaptureAnalysisTask,
    crack_detector: HrSegNetCrackDetector | None,
    workbook: InspectionWorkbook,
    output_directory: Path = CAPTURE_DIRECTORY,
):
    """Analyze and report the top-camera capture without rust inference."""

    if crack_detector is None:
        print("Top capture requires the full-frame HrSegNet detector.", file=sys.stderr)
        return None
    if not isinstance(frame, np.ndarray) or frame.shape != (
        FRAME_HEIGHT,
        FRAME_WIDTH,
        3,
    ):
        print(
            "Top capture frame must be the native 1280x720 BGR camera frame.",
            file=sys.stderr,
        )
        return None
    try:
        crack_result = crack_detector.detect(frame)
    except RuntimeError as exc:
        print(f"Top crack capture inference failed: {exc}", file=sys.stderr)
        return None
    if (
        crack_result.status != "ready"
        or not crack_result.method.startswith(HRSEGNET_CAPTURE_CRACK_DETECTOR_PREFIX)
        or crack_result.mask.shape != frame.shape[:2]
    ):
        print("Top crack capture result failed its runtime contract.", file=sys.stderr)
        return None

    zone_number = (
        task.phase_sequence
        if task.phase in (INITIAL_PHASE, RESCAN_PHASE)
        else None
    )
    display = annotate_cracks(frame, crack_result, zone_number)
    notice = (
        "top crack="
        f"{'yes' if crack_result.detected else 'no'} "
        f"({crack_result.crack_ratio * 100:.3f}%)"
    )
    cv2.putText(
        display,
        notice,
        (24, max(36, display.shape[0] - 24)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255) if crack_result.detected else (60, 220, 90),
        2,
        cv2.LINE_AA,
    )
    try:
        capture_path = save_capture(
            display,
            task.phase,
            task.phase_sequence,
            output_directory / "top_crack",
            task.captured_at,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Could not save top crack capture: {exc}", file=sys.stderr)
        return None
    crack_pixels = int(np.count_nonzero(crack_result.mask))
    crack_inspected_pixels = int(crack_result.mask.size)
    try:
        workbook.append_top_crack_capture(
            raw_capture_path=task.raw_capture_path,
            capture_path=capture_path,
            phase=task.phase,
            phase_sequence=task.phase_sequence,
            trigger=task.trigger,
            crack_detector=crack_result.method,
            crack_detected=crack_pixels > 0,
            crack_pixels=crack_pixels,
            crack_inspected_pixels=crack_inspected_pixels,
            captured_at=task.captured_at,
        )
    except Exception as exc:
        print(
            f"Could not record top crack capture in {workbook.path}: {exc}",
            file=sys.stderr,
        )
        return None
    try:
        dashboard_manifest = export_top_crack_record(
            frame=frame,
            crack_result=crack_result,
            workbook=workbook,
            capture_path=capture_path,
            raw_capture_path=task.raw_capture_path,
            phase=task.phase,
            phase_sequence=task.phase_sequence,
            trigger=task.trigger,
            captured_at=task.captured_at,
            output_directory=output_directory,
        )
        print(f"Dashboard TOP export updated: {dashboard_manifest}")
    except Exception as exc:
        print(
            "Dashboard TOP export failed; XLSX and captured images remain valid: "
            f"{exc}",
            file=sys.stderr,
        )
    print(f"Top crack capture saved: {capture_path}")
    print(notice)
    return display


class CaptureAnalysisWorker:
    """Run capture inference and XLSX writes on one FIFO worker thread."""

    _STOP = object()

    def __init__(
        self,
        detector: RustDetector,
        workbook: InspectionWorkbook,
        output_directory: Path = CAPTURE_DIRECTORY,
        *,
        crack_detector: HrSegNetCrackDetector | None = None,
        queue_capacity: int = CAPTURE_ANALYSIS_QUEUE_CAPACITY,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("Capture analysis queue capacity must be positive.")
        self.detector = detector
        self.workbook = workbook
        self.output_directory = output_directory
        self.crack_detector = crack_detector
        self._tasks: queue.Queue[object] = queue.Queue(maxsize=queue_capacity)
        self._thread = threading.Thread(
            target=self._run,
            name="capture-analysis",
            # A native TensorRT call cannot be cancelled safely from Python.  Keep
            # the normal path joined below, but do not let a hung CUDA worker keep
            # the UART mission process alive after the bounded fail-safe timeout.
            # The raw JPEG is saved before a task is submitted, so the input is
            # still recoverable if this exceptional path is taken.
            daemon=True,
        )
        self._state_lock = threading.Lock()
        self._failure: str | None = None
        self._skipped_raw_paths: list[Path] = []
        self._pending_tasks = 0
        self._idle = threading.Event()
        self._idle.set()
        self._started = False
        self._stopped = False
        self._stop_requested = False

    def start(self) -> None:
        if self._started:
            raise RuntimeError("Capture analysis worker was already started.")
        self._thread.start()
        self._started = True

    def submit(self, task: CaptureAnalysisTask) -> None:
        if not isinstance(task, CaptureAnalysisTask):
            raise TypeError("Capture analysis worker requires a capture task.")
        if not self._started:
            raise RuntimeError("Capture analysis worker is not running.")
        if self._stopped:
            raise RuntimeError("Capture analysis worker is already stopped.")
        failure = self.failure_message()
        if failure is not None:
            raise RuntimeError(failure)
        with self._state_lock:
            self._pending_tasks += 1
            self._idle.clear()
        try:
            self._tasks.put_nowait(task)
        except queue.Full as exc:
            with self._state_lock:
                self._pending_tasks -= 1
                if self._pending_tasks == 0:
                    self._idle.set()
            raise RuntimeError(
                "Capture analysis queue is full "
                f"({self._tasks.maxsize}); raw capture remains at "
                f"{task.raw_capture_path}."
            ) from exc

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        if timeout is not None and timeout < 0.0:
            raise ValueError("Capture analysis idle timeout must not be negative.")
        if not self._started:
            return True
        return self._idle.wait(timeout)

    def failure_message(self) -> str | None:
        with self._state_lock:
            if self._failure is None:
                return None
            message = self._failure
            if self._skipped_raw_paths:
                skipped = ", ".join(map(str, self._skipped_raw_paths))
                message += (
                    f" Skipped {len(self._skipped_raw_paths)} queued raw capture(s) "
                    f"after the failure: {skipped}."
                )
            return message

    def shutdown(
        self,
        timeout: float = CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS,
    ) -> str | None:
        if timeout < 0.0:
            raise ValueError("Capture analysis shutdown timeout must not be negative.")
        if self._started and not self._stopped:
            deadline = time.monotonic() + timeout
            if not self._stop_requested:
                try:
                    self._tasks.put(self._STOP, timeout=timeout)
                    self._stop_requested = True
                except queue.Full:
                    self._record_worker_failure(
                        "Capture analysis worker queue did not accept its stop "
                        f"request within {timeout:.1f} seconds."
                    )
            remaining = max(0.0, deadline - time.monotonic())
            self._thread.join(remaining)
            if self._thread.is_alive():
                self._record_worker_failure(
                    "Capture analysis worker did not stop within "
                    f"{timeout:.1f} seconds. Raw captures remain recoverable; "
                    "the mission process is stopping."
                )
            else:
                self._stopped = True
        return self.failure_message()

    def _record_worker_failure(self, message: str) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = message

    def _record_failure(self, task: CaptureAnalysisTask, exc: Exception) -> None:
        with self._state_lock:
            if self._failure is None:
                self._failure = (
                    "Background capture analysis failed for "
                    f"{task.raw_capture_path}: {exc}"
                )
            else:
                self._skipped_raw_paths.append(task.raw_capture_path)

    def _finish_task(self) -> None:
        with self._state_lock:
            self._pending_tasks -= 1
            if self._pending_tasks < 0:
                raise RuntimeError("Capture analysis pending task count became negative.")
            if self._pending_tasks == 0:
                self._idle.set()

    def _run(self) -> None:
        while True:
            queued_item = self._tasks.get()
            is_task = queued_item is not self._STOP
            try:
                if queued_item is self._STOP:
                    return
                task = queued_item
                assert isinstance(task, CaptureAnalysisTask)
                if self.failure_message() is not None:
                    with self._state_lock:
                        self._skipped_raw_paths.append(task.raw_capture_path)
                    continue
                try:
                    frame = cv2.imread(str(task.raw_capture_path), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError("Could not read the saved raw JPEG.")
                    if task.camera_role == TOP_CAMERA_ROLE:
                        analyzed = capture_top_crack_only(
                            frame,
                            task,
                            self.crack_detector,
                            self.workbook,
                            self.output_directory,
                        )
                    else:
                        analyzed = capture_and_analyze(
                            frame,
                            task.phase,
                            task.phase_sequence,
                            task.trigger,
                            self.detector,
                            self.workbook,
                            self.output_directory,
                            crack_detector=self.crack_detector,
                            raw_capture_path=task.raw_capture_path,
                            captured_at=task.captured_at,
                        )
                    if analyzed is None:
                        raise RuntimeError(
                            "Capture inference, analyzed-image save, or report "
                            "write did not complete."
                        )
                except Exception as exc:
                    self._record_failure(task, exc)
            finally:
                self._tasks.task_done()
                if is_task:
                    self._finish_task()


def prepare_realtime_transition(
    cleaner_controller: CleanerController,
    pump_controller: PumpController,
    analysis_worker: CaptureAnalysisWorker,
    *,
    additional_pairs: tuple[
        tuple[CleanerController, PumpController], ...
    ] = (),
) -> None:
    """Keep cleaning off and require a bounded capture-analysis drain."""

    first_error = None
    for cleaner, pump in (
        (cleaner_controller, pump_controller),
        *additional_pairs,
    ):
        for controller in (cleaner, pump):
            try:
                controller.force_off()
            except RuntimeError as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error
    if not analysis_worker.wait_until_idle(
        CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS
    ):
        raise RuntimeError(
            "Capture analysis did not become idle within "
            f"{CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS:.1f} seconds; "
            "cleaning remains OFF and the mission is stopping."
        )
    worker_failure = analysis_worker.failure_message()
    if worker_failure is not None:
        raise RuntimeError(worker_failure)


def close_runtime_resources(
    *,
    display=None,
    uart=None,
    student_detector=None,
    teacher_detector=None,
    capture_crack_detector=None,
    realtime_crack_detector=None,
    obstacle_detector=None,
    camera=None,
    top_camera=None,
    close_windows: bool = False,
) -> None:
    resources = (
        ("display", None if display is None else display.close),
        ("UART", None if uart is None else uart.close),
        (
            "student detector",
            None if student_detector is None else student_detector.close,
        ),
        (
            "teacher detector",
            None if teacher_detector is None else teacher_detector.close,
        ),
        (
            "capture crack detector",
            None if capture_crack_detector is None else capture_crack_detector.close,
        ),
        (
            "realtime crack detector",
            None if realtime_crack_detector is None else realtime_crack_detector.close,
        ),
        (
            "obstacle detector",
            None if obstacle_detector is None else obstacle_detector.close,
        ),
        ("camera", None if camera is None else camera.release),
        ("top camera", None if top_camera is None else top_camera.release),
        ("OpenCV windows", cv2.destroyAllWindows if close_windows else None),
    )
    for resource_name, close_resource in resources:
        if close_resource is None:
            continue
        try:
            close_resource()
        except Exception as exc:
            print(f"Could not close {resource_name}: {exc}", file=sys.stderr)


def run_dual_camera_realtime_test(args: argparse.Namespace) -> int:
    """Run the dual-camera pipeline in simulated or explicit UART test mode."""

    try:
        validate_distinct_camera_devices(
            args.side_camera_device,
            args.top_camera_device,
        )
        validate_tf32_runtime_environment()
        student_path, student_sha, optimized_rust = configured_student_engine(args)
        if student_path is None:
            raise ValueError("Dual-camera test requires a realtime rust engine.")
        student_path, student_sha = validate_engine(
            student_path,
            student_sha,
            (
                "--optimized-student-engine-sha256"
                if optimized_rust
                else "--student-engine-sha256"
            ),
        )
        crack_path, crack_sha = validate_engine(
            args.realtime_hrsegnet_crack_engine,
            args.realtime_hrsegnet_crack_engine_sha256,
            "--realtime-hrsegnet-crack-engine-sha256",
        )
        obstacle_path, obstacle_sha = validate_engine(
            args.obstacle_engine,
            args.obstacle_engine_sha256,
            "--obstacle-engine-sha256",
        )
        if len({student_sha, crack_sha, obstacle_sha}) != 3:
            raise ValueError("Dual-camera realtime TensorRT engines must be different.")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    student_detector = None
    crack_detector = None
    obstacle_detector = None
    side_camera = None
    top_camera = None
    display_controller = None
    uart = None
    actuator_arbiter = None
    actuator_shutdown_failure = None
    uart_safe_to_close = True
    uart_enabled = bool(getattr(args, "realtime_test_uart", False))
    front_cleaner = CleanerController(None, FRONT_ACTUATOR_COMMANDS)
    front_pump = PumpController(None, FRONT_ACTUATOR_COMMANDS)
    side_cleaner = CleanerController(None, SIDE_ACTUATOR_COMMANDS)
    side_pump = PumpController(None, SIDE_ACTUATOR_COMMANDS)
    control_off_logger = DualControlOffTransitionLogger()
    try:
        if optimized_rust:
            student_detector = OptimizedRustDetector(student_path, student_sha)
        else:
            student_detector = RustDetector(
                student_path,
                STUDENT_PROFILE,
                student_sha,
                gpu_argmax=True,
            )
        crack_detector = HrSegNetCrackDetector(
            crack_path,
            crack_sha,
            args.realtime_hrsegnet_crack_probability_threshold,
            args.realtime_hrsegnet_crack_min_component_pixels,
            role="realtime",
        )
        obstacle_detector = ObstacleDetector(
            obstacle_path,
            obstacle_sha,
            args.obstacle_confidence_threshold,
        )
        side_camera = open_latest_frame_camera(args.side_camera_device)
        try:
            top_camera = open_latest_frame_camera(args.top_camera_device)
        except Exception:
            side_camera.release()
            side_camera = None
            raise

        side_warmup = side_camera.read_latest()
        top_warmup = top_camera.read_latest()
        if side_warmup.frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                f"Side camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
            )
        if top_warmup.frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                f"Top camera must provide {FRAME_WIDTH}x{FRAME_HEIGHT} BGR frames."
            )
        side_rust_roi, side_crack_roi = extract_realtime_control_rois(
            side_warmup.frame
        )
        top_obstacle_roi, top_crack_roi = extract_realtime_control_rois(
            top_warmup.frame
        )
        student_detector.detect(side_rust_roi)
        crack_detector.detect(side_crack_roi)
        obstacle_detector.detect(top_obstacle_roi)
        crack_detector.detect(top_crack_roi)

        if uart_enabled:
            try:
                uart = serial.Serial(
                    args.serial_port,
                    args.baud_rate,
                    timeout=0,
                    write_timeout=UART_WRITE_TIMEOUT_SECONDS,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not open UART {args.serial_port}: {exc}"
                ) from exc
            enter_uart_actuator_test(uart)
            actuator_arbiter = RealtimeActuatorArbiter(uart)
            actuator_arbiter.start()

        side_inference = AlternatingRealtimeInference(
            student_detector,
            crack_detector,
            multitask_enabled=False,
            hybrid_enabled=False,
        )
        top_inference = AlternatingTopRealtimeInference(
            obstacle_detector,
            crack_detector,
        )
        front_control_history = FrontControlHistory()
        side_control_history = RealtimeControlHistory()
        front_command_hold = RealtimeActuatorDecisionHold()
        side_command_hold = RealtimeActuatorDecisionHold()
        side_display_cache = RealtimeDisplayCache()
        top_obstacle_display = None
        top_crack_display = None
        last_side_sequence = side_warmup.sequence
        last_top_sequence = top_warmup.sequence
        side_preview_frame = side_warmup.frame
        top_preview_frame = top_warmup.frame
        last_actuator_state = None
        last_actuator_log_at = None
        headless = bool(getattr(args, "headless", False))
        display_rate_limiter = None
        if not headless:
            display_controller = open_latest_frame_display()
            display_rate_limiter = DisplayRateLimiter(DUAL_GUI_FRAME_RATE)

        mode_label = (
            f"UART:{args.serial_port}" if uart_enabled else "SIMULATED:NO-UART"
        )
        print(f"Dual-camera realtime test: {mode_label}")
        print(f"Side camera: {args.side_camera_device}")
        print(f"Top camera: {args.top_camera_device}")
        print(f"Side rust detector: {student_detector.method}")
        print(f"Shared side/top crack detector: {crack_detector.method}")
        print(f"Top obstacle detector: {obstacle_detector.method}")
        print(
            "Obstacle control: cropped 1280x240 y=0:240 ROI inference, confidence>="
            f"{args.obstacle_confidence_threshold:g}."
        )
        timing_logger = (
            DualTimingLogger()
            if bool(getattr(args, "dual_timing", False))
            else None
        )
        if timing_logger is not None:
            print(
                "[DUAL_TIMING] enabled: first "
                f"{DUAL_TIMING_WARMUP_SECONDS:g}s excluded; summaries every "
                f"{DUAL_TIMING_REPORT_INTERVAL_SECONDS:g}s."
            )

        while True:
            if actuator_arbiter is not None:
                actuator_arbiter.raise_if_failed()
            if display_controller is not None and display_controller.check_status():
                break
            loop_started_at = time.perf_counter()
            actuator_state_before_frame_read = None
            try:
                actuator_state_before_frame_read = current_dual_actuator_state(
                    front_cleaner,
                    front_pump,
                    side_cleaner,
                    side_pump,
                    arbiter=actuator_arbiter,
                )
                frame_read = read_dual_realtime_camera_frames(
                    side_camera,
                    top_camera,
                    after_side_sequence=last_side_sequence,
                    after_top_sequence=last_top_sequence,
                    side_inference=side_inference,
                    top_inference=top_inference,
                    side_cleaner_controller=side_cleaner,
                    side_pump_controller=side_pump,
                    front_cleaner_controller=front_cleaner,
                    front_pump_controller=front_pump,
                    actuator_arbiter=actuator_arbiter,
                )
                actuator_state_before_frame_read = None
                side_frame = frame_read.side_frame
                top_frame = frame_read.top_frame
                if side_frame is not None:
                    last_side_sequence = side_frame.sequence
                    side_preview_frame = side_frame.frame
                if top_frame is not None:
                    last_top_sequence = top_frame.sequence
                    top_preview_frame = top_frame.frame
                if frame_read.side_cache_expired:
                    side_inference.expire_cache()
                    side_control_history.prune()
                if frame_read.top_cache_expired:
                    top_inference.expire_cache()
                    front_control_history.prune()
                side_outcome = None
                top_outcome = None
                side_update_seconds = None
                top_update_seconds = None
                side_inputs_fresh = realtime_role_inputs_are_fresh(
                    side_frame,
                    side_inference,
                )
                top_inputs_fresh = realtime_role_inputs_are_fresh(
                    top_frame,
                    top_inference,
                )
                if side_inputs_fresh:
                    update_started_at = time.perf_counter()
                    side_rust_roi, side_crack_roi = extract_realtime_control_rois(
                        side_frame.frame
                    )
                    side_outcome = side_inference.process(
                        side_frame,
                        side_rust_roi,
                        side_crack_roi,
                    )
                    side_update_seconds = time.perf_counter() - update_started_at
                if top_inputs_fresh:
                    update_started_at = time.perf_counter()
                    top_obstacle_roi, top_crack_roi = extract_realtime_control_rois(
                        top_frame.frame
                    )
                    top_outcome = top_inference.process(
                        top_frame,
                        top_obstacle_roi,
                        top_crack_roi,
                    )
                    top_update_seconds = time.perf_counter() - update_started_at
                if actuator_arbiter is not None:
                    actuator_arbiter.raise_if_failed()

                hold_snapshot_at = time.monotonic()
                side_hold_snapshot = side_command_hold.diagnostic_snapshot(
                    now=hold_snapshot_at
                )
                front_hold_snapshot = front_command_hold.diagnostic_snapshot(
                    now=hold_snapshot_at
                )
                side_history_decision = None
                front_history_decision = None

                if frame_read.side_error is not None:
                    side_control_history.reset()
                    side_inference.reset()
                    side_display_cache.reset()
                    side_command_hold.clear()
                    clear_realtime_actuator_role(
                        side_cleaner,
                        side_pump,
                        role="side",
                        arbiter=actuator_arbiter,
                    )
                elif frame_read.side_waiting:
                    maintain_waiting_role_control(
                        side_cleaner,
                        side_pump,
                        history=side_control_history,
                        inference=side_inference,
                        role="side",
                        hold=side_command_hold,
                        arbiter=actuator_arbiter,
                    )
                elif not side_inputs_fresh:
                    side_control_history.prune()
                    side_inference.expire_cache()
                    side_display_cache.reset()
                    maintain_waiting_role_control(
                        side_cleaner,
                        side_pump,
                        history=side_control_history,
                        inference=side_inference,
                        role="side",
                        hold=side_command_hold,
                        arbiter=actuator_arbiter,
                    )
                else:
                    side_control_history.update(side_outcome)
                    if side_outcome_requires_immediate_off(
                        side_outcome,
                        history_ready=side_control_history.ready,
                    ):
                        side_command_hold.clear()
                        clear_realtime_actuator_role(
                            side_cleaner,
                            side_pump,
                            role="side",
                            arbiter=actuator_arbiter,
                        )
                    elif side_control_history.ready:
                        side_history_decision = side_control_history.decision()
                        publish_realtime_actuator_decision(
                            side_cleaner,
                            side_pump,
                            role="side",
                            history=side_control_history,
                            decision=side_history_decision,
                            hold=side_command_hold,
                            arbiter=actuator_arbiter,
                        )
                    else:
                        maintain_waiting_role_control(
                            side_cleaner,
                            side_pump,
                            history=side_control_history,
                            inference=side_inference,
                            role="side",
                            hold=side_command_hold,
                            arbiter=actuator_arbiter,
                        )

                if frame_read.top_error is not None:
                    front_control_history.reset()
                    top_inference.reset()
                    top_obstacle_display = None
                    top_crack_display = None
                    front_command_hold.clear()
                    clear_realtime_actuator_role(
                        front_cleaner,
                        front_pump,
                        role="front",
                        arbiter=actuator_arbiter,
                    )
                elif frame_read.top_waiting:
                    maintain_waiting_role_control(
                        front_cleaner,
                        front_pump,
                        history=front_control_history,
                        inference=top_inference,
                        role="front",
                        hold=front_command_hold,
                        arbiter=actuator_arbiter,
                    )
                elif not top_inputs_fresh:
                    front_control_history.prune()
                    top_inference.expire_cache()
                    top_obstacle_display = None
                    top_crack_display = None
                    maintain_waiting_role_control(
                        front_cleaner,
                        front_pump,
                        history=front_control_history,
                        inference=top_inference,
                        role="front",
                        hold=front_command_hold,
                        arbiter=actuator_arbiter,
                    )
                else:
                    front_control_history.update(top_outcome)
                    if front_outcome_requires_immediate_off(top_outcome):
                        front_command_hold.clear()
                        clear_realtime_actuator_role(
                            front_cleaner,
                            front_pump,
                            role="front",
                            arbiter=actuator_arbiter,
                        )
                    elif front_control_history.ready:
                        front_history_decision = front_control_history.decision()
                        publish_realtime_actuator_decision(
                            front_cleaner,
                            front_pump,
                            role="front",
                            history=front_control_history,
                            decision=front_history_decision,
                            hold=front_command_hold,
                            arbiter=actuator_arbiter,
                        )
                    else:
                        maintain_waiting_role_control(
                            front_cleaner,
                            front_pump,
                            history=front_control_history,
                            inference=top_inference,
                            role="front",
                            hold=front_command_hold,
                            arbiter=actuator_arbiter,
                        )

                side_display_cache.update(side_outcome)
                if (
                    top_outcome is not None
                    and top_outcome.display_obstacle_result is not None
                ):
                    top_obstacle_display = top_outcome.display_obstacle_result
                if (
                    top_outcome is not None
                    and top_outcome.display_crack_result is not None
                ):
                    top_crack_display = top_outcome.display_crack_result

                actuator_state = current_dual_actuator_state(
                    front_cleaner,
                    front_pump,
                    side_cleaner,
                    side_pump,
                    arbiter=actuator_arbiter,
                )
                now = time.monotonic()
                front_off_diagnostic = None
                if control_off_logger.is_turning_off("front", actuator_state):
                    front_off_diagnostic = build_control_off_diagnostic(
                        role="front",
                        outcome=top_outcome,
                        history_decision=front_history_decision,
                        history_ready=front_control_history.ready_without_prune,
                        camera_error=frame_read.top_error is not None,
                        camera_error_detail=(
                            None
                            if frame_read.top_error is None
                            else str(frame_read.top_error)
                        ),
                        hold_snapshot=front_hold_snapshot,
                        hold=front_command_hold,
                        frame=top_frame,
                        now=now,
                    )
                side_off_diagnostic = None
                if control_off_logger.is_turning_off("side", actuator_state):
                    side_off_diagnostic = build_control_off_diagnostic(
                        role="side",
                        outcome=side_outcome,
                        history_decision=side_history_decision,
                        history_ready=side_control_history.ready_without_prune,
                        camera_error=frame_read.side_error is not None,
                        camera_error_detail=(
                            None
                            if frame_read.side_error is None
                            else str(frame_read.side_error)
                        ),
                        hold_snapshot=side_hold_snapshot,
                        hold=side_command_hold,
                        frame=side_frame,
                        now=now,
                    )
                for control_off_line in control_off_logger.observe(
                    actuator_state,
                    front=front_off_diagnostic,
                    side=side_off_diagnostic,
                ):
                    print(control_off_line)
                    print(control_off_line, flush=True)
                if (
                    actuator_state != last_actuator_state
                    or last_actuator_log_at is None
                    or now - last_actuator_log_at
                    >= REALTIME_TEST_ACTUATOR_LOG_INTERVAL_SECONDS
                ):
                    front_cleaner_label = (
                        "OFF"
                        if actuator_state[0] == 0.0
                        else f"{actuator_state[0]:g}%"
                    )
                    front_pump_label = "ON" if actuator_state[1] else "OFF"
                    side_cleaner_label = (
                        "OFF"
                        if actuator_state[2] == 0.0
                        else f"{actuator_state[2]:g}%"
                    )
                    side_pump_label = "ON" if actuator_state[3] else "OFF"
                    sent_label = "UART_TX_SENT " if uart_enabled else ""
                    print(
                        f"[dual-camera-test][{mode_label}] {sent_label}"
                        f"FRONT_CLEANER_PWM={front_cleaner_label} "
                        f"FRONT_WATER_PUMP={front_pump_label} "
                        f"SIDE_CLEANER_PWM={side_cleaner_label} "
                        f"SIDE_WATER_PUMP={side_pump_label}"
                        f"SIDE_WATER_PUMP={side_pump_label}",
                        flush=True,
                    )
                    last_actuator_state = actuator_state
                    last_actuator_log_at = now

                display_requested_stop = False
                if (
                    display_controller is not None
                    and display_rate_limiter is not None
                    and display_rate_limiter.should_render()
                ):
                    side_display = annotate_realtime_control_results(
                        side_preview_frame,
                        side_display_cache.rust_result,
                        side_display_cache.crack_result,
                    )
                    side_display = draw_realtime_roi_guide(
                        side_preview_frame,
                        side_display,
                    )
                    top_display = annotate_top_realtime_results(
                        top_preview_frame,
                        top_obstacle_display,
                        top_crack_display,
                    )
                    top_display = draw_realtime_roi_guide(
                        top_preview_frame,
                        top_display,
                        primary_roi_label="OBSTACLE CONTROL",
                    )
                    side_small = cv2.resize(side_display, (640, 360))
                    top_small = cv2.resize(top_display, (640, 360))
                    if not display_controller.submit(
                        np.ascontiguousarray(np.hstack((side_small, top_small)))
                    ):
                        display_requested_stop = True
                if timing_logger is not None:
                    timing_now = time.monotonic()
                    summary = timing_logger.record(
                        loop_seconds=time.perf_counter() - loop_started_at,
                        side_update_seconds=side_update_seconds,
                        top_update_seconds=top_update_seconds,
                        side_outcome=side_outcome,
                        top_outcome=top_outcome,
                        side_frame_age_seconds=(
                            None
                            if side_frame is None
                            else max(
                                0.0,
                                timing_now - side_frame.read_completed_at,
                            )
                        ),
                        top_frame_age_seconds=(
                            None
                            if top_frame is None
                            else max(
                                0.0,
                                timing_now - top_frame.read_completed_at,
                            )
                        ),
                        side_stale=(
                            frame_read.side_cache_expired
                            or (
                                not frame_read.side_waiting
                                and not side_inputs_fresh
                            )
                        ),
                        top_stale=(
                            frame_read.top_cache_expired
                            or (
                                not frame_read.top_waiting
                                and not top_inputs_fresh
                            )
                        ),
                        side_ready=side_control_history.ready,
                        front_ready=front_control_history.ready,
                        now=timing_now,
                    )
                    if summary is not None:
                        print(summary)
                if display_requested_stop:
                    break
            except (RuntimeError, TimeoutError, ValueError) as exc:
                actuator_state_before_off = actuator_state_before_frame_read
                if actuator_state_before_off is None:
                    actuator_state_before_off = current_dual_actuator_state(
                        front_cleaner,
                        front_pump,
                        side_cleaner,
                        side_pump,
                        arbiter=actuator_arbiter,
                    )
                if actuator_arbiter is not None:
                    actuator_arbiter.clear_all()
                else:
                    force_cleaning_safe_off(front_cleaner, front_pump)
                    force_cleaning_safe_off(side_cleaner, side_pump)
                runtime_reason, runtime_detail = classify_runtime_control_off(exc)
                for control_off_line in control_off_logger.observe_forced_off(
                    actuator_state_before_off,
                    reason=runtime_reason,
                    detail=runtime_detail,
                    now=time.monotonic(),
                ):
                    print(control_off_line, file=sys.stderr)
                front_control_history.reset()
                side_control_history.reset()
                front_command_hold.clear()
                side_command_hold.clear()
                print(f"Dual-camera realtime test failed: {exc}", file=sys.stderr)
                return 1
    except Exception as exc:
        actuator_state_before_off = current_dual_actuator_state(
            front_cleaner,
            front_pump,
            side_cleaner,
            side_pump,
            arbiter=actuator_arbiter,
        )
        runtime_reason, runtime_detail = classify_runtime_control_off(exc)
        for control_off_line in control_off_logger.observe_forced_off(
            actuator_state_before_off,
            reason=runtime_reason,
            detail=runtime_detail,
            now=time.monotonic(),
        ):
            print(control_off_line, file=sys.stderr)
        print(f"Dual-camera realtime test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if actuator_arbiter is not None:
            try:
                actuator_arbiter.close()
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                actuator_shutdown_failure = str(exc)
                uart_safe_to_close = not actuator_arbiter.worker_is_alive()
        if uart_safe_to_close:
            for controller in (
                front_cleaner,
                front_pump,
                side_cleaner,
                side_pump,
            ):
                try:
                    controller.force_off()
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
        if uart is not None and uart_safe_to_close:
            try:
                send_uart_test_command(uart, ACTUATOR_TEST_STOP_COMMAND)
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                actuator_shutdown_failure = str(exc)
        close_runtime_resources(
            display=display_controller,
            uart=uart if uart_safe_to_close else None,
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            obstacle_detector=obstacle_detector,
            camera=side_camera,
            top_camera=top_camera,
        )
    if actuator_shutdown_failure is not None:
        return 1
    return 0


def run_realtime_test(args: argparse.Namespace) -> int:
    """Run only the student rust and crack models on live camera frames."""

    if getattr(args, "dual_camera", False):
        return run_dual_camera_realtime_test(args)

    multitask_enabled = getattr(args, "realtime_multitask_engine", None) is not None
    hrsegnet_engine_argument = getattr(
        args, "realtime_hrsegnet_crack_engine", None
    )
    hrsegnet_engine_sha256_argument = getattr(
        args, "realtime_hrsegnet_crack_engine_sha256", None
    )
    hrsegnet_probability_threshold = getattr(
        args,
        "realtime_hrsegnet_crack_probability_threshold",
        HRSEGNET_DEFAULT_PROBABILITY_THRESHOLD,
    )
    hrsegnet_min_component_pixels = getattr(
        args,
        "realtime_hrsegnet_crack_min_component_pixels",
        HRSEGNET_DEFAULT_MIN_COMPONENT_PIXELS,
    )
    hrsegnet_enabled = hrsegnet_engine_argument is not None
    (
        configured_student_path,
        configured_student_sha256,
        optimized_rust_enabled,
    ) = configured_student_engine(args)
    hybrid_enabled = multitask_enabled and configured_student_path is not None
    if getattr(args, "no_crack", False):
        print("--realtime-test cannot be combined with --no-crack", file=sys.stderr)
        return 2
    if hrsegnet_enabled and hrsegnet_engine_sha256_argument is None:
        print(
            "--realtime-hrsegnet-crack-engine requires its approved "
            "--realtime-hrsegnet-crack-engine-sha256",
            file=sys.stderr,
        )
        return 2
    if (
        not multitask_enabled
        and not hrsegnet_enabled
        and args.realtime_crack_engine is None
    ):
        print(
            "--realtime-test requires --realtime-crack-engine or "
            "--realtime-multitask-engine or --realtime-hrsegnet-crack-engine",
            file=sys.stderr,
        )
        return 2

    try:
        if getattr(args, "dual_camera", False):
            validate_distinct_camera_devices(
                args.side_camera_device,
                args.top_camera_device,
            )
        validate_tf32_runtime_environment()
        if multitask_enabled:
            multitask_engine, multitask_engine_sha256 = validate_engine(
                args.realtime_multitask_engine,
                args.realtime_multitask_engine_sha256,
                "--realtime-multitask-engine-sha256",
            )
            if hybrid_enabled:
                student_engine, student_engine_sha256 = validate_engine(
                    configured_student_path,
                    configured_student_sha256,
                    (
                        "--optimized-student-engine-sha256"
                        if optimized_rust_enabled
                        else "--student-engine-sha256"
                    ),
                )
                if student_engine_sha256 == multitask_engine_sha256:
                    raise ValueError(
                        "Student rust and multitask TensorRT engines must be different."
                    )
            else:
                student_engine = multitask_engine
                student_engine_sha256 = multitask_engine_sha256
            realtime_crack_engine = multitask_engine
            realtime_crack_engine_sha256 = multitask_engine_sha256
        else:
            if configured_student_path is None:
                raise ValueError("A realtime student rust engine is required.")
            student_engine, student_engine_sha256 = validate_engine(
                configured_student_path,
                configured_student_sha256,
                (
                    "--optimized-student-engine-sha256"
                    if optimized_rust_enabled
                    else "--student-engine-sha256"
                ),
            )
            crack_engine_argument = (
                hrsegnet_engine_argument
                if hrsegnet_enabled
                else args.realtime_crack_engine
            )
            crack_sha256_argument = (
                hrsegnet_engine_sha256_argument
                if hrsegnet_enabled
                else args.realtime_crack_engine_sha256
            )
            crack_sha256_option = (
                "--realtime-hrsegnet-crack-engine-sha256"
                if hrsegnet_enabled
                else "--realtime-crack-engine-sha256"
            )
            realtime_crack_engine, realtime_crack_engine_sha256 = validate_engine(
                crack_engine_argument,
                crack_sha256_argument,
                crack_sha256_option,
            )
            if student_engine_sha256 == realtime_crack_engine_sha256:
                raise ValueError("Student and crack TensorRT engines must be different.")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    if hybrid_enabled:
        student_kind = "optimized student rust" if optimized_rust_enabled else "student rust"
        student_pin = "pinned" if configured_student_sha256 is not None else "unpinned"
        print(f"Realtime test {student_kind} engine ({student_pin}): {student_engine}")
        print(f"Realtime test student rust SHA-256: {student_engine_sha256}")
        print(f"Realtime test multitask crack engine (pinned): {multitask_engine}")
        print(f"Realtime test multitask crack SHA-256: {multitask_engine_sha256}")
    elif multitask_enabled:
        print(f"Realtime test optimized multitask engine (pinned): {student_engine}")
        print(f"Realtime test optimized multitask SHA-256: {student_engine_sha256}")
    else:
        student_pin = "pinned" if configured_student_sha256 is not None else "unpinned"
        crack_pin = (
            "pinned"
            if (
                hrsegnet_enabled
                or args.realtime_crack_engine_sha256 is not None
            )
            else "unpinned"
        )
        print(f"Realtime test student engine ({student_pin}): {student_engine}")
        print(f"Realtime test student SHA-256: {student_engine_sha256}")
        print(f"Realtime test crack engine ({crack_pin}): {realtime_crack_engine}")
        print(f"Realtime test crack SHA-256: {realtime_crack_engine_sha256}")

    student_detector = None
    crack_detector = None
    try:
        if hybrid_enabled:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
            crack_detector = OptimizedMultitaskDetector(
                realtime_crack_engine,
                realtime_crack_engine_sha256,
                args.realtime_crack_threshold,
                args.realtime_crack_min_component_pixels,
            )
        elif multitask_enabled:
            student_detector = OptimizedMultitaskDetector(
                student_engine,
                student_engine_sha256,
                args.realtime_crack_threshold,
                args.realtime_crack_min_component_pixels,
            )
        else:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
            if hrsegnet_enabled:
                crack_detector = HrSegNetCrackDetector(
                    realtime_crack_engine,
                    realtime_crack_engine_sha256,
                    hrsegnet_probability_threshold,
                    hrsegnet_min_component_pixels,
                    role="realtime",
                )
            else:
                crack_detector = CrackDetector(
                    realtime_crack_engine,
                    CRACK_REALTIME_PROFILE,
                    realtime_crack_engine_sha256,
                    args.realtime_crack_threshold,
                    args.realtime_crack_min_component_pixels,
                )
    except Exception as exc:
        print(f"Could not initialize realtime test models: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
        )
        return 2

    camera = None
    try:
        camera = open_latest_frame_camera(args.camera_index)
    except Exception as exc:
        print(f"Could not initialize camera: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            camera=camera,
        )
        return 1
    try:
        warmup_camera_frame = camera.read_latest()
        warmup_frame = warmup_camera_frame.frame
        warmup_water_roi, warmup_crack_roi = extract_realtime_control_rois(
            warmup_frame
        )
        detect_realtime_control(
            warmup_water_roi,
            warmup_crack_roi,
            student_detector,
            crack_detector,
            multitask_enabled=multitask_enabled,
            hybrid_enabled=hybrid_enabled,
        )
    except Exception as exc:
        print(f"Realtime test model warm-up failed: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            camera=camera,
        )
        return 1

    frame_height, frame_width = warmup_frame.shape[:2]
    print(f"Realtime student detector: {student_detector.method}")
    realtime_crack_method = (
        student_detector.crack_method
        if multitask_enabled and not hybrid_enabled
        else (
            crack_detector.crack_method if hybrid_enabled else crack_detector.method
        )
    )
    print(f"Realtime crack detector: {realtime_crack_method}")
    if hrsegnet_enabled:
        print(
            "Operator-selected pinned HrSegNet baseline: threshold="
            f"{hrsegnet_probability_threshold:g}, "
            "min_component_pixels="
            f"{hrsegnet_min_component_pixels}; field accuracy is not certified."
        )
    print(
        f"Camera index={args.camera_index}, requested={FRAME_WIDTH}x{FRAME_HEIGHT}"
        f"@{FRAME_RATE}, actual frame={frame_width}x{frame_height}, "
        f"rust ROI={warmup_water_roi.shape[1]}x{warmup_water_roi.shape[0]}, "
        f"crack ROI={warmup_crack_roi.shape[1]}x{warmup_crack_roi.shape[0]}"
    )
    headless = bool(getattr(args, "headless", False))
    if headless:
        headless_source = (
            "automatic: no DISPLAY/WAYLAND_DISPLAY"
            if getattr(args, "headless_auto", False)
            else "explicit --headless"
        )
        print(
            "Display mode: headless ("
            f"{headless_source}); preview and display-only overlays are disabled."
        )
        print(
            "Warm-up: 1 frame excluded. Camera inference and simulated control "
            "remain active; press Ctrl+C to stop."
        )
    else:
        print("Display mode: GUI")
        print("Warm-up: 1 frame excluded. Press Q/Esc to close the camera window.")

    measured_frames = 0
    first_decision_completed_at = None
    last_decision_completed_at = None
    total_control_latency_seconds = 0.0
    total_rust_seconds = 0.0
    total_crack_seconds = 0.0
    simulated_cleaner = CleanerController(None)
    simulated_pump = PumpController(None)
    realtime_inference = AlternatingRealtimeInference(
        student_detector,
        crack_detector,
        multitask_enabled=multitask_enabled,
        hybrid_enabled=hybrid_enabled,
    )
    realtime_control_history = RealtimeControlHistory()
    realtime_display_cache = RealtimeDisplayCache()
    last_camera_sequence = warmup_camera_frame.sequence
    last_simulated_state = None
    last_simulated_log_at = None
    display_controller = None
    if not headless:
        try:
            display_controller = open_latest_frame_display()
        except Exception as exc:
            print(f"Could not initialize display: {exc}", file=sys.stderr)
            close_runtime_resources(
                student_detector=student_detector,
                realtime_crack_detector=crack_detector,
                camera=camera,
            )
            return 1
    try:
        while True:
            if (
                display_controller is not None
                and display_controller.check_status()
            ):
                break
            camera_frame, cache_expired_while_waiting = read_realtime_camera_frame(
                camera,
                after_sequence=last_camera_sequence,
                inference=realtime_inference,
                cleaner_controller=simulated_cleaner,
                pump_controller=simulated_pump,
            )
            if cache_expired_while_waiting:
                realtime_control_history.reset()
            last_camera_sequence = camera_frame.sequence
            frame = camera_frame.frame

            try:
                water_roi, crack_roi = extract_realtime_control_rois(frame)
                if not realtime_camera_frame_is_fresh(camera_frame):
                    force_cleaning_safe_off(simulated_cleaner, simulated_pump)
                    realtime_control_history.reset()
                    outcome = None
                else:
                    outcome = realtime_inference.process(
                        camera_frame,
                        water_roi,
                        crack_roi,
                    )
                    if (
                        display_controller is not None
                        and display_controller.check_status()
                    ):
                        force_cleaning_safe_off(simulated_cleaner, simulated_pump)
                        realtime_control_history.reset()
                        break
                    realtime_control_history.update(outcome)
                    if outcome.ready and realtime_control_history.ready:
                        update_cleaning_actuators(
                            simulated_cleaner,
                            simulated_pump,
                            decision=realtime_control_history.decision(),
                        )
                    else:
                        force_cleaning_safe_off(simulated_cleaner, simulated_pump)
                if display_controller is not None:
                    realtime_display_cache.update(outcome)
                decision_completed_at = time.perf_counter()
                decision_completed_monotonic = time.monotonic()
                simulated_state = (
                    simulated_cleaner.duty_percent,
                    simulated_pump.is_on,
                )
                simulated_log_at = time.monotonic()
                if (
                    simulated_state != last_simulated_state
                    or last_simulated_log_at is None
                    or simulated_log_at - last_simulated_log_at
                    >= REALTIME_TEST_ACTUATOR_LOG_INTERVAL_SECONDS
                ):
                    cleaner_label = (
                        "OFF"
                        if simulated_state[0] == 0.0
                        else f"{simulated_state[0]:g}%"
                    )
                    pump_label = "ON" if simulated_state[1] else "OFF"
                    print(
                        "[realtime-test][SIMULATED:NO-UART] "
                        f"CLEANER_PWM={cleaner_label} WATER_PUMP={pump_label}"
                        f"CLEANER_PWM={cleaner_label} WATER_PUMP={pump_label}",
                        flush=True,
                    )
                    last_simulated_state = simulated_state
                    last_simulated_log_at = simulated_log_at

                if display_controller is not None:
                    annotated = annotate_realtime_control_results(
                        frame,
                        outcome.display_rust_result if outcome is not None else None,
                        outcome.display_crack_result if outcome is not None else None,
                    )
                    display = draw_realtime_roi_guide(frame, annotated)
                    display = draw_realtime_result_summary(
                        display,
                        realtime_display_cache.rust_result,
                        realtime_display_cache.crack_result,
                    )
            except (RuntimeError, ValueError) as exc:
                print(f"Realtime test inference failed: {exc}", file=sys.stderr)
                return 1

            measured_frames += 1
            if first_decision_completed_at is None:
                first_decision_completed_at = decision_completed_at
            last_decision_completed_at = decision_completed_at
            decision_span_seconds = max(
                0.0,
                last_decision_completed_at - first_decision_completed_at,
            )
            total_control_latency_seconds += max(
                0.0,
                decision_completed_monotonic - camera_frame.read_completed_at,
            )
            if outcome is not None:
                total_rust_seconds += outcome.rust_seconds
                total_crack_seconds += outcome.crack_seconds
            if display_controller is not None:
                completed_app_frames, total_app_seconds = (
                    display_controller.completed_statistics()
                )
                statistics_label = format_realtime_test_statistics(
                    measured_frames,
                    decision_span_seconds,
                    total_control_latency_seconds,
                    total_rust_seconds,
                    total_crack_seconds,
                    app_frame_count=completed_app_frames,
                    total_app_seconds=total_app_seconds,
                    multitask=multitask_enabled and not hybrid_enabled,
                )
                display = draw_realtime_test_statistics(display, statistics_label)
                if not display_controller.submit(display):
                    break
    except Exception as exc:
        print(f"Realtime test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if simulated_cleaner.is_on or simulated_pump.is_on:
            simulated_cleaner.force_off()
            simulated_pump.force_off()
            print(
                "[realtime-test][SIMULATED:NO-UART] "
                "CLEANER_PWM=OFF WATER_PUMP=OFF (test shutdown)"
            )
        close_runtime_resources(
            display=display_controller,
            student_detector=student_detector,
            realtime_crack_detector=crack_detector,
            camera=camera,
        )
    return 0


def run_capture_test(args: argparse.Namespace) -> int:
    """Run only the full-frame capture models when the operator presses S."""

    if getattr(args, "headless", False):
        print(
            "--capture-test requires a graphical display because S/Q/Esc are "
            "interactive. Use normal operation for headless STM-triggered captures.",
            file=sys.stderr,
        )
        return 2

    try:
        if getattr(args, "dual_camera", False):
            validate_distinct_camera_devices(
                args.side_camera_device,
                args.top_camera_device,
            )
        validate_tf32_runtime_environment()
        teacher_engine, teacher_engine_sha256 = validate_engine(
            args.teacher_engine,
            args.teacher_engine_sha256,
            "--teacher-engine-sha256",
        )
        capture_crack_engine, capture_crack_engine_sha256 = validate_engine(
            args.capture_hrsegnet_crack_engine,
            args.capture_hrsegnet_crack_engine_sha256,
            "--capture-hrsegnet-crack-engine-sha256",
        )
        if teacher_engine_sha256 == capture_crack_engine_sha256:
            raise ValueError("Capture rust and crack engines must be different.")
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    rust_pin = "pinned" if args.teacher_engine_sha256 is not None else "unpinned"
    crack_pin = (
        "pinned"
        if args.capture_hrsegnet_crack_engine_sha256 is not None
        else "unpinned"
    )
    print(f"Capture test rust engine ({rust_pin}): {teacher_engine}")
    print(f"Capture test rust SHA-256: {teacher_engine_sha256}")
    print(f"Capture test crack engine ({crack_pin}): {capture_crack_engine}")
    print(f"Capture test crack SHA-256: {capture_crack_engine_sha256}")

    teacher_detector = None
    capture_crack_detector = None
    try:
        teacher_detector = RustDetector(
            teacher_engine,
            TEACHER_PROFILE,
            teacher_engine_sha256,
        )
        capture_crack_detector = HrSegNetCrackDetector(
            capture_crack_engine,
            capture_crack_engine_sha256,
            args.capture_hrsegnet_crack_probability_threshold,
            args.capture_hrsegnet_crack_min_component_pixels,
            role="capture",
        )
    except Exception as exc:
        print(f"Could not initialize capture test models: {exc}", file=sys.stderr)
        close_runtime_resources(
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
        )
        return 2

    camera = None
    top_camera = None
    try:
        camera_source = (
            args.side_camera_device
            if getattr(args, "dual_camera", False)
            else args.camera_index
        )
        camera = open_latest_frame_camera(camera_source)
        if getattr(args, "dual_camera", False):
            top_camera = open_latest_frame_camera(args.top_camera_device)
        warmup_camera_frame = camera.read_latest()
        warmup_frame = warmup_camera_frame.frame
        warmup_top_frame = (
            top_camera.read_latest().frame
            if getattr(args, "dual_camera", False)
            else None
        )
        if warmup_frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
            raise RuntimeError(
                "Capture test requires the native 1280x720 BGR camera frame; "
                f"received {warmup_frame.shape}."
            )
        if warmup_top_frame is not None and warmup_top_frame.shape != (
            FRAME_HEIGHT,
            FRAME_WIDTH,
            3,
        ):
            raise RuntimeError(
                "Top capture test camera must provide a native 1280x720 BGR frame."
            )
        teacher_detector.detect(warmup_frame)
        capture_crack_detector.detect(warmup_frame)
        if warmup_top_frame is not None:
            capture_crack_detector.detect(warmup_top_frame)
    except Exception as exc:
        print(f"Capture test model warm-up failed: {exc}", file=sys.stderr)
        close_runtime_resources(
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    print(f"Capture rust detector: {teacher_detector.method}")
    print(f"Capture crack detector: {capture_crack_detector.method}")
    print(
        "Capture HrSegNet baseline: threshold="
        f"{args.capture_hrsegnet_crack_probability_threshold:g}, "
        "min_component_pixels="
        f"{args.capture_hrsegnet_crack_min_component_pixels}; "
        "operator-selected and not field-calibrated."
    )
    if getattr(args, "dual_camera", False):
        print(f"Capture side camera: {args.side_camera_device}")
        print(f"Capture top camera: {args.top_camera_device}")
        print(
            "Full-frame input: 1280x720. S runs side rust+crack and top "
            "crack-only; Q/Esc closes."
        )
    else:
        print("Full-frame input: 1280x720. Press S to infer; Q/Esc to close.")
    result_display = None
    last_camera_sequence = warmup_camera_frame.sequence
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        while True:
            try:
                camera_frame = camera.read_latest(
                    after_sequence=last_camera_sequence,
                    timeout=LATEST_FRAME_TIMEOUT_SECONDS,
                )
            except (RuntimeError, TimeoutError) as exc:
                print(f"Capture test camera stream failed: {exc}", file=sys.stderr)
                return 1
            last_camera_sequence = camera_frame.sequence
            frame = camera_frame.frame
            if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH, 3):
                print(
                    "Capture test requires the native 1280x720 BGR camera frame; "
                    f"received {frame.shape}.",
                    file=sys.stderr,
                )
                return 1

            display = draw_roi_guide(
                frame,
                result_display if result_display is not None else None,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key != ord("s"):
                continue

            try:
                total_started = time.perf_counter()
                top_capture_frame = None
                if getattr(args, "dual_camera", False):
                    side_capture, top_capture = read_stopped_capture_pair(
                        camera,
                        top_camera,
                    )
                    frame = side_capture.frame
                    top_capture_frame = top_capture.frame
                    last_camera_sequence = side_capture.sequence
                rust_started = total_started
                rust_result = teacher_detector.detect(frame)
                rust_finished = time.perf_counter()
                crack_result = capture_crack_detector.detect(frame)
                crack_finished = time.perf_counter()
                rust_result = prioritize_capture_crack_over_rust(
                    rust_result,
                    crack_result,
                )
                result_display = annotate(frame, rust_result)
                result_display = annotate_cracks(
                    result_display,
                    crack_result,
                    None,
                )
                if top_capture_frame is not None:
                    top_crack_started = time.perf_counter()
                    top_crack_result = capture_crack_detector.detect(
                        top_capture_frame
                    )
                    top_crack_finished = time.perf_counter()
                    top_display = annotate_cracks(
                        top_capture_frame,
                        top_crack_result,
                        None,
                    )
                    result_display = np.ascontiguousarray(
                        np.hstack(
                            (
                                cv2.resize(result_display, (640, 360)),
                                cv2.resize(top_display, (640, 360)),
                            )
                        )
                    )
            except Exception as exc:
                print(f"Capture test inference failed: {exc}", file=sys.stderr)
                return 1

            rust_ms = (rust_finished - rust_started) * 1000.0
            crack_ms = (crack_finished - rust_finished) * 1000.0
            total_ms = (crack_finished - total_started) * 1000.0
            crack_label = "YES" if crack_result.detected else "NO"
            print(
                "[capture-test] "
                f"rust={rust_ms:.1f} ms crack={crack_ms:.1f} ms "
                f"total={total_ms:.1f} ms | "
                f"rust_ratio={rust_result.rust_ratio * 100:.2f}% "
                f"crack_detected={crack_label} "
                f"crack_ratio={crack_result.crack_ratio * 100:.2f}%"
            )
            if top_capture_frame is not None:
                print(
                    "[capture-test][top] crack="
                    f"{(top_crack_finished - top_crack_started) * 1000.0:.1f} ms "
                    f"detected={'YES' if top_crack_result.detected else 'NO'} "
                    f"ratio={top_crack_result.crack_ratio * 100:.2f}%"
                )
    except Exception as exc:
        print(f"Capture test failed: {exc}", file=sys.stderr)
        return 1
    finally:
        close_runtime_resources(
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            camera=camera,
            top_camera=top_camera,
            close_windows=True,
        )
    return 0


def main() -> int:
    args = parse_args()
    if getattr(args, "capture_test", False):
        return run_capture_test(args)
    if getattr(args, "realtime_test", False):
        return run_realtime_test(args)
    headless = bool(getattr(args, "headless", False))
    dual_camera_enabled = bool(getattr(args, "dual_camera", False))
    multitask_enabled = getattr(args, "realtime_multitask_engine", None) is not None
    hrsegnet_enabled = (
        getattr(args, "realtime_hrsegnet_crack_engine", None) is not None
    )
    capture_hrsegnet_enabled = (
        getattr(args, "capture_hrsegnet_crack_engine", None) is not None
    )
    (
        configured_student_path,
        configured_student_sha256,
        optimized_rust_enabled,
    ) = configured_student_engine(args)
    hybrid_enabled = multitask_enabled and configured_student_path is not None
    if capture_hrsegnet_enabled or multitask_enabled or hrsegnet_enabled:
        try:
            validate_tf32_runtime_environment()
        except RuntimeError as exc:
            print(exc, file=sys.stderr)
            return 2
    capture_crack_engine_sha256 = None
    realtime_crack_engine_sha256 = None
    multitask_engine_sha256 = None
    hrsegnet_engine_sha256 = None
    obstacle_engine_sha256 = None
    try:
        if dual_camera_enabled:
            validate_distinct_camera_devices(
                args.side_camera_device,
                args.top_camera_device,
            )
        args.teacher_engine, teacher_engine_sha256 = validate_engine(
            args.teacher_engine,
            args.teacher_engine_sha256,
            "--teacher-engine-sha256",
        )
        if multitask_enabled:
            args.realtime_multitask_engine, multitask_engine_sha256 = validate_engine(
                args.realtime_multitask_engine,
                args.realtime_multitask_engine_sha256,
                "--realtime-multitask-engine-sha256",
            )
            if hybrid_enabled:
                student_engine, student_engine_sha256 = validate_engine(
                    configured_student_path,
                    configured_student_sha256,
                    (
                        "--optimized-student-engine-sha256"
                        if optimized_rust_enabled
                        else "--student-engine-sha256"
                    ),
                )
            else:
                student_engine = args.realtime_multitask_engine
                student_engine_sha256 = multitask_engine_sha256
        else:
            if configured_student_path is None:
                raise ValueError("A realtime student rust engine is required.")
            student_engine, student_engine_sha256 = validate_engine(
                configured_student_path,
                configured_student_sha256,
                (
                    "--optimized-student-engine-sha256"
                    if optimized_rust_enabled
                    else "--student-engine-sha256"
                ),
            )
        if teacher_engine_sha256 == student_engine_sha256:
            raise ValueError("Teacher and realtime TensorRT engines must be different.")
        if hrsegnet_enabled:
            (
                args.realtime_hrsegnet_crack_engine,
                hrsegnet_engine_sha256,
            ) = validate_engine(
                args.realtime_hrsegnet_crack_engine,
                args.realtime_hrsegnet_crack_engine_sha256,
                "--realtime-hrsegnet-crack-engine-sha256",
            )
        if capture_hrsegnet_enabled:
            (
                args.capture_hrsegnet_crack_engine,
                capture_crack_engine_sha256,
            ) = validate_engine(
                args.capture_hrsegnet_crack_engine,
                args.capture_hrsegnet_crack_engine_sha256,
                "--capture-hrsegnet-crack-engine-sha256",
            )
            if not multitask_enabled and not hrsegnet_enabled:
                args.realtime_crack_engine, realtime_crack_engine_sha256 = validate_engine(
                    args.realtime_crack_engine,
                    args.realtime_crack_engine_sha256,
                    "--realtime-crack-engine-sha256",
                )
            engine_digests = {
                teacher_engine_sha256,
                student_engine_sha256,
                capture_crack_engine_sha256,
            }
            if hybrid_enabled:
                engine_digests.add(multitask_engine_sha256)
            if hrsegnet_enabled:
                engine_digests.add(hrsegnet_engine_sha256)
            elif not multitask_enabled:
                engine_digests.add(realtime_crack_engine_sha256)
            expected_engine_count = 4 if hybrid_enabled or not multitask_enabled else 3
            if len(engine_digests) != expected_engine_count:
                raise ValueError(
                    "Capture and realtime TensorRT engine files must all be different."
                )
        if dual_camera_enabled:
            args.obstacle_engine, obstacle_engine_sha256 = validate_engine(
                args.obstacle_engine,
                args.obstacle_engine_sha256,
                "--obstacle-engine-sha256",
            )
            if obstacle_engine_sha256 in engine_digests:
                raise ValueError(
                    "Obstacle TensorRT engine must differ from every rust/crack engine."
                )
    except Exception as exc:
        print(exc, file=sys.stderr)
        return 2

    missing_engine_digests = []
    if args.teacher_engine_sha256 is None:
        missing_engine_digests.append(
            ("teacher", "--teacher-engine-sha256", teacher_engine_sha256)
        )
    if (not multitask_enabled or hybrid_enabled) and configured_student_sha256 is None:
        missing_engine_digests.append(
            (
                "optimized student" if optimized_rust_enabled else "student",
                (
                    "--optimized-student-engine-sha256"
                    if optimized_rust_enabled
                    else "--student-engine-sha256"
                ),
                student_engine_sha256,
            )
        )
    if (
        capture_hrsegnet_enabled
        and args.capture_hrsegnet_crack_engine_sha256 is None
    ):
        missing_engine_digests.append(
            (
                "capture crack",
                "--capture-hrsegnet-crack-engine-sha256",
                capture_crack_engine_sha256,
            )
        )
    if (
        not multitask_enabled
        and not hrsegnet_enabled
        and
        args.realtime_crack_engine is not None
        and args.realtime_crack_engine_sha256 is None
    ):
        missing_engine_digests.append(
            (
                "realtime crack",
                "--realtime-crack-engine-sha256",
                realtime_crack_engine_sha256,
            )
        )
    if hrsegnet_enabled and args.realtime_hrsegnet_crack_engine_sha256 is None:
        missing_engine_digests.append(
            (
                "realtime HrSegNet crack",
                "--realtime-hrsegnet-crack-engine-sha256",
                hrsegnet_engine_sha256,
            )
        )
    if missing_engine_digests:
        if not args.no_uart:
            missing_options = ", ".join(
                option for _, option, _ in missing_engine_digests
            )
            print(
                "UART operation requires approved digests for every enabled "
                "TensorRT engine. "
                f"Missing: {missing_options}. First verify them with --no-uart.",
                file=sys.stderr,
            )
            for role, _, digest in missing_engine_digests:
                print(f"Current {role} SHA-256: {digest}", file=sys.stderr)
            return 2
        for role, _, digest in missing_engine_digests:
            print(
                f"TensorRT {role} engine is unpinned because UART is disabled; "
                f"current SHA-256: {digest}"
            )

    teacher_detector = None
    student_detector = None
    capture_crack_detector = None
    realtime_crack_detector = None
    obstacle_detector = None
    try:
        teacher_detector = RustDetector(
            args.teacher_engine,
            TEACHER_PROFILE,
            teacher_engine_sha256,
        )
        if hybrid_enabled:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
        elif multitask_enabled:
            student_detector = OptimizedMultitaskDetector(
                args.realtime_multitask_engine,
                multitask_engine_sha256,
                args.realtime_crack_threshold,
                args.realtime_crack_min_component_pixels,
            )
        else:
            if optimized_rust_enabled:
                student_detector = OptimizedRustDetector(
                    student_engine,
                    student_engine_sha256,
                )
            else:
                student_detector = RustDetector(
                    student_engine,
                    STUDENT_PROFILE,
                    student_engine_sha256,
                    gpu_argmax=True,
                )
        if capture_hrsegnet_enabled:
            capture_crack_detector = HrSegNetCrackDetector(
                args.capture_hrsegnet_crack_engine,
                capture_crack_engine_sha256,
                args.capture_hrsegnet_crack_probability_threshold,
                args.capture_hrsegnet_crack_min_component_pixels,
                role="capture",
            )
            if hybrid_enabled:
                realtime_crack_detector = OptimizedMultitaskDetector(
                    args.realtime_multitask_engine,
                    multitask_engine_sha256,
                    args.realtime_crack_threshold,
                    args.realtime_crack_min_component_pixels,
                )
            elif hrsegnet_enabled:
                realtime_crack_detector = HrSegNetCrackDetector(
                    args.realtime_hrsegnet_crack_engine,
                    hrsegnet_engine_sha256,
                    args.realtime_hrsegnet_crack_probability_threshold,
                    args.realtime_hrsegnet_crack_min_component_pixels,
                    role="realtime",
                )
            elif not multitask_enabled:
                realtime_crack_detector = CrackDetector(
                    args.realtime_crack_engine,
                    CRACK_REALTIME_PROFILE,
                    realtime_crack_engine_sha256,
                    args.realtime_crack_threshold,
                    args.realtime_crack_min_component_pixels,
                )
        if dual_camera_enabled:
            obstacle_detector = ObstacleDetector(
                args.obstacle_engine,
                obstacle_engine_sha256,
                args.obstacle_confidence_threshold,
            )
    except Exception as exc:
        print(exc, file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
        )
        return 2

    camera = None
    top_camera = None
    try:
        camera_source = (
            args.side_camera_device if dual_camera_enabled else args.camera_index
        )
        camera = open_latest_frame_camera(camera_source)
        if dual_camera_enabled:
            top_camera = open_latest_frame_camera(args.top_camera_device)
    except Exception as exc:
        print(f"Could not initialize camera: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    try:
        warmup_camera_frame = camera.read_latest()
        warmup_frame = warmup_camera_frame.frame
        warmup_top_camera_frame = (
            top_camera.read_latest() if dual_camera_enabled else None
        )
    except Exception as exc:
        print(f"Could not read a camera frame for model warm-up: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1
    try:
        warmup_water_roi, warmup_crack_roi = extract_realtime_control_rois(
            warmup_frame
        )
        teacher_detector.detect(warmup_frame)
        if capture_crack_detector is not None:
            capture_crack_detector.detect(warmup_frame)
        detect_realtime_control(
            warmup_water_roi,
            warmup_crack_roi,
            student_detector,
            realtime_crack_detector,
            multitask_enabled=multitask_enabled,
            hybrid_enabled=hybrid_enabled,
        )
        if dual_camera_enabled:
            if warmup_top_camera_frame.frame.shape != (
                FRAME_HEIGHT,
                FRAME_WIDTH,
                3,
            ):
                raise RuntimeError(
                    "Top camera did not provide a native 1280x720 BGR frame."
                )
            (
                warmup_top_obstacle_roi,
                warmup_top_crack_roi,
            ) = extract_realtime_control_rois(warmup_top_camera_frame.frame)
            obstacle_detector.detect(warmup_top_obstacle_roi)
            realtime_crack_detector.detect(warmup_top_crack_roi)
    except Exception as exc:
        print(f"TensorRT model warm-up failed: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    try:
        workbook = InspectionWorkbook(
            args.report,
            capture_model_filename=args.teacher_engine.name,
            capture_model_sha256=teacher_detector.engine_sha256,
            realtime_model_filename=(
                args.realtime_multitask_engine.name
                if multitask_enabled and not hybrid_enabled
                else student_engine.name
            ),
            realtime_model_sha256=student_detector.engine_sha256,
            capture_detector=teacher_detector.method,
            realtime_detector=student_detector.method,
            crack_model_filename=(
                args.capture_hrsegnet_crack_engine.name
                if capture_hrsegnet_enabled
                else None
            ),
            crack_model_sha256=(
                None
                if capture_crack_detector is None
                else capture_crack_engine_sha256
            ),
            crack_detector=(
                None if capture_crack_detector is None else capture_crack_detector.method
            ),
            crack_probability_threshold=(
                args.capture_hrsegnet_crack_probability_threshold
            ),
            capture_crack_min_component_pixels=(
                args.capture_hrsegnet_crack_min_component_pixels
            ),
            realtime_crack_model_filename=(
                args.realtime_multitask_engine.name
                if multitask_enabled
                else (
                    args.realtime_hrsegnet_crack_engine.name
                    if hrsegnet_enabled
                    else (
                        None
                        if args.realtime_crack_engine is None
                        else args.realtime_crack_engine.name
                    )
                )
            ),
            realtime_crack_model_sha256=(
                multitask_engine_sha256
                if multitask_enabled
                else (
                    hrsegnet_engine_sha256
                    if hrsegnet_enabled
                    else (
                        None
                        if realtime_crack_detector is None
                        else realtime_crack_detector.engine_sha256
                    )
                )
            ),
            realtime_crack_detector=(
                (
                    realtime_crack_detector.crack_method
                    if hybrid_enabled
                    else student_detector.crack_method
                )
                if multitask_enabled
                else (
                    None
                    if realtime_crack_detector is None
                    else realtime_crack_detector.method
                )
            ),
            realtime_crack_probability_threshold=(
                args.realtime_hrsegnet_crack_probability_threshold
                if hrsegnet_enabled
                else args.realtime_crack_threshold
            ),
            realtime_crack_min_component_pixels=(
                args.realtime_hrsegnet_crack_min_component_pixels
                if hrsegnet_enabled
                else args.realtime_crack_min_component_pixels
            ),
        )
    except Exception as exc:
        print(f"Could not open inspection report {args.report}: {exc}", file=sys.stderr)
        close_runtime_resources(
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 2

    uart = None
    realtime_actuator_arbiter = None
    if not args.no_uart:
        try:
            uart = serial.Serial(
                args.serial_port,
                args.baud_rate,
                timeout=0,
                write_timeout=UART_WRITE_TIMEOUT_SECONDS,
            )
        except Exception as exc:
            print(f"Could not open UART {args.serial_port}: {exc}", file=sys.stderr)
            close_runtime_resources(
                student_detector=student_detector,
                teacher_detector=teacher_detector,
                capture_crack_detector=capture_crack_detector,
                realtime_crack_detector=realtime_crack_detector,
                obstacle_detector=obstacle_detector,
                camera=camera,
                top_camera=top_camera,
            )
            return 1

    if dual_camera_enabled:
        front_cleaner_controller = CleanerController(
            uart,
            FRONT_ACTUATOR_COMMANDS,
        )
        front_pump_controller = PumpController(uart, FRONT_ACTUATOR_COMMANDS)
        cleaner_controller = CleanerController(uart, SIDE_ACTUATOR_COMMANDS)
        pump_controller = PumpController(uart, SIDE_ACTUATOR_COMMANDS)
        actuator_pairs = (
            (front_cleaner_controller, front_pump_controller),
            (cleaner_controller, pump_controller),
        )
    else:
        front_cleaner_controller = None
        front_pump_controller = None
        cleaner_controller = CleanerController(uart)
        pump_controller = PumpController(uart)
        actuator_pairs = ((cleaner_controller, pump_controller),)
    analysis_worker = None
    display_controller = None
    dual_display_rate_limiter = None

    try:
        analysis_worker = CaptureAnalysisWorker(
            teacher_detector,
            workbook,
            crack_detector=capture_crack_detector,
        )
        analysis_worker.start()
        print(f"Capture detector: {teacher_detector.method}")
        print(f"Realtime detector: {student_detector.method}")
        if dual_camera_enabled:
            print(f"Side camera: {args.side_camera_device}")
            print(f"Top camera: {args.top_camera_device}")
            print(f"Top obstacle detector: {obstacle_detector.method}")
        if args.no_crack:
            print(
                "WARNING: --no-crack TEST MODE; crack detection and the "
                "fixed y=112:240 crack safety interlock is bypassed."
            )
        elif capture_crack_detector is None:
            print(
                "Crack detectors: disabled; provide both capture and realtime "
                "HrSegNet TensorRT engines."
            )
        else:
            print(f"Capture crack detector: {capture_crack_detector.method}")
            print(
                "Capture HrSegNet baseline: threshold="
                f"{args.capture_hrsegnet_crack_probability_threshold:g}, "
                "min_component_pixels="
                f"{args.capture_hrsegnet_crack_min_component_pixels}; "
                "operator-selected and not field-calibrated."
            )
            print(
                "Realtime crack detector: "
                f"{(realtime_crack_detector.crack_method if hybrid_enabled else student_detector.crack_method) if multitask_enabled else realtime_crack_detector.method}"
            )
            if hrsegnet_enabled:
                print(
                    "Realtime HrSegNet baseline: threshold="
                    f"{args.realtime_hrsegnet_crack_probability_threshold:g}, "
                    "min_component_pixels="
                    f"{args.realtime_hrsegnet_crack_min_component_pixels}."
                )
        print(
            "Realtime control inputs: "
            f"rust={warmup_water_roi.shape[1]}x{warmup_water_roi.shape[0]} "
            f"(full width, y={REALTIME_RUST_ROI_TOP}:{REALTIME_RUST_ROI_BOTTOM}), "
            f"crack={warmup_crack_roi.shape[1]}x{warmup_crack_roi.shape[0]} "
            f"(full width, y={REALTIME_CRACK_ROI_TOP}:{REALTIME_CRACK_ROI_BOTTOM})"
        )
        print(f"Inspection report: {workbook.path}")
        if uart is None:
            print("UART trigger: disabled (--no-uart); terminal start will not be sent.")
        else:
            print(f"UART trigger: {args.serial_port} at {args.baud_rate} baud")
            print("Type start and press Enter to send START to the STM.")
        if headless:
            headless_source = (
                "automatic: no DISPLAY/WAYLAND_DISPLAY"
                if getattr(args, "headless_auto", False)
                else "explicit --headless"
            )
            print(
                "Display mode: headless ("
                f"{headless_source}); preview, Q/Esc, and manual S capture are "
                "disabled. STM CAMERA_CAPTURE, inference, UART control, capture "
                "analysis, and reports remain active."
            )
        else:
            print("Display mode: GUI")
            print(
                "Press S to save a capture in CAPTURE_SCAN mode, "
                "or Q/Esc to close the camera window."
            )
        uart_buffer = bytearray()
        captured_until = 0.0
        captured_display = None
        mode = CAPTURE_SCAN_MODE
        initial_rail_section_count = 0
        rescan_rail_section_count = 0
        realtime_control_ready = False
        front_control_ready = False
        side_control_ready = False
        realtime_inference = AlternatingRealtimeInference(
            student_detector,
            realtime_crack_detector,
            multitask_enabled=multitask_enabled,
            hybrid_enabled=hybrid_enabled,
            no_crack=args.no_crack,
        )
        realtime_control_history = RealtimeControlHistory(no_crack=args.no_crack)
        top_realtime_inference = (
            AlternatingTopRealtimeInference(
                obstacle_detector,
                realtime_crack_detector,
            )
            if dual_camera_enabled
            else None
        )
        front_control_history = (
            FrontControlHistory() if dual_camera_enabled else None
        )
        front_command_hold = (
            RealtimeActuatorDecisionHold() if dual_camera_enabled else None
        )
        side_command_hold = (
            RealtimeActuatorDecisionHold() if dual_camera_enabled else None
        )
        realtime_display_cache = RealtimeDisplayCache()
        top_obstacle_display = None
        top_crack_display = None
        last_camera_sequence = warmup_camera_frame.sequence
        last_top_camera_sequence = (
            warmup_top_camera_frame.sequence if dual_camera_enabled else None
        )
        start_command_sent = False
        start_acknowledged = False
        terminal_input_active = True
        if not headless:
            display_controller = open_latest_frame_display()
            if dual_camera_enabled:
                dual_display_rate_limiter = DisplayRateLimiter(DUAL_GUI_FRAME_RATE)
        print(f"Mode: {mode}")
    except Exception as exc:
        print(f"Could not initialize runtime state: {exc}", file=sys.stderr)
        for cleaner, pump in actuator_pairs:
            for controller in (cleaner, pump):
                try:
                    controller.force_off()
                except RuntimeError as control_exc:
                    print(control_exc, file=sys.stderr)
        if analysis_worker is not None:
            worker_failure = analysis_worker.shutdown()
            if worker_failure is not None:
                print(worker_failure, file=sys.stderr)
        close_runtime_resources(
            display=display_controller,
            uart=uart,
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )
        return 1

    analysis_shutdown_failure = None
    realtime_actuator_shutdown_failure = None
    uart_safe_to_close = True
    worker_failure_reported = None
    mission_completed = False
    dashboard_finalized = False
    try:
        while True:
            if realtime_actuator_arbiter is not None:
                try:
                    realtime_actuator_arbiter.raise_if_failed()
                except RuntimeError as exc:
                    print(exc, file=sys.stderr)
                    return 1
            try:
                if (
                    display_controller is not None
                    and display_controller.check_status()
                ):
                    break
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 1
            worker_failure = analysis_worker.failure_message()
            if worker_failure is not None:
                print(worker_failure, file=sys.stderr)
                worker_failure_reported = worker_failure
                return 1

            try:
                cache_expired_while_waiting = False
                side_frame_available = True
                top_frame_available = not dual_camera_enabled
                if mode == REALTIME_MODE:
                    if dual_camera_enabled:
                        frame_read = read_dual_realtime_camera_frames(
                            camera,
                            top_camera,
                            after_side_sequence=last_camera_sequence,
                            after_top_sequence=last_top_camera_sequence,
                            side_inference=realtime_inference,
                            top_inference=top_realtime_inference,
                            side_cleaner_controller=cleaner_controller,
                            side_pump_controller=pump_controller,
                            front_cleaner_controller=front_cleaner_controller,
                            front_pump_controller=front_pump_controller,
                            actuator_arbiter=realtime_actuator_arbiter,
                        )
                        side_frame_available = frame_read.side_frame is not None
                        top_frame_available = frame_read.top_frame is not None
                        if side_frame_available:
                            camera_frame = frame_read.side_frame
                        if top_frame_available:
                            top_camera_frame = frame_read.top_frame
                        if frame_read.side_error is not None:
                            print(
                                f"Side camera realtime frame unavailable: "
                                f"{frame_read.side_error}",
                                file=sys.stderr,
                            )
                        if frame_read.top_error is not None:
                            print(
                                f"Top camera realtime frame unavailable: "
                                f"{frame_read.top_error}",
                                file=sys.stderr,
                            )
                    else:
                        (
                            camera_frame,
                            cache_expired_while_waiting,
                        ) = read_realtime_camera_frame(
                            camera,
                            after_sequence=last_camera_sequence,
                            inference=realtime_inference,
                            cleaner_controller=cleaner_controller,
                            pump_controller=pump_controller,
                        )
                else:
                    camera_frame = camera.read_latest(
                        after_sequence=last_camera_sequence,
                        timeout=LATEST_FRAME_TIMEOUT_SECONDS,
                    )
                    if dual_camera_enabled:
                        top_camera_frame = top_camera.read_latest(
                            after_sequence=last_top_camera_sequence,
                            timeout=LATEST_FRAME_TIMEOUT_SECONDS,
                        )
                        top_frame_available = True
            except (RuntimeError, TimeoutError) as exc:
                if mode == REALTIME_MODE:
                    if realtime_actuator_arbiter is not None:
                        realtime_actuator_arbiter.clear_all()
                    else:
                        force_cleaning_pairs_safe_off(*actuator_pairs)
                print(f"Camera frame stream failed: {exc}", file=sys.stderr)
                return 1
            if dual_camera_enabled and mode == REALTIME_MODE:
                if frame_read.side_cache_expired:
                    realtime_inference.expire_cache()
                    realtime_control_history.prune()
                if frame_read.top_cache_expired:
                    top_realtime_inference.expire_cache()
                    front_control_history.prune()
            elif cache_expired_while_waiting:
                realtime_control_history.reset()
                realtime_control_ready = False
            if side_frame_available:
                last_camera_sequence = camera_frame.sequence
            if dual_camera_enabled:
                if top_frame_available:
                    last_top_camera_sequence = top_camera_frame.sequence
            frame = camera_frame.frame

            terminal_command = None
            if terminal_input_active:
                terminal_command, terminal_input_active = read_terminal_command()
            if terminal_command is not None:
                if terminal_command != "start":
                    print("Unknown terminal command ignored; type start.")
                elif uart is None:
                    print("START not sent because UART is disabled (--no-uart).")
                elif start_command_sent:
                    print("START ignored because it was already sent to the STM.")
                elif mode != CAPTURE_SCAN_MODE or initial_rail_section_count != 0:
                    print(
                        f"START ignored because inspection is already in progress "
                        f"(mode={mode}, rail_sections={initial_rail_section_count}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET})."
                    )
                else:
                    try:
                        written = uart.write(START_COMMAND)
                    except (serial.SerialException, OSError) as exc:
                        print(f"Could not send START to the STM: {exc}", file=sys.stderr)
                        return 1
                    if written != len(START_COMMAND):
                        print(
                            f"Could not send the complete START command "
                            f"({written}/{len(START_COMMAND)} bytes).",
                            file=sys.stderr,
                        )
                        return 1
                    start_command_sent = True
                    print("UART TX: START")

            uart_messages = []
            captured_this_iteration = False
            realtime_started_this_iteration = False
            if uart is not None:
                try:
                    uart_messages = read_uart_messages(uart, uart_buffer)
                except (serial.SerialException, OSError) as exc:
                    print(f"UART connection failed: {exc}", file=sys.stderr)
                    return 1

            if (
                REALTIME_START_TRIGGER in uart_messages
                and RESCAN_RETURN_START_TRIGGER in uart_messages
            ):
                print(
                    f"Timing/protocol error: STM {REALTIME_START_TRIGGER} and "
                    f"{RESCAN_RETURN_START_TRIGGER} arrived in the same UART read "
                    "batch; no "
                    "realtime frame could be inspected.",
                    file=sys.stderr,
                )
                return 1

            if sum(message == CAPTURE_TRIGGER for message in uart_messages) > 1:
                print(
                    f"Multiple {CAPTURE_TRIGGER} messages arrived before one "
                    "frame could be saved. Stopping to avoid duplicate captures.",
                    file=sys.stderr,
                )
                return 1

            for message in uart_messages:
                print(f"UART: {message}")
                if message in MOTION_TRIGGERS and not start_acknowledged:
                    print(
                        f"Protocol error: STM {message} arrived before STARTED "
                        "acknowledgement; possible old firmware or automatic start.",
                        file=sys.stderr,
                    )
                    return 1
                if message == STARTED_TRIGGER:
                    if not start_command_sent:
                        print(
                            f"Unsolicited STM {STARTED_TRIGGER} received; inspection "
                            "state was not changed.",
                            file=sys.stderr,
                        )
                    elif start_acknowledged:
                        print(f"Duplicate STM {STARTED_TRIGGER} ignored.")
                    else:
                        start_acknowledged = True
                        print(
                            "STM STARTED acknowledged; waiting for encoder "
                            "Rail Sections."
                        )
                elif message == CAPTURE_TRIGGER:
                    if mode == CAPTURE_SCAN_MODE:
                        phase = INITIAL_PHASE
                        phase_sequence = initial_rail_section_count + 1
                        phase_rail_section_count = initial_rail_section_count
                        phase_label = "Automatic"
                    elif mode == RESCAN_MODE:
                        phase = RESCAN_PHASE
                        phase_sequence = rescan_rail_section_count + 1
                        phase_rail_section_count = rescan_rail_section_count
                        phase_label = "Rescan"
                    else:
                        print(
                            f"Protocol error: STM {CAPTURE_TRIGGER} arrived in "
                            f"{mode} mode; no capture was acknowledged.",
                            file=sys.stderr,
                        )
                        return 1
                    if phase_rail_section_count >= AUTOMATIC_RAIL_SECTION_TARGET:
                        print(
                            f"Protocol error: extra STM {CAPTURE_TRIGGER} after "
                            f"{AUTOMATIC_RAIL_SECTION_TARGET} {phase} Rail Sections.",
                            file=sys.stderr,
                        )
                        return 1
                    try:
                        top_capture_camera_frame = None
                        if dual_camera_enabled:
                            (
                                capture_camera_frame,
                                top_capture_camera_frame,
                            ) = read_stopped_capture_pair(camera, top_camera)
                            last_top_camera_sequence = (
                                top_capture_camera_frame.sequence
                            )
                        else:
                            capture_camera_frame = read_stopped_capture_frame(
                                camera,
                                after_sequence=last_camera_sequence,
                            )
                        last_camera_sequence = capture_camera_frame.sequence
                        capture_frame = capture_camera_frame.frame
                        shared_captured_at = datetime.now()
                        if dual_camera_enabled:
                            queue_capture_for_analysis(
                                capture_frame,
                                phase,
                                phase_sequence,
                                CAPTURE_TRIGGER,
                                analysis_worker,
                                camera_role=SIDE_CAMERA_ROLE,
                                captured_at=shared_captured_at,
                                frame_read_completed_at=(
                                    capture_camera_frame.read_completed_at
                                ),
                            )
                            queue_capture_for_analysis(
                                top_capture_camera_frame.frame,
                                phase,
                                phase_sequence,
                                CAPTURE_TRIGGER,
                                analysis_worker,
                                camera_role=TOP_CAMERA_ROLE,
                                captured_at=shared_captured_at,
                                frame_read_completed_at=(
                                    top_capture_camera_frame.read_completed_at
                                ),
                            )
                        else:
                            # Preserve the legacy single-camera call contract.
                            # Several downstream integrations wrap this function
                            # positionally and do not know the dual-camera metadata.
                            queue_capture_for_analysis(
                                capture_frame,
                                phase,
                                phase_sequence,
                                CAPTURE_TRIGGER,
                                analysis_worker,
                            )
                        send_capture_ok(uart)
                    except (OSError, RuntimeError, ValueError) as exc:
                        print(exc, file=sys.stderr)
                        return 1
                    if phase == INITIAL_PHASE:
                        initial_rail_section_count += 1
                    else:
                        rescan_rail_section_count += 1
                    if display_controller is not None:
                        captured_display = (
                            stack_dual_camera_displays(
                                capture_frame,
                                top_capture_camera_frame.frame,
                            )
                            if dual_camera_enabled
                            else capture_frame.copy()
                        )
                        captured_until = time.monotonic() + CAPTURE_NOTICE_SECONDS
                    captured_this_iteration = True
                    phase_rail_section_count += 1
                    print(
                        f"{phase_label} Rail Sections: "
                        f"{phase_rail_section_count}/"
                        f"{AUTOMATIC_RAIL_SECTION_TARGET}"
                    )
                    if phase_rail_section_count >= AUTOMATIC_RAIL_SECTION_TARGET:
                        next_trigger = (
                            RETURN_START_TRIGGER
                            if phase == INITIAL_PHASE
                            else RESCAN_DONE_TRIGGER
                        )
                        print(
                            f"{phase_label} Rail Section scan complete; waiting for STM "
                            f"{next_trigger}."
                        )
                elif message == RETURN_START_TRIGGER:
                    if mode != CAPTURE_SCAN_MODE:
                        print(
                            f"Out-of-order STM {RETURN_START_TRIGGER} ignored in "
                            f"{mode} mode.",
                            file=sys.stderr,
                        )
                        continue
                    if (
                        initial_rail_section_count
                        != AUTOMATIC_RAIL_SECTION_TARGET
                    ):
                        print(
                            f"Out-of-order STM {RETURN_START_TRIGGER} ignored: "
                            "initial Rail Sections are "
                            f"{initial_rail_section_count}/"
                            f"{AUTOMATIC_RAIL_SECTION_TARGET}.",
                            file=sys.stderr,
                        )
                        continue
                    mode = RETURN_MODE
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {RETURN_START_TRIGGER})")
                elif message == REALTIME_START_TRIGGER:
                    if mode != RETURN_MODE:
                        print(
                            f"Out-of-order STM {REALTIME_START_TRIGGER} ignored in "
                            f"{mode} mode.",
                            file=sys.stderr,
                        )
                        continue
                    try:
                        prepare_realtime_transition(
                            cleaner_controller,
                            pump_controller,
                            analysis_worker,
                            additional_pairs=(
                                (
                                    front_cleaner_controller,
                                    front_pump_controller,
                                ),
                            ) if dual_camera_enabled else (),
                        )
                    except RuntimeError as exc:
                        print(f"REALTIME transition blocked: {exc}", file=sys.stderr)
                        return 1
                    mode = REALTIME_MODE
                    realtime_started_this_iteration = True
                    realtime_inference.reset()
                    realtime_control_history.reset()
                    if dual_camera_enabled:
                        top_realtime_inference.reset()
                        front_control_history.reset()
                        front_command_hold.clear()
                        side_command_hold.clear()
                        if uart is not None:
                            realtime_actuator_arbiter = RealtimeActuatorArbiter(uart)
                            realtime_actuator_arbiter.start()
                    realtime_display_cache.reset()
                    realtime_control_ready = False
                    front_control_ready = False
                    side_control_ready = False
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    if args.no_crack:
                        print(
                            "REALTIME TEST MODE: --no-crack bypass is active; "
                            "the crack safety interlock is NOT protecting the "
                            "cleaner or pump."
                        )
                    elif realtime_crack_detector is None and not multitask_enabled:
                        print(
                            "Cleaning locked OFF: crack detector is disabled, so "
                            "the y=112:240 no-crack condition cannot be verified."
                        )
                    print(f"Mode: {mode} (STM {REALTIME_START_TRIGGER})")
                elif message == RESCAN_RETURN_START_TRIGGER:
                    if mode != REALTIME_MODE:
                        print(
                            f"Protocol error: STM {RESCAN_RETURN_START_TRIGGER} "
                            f"arrived in {mode} mode.",
                            file=sys.stderr,
                        )
                        return 1
                    try:
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.close()
                            realtime_actuator_arbiter = None
                        if dual_camera_enabled:
                            front_command_hold.clear()
                            side_command_hold.clear()
                        for cleaner, pump in actuator_pairs:
                            cleaner.force_off()
                            pump.force_off()
                    except RuntimeError as exc:
                        print(f"Cleaning actuator control failed: {exc}", file=sys.stderr)
                        return 1
                    mode = RESCAN_RETURN_MODE
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {RESCAN_RETURN_START_TRIGGER})")
                elif message == RESCAN_START_TRIGGER:
                    if mode != RESCAN_RETURN_MODE or rescan_rail_section_count != 0:
                        print(
                            f"Protocol error: STM {RESCAN_START_TRIGGER} arrived "
                            f"in {mode} mode with {rescan_rail_section_count} rescan "
                            "Rail Sections.",
                            file=sys.stderr,
                        )
                        return 1
                    mode = RESCAN_MODE
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {RESCAN_START_TRIGGER})")
                elif message == RESCAN_DONE_TRIGGER:
                    if (
                        mode != RESCAN_MODE
                        or rescan_rail_section_count
                        != AUTOMATIC_RAIL_SECTION_TARGET
                    ):
                        print(
                            f"Protocol error: STM {RESCAN_DONE_TRIGGER} arrived in "
                            f"{mode} mode with {rescan_rail_section_count}/"
                            f"{AUTOMATIC_RAIL_SECTION_TARGET} rescan Rail Sections.",
                            file=sys.stderr,
                        )
                        return 1
                    mode = RESCAN_DONE_MODE
                    print(f"Mode: {mode} (STM {RESCAN_DONE_TRIGGER}); waiting for DONE")
                elif message == DONE_TRIGGER:
                    if mode != RESCAN_DONE_MODE:
                        print(
                            f"Protocol error: STM {DONE_TRIGGER} arrived in {mode} "
                            f"mode before {RESCAN_DONE_TRIGGER}.",
                            file=sys.stderr,
                        )
                        return 1
                    force_cleaning_pairs_safe_off(*actuator_pairs)
                    if not analysis_worker.wait_until_idle(
                        CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS
                    ):
                        print(
                            "DONE blocked: capture analysis did not become idle "
                            "within "
                            f"{CAPTURE_ANALYSIS_DRAIN_TIMEOUT_SECONDS:.1f} seconds; "
                            "cleaning remains OFF and the mission is stopping.",
                            file=sys.stderr,
                        )
                        return 1
                    worker_failure = analysis_worker.failure_message()
                    if worker_failure is not None:
                        print(worker_failure, file=sys.stderr)
                        worker_failure_reported = worker_failure
                        return 1
                    mode = COMPLETE_MODE
                    mission_completed = True
                    try:
                        dashboard_finalized = (
                            finalize_dashboard_run(
                                workbook=workbook,
                                output_directory=CAPTURE_DIRECTORY,
                                status="complete",
                            )
                            is not None
                        )
                    except Exception as exc:
                        print(
                            "Dashboard mission finalization failed; XLSX and "
                            f"captured images remain valid: {exc}",
                            file=sys.stderr,
                        )
                    if not captured_this_iteration:
                        captured_display = None
                        captured_until = 0.0
                    print(f"Mode: {mode} (STM {DONE_TRIGGER})")

            if display_controller is None and mission_completed:
                print("Headless mission complete; exiting after STM DONE.")
                break

            if mode == REALTIME_MODE and realtime_started_this_iteration:
                try:
                    if dual_camera_enabled:
                        frame_read = read_dual_realtime_camera_frames(
                            camera,
                            top_camera,
                            after_side_sequence=last_camera_sequence,
                            after_top_sequence=last_top_camera_sequence,
                            side_inference=realtime_inference,
                            top_inference=top_realtime_inference,
                            side_cleaner_controller=cleaner_controller,
                            side_pump_controller=pump_controller,
                            front_cleaner_controller=front_cleaner_controller,
                            front_pump_controller=front_pump_controller,
                            actuator_arbiter=realtime_actuator_arbiter,
                        )
                        side_frame_available = frame_read.side_frame is not None
                        top_frame_available = frame_read.top_frame is not None
                        if side_frame_available:
                            camera_frame = frame_read.side_frame
                            last_camera_sequence = camera_frame.sequence
                        if top_frame_available:
                            top_camera_frame = frame_read.top_frame
                            last_top_camera_sequence = top_camera_frame.sequence
                        if frame_read.side_cache_expired:
                            realtime_inference.expire_cache()
                            realtime_control_history.prune()
                        if frame_read.top_cache_expired:
                            top_realtime_inference.expire_cache()
                            front_control_history.prune()
                    else:
                        camera_frame = camera.read_latest(
                            after_sequence=last_camera_sequence,
                            timeout=REALTIME_FRAME_TIMEOUT_SECONDS,
                        )
                except (RuntimeError, TimeoutError) as exc:
                    if realtime_actuator_arbiter is not None:
                        realtime_actuator_arbiter.clear_all()
                    else:
                        force_cleaning_pairs_safe_off(*actuator_pairs)
                    print(
                        f"Fresh frame after REALTIME_START failed: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                if not dual_camera_enabled:
                    last_camera_sequence = camera_frame.sequence
                frame = camera_frame.frame

            if display_controller is None and mode != REALTIME_MODE:
                continue

            if mode == REALTIME_MODE:
                captured_display = None
                captured_until = 0.0
                try:
                    realtime_outcome = None
                    top_realtime_outcome = None
                    side_waiting = (
                        dual_camera_enabled and frame_read.side_waiting
                    )
                    top_waiting = (
                        dual_camera_enabled and frame_read.top_waiting
                    )
                    side_inputs_fresh = (
                        side_frame_available
                        and realtime_role_inputs_are_fresh(
                            camera_frame,
                            realtime_inference,
                        )
                    )
                    if side_inputs_fresh:
                        water_roi, crack_roi = extract_realtime_control_rois(frame)
                        realtime_outcome = realtime_inference.process(
                            camera_frame,
                            water_roi,
                            crack_roi,
                        )
                    if side_inputs_fresh:
                        realtime_control_history.update(realtime_outcome)
                    elif not side_waiting:
                        realtime_outcome = None
                        realtime_control_history.prune()
                        realtime_inference.expire_cache()
                        realtime_display_cache.reset()

                    if dual_camera_enabled:
                        top_inputs_fresh = (
                            top_frame_available
                            and realtime_role_inputs_are_fresh(
                                top_camera_frame,
                                top_realtime_inference,
                            )
                        )
                        if top_inputs_fresh:
                            top_obstacle_roi, top_crack_roi = (
                                extract_realtime_control_rois(
                                    top_camera_frame.frame
                                )
                            )
                            top_realtime_outcome = top_realtime_inference.process(
                                top_camera_frame,
                                top_obstacle_roi,
                                top_crack_roi,
                            )
                        if top_inputs_fresh:
                            front_control_history.update(top_realtime_outcome)
                        elif not top_waiting:
                            top_realtime_outcome = None
                            front_control_history.prune()
                            top_realtime_inference.expire_cache()
                            top_obstacle_display = None
                            top_crack_display = None
                    if realtime_actuator_arbiter is not None:
                        realtime_actuator_arbiter.raise_if_failed()
                    if display_controller is not None:
                        realtime_display_cache.update(realtime_outcome)
                        if (
                            dual_camera_enabled
                            and top_realtime_outcome is not None
                        ):
                            if (
                                top_realtime_outcome.display_obstacle_result
                                is not None
                            ):
                                top_obstacle_display = (
                                    top_realtime_outcome.display_obstacle_result
                                )
                            if top_realtime_outcome.display_crack_result is not None:
                                top_crack_display = (
                                    top_realtime_outcome.display_crack_result
                                )
                except (RuntimeError, ValueError) as exc:
                    try:
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.clear_all()
                        else:
                            force_cleaning_pairs_safe_off(*actuator_pairs)
                    except RuntimeError as control_exc:
                        print(control_exc, file=sys.stderr)
                    print(f"Realtime inference failed: {exc}", file=sys.stderr)
                    return 1
                try:
                    if (
                        display_controller is not None
                        and display_controller.check_status()
                    ):
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.clear_all()
                        else:
                            force_cleaning_pairs_safe_off(*actuator_pairs)
                        break
                    if dual_camera_enabled:
                        side_has_new_decision = (
                            realtime_outcome is not None
                            and realtime_control_history.ready
                        )
                        side_immediate_off = side_outcome_requires_immediate_off(
                            realtime_outcome,
                            history_ready=realtime_control_history.ready,
                        )
                        if frame_read.side_error is not None or side_immediate_off:
                            if frame_read.side_error is not None:
                                realtime_control_history.reset()
                                realtime_inference.reset()
                                realtime_display_cache.reset()
                            side_command_hold.clear()
                            clear_realtime_actuator_role(
                                cleaner_controller,
                                pump_controller,
                                role="side",
                                arbiter=realtime_actuator_arbiter,
                            )
                            side_ready = False
                        elif side_has_new_decision:
                            if not side_control_ready:
                                print(
                                    "Side realtime control ready: four fresh rust "
                                    "and crack results are available."
                                )
                            publish_realtime_actuator_decision(
                                cleaner_controller,
                                pump_controller,
                                role="side",
                                history=realtime_control_history,
                                decision=realtime_control_history.decision(),
                                hold=side_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                            side_ready = True
                        else:
                            side_ready = maintain_waiting_role_control(
                                cleaner_controller,
                                pump_controller,
                                history=realtime_control_history,
                                inference=realtime_inference,
                                role="side",
                                hold=side_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                        if side_control_ready and not side_ready:
                            print(
                                "Side realtime control fail-closed: camera/error, "
                                "hazard, or 1.2 s command hold expired."
                            )
                        side_control_ready = side_ready

                        front_has_new_decision = (
                            top_realtime_outcome is not None
                            and front_control_history.ready
                        )
                        front_immediate_off = front_outcome_requires_immediate_off(
                            top_realtime_outcome
                        )
                        if frame_read.top_error is not None or front_immediate_off:
                            if frame_read.top_error is not None:
                                front_control_history.reset()
                                top_realtime_inference.reset()
                                top_obstacle_display = None
                                top_crack_display = None
                            front_command_hold.clear()
                            clear_realtime_actuator_role(
                                front_cleaner_controller,
                                front_pump_controller,
                                role="front",
                                arbiter=realtime_actuator_arbiter,
                            )
                            front_ready = False
                        elif front_has_new_decision:
                            if not front_control_ready:
                                print(
                                    "Front realtime control ready: four fresh "
                                    "obstacle and crack results are available."
                                )
                            publish_realtime_actuator_decision(
                                front_cleaner_controller,
                                front_pump_controller,
                                role="front",
                                history=front_control_history,
                                decision=front_control_history.decision(),
                                hold=front_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                            front_ready = True
                        else:
                            front_ready = maintain_waiting_role_control(
                                front_cleaner_controller,
                                front_pump_controller,
                                history=front_control_history,
                                inference=top_realtime_inference,
                                role="front",
                                hold=front_command_hold,
                                arbiter=realtime_actuator_arbiter,
                            )
                        if front_control_ready and not front_ready:
                            print(
                                "Front realtime control fail-closed: camera/error, "
                                "hazard, or 1.2 s command hold expired."
                            )
                        front_control_ready = front_ready
                    elif (
                        realtime_outcome is not None
                        and realtime_control_history.ready
                    ):
                        if not realtime_control_ready:
                            print(
                                "Realtime control ready: four fresh inference "
                                "results from each enabled model are available."
                            )
                        realtime_control_ready = True
                        update_cleaning_actuators(
                            cleaner_controller,
                            pump_controller,
                            decision=realtime_control_history.decision(),
                        )
                    else:
                        if realtime_control_ready:
                            print(
                                "Realtime control fail-closed: camera frame or "
                                "cached inference result exceeded 800 ms."
                            )
                        force_cleaning_safe_off(
                            cleaner_controller,
                            pump_controller,
                        )
                        realtime_control_ready = False
                except (RuntimeError, ValueError) as exc:
                    try:
                        if realtime_actuator_arbiter is not None:
                            realtime_actuator_arbiter.clear_all()
                        else:
                            force_cleaning_pairs_safe_off(*actuator_pairs)
                    except RuntimeError as safe_off_exc:
                        print(safe_off_exc, file=sys.stderr)
                    print(f"Cleaning actuator control failed: {exc}", file=sys.stderr)
                    return 1
                if display_controller is None:
                    continue
                if (
                    dual_camera_enabled
                    and dual_display_rate_limiter is not None
                    and not dual_display_rate_limiter.should_render()
                ):
                    continue
                try:
                    realtime_annotated = annotate_realtime_control_results(
                        frame,
                        (
                            realtime_outcome.display_rust_result
                            if realtime_outcome is not None
                            else None
                        ),
                        (
                            realtime_outcome.display_crack_result
                            if realtime_outcome is not None
                            else None
                        ),
                    )
                    display = draw_realtime_roi_guide(
                        frame,
                        realtime_annotated,
                    )
                    display = draw_realtime_result_summary(
                        display,
                        realtime_display_cache.rust_result,
                        realtime_display_cache.crack_result,
                        crack_bypassed=args.no_crack,
                    )
                    if dual_camera_enabled:
                        top_display = annotate_top_realtime_results(
                            top_camera_frame.frame,
                            top_obstacle_display,
                            top_crack_display,
                        )
                        top_display = draw_realtime_roi_guide(
                            top_camera_frame.frame,
                            top_display,
                            primary_roi_label="OBSTACLE CONTROL",
                        )
                        display = stack_dual_camera_displays(display, top_display)
                except (RuntimeError, ValueError) as exc:
                    print(f"Realtime display rendering failed: {exc}", file=sys.stderr)
                    return 1
            elif (
                dual_camera_enabled
                and dual_display_rate_limiter is not None
                and not dual_display_rate_limiter.should_render()
            ):
                continue
            elif captured_this_iteration and captured_display is not None:
                display = captured_display
            elif captured_display is not None and time.monotonic() < captured_until:
                display = captured_display
            else:
                captured_display = None
                try:
                    display = draw_roi_guide(frame)
                    if dual_camera_enabled:
                        top_display = draw_roi_guide(top_camera_frame.frame)
                        display = stack_dual_camera_displays(display, top_display)
                except ValueError as exc:
                    print(f"Could not draw ROI preview: {exc}", file=sys.stderr)
                    return 1
            if mode == CAPTURE_SCAN_MODE:
                mode_label = (
                    f"mode={mode} rail_sections="
                    f"{initial_rail_section_count}/"
                    f"{AUTOMATIC_RAIL_SECTION_TARGET}"
                )
            elif mode == RESCAN_MODE:
                mode_label = (
                    f"mode={mode} rail_sections="
                    f"{rescan_rail_section_count}/"
                    f"{AUTOMATIC_RAIL_SECTION_TARGET}"
                )
            else:
                mode_label = f"mode={mode}"
            (label_width, label_height), label_baseline = cv2.getTextSize(
                mode_label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2,
            )
            label_x = max(12, display.shape[1] - label_width - 24)
            label_y = max(
                label_height + 12,
                display.shape[0] - (100 if mode == REALTIME_MODE else 24),
            )
            cv2.rectangle(
                display,
                (label_x - 8, label_y - label_height - 8),
                (label_x + label_width + 8, label_y + label_baseline + 8),
                (20, 35, 50),
                -1,
            )
            cv2.putText(
                display,
                mode_label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (60, 220, 90) if mode == REALTIME_MODE else (0, 180, 255),
                2,
                cv2.LINE_AA,
            )
            try:
                if not display_controller.submit(display):
                    break
                key = display_controller.poll_key()
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                return 1
            if key in (ord("s"), ord("S")):
                if mode != CAPTURE_SCAN_MODE:
                    print(f"Manual capture ignored while {mode} mode is active.")
                    continue
                if captured_this_iteration or captured_display is not None:
                    print(
                        "Manual capture ignored while the previous capture result "
                        "is displayed."
                    )
                    continue
                try:
                    manual_top_frame = None
                    if dual_camera_enabled:
                        side_manual, top_manual = read_stopped_capture_pair(
                            camera,
                            top_camera,
                        )
                        last_camera_sequence = side_manual.sequence
                        last_top_camera_sequence = top_manual.sequence
                        shared_captured_at = datetime.now()
                        queue_capture_for_analysis(
                            side_manual.frame,
                            MANUAL_PHASE,
                            None,
                            "manual",
                            analysis_worker,
                            camera_role=SIDE_CAMERA_ROLE,
                            captured_at=shared_captured_at,
                            frame_read_completed_at=side_manual.read_completed_at,
                        )
                        queue_capture_for_analysis(
                            top_manual.frame,
                            MANUAL_PHASE,
                            None,
                            "manual",
                            analysis_worker,
                            camera_role=TOP_CAMERA_ROLE,
                            captured_at=shared_captured_at,
                            frame_read_completed_at=top_manual.read_completed_at,
                        )
                        frame = side_manual.frame
                        manual_top_frame = top_manual.frame
                    else:
                        queue_capture_for_analysis(
                            frame,
                            MANUAL_PHASE,
                            None,
                            "manual",
                            analysis_worker,
                        )
                except (OSError, RuntimeError, ValueError) as exc:
                    print(exc, file=sys.stderr)
                    return 1
                captured_display = (
                    stack_dual_camera_displays(frame, manual_top_frame)
                    if dual_camera_enabled
                    else frame.copy()
                )
                captured_until = time.monotonic() + CAPTURE_NOTICE_SECONDS
    finally:
        if realtime_actuator_arbiter is not None:
            try:
                realtime_actuator_arbiter.close()
            except RuntimeError as exc:
                print(exc, file=sys.stderr)
                realtime_actuator_shutdown_failure = str(exc)
                uart_safe_to_close = (
                    not realtime_actuator_arbiter.worker_is_alive()
                )
        if uart_safe_to_close:
            for cleaner, pump in actuator_pairs:
                for controller in (cleaner, pump):
                    try:
                        controller.force_off()
                    except RuntimeError as exc:
                        print(exc, file=sys.stderr)
                        if realtime_actuator_shutdown_failure is None:
                            realtime_actuator_shutdown_failure = str(exc)
        analysis_shutdown_failure = analysis_worker.shutdown()
        if (
            analysis_shutdown_failure is not None
            and analysis_shutdown_failure != worker_failure_reported
        ):
            print(analysis_shutdown_failure, file=sys.stderr)
        if not dashboard_finalized:
            dashboard_status = None
            dashboard_failure_reason = None
            if mission_completed:
                dashboard_status = "complete"
            elif (
                start_command_sent
                or start_acknowledged
                or initial_rail_section_count > 0
                or rescan_rail_section_count > 0
                or mode != CAPTURE_SCAN_MODE
            ):
                dashboard_status = "failed"
                dashboard_failure_reason = (
                    analysis_shutdown_failure
                    or realtime_actuator_shutdown_failure
                    or worker_failure_reported
                    or f"Runtime ended before STM DONE completion (mode={mode})."
                )
            if dashboard_status is not None:
                try:
                    finalize_dashboard_run(
                        workbook=workbook,
                        output_directory=CAPTURE_DIRECTORY,
                        status=dashboard_status,
                        failure_reason=dashboard_failure_reason,
                    )
                except Exception as exc:
                    print(
                        "Dashboard mission finalization failed; XLSX and captured "
                        f"images remain valid: {exc}",
                        file=sys.stderr,
                    )
        close_runtime_resources(
            display=display_controller,
            uart=uart if uart_safe_to_close else None,
            student_detector=student_detector,
            teacher_detector=teacher_detector,
            capture_crack_detector=capture_crack_detector,
            realtime_crack_detector=realtime_crack_detector,
            obstacle_detector=obstacle_detector,
            camera=camera,
            top_camera=top_camera,
        )

    if (
        analysis_shutdown_failure is not None
        or realtime_actuator_shutdown_failure is not None
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
