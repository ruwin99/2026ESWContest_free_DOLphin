#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SIDE_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
TOP_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0"
OBSTACLE_ENGINE="${HOME}/models/obstacle_yolo26n_roi_y0_240_20260821_1612/plans/obstacle-yolo26n-hardneg-all1410-roi-y0-240-w1280-h240-int-h256-fp32-notf32-trt10.3.plan"
OBSTACLE_ENGINE_SHA256="3def248110d5ead79491161049cb666322c337fa0dfdb39c41270e78c9bdf5e0"

FORWARDED_ARGUMENTS=()
HEADLESS_REQUESTED=0
UART_REQUESTED=0
TIMING_REQUESTED=0
for argument in "$@"; do
    case "${argument}" in
        --headless)
            if [[ "${HEADLESS_REQUESTED}" == "1" ]]; then
                echo "Usage: ./scripts/dual_camera_test.sh [--headless] [--uart] [--timing]" >&2
                exit 2
            fi
            HEADLESS_REQUESTED=1
            FORWARDED_ARGUMENTS+=("--headless")
            ;;
        --uart)
            if [[ "${UART_REQUESTED}" == "1" ]]; then
                echo "Usage: ./scripts/dual_camera_test.sh [--headless] [--uart] [--timing]" >&2
                exit 2
            fi
            UART_REQUESTED=1
            FORWARDED_ARGUMENTS+=("--realtime-test-uart")
            ;;
        --timing)
            if [[ "${TIMING_REQUESTED}" == "1" ]]; then
                echo "Usage: ./scripts/dual_camera_test.sh [--headless] [--uart] [--timing]" >&2
                exit 2
            fi
            TIMING_REQUESTED=1
            FORWARDED_ARGUMENTS+=("--dual-timing")
            ;;
        *)
            echo "Usage: ./scripts/dual_camera_test.sh [--headless] [--uart] [--timing]" >&2
            exit 2
            ;;
    esac
done

exec bash "${PROJECT_ROOT}/scripts/run.sh" \
    --realtime-test \
    --side-camera-device "${SIDE_CAMERA_DEVICE}" \
    --top-camera-device "${TOP_CAMERA_DEVICE}" \
    --obstacle-engine "${OBSTACLE_ENGINE}" \
    --obstacle-engine-sha256 "${OBSTACLE_ENGINE_SHA256}" \
    --obstacle-confidence-threshold 0.30 \
    "${FORWARDED_ARGUMENTS[@]}"
