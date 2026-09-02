#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_VENV="${PROJECT_ROOT}/.venv"
VENV="${RAIL_ROBOT_VENV:-${LOCAL_VENV}}"
export RAIL_ROBOT_VENV="${VENV}"
PYTHON="${VENV}/bin/python"

TRAINING=0
for argument in "$@"; do
    if [[ "${argument}" == "--training" ]]; then
        TRAINING=1
    fi
done
if [[ "${TRAINING}" == "1" ]]; then
    if [[ "$#" != "1" || "$1" != "--training" ]]; then
        echo "--training must be used alone." >&2
        exit 2
    fi
    if [[ ! -x "${PYTHON}" ]]; then
        echo "Python virtual environment is unavailable: ${PYTHON}" >&2
        exit 1
    fi
    if ! "${PYTHON}" -c 'import cv2' 2>/dev/null; then
        echo "OpenCV is unavailable in ${VENV}." >&2
        exit 1
    fi
    exec "${PYTHON}" "${PROJECT_ROOT}/jetson_code/training_capture.py"
fi

APPROVED_REALTIME_RUST_ENGINE_PATH="${HOME}/models/plans_new/realtime-rust-mnv2-os8-w1280-h240-fp16.plan"
APPROVED_REALTIME_RUST_ENGINE_SHA256="cb0b71128d9725a3b3d60e2282a2659278e5397823d7cea5e2e229c5ae3bded1"
APPROVED_CAPTURE_TEST_RUST_ENGINE_PATH="${HOME}/models/capture_r101_hardneg_v6_20260821/plans/capture-rust-r101-os8-hardneg-v6-w1280-h720-fp32-notf32.plan"
APPROVED_CAPTURE_TEST_RUST_ENGINE_SHA256="7c442abf7049bcc56d093e0dc7dd2a347caaee74dc8e20006b63ee032b6f22d4"
APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_PATH="${HOME}/models/hrsegnet_b32_20260816/plans/hrsegnet-b32-crackseg9k-realtime-w1280-h128-fp32-notf32.plan"
APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256="73a30156b3a7974748554f0fb328d2f118bd9dbec22863e334d43d0173a1e036"
APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_PATH="${HOME}/models/hrsegnet_b32_20260816/plans/hrsegnet-b32-crackseg9k-capture-w1280-h720-fp32-notf32.plan"
APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256="01c2dc7e4467e8888eda7eea68670ba6598ad4f7b664e57d420bda76e293111a"
APPROVED_SIDE_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.1:1.0-video-index0"
APPROVED_TOP_CAMERA_DEVICE="/dev/v4l/by-path/platform-3610000.usb-usb-0:2.3:1.0-video-index0"
APPROVED_OBSTACLE_ENGINE_PATH="${HOME}/models/obstacle_yolo26n_roi_y0_240_20260821_1612/plans/obstacle-yolo26n-hardneg-all1410-roi-y0-240-w1280-h240-int-h256-fp32-notf32-trt10.3.plan"
APPROVED_OBSTACLE_ENGINE_SHA256="3def248110d5ead79491161049cb666322c337fa0dfdb39c41270e78c9bdf5e0"
APPROVED_OBSTACLE_CONFIDENCE_THRESHOLD="0.30"

STUDENT_ENGINE_PATH="${RAIL_ROBOT_REALTIME_RUST_ENGINE:-${RAIL_ROBOT_STUDENT_ENGINE:-${RAIL_ROBOT_ENGINE:-${APPROVED_REALTIME_RUST_ENGINE_PATH}}}}"
OPTIMIZED_STUDENT_ENGINE_PATH="${RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE:-}"
TEACHER_ENGINE_PATH="${RAIL_ROBOT_CAPTURE_RUST_ENGINE:-${RAIL_ROBOT_TEACHER_ENGINE:-${HOME}/models/plans_new/corrosion-capture-r101-os8-w1280-h720-fp32.plan}}"
CAPTURE_HRSEGNET_CRACK_ENGINE_PATH="${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}"
REALTIME_CRACK_ENGINE_PATH="${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-${HOME}/models/plans_new/realtime-crack-bgcrack-w1280-h128-fp32.plan}"
REALTIME_MULTITASK_ENGINE_PATH="${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE:-}"
REALTIME_HRSEGNET_CRACK_ENGINE_PATH="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE:-}"
STUDENT_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256:-${RAIL_ROBOT_STUDENT_ENGINE_SHA256:-${RAIL_ROBOT_ENGINE_SHA256:-${APPROVED_REALTIME_RUST_ENGINE_SHA256}}}}"
OPTIMIZED_STUDENT_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE_SHA256:-}"
TEACHER_ENGINE_SHA256="${RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256:-${RAIL_ROBOT_TEACHER_ENGINE_SHA256:-}}"
CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256="${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}"
REALTIME_CRACK_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}"
REALTIME_MULTITASK_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256:-}"
REALTIME_HRSEGNET_CRACK_ENGINE_SHA256="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256:-}"
CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD="0.55"
REALTIME_CRACK_THRESHOLD="${RAIL_ROBOT_REALTIME_CRACK_THRESHOLD:-0.5}"
CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS="20"
REALTIME_CRACK_MIN_COMPONENT_PIXELS="${RAIL_ROBOT_REALTIME_CRACK_MIN_COMPONENT_PIXELS:-20}"
REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD:-0.55}"
REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS="${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS:-20}"

if [[ -n "${RAIL_ROBOT_CRACK_ENGINE:-}" || -n "${RAIL_ROBOT_CRACK_ENGINE_SHA256:-}" ]]; then
    echo "RAIL_ROBOT_CRACK_ENGINE is no longer supported; capture and realtime HrSegNet baselines are pinned separately." >&2
    exit 2
fi
if [[ -n "${RAIL_ROBOT_CRACK_THRESHOLD:-}" ]]; then
    echo "RAIL_ROBOT_CRACK_THRESHOLD is no longer supported; use the pinned role-specific HrSegNet baselines." >&2
    exit 2
fi
if [[ -n "${RAIL_ROBOT_CRACK_MIN_COMPONENT_PIXELS:-}" ]]; then
    echo "RAIL_ROBOT_CRACK_MIN_COMPONENT_PIXELS is no longer supported; use the pinned role-specific HrSegNet baselines." >&2
    exit 2
fi

if [[ -n "${RAIL_ROBOT_CAPTURE_CRACK_ENGINE:-}" ||
    -n "${RAIL_ROBOT_CAPTURE_CRACK_ENGINE_SHA256:-}" ||
    -n "${RAIL_ROBOT_CAPTURE_CRACK_THRESHOLD:-}" ||
    -n "${RAIL_ROBOT_CAPTURE_CRACK_MIN_COMPONENT_PIXELS:-}" ]]; then
    echo "Ignoring legacy capture BGCrack settings; capture now uses the pinned original HrSegNet-B32 full-frame baseline." >&2
fi
if [[ ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE}" != "${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}" ) ||
    ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}" != "${APPROVED_CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}" ) ||
    ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}" != "0.55" ) ||
    ( -n "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS:-}" &&
    "${RAIL_ROBOT_CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}" != "20" ) ]]; then
    echo "Ignoring unapproved capture HrSegNet overrides; run.sh uses the pinned plan, SHA-256, threshold, and component baseline." >&2
fi

CAPTURE_TEST=0
REALTIME_TEST=0
HRSEGNET_CLI_CONFIGURED=0
REALTIME_MODEL_CLI_CONFIGURED=0
CAPTURE_MODEL_CLI_CONFIGURED=0
TEACHER_CLI_CONFIGURED=0
NORMAL_CAMERA_OR_OBSTACLE_CLI_CONFIGURED=0
for argument in "$@"; do
    if [[ "${argument}" == "--capture-test" ]]; then
        CAPTURE_TEST=1
    elif [[ "${argument}" == "--realtime-test" ]]; then
        REALTIME_TEST=1
    elif [[ "${argument}" == --realtime-hrsegnet-crack-* ]]; then
        HRSEGNET_CLI_CONFIGURED=1
        REALTIME_MODEL_CLI_CONFIGURED=1
    elif [[ "${argument}" == --capture-crack-* ||
        "${argument}" == --capture-hrsegnet-crack-* ]]; then
        CAPTURE_MODEL_CLI_CONFIGURED=1
    elif [[ "${argument}" == --teacher-engine* ]]; then
        TEACHER_CLI_CONFIGURED=1
    elif [[ "${argument}" == --student-engine* ||
        "${argument}" == --engine* ||
        "${argument}" == --optimized-student-engine* ||
        "${argument}" == --realtime-crack-* ||
        "${argument}" == --realtime-multitask-* ]]; then
        REALTIME_MODEL_CLI_CONFIGURED=1
    elif [[ "${argument}" == --camera-index ||
        "${argument}" == --camera-index=* ||
        "${argument}" == --side-camera-device ||
        "${argument}" == --side-camera-device=* ||
        "${argument}" == --top-camera-device ||
        "${argument}" == --top-camera-device=* ||
        "${argument}" == --obstacle-engine ||
        "${argument}" == --obstacle-engine=* ||
        "${argument}" == --obstacle-engine-sha256 ||
        "${argument}" == --obstacle-engine-sha256=* ||
        "${argument}" == --obstacle-confidence-threshold ||
        "${argument}" == --obstacle-confidence-threshold=* ]]; then
        NORMAL_CAMERA_OR_OBSTACLE_CLI_CONFIGURED=1
    fi
done

if [[ "${CAPTURE_TEST}" == "0" && "${REALTIME_TEST}" == "0" &&
    "${NORMAL_CAMERA_OR_OBSTACLE_CLI_CONFIGURED}" == "1" ]]; then
    echo "Normal run.sh missions use pinned SIDE/TOP cameras and the pinned obstacle model; camera and obstacle CLI overrides are disabled." >&2
    exit 2
fi

if [[ "${CAPTURE_MODEL_CLI_CONFIGURED}" == "1" ]]; then
    echo "Capture model CLI overrides are disabled in run.sh; the pinned original HrSegNet-B32 full-frame baseline is mandatory." >&2
    exit 2
fi
if [[ "${CAPTURE_TEST}" == "1" && "${TEACHER_CLI_CONFIGURED}" == "1" ]]; then
    echo "Teacher engine CLI overrides are disabled for --capture-test." >&2
    exit 2
fi
if [[ "${CAPTURE_TEST}" == "1" ]]; then
    if [[ ( -n "${RAIL_ROBOT_CAPTURE_RUST_ENGINE:-}" &&
        "${RAIL_ROBOT_CAPTURE_RUST_ENGINE}" != "${APPROVED_CAPTURE_TEST_RUST_ENGINE_PATH}" ) ||
        ( -n "${RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256:-}" &&
        "${RAIL_ROBOT_CAPTURE_RUST_ENGINE_SHA256}" != "${APPROVED_CAPTURE_TEST_RUST_ENGINE_SHA256}" ) ||
        -n "${RAIL_ROBOT_TEACHER_ENGINE:-}" ||
        -n "${RAIL_ROBOT_TEACHER_ENGINE_SHA256:-}" ]]; then
        echo "Ignoring capture rust environment overrides; --capture-test uses the pinned hard-negative R101 plan." >&2
    fi
    TEACHER_ENGINE_PATH="${APPROVED_CAPTURE_TEST_RUST_ENGINE_PATH}"
    TEACHER_ENGINE_SHA256="${APPROVED_CAPTURE_TEST_RUST_ENGINE_SHA256}"
    echo "Capture test rust engine (pinned): ${TEACHER_ENGINE_PATH}" >&2
    echo "Capture test rust SHA-256: ${TEACHER_ENGINE_SHA256}" >&2
fi

if [[ "${CAPTURE_TEST}" == "0" ]]; then
    if [[ "${REALTIME_MODEL_CLI_CONFIGURED}" == "1" ]]; then
        echo "Realtime model CLI overrides are disabled in run.sh; the pinned Rust + HrSegNet baseline is mandatory." >&2
        exit 2
    fi
    if [[ -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-}" ||
        -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}" ||
        -n "${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE:-}" ||
        -n "${RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256:-}" ]]; then
        echo "Ignoring legacy BGCrack/multitask realtime environment settings; the approved HrSegNet baseline is pinned." >&2
    fi
    if [[ ( -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE:-}" &&
        "${RAIL_ROBOT_REALTIME_RUST_ENGINE}" != "${APPROVED_REALTIME_RUST_ENGINE_PATH}" ) ||
        ( -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256:-}" &&
        "${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256}" != "${APPROVED_REALTIME_RUST_ENGINE_SHA256}" ) ||
        ( -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE:-}" &&
        "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE}" != "${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" ) ||
        ( -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256:-}" &&
        "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}" != "${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}" ) ]]; then
        echo "Ignoring unapproved realtime Rust/HrSegNet environment overrides; run.sh uses the pinned baseline paths and SHA-256 values." >&2
    fi
    unset RAIL_ROBOT_REALTIME_CRACK_ENGINE
    unset RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256
    unset RAIL_ROBOT_REALTIME_MULTITASK_ENGINE
    unset RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256
    unset RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE
    unset RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE_SHA256
    REALTIME_CRACK_ENGINE_PATH=""
    REALTIME_CRACK_ENGINE_SHA256=""
    REALTIME_MULTITASK_ENGINE_PATH=""
    REALTIME_MULTITASK_ENGINE_SHA256=""
    OPTIMIZED_STUDENT_ENGINE_PATH=""
    OPTIMIZED_STUDENT_ENGINE_SHA256=""
    STUDENT_ENGINE_PATH="${APPROVED_REALTIME_RUST_ENGINE_PATH}"
    STUDENT_ENGINE_SHA256="${APPROVED_REALTIME_RUST_ENGINE_SHA256}"
    REALTIME_HRSEGNET_CRACK_ENGINE_PATH="${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_PATH}"
    REALTIME_HRSEGNET_CRACK_ENGINE_SHA256="${APPROVED_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}"
    if [[ "${REALTIME_TEST}" == "1" ]]; then
        REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD="0.50"
    else
        REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD="0.55"
    fi
    REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS="20"
fi

HRSEGNET_CONFIGURED=0
if [[ "${CAPTURE_TEST}" == "0" && (
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE:-}" ||
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256:-}" ||
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD:-}" ||
    -n "${RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS:-}"
) ]]; then
    HRSEGNET_CONFIGURED=1
fi
if [[ "${CAPTURE_TEST}" == "0" ]]; then
    HRSEGNET_CONFIGURED=1
fi
HRSEGNET_SELECTED=0
if [[ "${HRSEGNET_CONFIGURED}" == "1" || "${HRSEGNET_CLI_CONFIGURED}" == "1" ]]; then
    HRSEGNET_SELECTED=1
fi
if [[ -z "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" && "${HRSEGNET_CONFIGURED}" == "1" ]]; then
    echo "HrSegNet threshold, component, and SHA settings require RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE." >&2
    exit 2
fi
if [[ "${HRSEGNET_SELECTED}" == "1" && -n "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" && -z "${REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}" ]]; then
    echo "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE requires RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE_SHA256." >&2
    exit 2
fi
if [[ "${HRSEGNET_SELECTED}" == "1" && (
    -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-}" ||
    -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}" ||
    -n "${REALTIME_MULTITASK_ENGINE_PATH}" ||
    -n "${REALTIME_MULTITASK_ENGINE_SHA256}"
) ]]; then
    echo "RAIL_ROBOT_REALTIME_HRSEGNET_CRACK_ENGINE cannot be combined with BGCrack or multitask realtime crack settings." >&2
    exit 2
fi

SEPARATE_RUST_PATH_CONFIGURED=0
SEPARATE_RUST_SHA_CONFIGURED=0
if [[ "${CAPTURE_TEST}" == "0" ]]; then
    RAW_RUST_PATH_CONFIGURED=0
    RAW_RUST_SHA_CONFIGURED=0
    if [[ -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE:-}" ||
        -n "${RAIL_ROBOT_STUDENT_ENGINE:-}" ||
        -n "${RAIL_ROBOT_ENGINE:-}" ]]; then
        RAW_RUST_PATH_CONFIGURED=1
    fi
    if [[ -n "${RAIL_ROBOT_REALTIME_RUST_ENGINE_SHA256:-}" ||
        -n "${RAIL_ROBOT_STUDENT_ENGINE_SHA256:-}" ||
        -n "${RAIL_ROBOT_ENGINE_SHA256:-}" ]]; then
        RAW_RUST_SHA_CONFIGURED=1
    fi
    if [[ ( "${RAW_RUST_PATH_CONFIGURED}" == "1" ||
        "${RAW_RUST_SHA_CONFIGURED}" == "1" ) && (
        -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ||
        -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}"
    ) ]]; then
        echo "Raw realtime rust engine settings cannot be combined with RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE settings." >&2
        exit 2
    fi
    if [[ -z "${OPTIMIZED_STUDENT_ENGINE_PATH}" &&
        -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}" ]]; then
        echo "RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE_SHA256 requires RAIL_ROBOT_REALTIME_RUST_OPTIMIZED_ENGINE." >&2
        exit 2
    fi
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" && -z "${REALTIME_MULTITASK_ENGINE_SHA256}" ]]; then
        echo "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE requires RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256." >&2
        exit 2
    fi
    if [[ -z "${REALTIME_MULTITASK_ENGINE_PATH}" && -n "${REALTIME_MULTITASK_ENGINE_SHA256}" ]]; then
        echo "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE_SHA256 requires RAIL_ROBOT_REALTIME_MULTITASK_ENGINE." >&2
        exit 2
    fi
    if [[ -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ||
        "${RAW_RUST_PATH_CONFIGURED}" == "1" ]]; then
        SEPARATE_RUST_PATH_CONFIGURED=1
    fi
    if [[ -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}" ||
        "${RAW_RUST_SHA_CONFIGURED}" == "1" ]]; then
        SEPARATE_RUST_SHA_CONFIGURED=1
    fi
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" &&
        "${SEPARATE_RUST_PATH_CONFIGURED}" != "${SEPARATE_RUST_SHA_CONFIGURED}" ]]; then
        echo "Hybrid realtime mode requires both the separate realtime rust engine and its SHA-256." >&2
        exit 2
    fi
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" && (
        -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE:-}" ||
        -n "${RAIL_ROBOT_REALTIME_CRACK_ENGINE_SHA256:-}"
    ) ]]; then
        echo "RAIL_ROBOT_REALTIME_MULTITASK_ENGINE cannot be combined with a separate realtime crack engine; multitask supplies the crack decision." >&2
        exit 2
    fi
fi
if [[ -n "${NVIDIA_TF32_OVERRIDE:-}" && "${NVIDIA_TF32_OVERRIDE}" != "0" ]]; then
    echo "NVIDIA_TF32_OVERRIDE must be 0 because the crack plans were built with TF32 disabled; this setting applies to every active plan in this process." >&2
    exit 2
fi
export NVIDIA_TF32_OVERRIDE=0

ENGINE_ARGUMENTS=(--teacher-engine "${TEACHER_ENGINE_PATH}")
if [[ -n "${TEACHER_ENGINE_SHA256}" ]]; then
    ENGINE_ARGUMENTS+=(--teacher-engine-sha256 "${TEACHER_ENGINE_SHA256}")
fi

NORMAL_DUAL_CAMERA_ARGUMENTS=()
if [[ "${CAPTURE_TEST}" == "0" && "${REALTIME_TEST}" == "0" ]]; then
    NORMAL_DUAL_CAMERA_ARGUMENTS=(
        --side-camera-device "${APPROVED_SIDE_CAMERA_DEVICE}"
        --top-camera-device "${APPROVED_TOP_CAMERA_DEVICE}"
        --obstacle-engine "${APPROVED_OBSTACLE_ENGINE_PATH}"
        --obstacle-engine-sha256 "${APPROVED_OBSTACLE_ENGINE_SHA256}"
        --obstacle-confidence-threshold "${APPROVED_OBSTACLE_CONFIDENCE_THRESHOLD}"
    )
    echo "Normal mission dual-camera baseline: SIDE=${APPROVED_SIDE_CAMERA_DEVICE}, TOP=${APPROVED_TOP_CAMERA_DEVICE}." >&2
fi

if [[ "${CAPTURE_TEST}" == "1" ]]; then
    ENGINE_ARGUMENTS+=(
        --capture-hrsegnet-crack-engine "${CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}"
        --capture-hrsegnet-crack-engine-sha256 "${CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}"
        --capture-hrsegnet-crack-probability-threshold "${CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}"
        --capture-hrsegnet-crack-min-component-pixels "${CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}"
    )
else
    if [[ -n "${REALTIME_MULTITASK_ENGINE_PATH}" ]]; then
        ENGINE_ARGUMENTS+=(
            --realtime-multitask-engine "${REALTIME_MULTITASK_ENGINE_PATH}"
            --realtime-multitask-engine-sha256 "${REALTIME_MULTITASK_ENGINE_SHA256}"
        )
        if [[ "${SEPARATE_RUST_PATH_CONFIGURED}" == "1" ]]; then
            if [[ -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ]]; then
                ENGINE_ARGUMENTS+=(
                    --optimized-student-engine "${OPTIMIZED_STUDENT_ENGINE_PATH}"
                    --optimized-student-engine-sha256 "${OPTIMIZED_STUDENT_ENGINE_SHA256}"
                )
            else
                ENGINE_ARGUMENTS+=(
                    --student-engine "${STUDENT_ENGINE_PATH}"
                    --student-engine-sha256 "${STUDENT_ENGINE_SHA256}"
                )
            fi
        fi
    else
        if [[ -n "${OPTIMIZED_STUDENT_ENGINE_PATH}" ]]; then
            ENGINE_ARGUMENTS+=(
                --optimized-student-engine "${OPTIMIZED_STUDENT_ENGINE_PATH}"
            )
            if [[ -n "${OPTIMIZED_STUDENT_ENGINE_SHA256}" ]]; then
                ENGINE_ARGUMENTS+=(
                    --optimized-student-engine-sha256 "${OPTIMIZED_STUDENT_ENGINE_SHA256}"
                )
            fi
        else
            ENGINE_ARGUMENTS+=(--student-engine "${STUDENT_ENGINE_PATH}")
            if [[ -n "${STUDENT_ENGINE_SHA256}" ]]; then
                ENGINE_ARGUMENTS+=(--student-engine-sha256 "${STUDENT_ENGINE_SHA256}")
            fi
        fi
    fi

    NO_CRACK=0
    for argument in "$@"; do
        if [[ "${argument}" == "--no-crack" ]]; then
            NO_CRACK=1
            break
        fi
    done
    if [[ "${NO_CRACK}" == "0" && "${REALTIME_TEST}" == "0" ]]; then
        ENGINE_ARGUMENTS+=(
            --capture-hrsegnet-crack-engine "${CAPTURE_HRSEGNET_CRACK_ENGINE_PATH}"
            --capture-hrsegnet-crack-engine-sha256 "${CAPTURE_HRSEGNET_CRACK_ENGINE_SHA256}"
            --capture-hrsegnet-crack-probability-threshold "${CAPTURE_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}"
            --capture-hrsegnet-crack-min-component-pixels "${CAPTURE_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}"
        )
    fi
    if [[ "${HRSEGNET_SELECTED}" == "1" ]]; then
        if [[ -n "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}" ]]; then
            ENGINE_ARGUMENTS+=(
                --realtime-hrsegnet-crack-engine "${REALTIME_HRSEGNET_CRACK_ENGINE_PATH}"
                --realtime-hrsegnet-crack-engine-sha256 "${REALTIME_HRSEGNET_CRACK_ENGINE_SHA256}"
                --realtime-hrsegnet-crack-probability-threshold "${REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}"
                --realtime-hrsegnet-crack-min-component-pixels "${REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}"
            )
            echo "Realtime crack baseline: operator-selected pinned HrSegNet threshold=${REALTIME_HRSEGNET_CRACK_PROBABILITY_THRESHOLD}, min_component_pixels=${REALTIME_HRSEGNET_CRACK_MIN_COMPONENT_PIXELS}; field accuracy is not certified." >&2
        else
            echo "WARNING: UNLOCKED exploratory HrSegNet CLI settings; TEST ONLY." >&2
        fi
    elif [[ "${NO_CRACK}" == "0" ]]; then
        ENGINE_ARGUMENTS+=(
            --realtime-crack-threshold "${REALTIME_CRACK_THRESHOLD}"
            --realtime-crack-min-component-pixels "${REALTIME_CRACK_MIN_COMPONENT_PIXELS}"
        )
        if [[ -z "${REALTIME_MULTITASK_ENGINE_PATH}" ]]; then
            ENGINE_ARGUMENTS+=(--realtime-crack-engine "${REALTIME_CRACK_ENGINE_PATH}")
        fi
        if [[ -z "${REALTIME_MULTITASK_ENGINE_PATH}" && -n "${REALTIME_CRACK_ENGINE_SHA256}" ]]; then
            ENGINE_ARGUMENTS+=(--realtime-crack-engine-sha256 "${REALTIME_CRACK_ENGINE_SHA256}")
        fi
    fi
fi

bash "${PROJECT_ROOT}/scripts/requirement.sh"

exec "${PYTHON}" "${PROJECT_ROOT}/jetson_code/run.py" \
    "${ENGINE_ARGUMENTS[@]}" \
    "$@" \
    "${NORMAL_DUAL_CAMERA_ARGUMENTS[@]}"
