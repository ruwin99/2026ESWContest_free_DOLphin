#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" == "0" ]]; then
    CAMERA_ARGUMENTS=()
elif [[ "$#" == "2" && "$1" == "--camera-index" && "$2" =~ ^[0-9]+$ ]]; then
    CAMERA_ARGUMENTS=("$1" "$2")
else
    echo "Usage: $0 [--camera-index NONNEGATIVE_INTEGER]" >&2
    exit 2
fi

echo "dualhead96test.sh is deprecated; using the pinned Rust + HrSegNet realtime baseline." >&2
exec bash "${PROJECT_ROOT}/scripts/hrtest.sh" "${CAMERA_ARGUMENTS[@]}"
