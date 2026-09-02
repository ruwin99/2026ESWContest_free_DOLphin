from __future__ import annotations

import json
import re
import sys
from pathlib import Path


TRAINING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAINING_ROOT.parent
REQUIRED = (
    ".gitignore",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "code/README.md",
    "code/requirements-test.txt",
    "dashboard/.env.example",
    "dashboard/.firebaserc.example",
    "dashboard/README.md",
    "dashboard/firebase-client",
    "dashboard/firebase.json",
    "dashboard/firebase.vite.config.ts",
    "dashboard/firestore.rules",
    "dashboard/storage.rules",
    "docs/MODEL_STATUS.md",
    "docs/SAFETY_AND_LIMITATIONS.md",
    "stm_code/README.md",
    "data_training/.gitignore",
    "data_training/README_KR.md",
    "data_training/COMPETITION_CODE_GUIDE.md",
    "data_training/THIRD_PARTY_NOTICES.md",
    "data_training/PUBLIC_RELEASE_MANIFEST.txt",
    "data_training/vt_kd",
    "data_training/steelcrack",
    "data_training/realtime_w1280",
    "data_training/capture_1280x720",
    "data_training/YOLO26_hard_negative_1280x720",
    "data_training/yolo26_obstacle",
    "data_training/capture_rust_teacher_hardneg_v1",
    "data_training/realtime_multitask_5ch_hrseg_v1",
    "data_training/realtime_light_dualhead_96_v1/models/light_dualhead_96.py",
    "data_training/realtime_light_dualhead_96_v1/models/lraspp_96.py",
    "data_training/realtime_light_dualhead_96_v1/models/mobilenetv2_os8_encoder.py",
)
BINARY_MODEL_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".plan", ".trt"}
TEXT_SUFFIXES = {
    ".c",
    ".cmd",
    ".css",
    ".example",
    ".h",
    ".html",
    ".js",
    ".json",
    ".jsx",
    ".md",
    ".mjs",
    ".ps1",
    ".py",
    ".rules",
    ".s",
    ".sh",
    ".sql",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
ABSOLUTE_WINDOWS_PATH = re.compile(
    r"[A-Za-z]:(?:[\\/]|\\\\)+(?:Users|Documents and Settings)(?:[\\/]|\\\\)+",
    re.IGNORECASE,
)
ABSOLUTE_HOME_PATH = re.compile(r"/home/[A-Za-z0-9._-]+/")
SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?:api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*[\"']([^\"']+)[\"']"
)
PLACEHOLDER_MARKERS = ("본인의", "example", "placeholder", "redacted", "환경", "<", "[")


def relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []

    for name in REQUIRED:
        if not (REPO_ROOT / name).exists():
            errors.append(f"required path missing: {name}")

    for path in REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts:
            continue
        if path.suffix.lower() in BINARY_MODEL_SUFFIXES:
            errors.append(f"model artifact must not be committed by default: {relative(path)}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            warnings.append(f"text decode skipped: {relative(path)}")
            continue

        if path.suffix.lower() == ".py":
            try:
                compile(text, str(path), "exec")
            except SyntaxError as exc:
                errors.append(f"python syntax: {relative(path)}:{exc.lineno}: {exc.msg}")
        elif path.suffix.lower() == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                errors.append(f"json syntax: {relative(path)}:{exc.lineno}: {exc.msg}")

        if ABSOLUTE_WINDOWS_PATH.search(text) or ABSOLUTE_HOME_PATH.search(text):
            errors.append(f"personal absolute path: {relative(path)}")
        for match in SECRET_ASSIGNMENT.finditer(text):
            value = match.group(1).strip().lower()
            if value and not any(marker in value for marker in PLACEHOLDER_MARKERS):
                errors.append(f"possible embedded secret: {relative(path)}")

    report = {
        "root": ".",
        "passed": not errors,
        "errors": sorted(set(errors)),
        "warnings": sorted(set(warnings)),
        "note": "The verifier scans the complete repository. Model binaries, embedded secrets and personal absolute paths are release blockers.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
