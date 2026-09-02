from __future__ import annotations

from pathlib import Path

from common import audit_teacher, load_config, resolve_path


WORK = Path(__file__).resolve().parents[1]
CONFIG = WORK / "configs" / "light_dualhead_96_w1280_h240.yaml"


def test_rust_teacher_matches_handoff_contract() -> None:
    config = load_config(CONFIG)
    result = audit_teacher(resolve_path(config["paths"]["rust_teacher_onnx"]), config["teachers"]["rust"])
    assert not result["issues"]


def test_supplied_hrseg_teacher_matches_handoff_contract() -> None:
    config = load_config(CONFIG)
    result = audit_teacher(resolve_path(config["paths"]["hrseg_teacher_onnx"]), config["teachers"]["crack"])
    assert not result["issues"]
