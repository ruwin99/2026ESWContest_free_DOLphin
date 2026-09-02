from __future__ import annotations

import sys
from pathlib import Path

import pytest


WORK = Path(__file__).resolve().parents[1]
for directory in (WORK / "models", WORK / "tools"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))


@pytest.fixture
def official_training() -> Path:
    return WORK.parents[1] / "models" / "virginia_tech_cssd" / "official_code" / "Training - Testing"
