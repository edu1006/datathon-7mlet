from pathlib import Path

import pytest

from src.data import FIXTURE_CSV

FIXTURE = Path(FIXTURE_CSV)


@pytest.fixture
def fixture_csv() -> Path:
    return FIXTURE
