from pathlib import Path

import pytest

tomllib = pytest.importorskip("tomllib")  # stdlib from 3.11

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def _build_targets():
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["tool"]["hatch"]["build"]["targets"]


def test_wheel_ships_only_the_mldna_package():
    # Anything outside src/mldna (examples/, tests/, docs/) stays out of the wheel.
    assert _build_targets()["wheel"]["packages"] == ["src/mldna"]


def test_sdist_excludes_the_examples():
    assert "examples" in _build_targets()["sdist"]["exclude"]
