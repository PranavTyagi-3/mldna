from typer.testing import CliRunner

import mldna
from mldna.cli.main import app

runner = CliRunner()


def test_package_version_has_a_single_source():
    # __version__ is the only version literal: pyproject declares the version
    # dynamic and hatchling reads it from this module. Guard that wiring rather
    # than a literal - scaffolding emits `mldna-lineage=={__version__}` pins, so
    # a static pyproject version drifting from the module would ship a pin to a
    # release that does not exist.
    # (Regex, not tomllib: tomllib is 3.11+ and this package supports 3.10.)
    import re
    from pathlib import Path
    root = Path(__file__).parent.parent
    pyproject = (root / "pyproject.toml").read_text()
    project = re.split(r"\n\[", pyproject.split("\n[project]\n", 1)[1], maxsplit=1)[0]
    assert not re.search(r"^version = ", project, re.MULTILINE)
    assert re.search(r'^dynamic = \[[^]]*"version"', project, re.MULTILINE)
    source = re.search(
        r'^\[tool\.hatch\.version\]\npath = "([^"]+)"', pyproject, re.MULTILINE
    )
    assert source and (root / source.group(1)).resolve() == Path(mldna.__file__).resolve()


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert mldna.__version__ in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
