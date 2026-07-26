from typer.testing import CliRunner

import mldna
from mldna.cli.main import app

runner = CliRunner()


def test_package_has_version():
    # Assert agreement with pyproject.toml rather than a literal, so a
    # version bump can't silently leave the two sources out of sync.
    # (Regex, not tomllib: tomllib is 3.11+ and this package supports 3.10.)
    import re
    from pathlib import Path
    pyproject = (Path(__file__).parent.parent / "pyproject.toml").read_text()
    declared = re.search(r'^version = "(.+)"$', pyproject, re.MULTILINE).group(1)
    assert mldna.__version__ == declared


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert mldna.__version__ in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
