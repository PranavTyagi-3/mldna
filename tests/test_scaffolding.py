from typer.testing import CliRunner

import mlops
from mlops.cli.main import app

runner = CliRunner()


def test_package_has_version():
    assert mlops.__version__ == "0.0.1"


def test_cli_version_flag():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert mlops.__version__ in result.stdout


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
