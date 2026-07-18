from typer.testing import CliRunner

from mlops.cli.main import app

runner = CliRunner()


def test_init_scaffolds_project(tmp_path):
    result = runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path)])

    assert result.exit_code == 0
    assert (tmp_path / "mlops.yaml").exists()
    assert "Initialized producer project 'fraud-detector'" in result.stdout
