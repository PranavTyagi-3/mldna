from typer.testing import CliRunner

from mlops.cli.main import app

runner = CliRunner()


def test_push_commits_and_pushes(producer_repo):
    result = runner.invoke(app, ["push", "--message", "cli push", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "Committed: cli push" in result.stdout
    assert "Pushed." in result.stdout
