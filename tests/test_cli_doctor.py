import subprocess

from typer.testing import CliRunner

from mlops.cli.main import app

runner = CliRunner()


def test_doctor_reports_all_checks(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)

    result = runner.invoke(app, ["doctor", "--dir", str(producer_repo)])

    assert "tools:" in result.output
    assert "gitignore:" in result.output
    assert "run_lock:" in result.output
    assert "git_remote:" in result.output
    assert "committed_secrets:" in result.output
    assert "aml_auth:" in result.output
    assert "blob_auth:" in result.output


def test_doctor_exit_code_reflects_failures(producer_repo):
    (producer_repo / ".gitignore").write_text("models/\n")

    result = runner.invoke(app, ["doctor", "--dir", str(producer_repo)])

    assert result.exit_code == 1
    assert "[FAIL] gitignore" in result.output
