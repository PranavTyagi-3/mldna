import subprocess

from typer.testing import CliRunner

from mldna.cli.main import app
from mldna.core.push import push as core_push
from mldna.tracking import track_run

runner = CliRunner()


def test_runs_cli_no_runs_yet(producer_repo):
    result = runner.invoke(app, ["runs", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "No runs found." in result.output


def test_runs_cli_lists_a_run(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        pass

    result = runner.invoke(app, ["runs", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert run.run_id[:8] in result.output


def test_runs_cli_missing_mldna_yaml_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["runs", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "runs failed" in result.output
