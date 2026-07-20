import subprocess

from typer.testing import CliRunner

from mlops.cli.main import app
from mlops.core.push import push as core_push
from mlops.publish import publish
from mlops.tracking import track_run

runner = CliRunner()


def _commit_and_push(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)


def test_pull_cli_reports_no_published_model(producer_repo):
    _commit_and_push(producer_repo)

    result = runner.invoke(app, ["pull", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "Pulled code and data." in result.output
    assert "No published model found" in result.output


def test_pull_cli_reports_published_model(producer_repo):
    _commit_and_push(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    publish(pkl, workspace_dir=producer_repo, confirm=lambda s: True)

    result = runner.invoke(app, ["pull", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert f"Pulled published model bundle from run {run.run_id}" in result.output


def test_pull_cli_fails_in_consumer_repo(tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--consumer"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["pull", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "mlops fetch" in result.output
