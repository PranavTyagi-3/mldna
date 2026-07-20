import subprocess

from typer.testing import CliRunner

from mlops.cli.main import app
from mlops.core.push import push as core_push
from mlops.tracking import track_run

runner = CliRunner()


def _tracked_pkl(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    return pkl, run.run_id


def test_publish_yes_succeeds_noninteractively(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    result = runner.invoke(app, [
        "publish", str(pkl), "--dir", str(producer_repo), "--yes",
    ])

    assert result.exit_code == 0
    assert "Published test-model v" in result.stdout


def test_publish_without_yes_declines(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    result = runner.invoke(app, [
        "publish", str(pkl), "--dir", str(producer_repo),
    ], input="n\n")

    assert result.exit_code == 1
    assert "Aborted" in result.output
