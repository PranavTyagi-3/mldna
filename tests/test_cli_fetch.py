import subprocess

from typer.testing import CliRunner

from mlops.cli.main import app
from mlops.core.push import push as core_push
from mlops.publish import publish
from mlops.tracking import track_run

runner = CliRunner()


def _publish_pkl_only(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    return publish(pkl, workspace_dir=repo, confirm=lambda s: True)


def test_fetch_cli_happy_path(producer_repo):
    result = _publish_pkl_only(producer_repo)
    (producer_repo / "models.lock.yaml").write_text(
        f"test-model:\n  version: {result['version']}\n  format: pkl\n  dest: app/models/test\n"
    )

    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 0
    assert "test-model v" in invocation.stdout
    assert (producer_repo / "app" / "models" / "test" / "model.pkl").exists()


def test_fetch_cli_missing_lockfile_fails_cleanly(producer_repo):
    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 1
    assert "fetch failed" in invocation.output
    assert "models.lock.yaml" in invocation.output
