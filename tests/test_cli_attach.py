import subprocess

import mlflow
from typer.testing import CliRunner

from mldna.cli.main import app
from mldna.core.push import push as core_push
from mldna.tracking import track_run

runner = CliRunner()


def _tracked_pkl_with_network_blip(repo, monkeypatch):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)

    def boom(*a, **kw):
        raise RuntimeError("network blip")

    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
        monkeypatch.setattr(mlflow, "set_tag", boom)
        monkeypatch.setattr(mlflow, "end_run", boom)
    return pkl, run.run_id


def test_attach_recovers_a_run_left_running(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl_with_network_blip(producer_repo, monkeypatch)

    result = runner.invoke(app, ["attach", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert run_id in result.stdout
    assert "1 checksum tag(s) written" in result.stdout


def test_attach_without_a_marker_prints_refusal_without_traceback(producer_repo):
    result = runner.invoke(app, ["attach", "--dir", str(producer_repo)])

    assert result.exit_code == 1
    assert "attach failed:" in result.output
    assert "last_run.json" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_attach_nothing_to_do_prints_refusal_without_traceback(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)
    core_push(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")

    result = runner.invoke(app, ["attach", "--dir", str(producer_repo)])

    assert result.exit_code == 1
    assert "attach failed:" in result.output
    assert "nothing" in result.output.lower()
    assert result.exception is None or isinstance(result.exception, SystemExit)
