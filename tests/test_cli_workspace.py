import json
import os
import subprocess

from typer.testing import CliRunner

from mldna.cli.main import app

runner = CliRunner()


def _commit(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, capture_output=True)


def test_new_outside_a_repo_prints_refusal_without_traceback(tmp_path):
    result = runner.invoke(app, ["workspace", "new", "a", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "workspace new failed:" in result.output
    assert "Fix:" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_new_list_remove_round_trip(producer_repo):
    _commit(producer_repo)
    dir = str(producer_repo)

    created = runner.invoke(app, ["workspace", "new", "a", "--dir", dir])
    listed = runner.invoke(app, ["workspace", "list", "--dir", dir])
    removed = runner.invoke(app, ["workspace", "remove", "a", "--dir", dir])

    assert created.exit_code == 0
    assert f"{producer_repo.name}-a" in listed.stdout
    assert removed.exit_code == 0
    assert not (producer_repo.parent / f"{producer_repo.name}-a").exists()


def test_remove_refuses_under_a_live_lock(producer_repo):
    _commit(producer_repo)
    dir = str(producer_repo)
    runner.invoke(app, ["workspace", "new", "a", "--dir", dir])
    path = producer_repo.parent / f"{producer_repo.name}-a"
    (path / ".mldna").mkdir(exist_ok=True)
    (path / ".mldna" / "run.lock").write_text(json.dumps({"pid": os.getpid()}))

    result = runner.invoke(app, ["workspace", "remove", "a", "--dir", dir])

    assert result.exit_code == 1
    assert "live run lock" in result.output
    assert path.exists()


def _workspace_with_model(producer_repo):
    _commit(producer_repo)
    dir = str(producer_repo)
    runner.invoke(app, ["workspace", "new", "a", "--dir", dir])
    path = producer_repo.parent / f"{producer_repo.name}-a"
    (path / "models").mkdir(exist_ok=True)
    model = path / "models" / "best.pkl"
    model.write_bytes(b"unpublished")
    return dir, path, model


def test_remove_non_interactive_without_yes_refuses_naming_yes(producer_repo):
    dir, path, model = _workspace_with_model(producer_repo)

    result = runner.invoke(app, ["workspace", "remove", "a", "--dir", dir])

    assert result.exit_code == 1
    assert "--yes" in result.output
    assert "models/best.pkl" in result.output
    assert model.exists()


def test_remove_yes_deletes_the_ignored_model(producer_repo):
    dir, path, model = _workspace_with_model(producer_repo)

    result = runner.invoke(app, ["workspace", "remove", "a", "--dir", dir, "--yes"])

    assert result.exit_code == 0
    assert not path.exists()


def test_remove_interactive_decline_keeps_the_workspace(producer_repo, monkeypatch):
    dir, path, model = _workspace_with_model(producer_repo)
    monkeypatch.setattr("mldna.cli.main._stdin_is_interactive", lambda: True)

    result = runner.invoke(app, ["workspace", "remove", "a", "--dir", dir], input="n\n")

    assert result.exit_code == 1
    assert model.exists()


def test_remove_interactive_accept_removes_the_workspace(producer_repo, monkeypatch):
    dir, path, model = _workspace_with_model(producer_repo)
    monkeypatch.setattr("mldna.cli.main._stdin_is_interactive", lambda: True)

    result = runner.invoke(app, ["workspace", "remove", "a", "--dir", dir], input="y\n")

    assert result.exit_code == 0
    assert not path.exists()
