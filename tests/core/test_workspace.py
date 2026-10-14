import json
import os
import subprocess
from pathlib import Path

import pytest

from mldna.core import dvc
from mldna.core.workspace import WorkspaceError, list_workspaces, new, remove

DEAD_PID = 2**30


@pytest.fixture
def repo(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=producer_repo, check=True, capture_output=True)
    return producer_repo


def _cache_dir(path):
    return Path(dvc.run(["cache", "dir"], cwd=path).stdout.strip()).resolve()


def _write_lock(path, pid):
    lock = path / ".mldna" / "run.lock"
    lock.parent.mkdir(exist_ok=True)
    lock.write_text(json.dumps({"pid": pid, "run_id": None, "started_at": "now"}))


def test_new_outside_a_git_repo_refuses(tmp_path):
    with pytest.raises(WorkspaceError, match="needs a git repository"):
        new(tmp_path, "a")


@pytest.mark.parametrize("name", ["../x", "a/b", "..", "", "a\\b", "/abs"])
def test_new_refuses_names_that_are_not_plain(repo, name):
    with pytest.raises(WorkspaceError, match="not a plain name"):
        new(repo, name)


def test_new_refuses_an_existing_name(repo):
    new(repo, "a")
    with pytest.raises(WorkspaceError, match="already exists") as e:
        new(repo, "a")
    assert "mldna workspace remove a" in str(e.value)


def test_new_creates_a_sibling_worktree(repo):
    path = new(repo, "a")

    assert path == repo.resolve().parent / f"{repo.name}-a"
    assert (path / "mldna.yaml").exists()


def test_new_survives_a_failed_checkout_and_prints_the_hint(repo, monkeypatch, capsys):
    real_run = dvc.run

    def run(args, cwd, check=True):
        if args == ["checkout"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="ERROR: object missing")
        return real_run(args, cwd, check)

    monkeypatch.setattr("mldna.core.workspace.dvc.run", run)

    path = new(repo, "a")

    assert (path / "mldna.yaml").exists()
    out = capsys.readouterr().out
    assert "data not checked out: ERROR: object missing" in out
    assert f"Fix: run `dvc pull` in {path}" in out


def test_new_checks_out_the_tracked_data(repo, capsys):
    path = new(repo, "a")

    files = [f.name for f in (repo / "dataset").iterdir()]
    assert files
    assert sorted(f.name for f in (path / "dataset").iterdir()) == sorted(files)
    assert "data checked out" in capsys.readouterr().out


def test_new_shares_the_dvc_cache(repo):
    path = new(repo, "a")

    assert _cache_dir(path) == _cache_dir(repo)
    assert _cache_dir(path) != (path / ".dvc" / "cache").resolve()


def test_new_gives_fresh_mldna_state(repo):
    _write_lock(repo, DEAD_PID)
    (repo / ".mldna" / "last_run.json").write_text("{}")

    path = new(repo, "a")

    assert not (path / ".mldna" / "run.lock").exists()
    assert not (path / ".mldna" / "last_run.json").exists()


def test_list_shows_main_and_new_workspace(repo):
    path = new(repo, "a")

    assert list_workspaces(repo) == [("(main)", repo.resolve()), ("a", path.resolve())]


def test_remove_refuses_a_name_that_is_not_a_worktree(repo):
    with pytest.raises(WorkspaceError, match="not a workspace") as e:
        remove(repo, "nope")
    assert "mldna workspace list" in str(e.value)


def test_remove_refuses_the_main_worktree(repo):
    with pytest.raises(WorkspaceError, match="main working copy"):
        remove(repo, repo.name)
    assert repo.exists()


def test_remove_refuses_while_the_lock_is_live(repo):
    path = new(repo, "a")
    _write_lock(path, os.getpid())

    with pytest.raises(WorkspaceError, match="live run lock") as e:
        remove(repo, "a")

    assert "mldna workspace remove a" in str(e.value)
    assert path.exists()
    assert ("a", path.resolve()) in list_workspaces(repo)


def test_remove_succeeds_without_a_lock(repo):
    path = new(repo, "a")

    assert remove(repo, "a") is False
    assert not path.exists()
    assert list_workspaces(repo) == [("(main)", repo.resolve())]


def test_remove_proceeds_past_a_stale_lock_and_reports_it(repo):
    path = new(repo, "a")
    _write_lock(path, DEAD_PID)

    assert remove(repo, "a") is True
    assert not path.exists()


def _ignored_model(path):
    (path / "models").mkdir(exist_ok=True)
    model = path / "models" / "best.pkl"
    model.write_bytes(b"unpublished")
    return model


def test_remove_refuses_an_ignored_model_without_confirmation(repo):
    path = new(repo, "a")
    model = _ignored_model(path)

    with pytest.raises(WorkspaceError, match="cannot restore") as e:
        remove(repo, "a")

    assert "models/best.pkl" in str(e.value)
    assert "--yes" in str(e.value)
    assert model.exists()


def test_remove_proceeds_when_confirmed(repo):
    path = new(repo, "a")
    _ignored_model(path)
    seen = []

    remove(repo, "a", lambda msg: seen.append(msg) or True)

    assert "models/best.pkl" in seen[0]
    assert not path.exists()


def test_remove_warns_about_a_local_mlruns(repo):
    path = new(repo, "a")
    (path / "mlruns").mkdir()
    (path / "mlruns" / "meta.yaml").write_text("x")

    with pytest.raises(WorkspaceError, match="mlruns/"):
        remove(repo, "a")

    assert path.exists()


def test_remove_without_ignored_models_never_asks(repo):
    path = new(repo, "a")

    remove(repo, "a", lambda msg: pytest.fail("must not prompt"))

    assert not path.exists()


def test_remove_live_lock_refusal_comes_before_the_model_check(repo):
    path = new(repo, "a")
    _ignored_model(path)
    _write_lock(path, os.getpid())

    with pytest.raises(WorkspaceError, match="live run lock"):
        remove(repo, "a", lambda msg: pytest.fail("must not prompt"))
