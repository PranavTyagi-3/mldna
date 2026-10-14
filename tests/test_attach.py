import json
import subprocess

import mlflow
import pytest

from mldna.attach import AttachError, attach
from mldna.core import hashing
from mldna.core.rungates import RunGateError, check_run_gates
from mldna.tracking import track_run


def _get_run(workspace_dir, run_id):
    mlflow.set_tracking_uri(f"file:{workspace_dir / 'mlruns'}")
    return mlflow.get_run(run_id)


@pytest.fixture
def dvc_project(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True, capture_output=True
    )
    return producer_repo


def _train_with_network_blip_at_exit(workspace_dir, monkeypatch):
    """issue #10: the exit-time checksum tagging AND end_run both fail, as a
    single network blip would cause. The run is left RUNNING server-side with
    none of its model_sha256__<n>/save_context tags - but the hash itself was
    already computed and saved locally by _write_last_run."""
    def boom(*a, **kw):
        raise RuntimeError("network blip")

    with track_run(workspace_dir=workspace_dir) as run:
        run_id = run.run_id
        (workspace_dir / "model.pkl").write_bytes(b"trained model bytes")
        monkeypatch.setattr(mlflow, "set_tag", boom)
        monkeypatch.setattr(mlflow, "end_run", boom)
    return run_id


# --- happy path ---

def test_attach_recovers_a_run_left_running(git_repo, monkeypatch):
    run_id = _train_with_network_blip_at_exit(git_repo, monkeypatch)

    run = _get_run(git_repo, run_id)
    assert run.info.status == "RUNNING"
    assert "model_sha256__0" not in run.data.tags

    result = attach(workspace_dir=git_repo)

    assert result == {"run_id": run_id, "checksums_written": 1}

    run = _get_run(git_repo, run_id)
    assert run.info.status == "FINISHED"
    expected_sha = hashing.sha256_file(git_repo / "model.pkl")
    assert run.data.tags["model_sha256__0"] == f"{expected_sha}:model.pkl"
    assert run.data.tags["save_context"] == "in_run"


def test_attached_run_carries_the_retroactive_label(git_repo, monkeypatch):
    run_id = _train_with_network_blip_at_exit(git_repo, monkeypatch)

    attach(workspace_dir=git_repo)

    tags = _get_run(git_repo, run_id).data.tags
    assert tags["attached"] == "true"
    assert "attached_at" in tags


def test_attach_makes_the_run_publishable(dvc_project, monkeypatch):
    run_id = _train_with_network_blip_at_exit(dvc_project, monkeypatch)

    # Before attach: still RUNNING, so publish's gates must refuse.
    with pytest.raises(RunGateError):
        check_run_gates(run_id)

    attach(workspace_dir=dvc_project)

    run = check_run_gates(run_id)
    assert run.info.status == "FINISHED"


# --- refusal paths ---

def test_attach_refuses_without_a_marker(git_repo):
    assert not (git_repo / ".mldna" / "last_run.json").exists()

    with pytest.raises(AttachError) as e:
        attach(workspace_dir=git_repo)

    assert "last_run.json" in str(e.value)


def test_attach_refuses_an_unknown_run_id(git_repo):
    marker_path = git_repo / ".mldna" / "last_run.json"
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(json.dumps({
        "run_id": "doesnotexist", "git_commit": "a" * 40,
        "status": "FINISHED", "checksums": {},
    }))

    with pytest.raises(AttachError) as e:
        attach(workspace_dir=git_repo)

    assert "doesnotexist" in str(e.value)


def test_attach_refuses_a_record_that_does_not_match_the_run(git_repo, monkeypatch):
    run_id = _train_with_network_blip_at_exit(git_repo, monkeypatch)

    marker_path = git_repo / ".mldna" / "last_run.json"
    marker = json.loads(marker_path.read_text())
    marker["git_commit"] = "0" * 40
    marker_path.write_text(json.dumps(marker))

    with pytest.raises(AttachError) as e:
        attach(workspace_dir=git_repo)

    assert "does not match" in str(e.value)
    assert run_id in str(e.value)


def test_attach_refuses_when_nothing_to_do(git_repo):
    # A normal successful run already carries its tags and FINISHED status -
    # attach has nothing to add.
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "model.pkl").write_bytes(b"trained model bytes")

    with pytest.raises(AttachError) as e:
        attach(workspace_dir=git_repo)

    assert "nothing" in str(e.value).lower()
    assert run_id in str(e.value)


# --- an earlier run is not lost when a later run overwrites last_run.json ---

def test_attach_run_id_recovers_an_earlier_run_after_a_later_run(git_repo, monkeypatch):
    first_run_id = _train_with_network_blip_at_exit(git_repo, monkeypatch)
    monkeypatch.undo()
    mlflow.end_run()  # a retry is a new process; clear the still-active run

    with track_run(workspace_dir=git_repo, allow_dirty=True) as second:
        (git_repo / "model2.pkl").write_bytes(b"second model bytes")

    latest = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert latest["run_id"] == second.run_id
    with pytest.raises(AttachError):
        attach(workspace_dir=git_repo)

    result = attach(workspace_dir=git_repo, run_id=first_run_id[:8])

    assert result["run_id"] == first_run_id
    assert _get_run(git_repo, first_run_id).info.status == "FINISHED"


def test_attach_run_id_matching_no_saved_record_lists_what_is_saved(git_repo, monkeypatch):
    run_id = _train_with_network_blip_at_exit(git_repo, monkeypatch)

    with pytest.raises(AttachError) as e:
        attach(workspace_dir=git_repo, run_id="zzzz")

    assert "--run-id zzzz" in str(e.value)
    assert run_id in str(e.value)


def test_attach_run_id_listing_shows_only_the_ten_newest_records(git_repo):
    import os

    runs_dir = git_repo / ".mldna" / "runs"
    runs_dir.mkdir(parents=True)
    for i in range(15):
        record = runs_dir / f"run{i:02d}.json"
        record.write_text("{}")
        os.utime(record, (1_700_000_000 + i, 1_700_000_000 + i))

    with pytest.raises(AttachError) as e:
        attach(workspace_dir=git_repo, run_id="zzzz")

    message = str(e.value)
    assert "run14" in message and "run05" in message
    assert "run04" not in message
    assert "... and 5 more" in message
