import json

import mlflow
import pytest

from mlops.core.lock import RunLock, RunLockError
from mlops.tracking import TrackRunError, track_run


def _get_run(workspace_dir, run_id):
    mlflow.set_tracking_uri(f"file:{workspace_dir / 'mlruns'}")
    return mlflow.get_run(run_id)


# --- refusal paths ---

def test_dirty_tree_refuses(git_repo):
    (git_repo / "README.md").write_text("dirty\n")

    with pytest.raises(TrackRunError):
        with track_run("test-model", workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False


def test_dirty_tree_allowed_with_allow_dirty(git_repo):
    (git_repo / "README.md").write_text("dirty\n")

    with track_run("test-model", workspace_dir=git_repo, allow_dirty=True) as run:
        run_id = run.run_id

    assert _get_run(git_repo, run_id).data.tags["dirty"] == "true"


def test_no_identity_refuses(tmp_path, monkeypatch):
    empty_config = tmp_path / "empty_gitconfig"
    empty_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_config))

    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    with pytest.raises(TrackRunError):
        with track_run("test-model", workspace_dir=tmp_path):
            pass

    assert RunLock(tmp_path).is_locked() is False


def test_lock_already_held_refuses(git_repo):
    RunLock(git_repo).acquire()

    with pytest.raises(RunLockError):
        with track_run("test-model", workspace_dir=git_repo):
            pass


# --- happy path ---

def test_run_has_lineage_tags(git_repo):
    with track_run("test-model", workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags = _get_run(git_repo, run_id).data.tags
    assert len(tags["git_commit"]) == 40
    assert tags["git_branch"] in ("main", "master")
    assert tags["user"] == "dev@example.com"
    assert "mlops_version" in tags
    assert "dirty" not in tags
    assert "dvc_data_hash" not in tags


def test_dvc_data_hash_tag_set_when_dataset_tracked(producer_repo):
    import subprocess
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)

    with track_run("test-model", workspace_dir=producer_repo) as run:
        run_id = run.run_id

    tags = _get_run(producer_repo, run_id).data.tags
    assert tags["dvc_data_hash"].endswith(".dir")


def test_model_file_is_discovered_and_hashed(git_repo):
    with track_run("test-model", workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "model.pkl").write_bytes(b"trained model bytes")

    tags = _get_run(git_repo, run_id).data.tags
    assert tags["save_context"] == "in_run"
    assert tags["model_sha256__0"].endswith(":model.pkl")

    checksums_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checksums.json"
    )
    checksums = json.loads(open(checksums_path).read())
    assert checksums["model.pkl"]["context"] == "in_run"


def test_last_run_marker_written_on_success(git_repo):
    with track_run("test-model", workspace_dir=git_repo) as run:
        run_id = run.run_id

    marker = json.loads((git_repo / ".mlops" / "last_run.json").read_text())
    assert marker["run_id"] == run_id
    assert marker["status"] == "FINISHED"


def test_failed_run_marks_status_and_releases_lock(git_repo):
    with pytest.raises(RuntimeError):
        with track_run("test-model", workspace_dir=git_repo) as run:
            run_id = run.run_id
            raise RuntimeError("boom")

    assert _get_run(git_repo, run_id).info.status == "FAILED"
    marker = json.loads((git_repo / ".mlops" / "last_run.json").read_text())
    assert marker["status"] == "FAILED"
    assert RunLock(git_repo).is_locked() is False
