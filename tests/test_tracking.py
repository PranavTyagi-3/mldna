import json
import subprocess

import mlflow
import pytest

from mldna.core.lock import RunLock, RunLockError
from mldna.tracking import TrackRunError, track_run


def _get_run(workspace_dir, run_id):
    mlflow.set_tracking_uri(f"file:{workspace_dir / 'mlruns'}")
    return mlflow.get_run(run_id)


# --- refusal paths ---

def test_dirty_tree_refuses(git_repo):
    (git_repo / "README.md").write_text("dirty\n")

    with pytest.raises(TrackRunError):
        with track_run(workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False


def test_dirty_tree_allowed_with_allow_dirty(git_repo):
    (git_repo / "README.md").write_text("dirty\n")

    with track_run(workspace_dir=git_repo, allow_dirty=True) as run:
        run_id = run.run_id

    assert _get_run(git_repo, run_id).data.tags["dirty"] == "true"


def test_no_identity_refuses(tmp_path, monkeypatch):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "mldna.yaml").write_text("project:\n  model_name: test-model\n  repo_role: producer\n")
    subprocess.run(["git", "add", "mldna.yaml"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=dev@example.com", "-c", "user.name=dev",
         "commit", "-q", "-m", "init"],
        cwd=tmp_path, check=True,
    )

    empty_config = tmp_path / "empty_gitconfig"
    empty_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_config))

    with pytest.raises(TrackRunError):
        with track_run(workspace_dir=tmp_path):
            pass

    assert RunLock(tmp_path).is_locked() is False


def test_lock_already_held_refuses(git_repo):
    RunLock(git_repo).acquire()

    with pytest.raises(RunLockError):
        with track_run(workspace_dir=git_repo):
            pass


# --- happy path ---

def test_run_has_lineage_tags(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags = _get_run(git_repo, run_id).data.tags
    assert len(tags["git_commit"]) == 40
    assert tags["git_branch"] in ("main", "master")
    assert tags["user"] == "dev@example.com"
    assert "mldna_version" in tags
    assert "dirty" not in tags
    assert "dvc_data_hash" not in tags


def test_dvc_data_hash_tag_set_when_dataset_tracked(producer_repo):
    import subprocess
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    tags = _get_run(producer_repo, run_id).data.tags
    assert tags["dvc_data_hash"].endswith(".dir")


def test_model_file_is_discovered_and_hashed(git_repo):
    with track_run(workspace_dir=git_repo) as run:
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
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["run_id"] == run_id
    assert marker["status"] == "FINISHED"


def test_custom_model_glob_from_mldna_yaml_is_honored(git_repo):
    (git_repo / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  repo_role: producer\n"
        "tracking:\n  model_globs: [\"*.custom\"]\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add custom glob"], cwd=git_repo, check=True)

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "model.custom").write_bytes(b"custom format model bytes")

    tags = _get_run(git_repo, run_id).data.tags
    assert tags["model_sha256__0"].endswith(":model.custom")


def test_custom_scan_exclude_from_mldna_yaml_is_honored(git_repo):
    (git_repo / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  repo_role: producer\n"
        "tracking:\n  scan_exclude: [\"ignored_dir/\"]\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add scan exclude"], cwd=git_repo, check=True)

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "ignored_dir").mkdir()
        (git_repo / "ignored_dir" / "model.pkl").write_bytes(b"should be excluded")

    tags = _get_run(git_repo, run_id).data.tags
    assert "model_sha256__0" not in tags


def test_failed_run_marks_status_and_releases_lock(git_repo):
    with pytest.raises(RuntimeError):
        with track_run(workspace_dir=git_repo) as run:
            run_id = run.run_id
            raise RuntimeError("boom")

    assert _get_run(git_repo, run_id).info.status == "FAILED"
    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["status"] == "FAILED"
    assert RunLock(git_repo).is_locked() is False


def test_lock_and_marker_survive_end_run_crashing(git_repo, monkeypatch):
    # Live-verified bug: mlflow.end_run() can raise mid-call against a real
    # remote tracking server (Windows console can't encode the emoji mlflow
    # writes for a real run URL) - the lock/marker must not depend on it
    # succeeding, or every crash there permanently locks the workspace.
    real_end_run = mlflow.end_run

    def boom(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "x", 0, 1, "boom")

    with pytest.raises(UnicodeEncodeError):
        with track_run(workspace_dir=git_repo) as run:
            run_id = run.run_id
            monkeypatch.setattr(mlflow, "end_run", boom)

    monkeypatch.setattr(mlflow, "end_run", real_end_run)
    assert RunLock(git_repo).is_locked() is False
    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["run_id"] == run_id
