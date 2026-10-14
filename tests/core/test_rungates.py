import subprocess

import mlflow
import pytest

from mldna.core.rungates import RunGateError, check_run_gates
from mldna.tracking import track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def test_healthy_run_passes(producer_repo):
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    check_run_gates(run_id)


def test_dirty_run_refused(git_repo):
    (git_repo / "README.md").write_text("dirty\n")

    with track_run(workspace_dir=git_repo, allow_dirty=True) as run:
        run_id = run.run_id

    with pytest.raises(RunGateError, match="dirty") as excinfo:
        check_run_gates(run_id)

    # same rule, same fix as resolve.py's all-matches-dirty refusal (SPEC 15)
    message = str(excinfo.value)
    assert "Fix: commit or stash your changes" in message
    assert "mldna.track_run()" in message


def test_missing_lineage_tag_refused(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    # D#63: the data hash is one tag per out, so the requirement is "at least
    # one", and the refusal names that shape rather than a tag nothing writes.
    with pytest.raises(RunGateError, match=r"dvc_data_hash__<n>"):
        check_run_gates(run_id)


def test_legacy_single_hash_run_still_satisfies_the_data_requirement(git_repo):
    # Runs recorded before D#63 carry one bare dvc_data_hash and can never
    # grow per-out tags (tags are written when their claim becomes true, SPEC
    # 6). They stay publishable rather than being refused for a shape they
    # could not have written.
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
    mlflow.tracking.MlflowClient().set_tag(run_id, "dvc_data_hash", "d4e5f6.dir")

    check_run_gates(run_id)


def test_legacy_published_run_refused_honestly(producer_repo):
    # Published by an mldna older than D#78: registry_version, no per-file tag.
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    mlflow.tracking.MlflowClient().set_tag(run_id, "registry_version", "3")

    with pytest.raises(RunGateError, match="already published") as excinfo:
        check_run_gates(run_id, SHA_A)
    message = str(excinfo.value)
    assert "older mldna" in message
    assert "runs published from now on" in message
    assert "identical content" not in message


SHA_A = "a" * 64
SHA_B = "b" * 64


def _published_file_run(producer_repo):
    _commit_scaffold(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "registry_version", "3")
    client.set_tag(run_id, f"published__{SHA_A[:16]}", "3")
    return run_id


def test_same_file_published_twice_from_one_run_refused(producer_repo):
    run_id = _published_file_run(producer_repo)

    with pytest.raises(RunGateError, match="already published this exact file") as excinfo:
        check_run_gates(run_id, SHA_A)
    message = str(excinfo.value)
    assert SHA_A in message
    assert "as version 3" in message
    assert "Fix: publish a different model file, or retrain" in message


def test_different_file_from_a_published_run_passes(producer_repo):
    run_id = _published_file_run(producer_repo)

    check_run_gates(run_id, SHA_B)


def test_failed_run_refused(git_repo):
    with pytest.raises(RuntimeError):
        with track_run(workspace_dir=git_repo) as run:
            run_id = run.run_id
            raise RuntimeError("boom")

    with pytest.raises(RunGateError, match="status"):
        check_run_gates(run_id)
