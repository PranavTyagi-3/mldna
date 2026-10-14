import subprocess

import mlflow
import pytest

from mldna.core import git
from mldna.core.commitcheck import CommitReachabilityError, check_commit_reachable, commit_needs_push
from mldna.core.push import push, stage
from mldna.tracking import track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def test_commit_already_on_remote_passes(producer_repo):
    _commit_scaffold(producer_repo)
    stage(producer_repo)
    push(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    check_commit_reachable(producer_repo, run_id)


def test_commit_local_only_gets_pushed(producer_repo):
    _commit_scaffold(producer_repo)
    sha = git.get_commit_sha(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    check_commit_reachable(producer_repo, run_id)

    on_remote = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], cwd=producer_repo,
        capture_output=True, text=True,
    ).stdout
    assert "origin/" in on_remote


def test_commit_not_found_locally_raises(producer_repo):
    _commit_scaffold(producer_repo)
    stage(producer_repo)
    push(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "git_commit", "f" * 40)

    with pytest.raises(CommitReachabilityError, match="does not exist"):
        check_commit_reachable(producer_repo, run_id)


# --- commit_needs_push: the pre-check publish's summary relies on (D#56) ---

def test_commit_needs_push_raises_when_commit_missing_locally(producer_repo):
    _commit_scaffold(producer_repo)
    stage(producer_repo)
    push(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "git_commit", "f" * 40)

    with pytest.raises(CommitReachabilityError, match="does not exist"):
        commit_needs_push(producer_repo, run_id)


def test_commit_needs_push_true_when_not_yet_on_remote(producer_repo):
    _commit_scaffold(producer_repo)
    sha = git.get_commit_sha(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    branch, needs_push = commit_needs_push(producer_repo, run_id)
    assert needs_push is True
    assert branch

    # Must only report the pending push, never perform it.
    on_remote = subprocess.run(
        ["git", "branch", "-r", "--contains", sha], cwd=producer_repo,
        capture_output=True, text=True,
    ).stdout
    assert "origin/" not in on_remote


def test_commit_needs_push_false_when_already_on_remote(producer_repo):
    _commit_scaffold(producer_repo)
    stage(producer_repo)
    push(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    _, needs_push = commit_needs_push(producer_repo, run_id)
    assert needs_push is False


# --- no remote configured: must refuse cleanly, never a bare exit status 128 ---

def test_check_commit_reachable_no_remote_refuses_cleanly(producer_repo):
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    subprocess.run(["git", "remote", "remove", "origin"], cwd=producer_repo, check=True)

    with pytest.raises(CommitReachabilityError, match="git remote add origin"):
        check_commit_reachable(producer_repo, run_id)


def test_commit_needs_push_no_remote_refuses_cleanly(producer_repo):
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    subprocess.run(["git", "remote", "remove", "origin"], cwd=producer_repo, check=True)

    with pytest.raises(CommitReachabilityError, match="git remote add origin"):
        commit_needs_push(producer_repo, run_id)


# --- remote configured but unreachable: a clean refusal, never CalledProcessError ---

def _unreachable_remote_run(repo, tmp_path):
    _commit_scaffold(repo)
    with track_run(workspace_dir=repo) as run:
        run_id = run.run_id
    missing = tmp_path / "no-such-remote.git"
    subprocess.run(["git", "remote", "set-url", "origin", str(missing)], cwd=repo, check=True)
    return run_id


def test_commit_needs_push_unreachable_remote_refuses_cleanly(producer_repo, tmp_path):
    run_id = _unreachable_remote_run(producer_repo, tmp_path)

    with pytest.raises(CommitReachabilityError, match="git ls-remote origin") as exc:
        commit_needs_push(producer_repo, run_id)
    assert "no-such-remote.git" in str(exc.value)


def test_check_commit_reachable_unreachable_remote_refuses_cleanly(producer_repo, tmp_path):
    run_id = _unreachable_remote_run(producer_repo, tmp_path)

    with pytest.raises(CommitReachabilityError, match="git ls-remote origin") as exc:
        check_commit_reachable(producer_repo, run_id)
    assert "no-such-remote.git" in str(exc.value)
