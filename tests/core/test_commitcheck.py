import subprocess

import mlflow
import pytest

from mlops.core import git
from mlops.core.commitcheck import CommitReachabilityError, check_commit_reachable
from mlops.core.push import push
from mlops.tracking import track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def test_commit_already_on_remote_passes(producer_repo):
    _commit_scaffold(producer_repo)
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
    push(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "git_commit", "f" * 40)

    with pytest.raises(CommitReachabilityError, match="does not exist"):
        check_commit_reachable(producer_repo, run_id)
