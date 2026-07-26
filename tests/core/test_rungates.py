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

    with pytest.raises(RunGateError, match="dirty"):
        check_run_gates(run_id)


def test_missing_lineage_tag_refused(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    with pytest.raises(RunGateError, match="dvc_data_hash"):
        check_run_gates(run_id)


def test_already_published_run_refused(producer_repo):
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    mlflow.tracking.MlflowClient().set_tag(run_id, "registry_version", "3")

    with pytest.raises(RunGateError, match="already published"):
        check_run_gates(run_id)


def test_failed_run_refused(git_repo):
    with pytest.raises(RuntimeError):
        with track_run(workspace_dir=git_repo) as run:
            run_id = run.run_id
            raise RuntimeError("boom")

    with pytest.raises(RunGateError, match="status"):
        check_run_gates(run_id)
