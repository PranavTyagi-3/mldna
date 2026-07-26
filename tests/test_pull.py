import shutil
import subprocess

import pytest

from mldna.core.pull import PullError
from mldna.core.push import push as core_push
from mldna.publish import publish
from mldna.pull import pull
from mldna.tracking import track_run


def _commit_and_push(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)


def _train_and_publish(repo, pkl_bytes):
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(pkl_bytes)
    result = publish(pkl, workspace_dir=repo, confirm=lambda s: True)
    return result


def test_pull_reports_no_published_model_for_commit(producer_repo):
    _commit_and_push(producer_repo)

    result = pull(producer_repo)

    assert result["model_pulled"] is False


def test_pull_downloads_single_published_run(producer_repo):
    _commit_and_push(producer_repo)
    published = _train_and_publish(producer_repo, b"v1 bytes")
    shutil.rmtree(producer_repo / "models")

    result = pull(producer_repo)

    assert result["model_pulled"] is True
    assert result["run_id"] == published["run_id"]
    assert (producer_repo / "models" / "model.pkl").read_bytes() == b"v1 bytes"


def test_pull_multiple_published_runs_without_select_refuses(producer_repo):
    _commit_and_push(producer_repo)
    _train_and_publish(producer_repo, b"v1 bytes")
    _train_and_publish(producer_repo, b"v2 bytes")

    with pytest.raises(PullError, match="published runs match"):
        pull(producer_repo)


def test_pull_multiple_published_runs_with_select_downloads_chosen(producer_repo):
    _commit_and_push(producer_repo)
    first = _train_and_publish(producer_repo, b"v1 bytes")
    _train_and_publish(producer_repo, b"v2 bytes")
    shutil.rmtree(producer_repo / "models")

    result = pull(producer_repo, select=lambda candidates: next(
        c for c in candidates if c.info.run_id == first["run_id"]
    ))

    assert result["run_id"] == first["run_id"]
    assert (producer_repo / "models" / "model.pkl").read_bytes() == b"v1 bytes"
