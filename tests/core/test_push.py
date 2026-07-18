import subprocess
import sys

import pytest

from mlops.core.push import PushError, push


def test_push_commits_and_pushes(producer_repo):
    result = push(producer_repo, message="first push")

    assert result["committed"] is True

    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True, text=True
    ).stdout.strip()
    remote_head = subprocess.run(
        ["git", "rev-parse", "origin/master"], cwd=producer_repo, capture_output=True, text=True
    )
    if remote_head.returncode != 0:
        remote_head = subprocess.run(
            ["git", "rev-parse", "origin/main"], cwd=producer_repo, capture_output=True, text=True
        )
    assert remote_head.stdout.strip() == local_head


def test_push_second_call_is_a_noop(producer_repo):
    push(producer_repo, message="first push")

    result = push(producer_repo, message="second push")

    assert result["committed"] is False


def test_push_fails_clearly_without_git_remote(tmp_path):
    from mlops.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    dvc_remote = tmp_path.parent / "dvc-remote"
    subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "add", "-d", "local-remote", str(dvc_remote)],
        cwd=tmp_path, check=True,
    )

    with pytest.raises(PushError, match="git push failed"):
        push(tmp_path)


def test_push_fails_clearly_without_dvc_remote(tmp_path):
    from mlops.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    origin = tmp_path.parent / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True)

    with pytest.raises(PushError, match="dvc push failed"):
        push(tmp_path)
