import subprocess
import sys

import pytest


def run(repo_dir, *args):
    subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    run(tmp_path, "init", "-q")
    run(tmp_path, "config", "user.email", "dev@example.com")
    run(tmp_path, "config", "user.name", "dev")
    (tmp_path / ".gitignore").write_text("models/\noutputs/\n.mlops/\nmlruns/\n")
    (tmp_path / "README.md").write_text("hello\n")
    run(tmp_path, "add", ".gitignore", "README.md")
    run(tmp_path, "commit", "-q", "-m", "initial commit")
    return tmp_path


@pytest.fixture
def producer_repo(tmp_path):
    from mlops.core.scaffold import scaffold_producer

    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True, capture_output=True)

    project_dir = tmp_path / "project"
    scaffold_producer(project_dir, "test-model")

    run(project_dir, "config", "user.email", "dev@example.com")
    run(project_dir, "config", "user.name", "dev")
    run(project_dir, "remote", "add", "origin", str(origin))

    dvc_remote = tmp_path / "dvc-remote"
    subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "add", "-d", "local-remote", str(dvc_remote)],
        cwd=project_dir, check=True, capture_output=True,
    )

    return project_dir
