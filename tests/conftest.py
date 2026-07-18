import subprocess

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
