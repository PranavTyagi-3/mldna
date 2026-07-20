import subprocess

import pytest

from mlops.core.lock import RunLock
from mlops.core.pull import PullError, pull
from mlops.core.push import push


def _clone(origin, dest):
    subprocess.run(["git", "clone", "-q", str(origin), str(dest)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=dest, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=dest, check=True)


def test_pull_syncs_code_and_data(producer_repo, tmp_path):
    (producer_repo / "dataset" / "data.csv").write_text("a,b\n1,2\n")
    push(producer_repo, message="add data")

    origin = producer_repo.parent / "origin.git"
    clone_dir = tmp_path / "clone"
    _clone(origin, clone_dir)

    (producer_repo / "main.py").write_text("# updated\n")
    (producer_repo / "dataset" / "data.csv").write_text("a,b\n1,2\n3,4\n")
    push(producer_repo, message="update code and data")

    pull(clone_dir)

    assert (clone_dir / "main.py").read_text() == "# updated\n"
    assert (clone_dir / "dataset" / "data.csv").read_text() == "a,b\n1,2\n3,4\n"


def test_pull_fails_in_consumer_repo(tmp_path):
    (tmp_path / "mlops.yaml").write_text("project:\n  repo_role: consumer\n")

    with pytest.raises(PullError, match="mlops fetch"):
        pull(tmp_path)


def test_pull_fails_under_live_lock(producer_repo):
    RunLock(producer_repo).acquire()

    with pytest.raises(PullError, match="run is active"):
        pull(producer_repo)


def _force_rmtree(path):
    import os
    import shutil
    import stat

    def _on_exc(func, p, exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    shutil.rmtree(path, onexc=_on_exc)


def test_pull_fails_clearly_without_dvc_remote(tmp_path):
    from mlops.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    (tmp_path / "dataset" / "data.csv").write_text("a,b\n1,2\n")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    origin = tmp_path.parent / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True)
    from mlops.core import dvc
    dvc.run(["add", "dataset", "-q"], cwd=tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=tmp_path, check=True, capture_output=True)
    _force_rmtree(tmp_path / ".dvc" / "cache")
    _force_rmtree(tmp_path / "dataset")

    with pytest.raises(PullError, match="dvc pull failed"):
        pull(tmp_path)
