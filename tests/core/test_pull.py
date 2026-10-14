import subprocess
import sys

import pytest

from mldna.core import dvc
from mldna.core.lock import RunLock
from mldna.core.pull import PullError, pull
from mldna.core.push import push, stage


def _clone(origin, dest):
    subprocess.run(["git", "clone", "-q", str(origin), str(dest)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=dest, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=dest, check=True)


def test_pull_syncs_code_and_data(producer_repo, tmp_path):
    (producer_repo / "dataset" / "data.csv").write_text("a,b\n1,2\n")
    stage(producer_repo)
    push(producer_repo, message="add data")

    origin = producer_repo.parent / "origin.git"
    clone_dir = tmp_path / "clone"
    _clone(origin, clone_dir)

    (producer_repo / "main.py").write_text("# updated\n")
    (producer_repo / "dataset" / "data.csv").write_text("a,b\n1,2\n3,4\n")
    stage(producer_repo)
    push(producer_repo, message="update code and data")

    pull(clone_dir)

    assert (clone_dir / "main.py").read_text() == "# updated\n"
    assert (clone_dir / "dataset" / "data.csv").read_text() == "a,b\n1,2\n3,4\n"


def test_pull_fails_in_consumer_repo(tmp_path):
    (tmp_path / "mldna.yaml").write_text("project:\n  repo_role: consumer\n")

    with pytest.raises(PullError, match="mldna fetch"):
        pull(tmp_path)


def test_pull_fails_under_live_lock(producer_repo):
    RunLock(producer_repo).acquire()

    with pytest.raises(PullError, match="run is active"):
        pull(producer_repo)


def _force_rmtree(path):
    import os
    import shutil
    import stat
    import sys

    def _on_exc(func, p, exc):
        os.chmod(p, stat.S_IWRITE)
        func(p)

    # onexc is 3.12+; onerror is its pre-3.12 spelling. The handler ignores
    # the third argument, so the signature difference doesn't matter.
    kwarg = "onexc" if sys.version_info >= (3, 12) else "onerror"
    shutil.rmtree(path, **{kwarg: _on_exc})


def test_pull_fails_clearly_without_dvc_remote(tmp_path):
    from mldna.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    (tmp_path / "dataset" / "data.csv").write_text("a,b\n1,2\n")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    origin = tmp_path.parent / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True)
    from mldna.core import dvc
    dvc.run(["add", "dataset", "-q"], cwd=tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=tmp_path, check=True, capture_output=True)
    _force_rmtree(tmp_path / ".dvc" / "cache")
    _force_rmtree(tmp_path / "dataset")

    with pytest.raises(PullError, match="dvc pull failed"):
        pull(tmp_path)


def test_pull_error_includes_conflicting_files(tmp_path):
    from mldna.core.scaffold import scaffold_producer
    # Create initial producer repo
    producer = tmp_path / "producer"
    scaffold_producer(producer, "test-model")
    (producer / "dataset" / "data.csv").write_text("a,b\n1,2\n")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=producer, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=producer, check=True)

    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=producer, check=True)

    from mldna.core import dvc
    dvc.run(["add", "dataset", "-q"], cwd=producer)
    subprocess.run(["git", "add", "-A"], cwd=producer, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "initial"], cwd=producer, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=producer, check=True, capture_output=True)

    # Clone it
    clone = tmp_path / "clone"
    _clone(origin, clone)

    # Make conflicting change in producer and push
    (producer / "main.py").write_text("# producer version\n")
    subprocess.run(["git", "add", "-A"], cwd=producer, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "producer edit"], cwd=producer, check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], cwd=producer, check=True, capture_output=True)

    # Make different conflicting change in clone
    (clone / "main.py").write_text("# clone version\n")
    subprocess.run(["git", "add", "-A"], cwd=clone, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "clone edit"], cwd=clone, check=True)

    # Pull should fail and error message should include the conflicting file name
    with pytest.raises(PullError) as exc_info:
        pull(clone)

    error_msg = str(exc_info.value)
    assert "main.py" in error_msg, f"Conflicting file name not in error message: {error_msg}"
    assert "git pull failed" in error_msg


def _config_local_content(workspace_dir):
    config_local = workspace_dir / ".dvc" / "config.local"
    return config_local.read_text() if config_local.exists() else ""


def test_pull_skips_blob_override_for_local_remote(producer_repo, tmp_path, monkeypatch):
    (producer_repo / "dataset" / "data.csv").write_text("a,b\n1,2\n")
    stage(producer_repo)
    push(producer_repo, message="add data")

    origin = producer_repo.parent / "origin.git"
    clone_dir = tmp_path / "clone"
    _clone(origin, clone_dir)

    monkeypatch.setenv("MLDNA_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;test=1")

    pull(clone_dir)

    config_content = _config_local_content(clone_dir)
    assert "test=1" not in config_content
    assert "connection_string" not in config_content


def test_pull_applies_blob_override_for_azure_remote(tmp_path, monkeypatch):
    from mldna.core.scaffold import scaffold_producer

    producer_dir = tmp_path / "producer"
    scaffold_producer(producer_dir, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=producer_dir, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=producer_dir, check=True)

    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=producer_dir, check=True)


    # The producer keeps its local remote so setup can push; only the clone
    # gets an azure:// remote, since the clone's pull is what is measured.
    (producer_dir / "dataset" / "data.csv").write_text("a,b\n1,2\n")
    stage(producer_dir)
    push(producer_dir, message="add data")

    clone_dir = tmp_path / "clone"
    _clone(origin, clone_dir)

    dvc.run(["remote", "remove", "local-remote"], cwd=clone_dir, check=False)
    dvc.run(["remote", "add", "-d", "azure-remote", "azure://container/prefix"], cwd=clone_dir)

    monkeypatch.setenv("MLDNA_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;test=1")

    # dvc push/pull to a real azure:// remote cannot succeed here - there is no
    # Azure. The connection string is written BEFORE that attempt, which is what
    # this test is about, so expect the transfer failure and assert the write.
    with pytest.raises(PullError):
        pull(clone_dir)

    config_content = _config_local_content(clone_dir)
    assert "test=1" in config_content
