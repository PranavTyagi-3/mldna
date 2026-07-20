import subprocess

import mlflow
import pytest

from mlops.core.lockfile import LockfileError
from mlops.core.manifest import ManifestError
from mlops.core.push import push as core_push
from mlops.fetch import FetchError, fetch
from mlops.publish import publish
from mlops.tracking import track_convert, track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def _publish_pkl_only(repo, pkl_bytes=b"trained model bytes"):
    _commit_scaffold(repo)
    core_push(repo)
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(pkl_bytes)
    result = publish(pkl, workspace_dir=repo, confirm=lambda s: True)
    return pkl, result


def _write_lockfile(repo, model_name="test-model", version=1, format="pkl", dest="app/models/test"):
    (repo / "models.lock.yaml").write_text(
        f"{model_name}:\n  version: {version}\n  format: {format}\n  dest: {dest}\n"
    )
    return repo / dest


def test_fetch_missing_lockfile_refuses(producer_repo):
    with pytest.raises(LockfileError):
        fetch(workspace_dir=producer_repo)


def test_fetch_unknown_version_refuses(producer_repo):
    _write_lockfile(producer_repo, model_name="ghost-model", version=1)

    with pytest.raises(FetchError, match="not found in the registry"):
        fetch(workspace_dir=producer_repo)


def test_fetch_recalled_version_refuses(producer_repo):
    _, result = _publish_pkl_only(producer_repo)
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag("test-model", result["version"], "recalled", "true")
    _write_lockfile(producer_repo, version=result["version"])

    with pytest.raises(FetchError, match="recalled"):
        fetch(workspace_dir=producer_repo)


def test_fetch_missing_format_refuses(producer_repo):
    _, result = _publish_pkl_only(producer_repo)
    _write_lockfile(producer_repo, version=result["version"], format="onnx")

    with pytest.raises(ManifestError, match="onnx"):
        fetch(workspace_dir=producer_repo)


def test_fetch_pkl_only_happy_path(producer_repo):
    pkl, result = _publish_pkl_only(producer_repo)
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched == [{"model_name": "test-model", "version": str(result["version"]), "skipped": False}]
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()
    assert (dest / ".mlops-version").read_text().strip() == str(result["version"])


def test_fetch_converted_format_happy_path(producer_repo):
    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    with track_convert(pkl, workspace_dir=producer_repo):
        onnx_dir = producer_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted bytes")
    result = publish(pkl, workspace_dir=producer_repo,
                      format_dirs=[onnx_dir], confirm=lambda s: True)
    dest = _write_lockfile(producer_repo, version=result["version"], format="onnx")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched == [{"model_name": "test-model", "version": str(result["version"]), "skipped": False}]
    assert (dest / "model.onnx").read_bytes() == b"converted bytes"


def test_fetch_idempotent_skip_avoids_redownload(producer_repo, monkeypatch):
    _, result = _publish_pkl_only(producer_repo)
    _write_lockfile(producer_repo, version=result["version"], format="pkl")

    first = fetch(workspace_dir=producer_repo)
    assert first[0]["skipped"] is False

    import mlops.fetch as fetch_module

    def _boom(*args, **kwargs):
        raise AssertionError("subtree should not be re-downloaded on an idempotent fetch")

    monkeypatch.setattr(fetch_module, "_stage_download", _boom)

    second = fetch(workspace_dir=producer_repo)
    assert second[0]["skipped"] is True


def test_fetch_version_bump_refetches(producer_repo):
    # Both publishes happen before any lockfile/fetch writes touch the working
    # tree - fetch's own output files would otherwise dirty the tree that the
    # second track_run() needs clean.
    _, result1 = _publish_pkl_only(producer_repo, pkl_bytes=b"v1 bytes")
    with track_run(workspace_dir=producer_repo):
        pkl2 = producer_repo / "models" / "model.pkl"
        pkl2.write_bytes(b"v2 bytes")
    result2 = publish(pkl2, workspace_dir=producer_repo, confirm=lambda s: True)
    assert result2["version"] != result1["version"]

    dest = _write_lockfile(producer_repo, version=result1["version"], format="pkl")
    fetch(workspace_dir=producer_repo)
    assert (dest / "model.pkl").read_bytes() == b"v1 bytes"

    _write_lockfile(producer_repo, version=result2["version"], format="pkl")
    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == b"v2 bytes"
    assert (dest / ".mlops-version").read_text().strip() == str(result2["version"])
