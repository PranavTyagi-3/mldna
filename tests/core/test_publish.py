import subprocess

import mlflow
import pytest

from mlops.core.commitcheck import CommitReachabilityError
from mlops.core.conversions import ConversionError
from mlops.core.publish import PublishError, publish
from mlops.core.push import push as core_push
from mlops.core.resolve import ResolveError
from mlops.core.rungates import RunGateError
from mlops.tracking import track_convert, track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def _tracked_pkl(repo):
    _commit_scaffold(repo)
    core_push(repo)
    with track_run("test-model", workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    return pkl, run.run_id


def test_publish_pkl_only(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    result = publish(pkl, "test-model", workspace_dir=producer_repo)

    assert result["run_id"] == run_id
    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    run = mlflow.get_run(run_id)
    assert run.data.tags["registry_version"] == str(result["version"])
    assert "published_by" in run.data.tags


def test_publish_with_converted_format(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)
    with track_convert(pkl, "test-model", workspace_dir=producer_repo):
        onnx_dir = producer_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted bytes")

    result = publish(pkl, "test-model", workspace_dir=producer_repo,
                      format_dirs=[producer_repo / "models" / "onnx"])

    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    manifest_path = mlflow.artifacts.download_artifacts(
        run_id=result["run_id"], artifact_path="models/manifest.yaml"
    )
    import yaml
    manifest = yaml.safe_load(open(manifest_path).read())
    assert "onnx" in manifest["formats"]
    assert manifest["conversions"]


def test_publish_untracked_pkl_refuses(git_repo):
    pkl = git_repo / "models" / "untracked.pkl"
    pkl.parent.mkdir(exist_ok=True)
    pkl.write_bytes(b"never tracked")
    with track_run("test-model", workspace_dir=git_repo):
        pass

    with pytest.raises(ResolveError):
        publish(pkl, "test-model", workspace_dir=git_repo)


def test_publish_dirty_run_refuses(git_repo):
    (git_repo / "README.md").write_text("dirty\n")
    with track_run("test-model", workspace_dir=git_repo, allow_dirty=True) as run:
        pkl = git_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained on a dirty tree")

    with pytest.raises(RunGateError):
        publish(pkl, "test-model", workspace_dir=git_repo)


def test_publish_unrecorded_conversion_refuses(producer_repo):
    pkl, _ = _tracked_pkl(producer_repo)
    fake_dir = producer_repo / "models" / "onnx"
    fake_dir.mkdir()
    (fake_dir / "model.onnx").write_bytes(b"never went through track_convert")

    with pytest.raises(ConversionError):
        publish(pkl, "test-model", workspace_dir=producer_repo, format_dirs=[fake_dir])


def test_publish_aborts_when_confirm_declines(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    with pytest.raises(PublishError, match="Aborted"):
        publish(pkl, "test-model", workspace_dir=producer_repo, confirm=lambda summary: False)

    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    run = mlflow.get_run(run_id)
    assert "registry_version" not in run.data.tags


def test_publish_missing_local_commit_refuses(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "git_commit", "f" * 40)

    with pytest.raises(CommitReachabilityError):
        publish(pkl, "test-model", workspace_dir=producer_repo)
