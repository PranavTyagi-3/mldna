import json

import mlflow
import pytest

from mlops.core.conversions import ConversionError, verify_bundle
from mlops.core.hashing import sha256_file
from mlops.core.lock import RunLock, RunLockError
from mlops.core.resolve import ResolveError
from mlops.tracking import track_convert, track_run

MODEL_BYTES = b"trained model bytes"


def _tracked_pkl(git_repo):
    with track_run("test-model", workspace_dir=git_repo) as run:
        pkl = git_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(MODEL_BYTES)
    return pkl, run.run_id


def _read_conversions(git_repo, run_id):
    mlflow.set_tracking_uri(f"file:{git_repo / 'mlruns'}")
    path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="conversions.json")
    return json.loads(open(path).read())


# --- refusal paths ---

def test_untracked_pkl_refuses(git_repo):
    pkl = git_repo / "models" / "untracked.pkl"
    pkl.parent.mkdir(exist_ok=True)
    pkl.write_bytes(b"never tracked")

    with track_run("test-model", workspace_dir=git_repo):
        pass

    with pytest.raises(ResolveError):
        with track_convert(pkl, "test-model", workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False


def test_lock_already_held_refuses(git_repo):
    pkl, _ = _tracked_pkl(git_repo)
    RunLock(git_repo).acquire()

    with pytest.raises(RunLockError):
        with track_convert(pkl, "test-model", workspace_dir=git_repo):
            pass


# --- happy paths ---

def test_conversions_recorded_on_source_run(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with track_convert(pkl, "test-model", workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    conversions = _read_conversions(git_repo, run_id)
    onnx_sha = sha256_file(git_repo / "models" / "onnx" / "model.onnx")
    assert conversions[onnx_sha]["relpath"] == "models/onnx/model.onnx"
    assert conversions[onnx_sha]["source_pkl_sha256"] == sha256_file(pkl)


def test_second_session_merges(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with track_convert(pkl, "test-model", workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    with track_convert(pkl, "test-model", workspace_dir=git_repo):
        ov_dir = git_repo / "models" / "openvino"
        ov_dir.mkdir()
        (ov_dir / "model.xml").write_bytes(b"openvino xml")

    conversions = _read_conversions(git_repo, run_id)
    relpaths = {e["relpath"] for e in conversions.values()}
    assert "models/onnx/model.onnx" in relpaths
    assert "models/openvino/model.xml" in relpaths


def test_verify_bundle_end_to_end(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with track_convert(pkl, "test-model", workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    conversions = _read_conversions(git_repo, run_id)
    verify_bundle(conversions, [git_repo / "models" / "onnx"], sha256_file(pkl))

    pkl.write_bytes(b"retrained different model")
    with pytest.raises(ConversionError, match="different pkl"):
        verify_bundle(conversions, [git_repo / "models" / "onnx"], sha256_file(pkl))


def test_exception_in_block_records_nothing(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with pytest.raises(RuntimeError):
        with track_convert(pkl, "test-model", workspace_dir=git_repo):
            partial = git_repo / "models" / "partial.onnx"
            partial.write_bytes(b"half-written")
            raise RuntimeError("conversion tool crashed")

    assert RunLock(git_repo).is_locked() is False
    mlflow.set_tracking_uri(f"file:{git_repo / 'mlruns'}")
    with pytest.raises(Exception):
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="conversions.json")
