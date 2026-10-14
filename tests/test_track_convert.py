import json
import subprocess

import mlflow
import pytest

from mldna.core.conversions import ConversionError, verify_bundle
from mldna.core.hashing import sha256_file
from mldna.core.lock import RunLock, RunLockError
from mldna.core.resolve import AmbiguousRunError, ResolveError
from mldna.tracking import track_convert, track_run

MODEL_BYTES = b"trained model bytes"


def _tracked_pkl(git_repo):
    with track_run(workspace_dir=git_repo) as run:
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

    with track_run(workspace_dir=git_repo):
        pass

    with pytest.raises(ResolveError):
        with track_convert(pkl, workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False


def test_lock_already_held_refuses(git_repo):
    pkl, _ = _tracked_pkl(git_repo)
    RunLock(git_repo).acquire()

    with pytest.raises(RunLockError):
        with track_convert(pkl, workspace_dir=git_repo):
            pass


def test_unreadable_conversions_refuses_instead_of_overwriting(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with track_convert(pkl, workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    with pytest.MonkeyPatch.context() as mp:
        def boom(*a, **kw):
            raise RuntimeError("500 Server Error")
        mp.setattr(mlflow.artifacts, "download_artifacts", boom)
        with pytest.raises(ConversionError, match="overwrite"):
            with track_convert(pkl, workspace_dir=git_repo):
                ov_dir = git_repo / "models" / "openvino"
                ov_dir.mkdir()
                (ov_dir / "model.xml").write_bytes(b"openvino xml")

    # the earlier session's provenance must still be on the run
    conversions = _read_conversions(git_repo, run_id)
    relpaths = {e["relpath"] for e in conversions.values()}
    assert relpaths == {"models/onnx/model.onnx"}


def test_lock_released_when_conversions_read_fails(git_repo):
    pkl, _ = _tracked_pkl(git_repo)

    with track_convert(pkl, workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    with pytest.MonkeyPatch.context() as mp:
        def boom(*a, **kw):
            raise RuntimeError("500 Server Error")
        mp.setattr(mlflow.artifacts, "download_artifacts", boom)
        with pytest.raises(ConversionError):
            with track_convert(pkl, workspace_dir=git_repo):
                ov_dir = git_repo / "models" / "openvino"
                ov_dir.mkdir()
                (ov_dir / "model.xml").write_bytes(b"openvino xml")

    assert RunLock(git_repo).is_locked() is False


def test_artifact_listing_failure_refuses(git_repo, monkeypatch):
    pkl, _ = _tracked_pkl(git_repo)

    def boom(self, run_id, *a, **kw):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(mlflow.tracking.MlflowClient, "list_artifacts", boom)

    with pytest.raises(ConversionError, match="Could not list the artifacts"):
        with track_convert(pkl, workspace_dir=git_repo):
            onnx_dir = git_repo / "models" / "onnx"
            onnx_dir.mkdir()
            (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    assert RunLock(git_repo).is_locked() is False


def test_first_conversion_never_downloads_conversions(git_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(git_repo)

    calls = []
    real = mlflow.artifacts.download_artifacts

    def counting(*a, **kw):
        calls.append(kw.get("artifact_path"))
        return real(*a, **kw)
    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", counting)

    with track_convert(pkl, workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    assert calls == []


def _two_runs_differing_commits(git_repo):
    # D#64: identical bytes from two clean runs at different commits.
    with track_run(workspace_dir=git_repo):
        (git_repo / "models").mkdir(exist_ok=True)
        (git_repo / "models" / "model_a.pkl").write_bytes(MODEL_BYTES)
    (git_repo / "README.md").write_text("docs only\n")
    subprocess.run(["git", "commit", "-q", "-am", "docs only"], cwd=git_repo, check=True)
    with track_run(workspace_dir=git_repo) as run_b:
        pkl = git_repo / "models" / "model_b.pkl"
        pkl.write_bytes(MODEL_BYTES)
    commit_b = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=git_repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return pkl, run_b.run_id, commit_b


def test_differing_lineage_refuses_naming_commit_kwarg(git_repo):
    pkl, _, _ = _two_runs_differing_commits(git_repo)

    with pytest.raises(AmbiguousRunError, match='commit="<sha>"'):
        with track_convert(pkl, workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False


# --- happy paths ---

def test_commit_kwarg_selects_run_on_differing_lineage(git_repo):
    pkl, run_b, commit_b = _two_runs_differing_commits(git_repo)

    with track_convert(pkl, workspace_dir=git_repo, commit=commit_b) as tc:
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    assert tc.run_id == run_b
    assert _read_conversions(git_repo, run_b)


def test_conversions_recorded_on_source_run(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with track_convert(pkl, workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    conversions = _read_conversions(git_repo, run_id)
    onnx_sha = sha256_file(git_repo / "models" / "onnx" / "model.onnx")
    assert conversions[onnx_sha]["relpath"] == "models/onnx/model.onnx"
    assert conversions[onnx_sha]["source_pkl_sha256"] == sha256_file(pkl)


def test_second_session_merges(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with track_convert(pkl, workspace_dir=git_repo):
        onnx_dir = git_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted onnx bytes")

    with track_convert(pkl, workspace_dir=git_repo):
        ov_dir = git_repo / "models" / "openvino"
        ov_dir.mkdir()
        (ov_dir / "model.xml").write_bytes(b"openvino xml")

    conversions = _read_conversions(git_repo, run_id)
    relpaths = {e["relpath"] for e in conversions.values()}
    assert relpaths == {"models/onnx/model.onnx", "models/openvino/model.xml"}


def test_verify_bundle_end_to_end(git_repo):
    pkl, run_id = _tracked_pkl(git_repo)

    with track_convert(pkl, workspace_dir=git_repo):
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
        with track_convert(pkl, workspace_dir=git_repo):
            partial = git_repo / "models" / "partial.onnx"
            partial.write_bytes(b"half-written")
            raise RuntimeError("conversion tool crashed")

    assert RunLock(git_repo).is_locked() is False
    mlflow.set_tracking_uri(f"file:{git_repo / 'mlruns'}")
    with pytest.raises(Exception):
        mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path="conversions.json")
