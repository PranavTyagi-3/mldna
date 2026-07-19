from types import SimpleNamespace

from mlops.core.hashing import sha256_file
from mlops.core.manifest import build_manifest
from mlops.core.resolve import ResolvedRun


def _fake_run(tags):
    return SimpleNamespace(data=SimpleNamespace(tags=tags))


def test_build_manifest_pkl_only(tmp_path):
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(b"trained model")
    run = _fake_run({"git_commit": "abc123", "dvc_data_hash": "d4e5f6", "save_context": "in_run"})
    resolved = ResolvedRun(run_id="run-1", relpath="models/model.pkl")

    manifest = build_manifest("fraud-detector", run, resolved, pkl, [], {}, "priya", "2026-01-01T00:00:00Z")

    assert manifest["model_name"] == "fraud-detector"
    assert manifest["source_run_id"] == "run-1"
    assert manifest["formats"]["pkl"]["sha256"] == sha256_file(pkl)
    assert manifest["formats"].keys() == {"pkl"}
    assert manifest["lineage"] == {"git_commit": "abc123", "dvc_data_hash": "d4e5f6"}
    assert manifest["conversions"] == {}


def test_build_manifest_with_format_dir_and_conversions(tmp_path):
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(b"trained model")
    onnx_dir = tmp_path / "onnx"
    onnx_dir.mkdir()
    (onnx_dir / "model.onnx").write_bytes(b"converted bytes")
    onnx_sha = sha256_file(onnx_dir / "model.onnx")

    run = _fake_run({"git_commit": "abc123", "dvc_data_hash": "d4e5f6"})
    resolved = ResolvedRun(run_id="run-1", relpath="models/model.pkl")
    conversions = {
        onnx_sha: {"relpath": "models/onnx/model.onnx", "source_pkl_sha256": "x", "at": "t"},
        "unrelated-sha": {"relpath": "other.bin", "source_pkl_sha256": "y", "at": "t"},
    }

    manifest = build_manifest("fraud-detector", run, resolved, pkl, [onnx_dir],
                               conversions, "priya", "2026-01-01T00:00:00Z")

    assert manifest["formats"]["onnx"]["files"] == {"model.onnx": onnx_sha}
    assert manifest["conversions"] == {onnx_sha: conversions[onnx_sha]}
    assert manifest["save_context"] == "in_run"
