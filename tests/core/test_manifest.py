from types import SimpleNamespace

import pytest

from mldna.core.hashing import sha256_file
from mldna.core.manifest import ManifestError, build_manifest, resolve_format, verify_format_files
from mldna.core.resolve import ResolvedRun


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


def test_resolve_format_missing_lists_available():
    manifest = {"model_name": "fraud-detector", "formats": {"pkl": {}, "onnx": {}}}

    with pytest.raises(ManifestError, match=r"\[onnx, pkl\]"):
        resolve_format(manifest, "openvino")


def test_resolve_format_returns_entry():
    manifest = {"formats": {"onnx": {"path": "onnx/", "files": {"model.onnx": "abc"}}}}

    assert resolve_format(manifest, "onnx") == {"path": "onnx/", "files": {"model.onnx": "abc"}}


def test_verify_format_files_single_file_ok(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"bytes")
    format_entry = {"path": "model.pkl", "sha256": sha256_file(tmp_path / "model.pkl")}

    verify_format_files(tmp_path, format_entry)  # no raise


def test_verify_format_files_single_file_mismatch_refuses(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"bytes")
    format_entry = {"path": "model.pkl", "sha256": "wrong"}

    with pytest.raises(ManifestError, match="checksum mismatch"):
        verify_format_files(tmp_path, format_entry)


def test_verify_format_files_missing_file_refuses(tmp_path):
    format_entry = {"path": "model.pkl", "sha256": "abc"}

    with pytest.raises(ManifestError, match="missing"):
        verify_format_files(tmp_path, format_entry)


def test_verify_format_files_dir_ok(tmp_path):
    (tmp_path / "model.xml").write_bytes(b"xml")
    (tmp_path / "model.bin").write_bytes(b"bin")
    format_entry = {
        "path": "openvino/",
        "files": {
            "model.xml": sha256_file(tmp_path / "model.xml"),
            "model.bin": sha256_file(tmp_path / "model.bin"),
        },
    }

    verify_format_files(tmp_path, format_entry)  # no raise


def test_verify_format_files_dir_mismatch_refuses(tmp_path):
    (tmp_path / "model.xml").write_bytes(b"xml")
    format_entry = {"path": "openvino/", "files": {"model.xml": "wrong"}}

    with pytest.raises(ManifestError, match="checksum mismatch"):
        verify_format_files(tmp_path, format_entry)
