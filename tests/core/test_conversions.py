import pytest

from mlops.core.conversions import ConversionError, build_entries, verify_bundle
from mlops.core.hashing import sha256_file

PKL_SHA = "a" * 64
OTHER_PKL_SHA = "b" * 64


def _make_format_dir(tmp_path, name="onnx"):
    d = tmp_path / name
    d.mkdir()
    (d / "model.onnx").write_bytes(b"converted model bytes")
    return d


def test_verify_passes_when_all_files_recorded(tmp_path):
    d = _make_format_dir(tmp_path)
    sha = sha256_file(d / "model.onnx")
    conversions = build_entries({"onnx/model.onnx": sha}, PKL_SHA, "2026-01-01T00:00:00Z")

    verify_bundle(conversions, [d], PKL_SHA)


def test_verify_refuses_unrecorded_file(tmp_path):
    d = _make_format_dir(tmp_path)

    with pytest.raises(ConversionError, match="not recorded as a tracked conversion"):
        verify_bundle({}, [d], PKL_SHA)


def test_verify_refuses_wrong_source_pkl(tmp_path):
    d = _make_format_dir(tmp_path)
    sha = sha256_file(d / "model.onnx")
    conversions = build_entries({"onnx/model.onnx": sha}, OTHER_PKL_SHA, "2026-01-01T00:00:00Z")

    with pytest.raises(ConversionError, match="converted from a different pkl"):
        verify_bundle(conversions, [d], PKL_SHA)


def test_verify_empty_format_dirs_passes():
    verify_bundle({}, [], PKL_SHA)
