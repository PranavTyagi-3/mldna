import pytest

from mlops.core.structure import StructureError, check_structure


def _make_bundle(tmp_path, formats):
    pkl = tmp_path / "model.pkl"
    pkl.write_bytes(b"x")
    dirs = []
    for name, files in formats.items():
        d = tmp_path / name
        d.mkdir()
        for fname, content in files.items():
            (d / fname).write_bytes(content)
        dirs.append(d)
    return pkl, dirs


def test_pkl_only_bundle_allowed(tmp_path):
    pkl, _ = _make_bundle(tmp_path, {})

    check_structure(pkl, [])


def test_missing_required_format_refused(tmp_path):
    pkl, dirs = _make_bundle(tmp_path, {"onnx": {"model.onnx": b"x"}})

    with pytest.raises(StructureError, match="openvino"):
        check_structure(pkl, dirs, require_formats=["openvino"])


def test_passes_with_required_format_present(tmp_path):
    pkl, dirs = _make_bundle(
        tmp_path, {"openvino": {"model.xml": b"x", "model.bin": b"y"}}
    )

    check_structure(pkl, dirs, require_formats=["openvino"])


def test_junk_files_refused(tmp_path):
    pkl, dirs = _make_bundle(tmp_path, {"openvino": {"model.xml": b"x"}})
    (dirs[0] / "__pycache__").mkdir()
    (dirs[0] / "__pycache__" / "cache.pyc").write_bytes(b"junk")

    with pytest.raises(StructureError, match="Junk files"):
        check_structure(pkl, dirs)
