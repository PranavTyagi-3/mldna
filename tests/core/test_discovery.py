from mldna.core import discovery


def test_snapshot_and_diff_detects_new_file(tmp_path):
    before = discovery.snapshot(tmp_path)
    (tmp_path / "model.pkl").write_bytes(b"data")
    after = discovery.snapshot(tmp_path)

    changed = discovery.diff(before, after)
    assert changed == ["model.pkl"]


def test_diff_detects_modified_file(tmp_path):
    path = tmp_path / "model.pkl"
    path.write_bytes(b"data")
    before = discovery.snapshot(tmp_path)
    path.write_bytes(b"more data")
    after = discovery.snapshot(tmp_path)

    assert discovery.diff(before, after) == ["model.pkl"]


def test_snapshot_excludes_default_dirs(tmp_path):
    (tmp_path / "dataset").mkdir()
    (tmp_path / "dataset" / "data.csv").write_text("a,b\n")
    (tmp_path / "model.pkl").write_bytes(b"data")

    snap = discovery.snapshot(tmp_path)
    assert "model.pkl" in snap
    assert "dataset/data.csv" not in snap


def test_filter_by_globs_keeps_only_model_files():
    paths = ["model.pkl", "notes.txt", "weights.onnx"]
    assert discovery.filter_by_globs(paths) == ["model.pkl", "weights.onnx"]
