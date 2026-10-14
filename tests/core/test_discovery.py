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


def test_snapshot_excludes_mlruns_dir(tmp_path):
    (tmp_path / "mlruns").mkdir()
    (tmp_path / "mlruns" / "0" / "abc123def456" / "artifacts").mkdir(parents=True)
    (tmp_path / "mlruns" / "0" / "abc123def456" / "artifacts" / "model.pkl").write_bytes(b"data")
    (tmp_path / "model.pkl").write_bytes(b"data")

    snap = discovery.snapshot(tmp_path)
    assert "model.pkl" in snap
    assert "mlruns/0/abc123def456/artifacts/model.pkl" not in snap


def test_excluded_model_hint_names_outputs_dir_and_says_save_under_models(tmp_path):
    # #26: a model saved under outputs/ is never hashed; the hint must
    # explain why instead of reading as user error.
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "model.pkl").write_bytes(b"data")

    hint = discovery.excluded_model_hint(tmp_path)

    assert hint is not None
    assert "outputs/" in hint
    assert "model.pkl" in hint
    assert "save models under models/" in hint
    assert "model_globs" not in hint


def test_excluded_model_hint_none_when_no_candidates(tmp_path):
    (tmp_path / "model.pkl").write_bytes(b"data")

    assert discovery.excluded_model_hint(tmp_path) is None


def test_snapshot_still_excludes_outputs_dir(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "model.pkl").write_bytes(b"data")

    snap = discovery.snapshot(tmp_path)
    assert "outputs/model.pkl" not in snap


def test_filter_by_globs_includes_new_model_extensions():
    paths = [
        "model.keras", "weights.cbm", "data.ubj", "checkpoint.ckpt",
        "model.pb", "config.json", "notes.txt"
    ]
    result = discovery.filter_by_globs(paths)
    assert set(result) == {"model.keras", "weights.cbm", "data.ubj", "checkpoint.ckpt", "model.pb"}
    assert "config.json" not in result
    assert "notes.txt" not in result
