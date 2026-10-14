from mldna.core.bigfiles import MODEL_SIZE_FLOOR, is_sharp_drop, pointer_totals, risky_files, shrink_note
from mldna.core.discovery import DEFAULT_MODEL_GLOBS

MB = 1024 * 1024


def _risky(tmp_path, *paths, max_file_mb=5):
    return risky_files(tmp_path, list(paths), DEFAULT_MODEL_GLOBS, max_file_mb)


def test_large_file_is_flagged_with_its_size(tmp_path):
    (tmp_path / "big.csv").write_bytes(b"x" * (6 * MB))

    [(path, reason)] = _risky(tmp_path, "big.csv")

    assert path == "big.csv"
    assert "larger than 5 MB" in reason
    assert "6.0 MB" in reason


def test_file_under_the_limit_is_not_flagged(tmp_path):
    (tmp_path / "small.csv").write_bytes(b"x" * MB)

    assert _risky(tmp_path, "small.csv") == []


def test_model_shaped_file_above_the_floor_is_flagged(tmp_path):
    (tmp_path / "net.pkl").write_bytes(b"x" * MODEL_SIZE_FLOOR)

    [(path, reason)] = _risky(tmp_path, "net.pkl")

    assert path == "net.pkl"
    assert "model-shaped" in reason


def test_small_xml_bin_pb_are_not_flagged(tmp_path):
    for name in ("pom.xml", "asset.bin", "schema.pb"):
        (tmp_path / name).write_bytes(b"x" * 1024)

    assert _risky(tmp_path, "pom.xml", "asset.bin", "schema.pb") == []


def test_large_signal_has_no_floor_and_applies_to_any_name(tmp_path):
    (tmp_path / "pom.xml").write_bytes(b"x" * 1024)

    assert _risky(tmp_path, "pom.xml", max_file_mb=0) != []


def test_missing_path_is_skipped(tmp_path):
    assert _risky(tmp_path, "gone.pkl") == []


def _pointer(nfiles, size):
    return f"outs:\n- md5: abc.dir\n  size: {size}\n  nfiles: {nfiles}\n  path: dataset\n"


def test_pointer_totals_reads_nfiles_and_size():
    assert pointer_totals(_pointer(2, 88064)) == (2, 88064)


def test_pointer_totals_treats_a_single_file_out_as_one_file():
    assert pointer_totals("outs:\n- md5: abc\n  size: 10\n  path: data.csv\n") == (1, 10)


def test_sharp_drop_to_zero_files_is_flagged_with_both_numbers():
    note = shrink_note("dataset.dvc", _pointer(2, 88064), _pointer(0, 0))

    assert "files 2 -> 0" in note
    assert "86 KB -> 0 B" in note


def test_sharp_drop_of_size_alone_is_flagged():
    assert shrink_note("d.dvc", _pointer(2, 1000), _pointer(2, 0)) is not None


def test_drop_over_half_is_flagged():
    assert shrink_note("d.dvc", _pointer(10, 1000), _pointer(4, 400)) is not None


def test_drop_of_ten_percent_is_fine():
    assert shrink_note("d.dvc", _pointer(10, 1000), _pointer(9, 900)) is None


def test_growth_is_fine():
    assert shrink_note("d.dvc", _pointer(1, 10), _pointer(5, 500)) is None


def test_is_sharp_drop_ignores_an_empty_old_value():
    assert is_sharp_drop(0, 0) is False
