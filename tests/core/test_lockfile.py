import pytest

from mlops.core.lockfile import LockfileError, load_lockfile


def test_load_lockfile_missing_file_refuses(tmp_path):
    with pytest.raises(LockfileError, match="not found"):
        load_lockfile(tmp_path)


def test_load_lockfile_empty_refuses(tmp_path):
    (tmp_path / "models.lock.yaml").write_text("")
    with pytest.raises(LockfileError, match="no entries"):
        load_lockfile(tmp_path)


def test_load_lockfile_missing_required_key_refuses(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "fraud-detector:\n  version: 9\n  format: onnx\n"
    )
    with pytest.raises(LockfileError, match="dest"):
        load_lockfile(tmp_path)


def test_load_lockfile_parses_entries(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "fraud-detector:\n"
        "  version: 9\n"
        "  format: openvino\n"
        "  dest: ./app/models/fraud\n"
        "churn-model:\n"
        "  version: 3\n"
        "  format: onnx\n"
        "  dest: ./app/models/churn\n"
    )

    entries = load_lockfile(tmp_path)

    assert len(entries) == 2
    fraud = next(e for e in entries if e.model_name == "fraud-detector")
    assert fraud.version == "9"
    assert fraud.format == "openvino"
    assert fraud.dest == "./app/models/fraud"
