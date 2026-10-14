import pytest

from mldna.core.lockfile import LockfileError, load_lockfile


def test_load_lockfile_missing_file_refuses(tmp_path):
    with pytest.raises(LockfileError, match="not found"):
        load_lockfile(tmp_path)


def test_load_lockfile_empty_refuses(tmp_path):
    (tmp_path / "models.lock.yaml").write_text("")
    with pytest.raises(LockfileError, match="no entries"):
        load_lockfile(tmp_path)


def test_load_lockfile_duplicate_model_name_refuses(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "fraud-detector:\n"
        "  version: 1\n"
        "  format: onnx\n"
        "  dest: ./app/models/fraud\n"
        "fraud-detector:\n"
        "  version: 2\n"
        "  format: onnx\n"
        "  dest: ./app/models/fraud\n"
    )
    with pytest.raises(LockfileError, match="fraud-detector"):
        load_lockfile(tmp_path)


def test_load_lockfile_different_model_names_still_loads(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "fraud-detector:\n"
        "  version: 1\n"
        "  format: onnx\n"
        "  dest: ./app/models/fraud\n"
        "churn-model:\n"
        "  version: 1\n"
        "  format: onnx\n"
        "  dest: ./app/models/churn\n"
    )

    entries = load_lockfile(tmp_path)

    assert {e.model_name for e in entries} == {"fraud-detector", "churn-model"}


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


def test_load_lockfile_entry_without_model_key_resolves_alias_as_model(tmp_path):
    # Every lockfile written before aliases existed still loads unchanged:
    # 'model' defaults to the key itself.
    (tmp_path / "models.lock.yaml").write_text(
        "fraud-detector:\n  version: 9\n  format: openvino\n  dest: ./app/models/fraud\n"
    )

    entries = load_lockfile(tmp_path)

    assert len(entries) == 1
    assert entries[0].alias == "fraud-detector"
    assert entries[0].model_name == "fraud-detector"


def test_load_lockfile_two_aliases_pin_same_model_at_different_versions(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "churn-model:\n"
        "  version: 2\n"
        "  format: pkl\n"
        "  dest: ./models/churn\n"
        "churn-canary:\n"
        "  model: churn-model\n"
        "  version: 3\n"
        "  format: onnx\n"
        "  dest: ./models/churn-canary\n"
    )

    entries = load_lockfile(tmp_path)

    assert len(entries) == 2
    stable = next(e for e in entries if e.alias == "churn-model")
    canary = next(e for e in entries if e.alias == "churn-canary")
    assert stable.model_name == "churn-model" and stable.version == "2"
    assert canary.model_name == "churn-model" and canary.version == "3"


def test_load_lockfile_shared_dest_refuses(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "churn-model:\n"
        "  version: 2\n"
        "  format: pkl\n"
        "  dest: ./models/churn\n"
        "churn-canary:\n"
        "  model: churn-model\n"
        "  version: 3\n"
        "  format: onnx\n"
        "  dest: ./models/churn\n"
    )

    with pytest.raises(LockfileError, match="dest"):
        load_lockfile(tmp_path)


def test_load_lockfile_shared_dest_refuses_with_dot_slash_vs_bare_path(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "model-a:\n  version: 1\n  format: pkl\n  dest: ./models/x\n"
        "model-b:\n  version: 1\n  format: pkl\n  dest: models/x\n"
    )

    with pytest.raises(LockfileError, match="dest"):
        load_lockfile(tmp_path)


def test_load_lockfile_shorthand_int_pin_refuses_not_a_typeerror(tmp_path):
    (tmp_path / "models.lock.yaml").write_text("churn-model: 3\n")

    with pytest.raises(LockfileError, match="not a mapping"):
        load_lockfile(tmp_path)


def test_load_lockfile_yaml_syntax_error_refuses_cleanly(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "churn-model:\n  version: 1\n  format: onnx\n dest: ./models/churn\n"
    )

    with pytest.raises(LockfileError, match="syntax error"):
        load_lockfile(tmp_path)


def test_load_lockfile_duplicate_key_error_not_mistaken_for_syntax_error(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "fraud-detector:\n"
        "  version: 1\n"
        "  format: onnx\n"
        "  dest: ./app/models/fraud\n"
        "fraud-detector:\n"
        "  version: 2\n"
        "  format: onnx\n"
        "  dest: ./app/models/fraud\n"
    )

    with pytest.raises(LockfileError, match=r"lines \d+ and \d+") as excinfo:
        load_lockfile(tmp_path)
    assert "syntax error" not in str(excinfo.value)


@pytest.mark.parametrize("content", ["- a\n", "just text\n"])
def test_load_lockfile_non_mapping_top_level_refuses_cleanly(tmp_path, content):
    (tmp_path / "models.lock.yaml").write_text(content)

    with pytest.raises(LockfileError, match="not a mapping of model pins") as excinfo:
        load_lockfile(tmp_path)
    assert "dest: ./app/models/<name>" in str(excinfo.value)


@pytest.mark.parametrize("key,value", [
    ("dest", "5"), ("dest", "[a, b]"), ("dest", ""),
    ("format", "5"), ("format", "[onnx]"),
])
def test_load_lockfile_non_text_value_refuses_cleanly(tmp_path, key, value):
    entry = {"version": "1", "format": "onnx", "dest": "./models/churn"}
    entry[key] = value
    body = "".join(f"  {k}: {v}\n" for k, v in entry.items())
    (tmp_path / "models.lock.yaml").write_text(f"churn-model:\n{body}")

    with pytest.raises(LockfileError, match=f"{key}: .* not text") as excinfo:
        load_lockfile(tmp_path)
    assert "dest: ./app/models/<name>" in str(excinfo.value)


def test_load_lockfile_non_text_model_refuses_cleanly(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "churn-model:\n  model: [a]\n  version: 1\n  format: onnx\n  dest: ./models/churn\n"
    )

    with pytest.raises(LockfileError, match="model: .* not text"):
        load_lockfile(tmp_path)


@pytest.mark.parametrize("version", ["[1]", "one", "0", "-1", "1.5", "true", "'0'", "'1.5'", "''"])
def test_load_lockfile_invalid_version_refuses_cleanly(tmp_path, version):
    (tmp_path / "models.lock.yaml").write_text(
        f"churn-model:\n  version: {version}\n  format: onnx\n  dest: ./models/churn\n"
    )

    with pytest.raises(LockfileError, match="entry 'churn-model' has version: .* not a positive whole number") as excinfo:
        load_lockfile(tmp_path)
    assert "dest: ./app/models/<name>" in str(excinfo.value)


@pytest.mark.parametrize("version", ["3", "'3'"])
def test_load_lockfile_whole_number_version_loads_as_text(tmp_path, version):
    (tmp_path / "models.lock.yaml").write_text(
        f"churn-model:\n  version: {version}\n  format: onnx\n  dest: ./models/churn\n"
    )

    assert load_lockfile(tmp_path)[0].version == "3"


def test_load_lockfile_dest_that_is_an_existing_file_refuses(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "churn").write_text("not a folder")
    (tmp_path / "models.lock.yaml").write_text(
        "churn-model:\n  version: 1\n  format: onnx\n  dest: ./models/churn\n"
    )

    with pytest.raises(LockfileError, match="entry 'churn-model' has dest ./models/churn, which is an existing file") as excinfo:
        load_lockfile(tmp_path)
    assert "Fix: point dest at a folder" in str(excinfo.value)


@pytest.mark.parametrize("outer_first", [True, False])
def test_load_lockfile_nested_dest_refuses_in_either_order(tmp_path, outer_first):
    outer = "outer:\n  version: 1\n  format: pkl\n  dest: ./m\n"
    inner = "inner:\n  version: 1\n  format: pkl\n  dest: ./m/inner\n"
    (tmp_path / "models.lock.yaml").write_text(outer + inner if outer_first else inner + outer)

    with pytest.raises(LockfileError, match="nested dests"):
        load_lockfile(tmp_path)


def test_load_lockfile_sibling_dests_with_shared_prefix_still_load(tmp_path):
    (tmp_path / "models.lock.yaml").write_text(
        "a:\n  version: 1\n  format: pkl\n  dest: ./m\n"
        "b:\n  version: 1\n  format: pkl\n  dest: ./m2\n"
    )

    assert len(load_lockfile(tmp_path)) == 2
