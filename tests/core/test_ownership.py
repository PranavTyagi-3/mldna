import pytest

from mldna.core import ownership


def test_validate_contained_rejects_absolute_dest(tmp_path):
    with pytest.raises(ownership.OwnershipError, match="absolute"):
        ownership.validate_contained(tmp_path, tmp_path / "elsewhere", "entry")


def test_validate_contained_rejects_workspace_root_itself(tmp_path):
    with pytest.raises(ownership.OwnershipError):
        ownership.validate_contained(tmp_path, ".", "entry")


def test_validate_contained_rejects_escaping_dest(tmp_path):
    with pytest.raises(ownership.OwnershipError):
        ownership.validate_contained(tmp_path, "../victim", "entry")


def test_validate_contained_accepts_nested_relative_dest(tmp_path):
    resolved = ownership.validate_contained(tmp_path, "app/models/thing", "entry")
    assert resolved == (tmp_path / "app" / "models" / "thing").resolve()


def test_is_safe_to_replace_true_for_absent_dest(tmp_path):
    assert ownership.is_safe_to_replace(tmp_path / "missing") is True


def test_is_safe_to_replace_true_for_empty_dest(tmp_path):
    dest = tmp_path / "empty"
    dest.mkdir()
    assert ownership.is_safe_to_replace(dest) is True


def test_is_safe_to_replace_true_for_marked_dest(tmp_path):
    dest = tmp_path / "owned"
    dest.mkdir()
    (dest / ownership.MARKER_NAME).write_text("1")
    assert ownership.is_safe_to_replace(dest) is True


def test_is_safe_to_replace_true_for_legacy_bare_version_marker(tmp_path):
    # Pre-D#53 markers are a bare version string, not JSON. Ownership only
    # checks presence, so this must still count as owned.
    dest = tmp_path / "owned"
    dest.mkdir()
    (dest / ownership.MARKER_NAME).write_text("1")
    assert ownership.is_safe_to_replace(dest) is True


def test_is_safe_to_replace_false_for_foreign_dest(tmp_path):
    dest = tmp_path / "src"
    dest.mkdir()
    (dest / "server.py").write_text("print('hi')")
    assert ownership.is_safe_to_replace(dest) is False


def test_require_owned_or_absent_refuses_foreign_dest(tmp_path):
    dest = tmp_path / "src"
    dest.mkdir()
    (dest / "server.py").write_text("print('hi')")

    with pytest.raises(ownership.OwnershipError, match="did not create"):
        ownership.require_owned_or_absent(dest, "entry")

    assert (dest / "server.py").exists()


# --- marker contents: ownership (presence) vs. identity (D#53) -----------

def test_write_marker_then_read_marker_round_trips_identity(tmp_path):
    dest = tmp_path / "d"
    dest.mkdir()
    ownership.write_marker(dest, "test-model", "3", "pkl", "deadbeef")
    assert ownership.read_marker(dest) == {
        "model_name": "test-model", "version": "3", "format": "pkl",
        "manifest_digest": "deadbeef",
    }


def test_write_marker_without_format_or_digest_omits_them(tmp_path):
    # pull.py's call site has no format/digest to offer - it only needs
    # ownership, not fetch's identity check.
    dest = tmp_path / "d"
    dest.mkdir()
    ownership.write_marker(dest, "test-model", "3")
    marker = ownership.read_marker(dest)
    assert marker["model_name"] == "test-model"
    assert marker["version"] == "3"
    assert "format" not in marker
    assert "manifest_digest" not in marker


def test_read_marker_flags_bare_old_format_version_as_legacy(tmp_path):
    dest = tmp_path / "d"
    dest.mkdir()
    (dest / ownership.MARKER_NAME).write_text("2")
    assert ownership.read_marker(dest) == {"version": "2", "legacy": True}


def test_read_marker_returns_none_when_absent(tmp_path):
    assert ownership.read_marker(tmp_path / "missing") is None


def test_swap_into_dest_refuses_foreign_dest_and_deletes_nothing(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.onnx").write_bytes(b"new")

    dest = tmp_path / "src"
    dest.mkdir()
    (dest / "server.py").write_text("print('hi')")

    with pytest.raises(ownership.OwnershipError):
        ownership.swap_into_dest(staged, dest, "entry")

    assert (dest / "server.py").exists()
    assert (staged / "model.onnx").exists()


def test_swap_into_dest_into_nonexistent_dest(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.onnx").write_bytes(b"new")
    dest = tmp_path / "app" / "models" / "thing"

    ownership.swap_into_dest(staged, dest, "entry")

    assert (dest / "model.onnx").read_bytes() == b"new"


def test_swap_into_dest_replaces_owned_dest(tmp_path):
    dest = tmp_path / "owned"
    dest.mkdir()
    (dest / "model.onnx").write_bytes(b"old")
    (dest / ownership.MARKER_NAME).write_text("1")

    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "model.onnx").write_bytes(b"new")

    ownership.swap_into_dest(staged, dest, "entry")

    assert (dest / "model.onnx").read_bytes() == b"new"
    assert not (dest / ownership.MARKER_NAME).exists()
    # no leftover sibling from the old directory
    assert list(tmp_path.iterdir()) == [dest]
