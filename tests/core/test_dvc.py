import subprocess
import sys

import pytest

from mldna.core.dvc import (
    DvcStatusError,
    _tracks_data,
    cloud_status,
    data_status,
    find_pointer_files,
    read_data_hashes,
    split_status,
)


# --- per-out hashes and the scope they share with the gate (D#63) ---

def test_nested_pointer_is_in_scope_for_both_the_hashes_and_the_gate(tmp_path):
    # The fail-open D#63 had to close: if hash discovery sees a pointer below
    # the workspace root while _tracks_data does not, data_status skips while
    # the run still records hashes for that data - gated nothing, claimed
    # everything. The two scopes must be identical, so a pointer only
    # discovery can see must not exist.
    workspace = tmp_path / "project"
    (workspace / "sub").mkdir(parents=True)
    (workspace / "sub" / "dataset.dvc").write_text("outs:\n- md5: abc123.dir\n  path: dataset\n")

    assert read_data_hashes(workspace) == {"sub/dataset": "abc123.dir"}
    assert _tracks_data(workspace) is True


def test_pointer_discovery_does_not_walk_into_tracked_data(producer_repo):
    # dataset.dvc sits beside dataset/, so dataset/ is pruned rather than
    # descended into - that is what keeps a multi-GB out off every track_run
    # entry. A pointer planted inside it must therefore be invisible.
    nested = producer_repo / "dataset" / "nested"
    nested.mkdir()
    (nested / "decoy.dvc").write_text("outs:\n- md5: nope\n  path: x\n")

    found = [p.name for p in find_pointer_files(producer_repo)]
    assert "dataset.dvc" in found
    assert "decoy.dvc" not in found


def test_read_data_hashes_from_real_pointer_file(producer_repo):
    hashes = read_data_hashes(producer_repo)

    assert set(hashes) == {"dataset"}
    assert hashes["dataset"].endswith(".dir")


def test_read_data_hashes_records_one_hash_per_out(producer_repo):
    second = producer_repo / "dataset2"
    second.mkdir()
    (second / "rows.csv").write_text("a,b\n1,2\n")
    subprocess.run([sys.executable, "-m", "dvc", "add", "dataset2", "-q"],
                   cwd=producer_repo, check=True)

    hashes = read_data_hashes(producer_repo)

    assert set(hashes) == {"dataset", "dataset2"}
    assert hashes["dataset"] != hashes["dataset2"]


def test_read_data_hashes_empty_when_no_pointers_exist(tmp_path):
    assert read_data_hashes(tmp_path) == {}


def test_read_data_hashes_skips_outs_with_no_recorded_hash(tmp_path):
    # A dvc.yaml stage declares outs but carries no md5 until dvc.lock exists:
    # absent, never a null hash standing in for one.
    (tmp_path / "dvc.yaml").write_text(
        "stages:\n  prep:\n    cmd: echo hi\n    outs:\n    - out.csv\n"
    )

    assert read_data_hashes(tmp_path) == {}


# --- data_status: missing vs modified (D#74) ---

def test_data_status_tells_deleted_data_from_modified_data(producer_repo):
    rows = next((producer_repo / "dataset").iterdir())
    rows.write_text("edited")
    assert split_status(data_status(producer_repo), producer_repo) == ([], ["dataset.dvc"])

    rows.unlink()
    assert split_status(data_status(producer_repo), producer_repo) == (["dataset.dvc"], [])


def test_split_status_mixed_targets_and_unknown_shapes_are_changed():
    status = {
        "a.dvc": [{"changed outs": {"a": "deleted"}}],
        "b.dvc": [{"changed outs": {"b": "modified"}}],
        "c.dvc": [{"changed outs": {"c": "deleted", "d": "new"}}],
    }
    assert split_status(status) == (["a.dvc"], ["b.dvc", "c.dvc"])


# --- data_status: failure paths first ---

def test_data_status_raises_when_dvc_fails(producer_repo, monkeypatch):
    import subprocess as sp

    def failing(args, cwd, check=True):
        return sp.CompletedProcess(
            args, 253, stdout="", stderr="ERROR: you are not inside of a DVC repository\n"
        )

    monkeypatch.setattr("mldna.core.dvc.run", failing)
    with pytest.raises(DvcStatusError) as e:
        data_status(producer_repo)
    assert "not inside of a DVC repository" in str(e.value)


def test_data_status_raises_on_unparseable_output(producer_repo, monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(
        "mldna.core.dvc.run",
        lambda args, cwd, check=True: sp.CompletedProcess(args, 0, stdout="not json", stderr=""),
    )
    with pytest.raises(DvcStatusError):
        data_status(producer_repo)


# --- data_status: answers ---

def test_data_status_reports_data_modified_without_dvc_add(producer_repo):
    (producer_repo / "dataset" / "rows.csv").write_text("a,b\n1,2\n")

    assert "dataset.dvc" in data_status(producer_repo)


def test_data_status_empty_when_in_sync(producer_repo):
    assert data_status(producer_repo) == {}


def test_data_status_empty_without_dvc_directory(tmp_path):
    assert data_status(tmp_path) == {}


def test_data_status_skip_rule_is_pointer_presence_not_dvc_dir(tmp_path):
    # A project nested in a DVC repo keeps its pointers locally while .dvc
    # stays at the parent, so .dvc presence is the wrong discriminator.
    assert _tracks_data(tmp_path) is False
    for marker in ("dataset.dvc", "dvc.lock", "dvc.yaml"):
        workspace = tmp_path / marker.replace(".", "_")
        workspace.mkdir()
        (workspace / marker).write_text("")
        assert _tracks_data(workspace) is True


# --- cloud_status: failure paths first ---

def test_cloud_status_raises_when_dvc_fails(producer_repo, monkeypatch):
    import subprocess as sp

    def failing(args, cwd, check=True):
        return sp.CompletedProcess(
            args, 1, stdout="", stderr="ERROR: no remote specified\n"
        )

    monkeypatch.setattr("mldna.core.dvc.run", failing)
    with pytest.raises(DvcStatusError) as e:
        cloud_status(producer_repo)
    assert "no remote specified" in str(e.value)


def test_cloud_status_raises_on_unparseable_output(producer_repo, monkeypatch):
    import subprocess as sp

    monkeypatch.setattr(
        "mldna.core.dvc.run",
        lambda args, cwd, check=True: sp.CompletedProcess(args, 0, stdout="not json", stderr=""),
    )
    with pytest.raises(DvcStatusError):
        cloud_status(producer_repo)


# --- cloud_status: answers ---

def test_cloud_status_reports_not_present_before_push(producer_repo):
    # scaffold already ran `dvc add dataset`, but nothing has pushed it to
    # the remote the fixture configures - the far side must show it missing.
    assert cloud_status(producer_repo) != {}


def test_cloud_status_empty_after_push(producer_repo):
    from mldna.core.push import push, stage

    stage(producer_repo)
    push(producer_repo)
    assert cloud_status(producer_repo) == {}
