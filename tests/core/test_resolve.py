import subprocess

import pytest

from mlops.core.resolve import ResolveError, resolve_run
from mlops.tracking import track_run

MODEL_BYTES = b"identical model bytes for hash matching"


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def _commit_change(repo, message):
    (repo / "README.md").write_text(message + "\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", message], cwd=repo, check=True)


def test_resolve_finds_single_match(producer_repo, tmp_path):
    _commit_scaffold(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id
        (producer_repo / "models" / "model.pkl").write_bytes(MODEL_BYTES)

    candidate = tmp_path / "to_publish.pkl"
    candidate.write_bytes(MODEL_BYTES)

    resolved = resolve_run("test-model", candidate)
    assert resolved.run_id == run_id
    assert resolved.relpath == "models/model.pkl"
    assert resolved.auto_resolved_note is None


def test_resolve_zero_match_raises(producer_repo, tmp_path):
    _commit_scaffold(producer_repo)
    with track_run(workspace_dir=producer_repo):
        (producer_repo / "models" / "model.pkl").write_bytes(MODEL_BYTES)

    candidate = tmp_path / "never_tracked.pkl"
    candidate.write_bytes(b"different bytes")

    with pytest.raises(ResolveError, match="No tracked run matches"):
        resolve_run("test-model", candidate)


def test_resolve_ambiguous_identical_lineage_auto_resolves(producer_repo, tmp_path):
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo) as run1:
        run1_id = run1.run_id
        (producer_repo / "models" / "model_a.pkl").write_bytes(MODEL_BYTES)

    with track_run(workspace_dir=producer_repo):
        (producer_repo / "models" / "model_b.pkl").write_bytes(MODEL_BYTES)

    candidate = tmp_path / "to_publish.pkl"
    candidate.write_bytes(MODEL_BYTES)

    resolved = resolve_run("test-model", candidate)
    assert resolved.run_id == run1_id
    assert resolved.relpath == "models/model_a.pkl"
    assert resolved.auto_resolved_note is not None


def test_resolve_ambiguous_differing_lineage_raises(producer_repo, tmp_path):
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo):
        (producer_repo / "models" / "model_a.pkl").write_bytes(MODEL_BYTES)

    _commit_change(producer_repo, "unrelated code change")

    with track_run(workspace_dir=producer_repo):
        (producer_repo / "models" / "model_b.pkl").write_bytes(MODEL_BYTES)

    candidate = tmp_path / "to_publish.pkl"
    candidate.write_bytes(MODEL_BYTES)

    with pytest.raises(ResolveError, match="DIFFERING lineage"):
        resolve_run("test-model", candidate)
