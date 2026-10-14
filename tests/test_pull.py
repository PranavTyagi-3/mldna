import subprocess
from pathlib import Path

import pytest

from mldna.core.ownership import MARKER_NAME
from mldna.core.pull import PullError
from mldna.core.push import push as core_push
from mldna.core.config import ProjectConfig
from mldna.pull import pull


def _commit_and_push(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)


def test_pull_reports_no_published_model_for_commit(producer_repo):
    _commit_and_push(producer_repo)

    result = pull(producer_repo)

    assert result["model_pulled"] is False


# --- failing-case tests first: the owner's repro, and the containment guard ---

def test_pull_preserves_preexisting_unpublished_files_in_models(published):
    """The owner's repro: pulling a published run at this commit must not
    touch any unpublished experiment file already sitting in models/."""
    producer_repo, _, _ = published

    unpublished = [
        producer_repo / "models" / "experiment_a.pkl",
        producer_repo / "models" / "experiment_b.pkl",
        producer_repo / "models" / "nested" / "experiment_c.pkl",
    ]
    for f in unpublished:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"irreplaceable, never published, gitignored")

    result = pull(producer_repo)

    assert result["model_pulled"] is True
    for f in unpublished:
        assert f.exists(), f"{f} was destroyed by pull"
        assert f.read_bytes() == b"irreplaceable, never published, gitignored"
    # the bundle landed beside them, not over them
    assert (producer_repo / result["dest"] / "model.pkl").read_bytes() == b"trained model bytes"


def test_pull_refuses_to_replace_a_foreign_version_directory(published):
    """If something other than mldna already occupies the exact version
    subdirectory pull would write to - and it carries no marker - pull must
    refuse rather than delete it."""
    producer_repo, _, _ = published

    # discover the dest pull would use by running the real thing once, then
    # recreate the collision for a second, independent run.
    first = pull(producer_repo)
    foreign_dest = producer_repo / first["dest"]
    # strip the marker so it looks foreign, and add an unrelated file
    (foreign_dest / MARKER_NAME).unlink()
    (foreign_dest / "not_mldna.txt").write_text("foreign")

    with pytest.raises(PullError, match="no .mldna-version marker|not empty"):
        pull(producer_repo)


def test_pull_refuses_an_abandoned_version(published):
    # D#76: the version the run points at lost a concurrent publish.
    import mlflow

    producer_repo, _, result = published
    mlflow.tracking.MlflowClient().set_model_version_tag(
        result["model_name"], result["version"], "abandoned", "true")

    with pytest.raises(PullError, match="tagged abandoned") as excinfo:
        pull(producer_repo)
    assert "Fix: fetch the winning version" in str(excinfo.value)
    assert not (producer_repo / "models" / f"{result['model_name']}-v{result['version']}").exists()


def test_pull_model_id_mismatch_offers_restoring_the_registered_id(published):
    import mlflow

    producer_repo, _, result = published
    mlflow.tracking.MlflowClient().set_registered_model_tag(
        result["model_name"], "mldna_model_id", "99999999-2222-4333-8444-555555555555")

    with pytest.raises(PullError, match="belongs to another project") as excinfo:
        pull(producer_repo)
    assert ("restore it: set project.model_id: 99999999-2222-4333-8444-555555555555 "
            "in mldna.yaml and commit") in str(excinfo.value)


# --- passing-case tests ---

def test_pull_downloads_single_published_run_into_versioned_subdir(published):
    producer_repo, _, published_result = published

    result = pull(producer_repo)

    assert result["model_pulled"] is True
    assert result["run_id"] == published_result["run_id"]
    full = ProjectConfig.load(producer_repo).full_model_name()
    expected_dest = str(Path("models") / f"{full}-v{result['registry_version']}")
    assert result["dest"] == expected_dest
    dest = producer_repo / result["dest"]
    assert (dest / "model.pkl").read_bytes() == b"trained model bytes"
    assert (dest / MARKER_NAME).exists()
    # models/ itself was never deleted
    assert (producer_repo / "models").exists()


def test_pull_same_version_twice_replaces_only_that_subdir(published):
    producer_repo, _, _ = published

    sibling = producer_repo / "models" / "sibling_experiment.pkl"
    sibling.write_bytes(b"keep me")

    first = pull(producer_repo)
    second = pull(producer_repo)

    assert first["dest"] == second["dest"]
    dest = producer_repo / second["dest"]
    assert (dest / "model.pkl").read_bytes() == b"trained model bytes"
    assert sibling.exists()
    assert sibling.read_bytes() == b"keep me"


def test_pull_multiple_published_runs_without_select_refuses(published_twice):
    producer_repo, _ = published_twice

    with pytest.raises(PullError, match="published runs match"):
        pull(producer_repo)


def test_pull_multiple_published_runs_with_select_downloads_chosen(published_twice):
    producer_repo, (first, _) = published_twice

    result = pull(producer_repo, select=lambda candidates: next(
        c for c in candidates if c.info.run_id == first["run_id"]
    ))

    assert result["run_id"] == first["run_id"]
    assert (producer_repo / result["dest"] / "model.pkl").read_bytes() == b"model bytes v1"
    # the other (unselected) candidate's sibling files, if any, are untouched -
    # in particular models/ as a whole still exists and was never cleared.
    assert (producer_repo / "models").exists()


def test_pull_without_mldna_yaml_raises_parent_project_hint_and_pulls_nothing(
    producer_repo, monkeypatch
):
    import mldna.pull as pull_module

    called = []
    monkeypatch.setattr(pull_module, "pull_code_and_data", lambda *a, **k: called.append(a))
    sub = producer_repo / "notebooks"
    sub.mkdir()

    with pytest.raises(PullError) as exc_info:
        pull(sub)

    msg = str(exc_info.value)
    assert str(producer_repo / "mldna.yaml") in msg
    assert "mldna init" not in msg
    assert called == []
