import json
import subprocess
import sys

import mlflow
import pytest

from mldna.core import dvc, git, hashing
from mldna.core.lock import RunLock, RunLockError
from mldna.tracking import TrackRunError, track_run


def _get_run(workspace_dir, run_id):
    mlflow.set_tracking_uri(f"file:{workspace_dir / 'mlruns'}")
    return mlflow.get_run(run_id)


@pytest.fixture
def dvc_project(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True, capture_output=True
    )
    return producer_repo


@pytest.fixture
def nested_dvc_project(tmp_path):
    """A project living inside an outer DVC repo: pointers here, .dvc at the parent."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        [sys.executable, "-m", "dvc", "init", "-q"], cwd=tmp_path, check=True, capture_output=True
    )
    project_dir = tmp_path / "models" / "churn"
    (project_dir / "dataset").mkdir(parents=True)
    (project_dir / "dataset" / "rows.csv").write_text("5000 rows")
    (project_dir / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  namespace: test-ns\n  repo_role: producer\n"
    )
    (project_dir / ".gitignore").write_text("models/\noutputs/\n.mldna/\nmlruns/\n")
    subprocess.run(
        [sys.executable, "-m", "dvc", "add", "dataset", "-q"],
        cwd=project_dir, check=True, capture_output=True,
    )
    for args in (
        ["config", "user.email", "dev@example.com"],
        ["config", "user.name", "dev"],
        ["add", "-A"],
        ["commit", "-q", "-m", "nested project"],
    ):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)
    assert not (project_dir / ".dvc").exists()
    return project_dir


# --- refusal paths ---

def test_no_namespace_refuses_naming_the_fix(tmp_path):
    # issue #16: no legacy fallback to the bare name - a producer project
    # without project.namespace must refuse loudly, not silently train under
    # an unnamespaced experiment that could collide with another team's.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "mldna.yaml").write_text("project:\n  model_name: test-model\n  repo_role: producer\n")
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=dev@example.com", "-c", "user.name=dev",
         "commit", "-q", "-m", "init"],
        cwd=tmp_path, check=True,
    )

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=tmp_path):
            pass

    message = str(e.value)
    assert "project.namespace" in message
    assert str(tmp_path) in message
    assert RunLock(tmp_path).is_locked() is False


def test_dirty_tree_refuses(git_repo):
    (git_repo / "README.md").write_text("dirty\n")

    with pytest.raises(TrackRunError):
        with track_run(workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False


def test_dirty_tree_allowed_with_allow_dirty(git_repo):
    (git_repo / "README.md").write_text("dirty\n")

    with track_run(workspace_dir=git_repo, allow_dirty=True) as run:
        run_id = run.run_id

    assert _get_run(git_repo, run_id).data.tags["dirty"] == "true"


def test_no_identity_refuses(tmp_path, monkeypatch):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  namespace: test-ns\n  repo_role: producer\n"
    )
    subprocess.run(["git", "add", "mldna.yaml"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "-c", "user.email=dev@example.com", "-c", "user.name=dev",
         "commit", "-q", "-m", "init"],
        cwd=tmp_path, check=True,
    )

    empty_config = tmp_path / "empty_gitconfig"
    empty_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_config))

    with pytest.raises(TrackRunError):
        with track_run(workspace_dir=tmp_path):
            pass

    assert RunLock(tmp_path).is_locked() is False


def test_lock_already_held_refuses(git_repo):
    RunLock(git_repo).acquire()

    with pytest.raises(RunLockError):
        with track_run(workspace_dir=git_repo):
            pass


def test_data_modified_without_dvc_add_refuses_naming_the_dvc_plane(dvc_project):
    # The user's reported bug: rows edited in place, so git (which ignores
    # dataset/) sees nothing and the dataset.dvc pointer still claims the old
    # data. The run must refuse, and name the data plane - the fix differs.
    (dvc_project / "dataset" / "rows.csv").write_text("a,b,c")

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=dvc_project):
            pass

    message = str(e.value)
    assert "DVC-tracked data has changes its pointers do not record" in message
    assert "dataset.dvc" in message
    assert "dvc add" in message
    assert "uncommitted changes" not in message
    assert RunLock(dvc_project).is_locked() is False


def test_data_deleted_refuses_with_checkout_advice_not_stage_push(dvc_project):
    # D#74: missing data is not a change to record; stage/push would pin an empty dataset.
    for f in (dvc_project / "dataset").iterdir():
        f.unlink()

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=dvc_project):
            pass

    message = str(e.value)
    assert "DVC-tracked data is missing from the working tree (dataset.dvc)" in message
    assert "dvc checkout" in message
    assert "mldna stage" not in message
    assert "mldna push" not in message
    assert "do not record" not in message
    assert RunLock(dvc_project).is_locked() is False


def test_data_modified_allowed_with_allow_dirty_tags_dirty(dvc_project):
    (dvc_project / "dataset" / "rows.csv").write_text("a,b,c")

    with track_run(workspace_dir=dvc_project, allow_dirty=True) as run:
        run_id = run.run_id

    assert _get_run(dvc_project, run_id).data.tags["dirty"] == "true"


def test_undeterminable_data_status_refuses_fail_closed(dvc_project, monkeypatch):
    def boom(workspace_dir):
        raise dvc.DvcStatusError("ERROR: unable to read '.dvc/tmp'")

    monkeypatch.setattr(dvc, "data_status", boom)

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=dvc_project):
            pass

    message = str(e.value)
    assert "Could not determine whether the DVC-tracked data is clean" in message
    assert "an unanswered check is not an answer of clean" in message
    assert "DVC-tracked data has changes" not in message
    assert RunLock(dvc_project).is_locked() is False


def test_undeterminable_git_status_refuses_fail_closed(git_repo, monkeypatch):
    def boom(repo_dir):
        raise git.GitStatusError("fatal: detected dubious ownership in repository")

    monkeypatch.setattr(git, "is_clean", boom)

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=git_repo):
            pass

    message = str(e.value)
    assert "Could not determine whether the git working tree is clean" in message
    assert "an unanswered check is not an answer of clean" in message
    assert "uncommitted changes" not in message
    assert RunLock(git_repo).is_locked() is False


def test_both_planes_dirty_names_both(dvc_project):
    (dvc_project / "main.py").write_text("# edited")
    (dvc_project / "dataset" / "rows.csv").write_text("a,b,c")

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=dvc_project):
            pass

    message = str(e.value)
    assert "uncommitted changes" in message
    assert "DVC-tracked data has changes its pointers do not record" in message


def test_git_dirty_only_names_only_the_git_plane(git_repo):
    (git_repo / "README.md").write_text("dirty")

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=git_repo):
            pass

    message = str(e.value)
    assert "uncommitted changes" in message
    assert "mldna push" in message
    assert "DVC-tracked data" not in message


def test_project_without_dvc_outs_is_unaffected_by_the_data_gate(git_repo):
    # No .dvc at all: the data gate must stay silent and the run must start
    # exactly as before, leaving the missing dvc_data_hash for publish to
    # refuse on later.
    assert not (git_repo / ".dvc").exists()

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags = _get_run(git_repo, run_id).data.tags
    assert "dvc_data_hash" not in tags
    assert "dirty" not in tags


def test_data_modified_in_a_project_nested_in_a_dvc_repo_refuses(nested_dvc_project):
    # `dvc add` run here wrote dataset.dvc beside the data while .dvc stayed at
    # the parent repo - so a missing local .dvc must not be read as "no data to
    # check", or this layout keeps the original bug.
    (nested_dvc_project / "dataset" / "rows.csv").write_text("3318 rows")

    with pytest.raises(TrackRunError) as e:
        with track_run(workspace_dir=nested_dvc_project):
            pass

    message = str(e.value)
    assert "DVC-tracked data has changes its pointers do not record" in message
    assert "dataset.dvc" in message
    assert "uncommitted changes" not in message
    assert RunLock(nested_dvc_project).is_locked() is False


# --- happy path ---

def test_run_is_recorded_under_the_namespaced_experiment(git_repo):
    # issue #16: the experiment name is namespace-model_name, never the bare
    # model_name - that's what keeps two teams' same-named models apart.
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    mlflow.set_tracking_uri(f"file:{git_repo / 'mlruns'}")
    run_info = mlflow.get_run(run_id).info
    experiment = mlflow.get_experiment(run_info.experiment_id)
    assert experiment.name == "test-ns-test-model"
    assert mlflow.get_experiment_by_name("test-model") is None


def test_run_has_lineage_tags(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags = _get_run(git_repo, run_id).data.tags
    assert len(tags["git_commit"]) == 40
    assert tags["git_branch"] in ("main", "master")
    assert tags["user"] == "dev@example.com"
    assert "mldna_version" in tags
    assert "dirty" not in tags
    assert "dvc_data_hash" not in tags


def test_dvc_data_hash_tag_set_when_dataset_tracked(producer_repo):
    import subprocess
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    tags = _get_run(producer_repo, run_id).data.tags
    assert tags["dvc_data_hash__0"].endswith(".dir:dataset")
    assert "dvc_data_hash" not in tags


def test_dvc_data_hash_tag_written_per_out(producer_repo):
    # D#63: one tag per DVC out, same idiom as model_sha256__<n>, so the run
    # records exactly the scope D#45's gate checks.
    import subprocess
    second = producer_repo / "dataset2"
    second.mkdir()
    (second / "rows.csv").write_text("a,b\n1,2\n")
    subprocess.run([sys.executable, "-m", "dvc", "add", "dataset2", "-q"],
                   cwd=producer_repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "two outs"], cwd=producer_repo, check=True)

    with track_run(workspace_dir=producer_repo) as run:
        run_id = run.run_id

    tags = _get_run(producer_repo, run_id).data.tags
    recorded = {
        value.partition(":")[2]: value.partition(":")[0]
        for key, value in tags.items() if key.startswith("dvc_data_hash__")
    }
    assert set(recorded) == {"dataset", "dataset2"}
    assert recorded["dataset"] != recorded["dataset2"]


def test_model_file_is_discovered_and_hashed(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "model.pkl").write_bytes(b"trained model bytes")

    tags = _get_run(git_repo, run_id).data.tags
    assert tags["save_context"] == "in_run"
    assert tags["model_sha256__0"].endswith(":model.pkl")

    checksums_path = mlflow.artifacts.download_artifacts(
        run_id=run_id, artifact_path="checksums.json"
    )
    checksums = json.loads(open(checksums_path).read())
    assert checksums["model.pkl"]["context"] == "in_run"


def test_last_run_marker_written_on_success(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["run_id"] == run_id
    assert marker["status"] == "FINISHED"


def test_custom_model_glob_from_mldna_yaml_is_honored(git_repo):
    (git_repo / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  namespace: test-ns\n  repo_role: producer\n"
        "tracking:\n  model_globs: [\"*.custom\"]\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add custom glob"], cwd=git_repo, check=True)

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "model.custom").write_bytes(b"custom format model bytes")

    tags = _get_run(git_repo, run_id).data.tags
    assert tags["model_sha256__0"].endswith(":model.custom")


def test_custom_scan_exclude_from_mldna_yaml_is_honored(git_repo):
    (git_repo / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  namespace: test-ns\n  repo_role: producer\n"
        "tracking:\n  scan_exclude: [\"ignored_dir/\"]\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add scan exclude"], cwd=git_repo, check=True)

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "ignored_dir").mkdir()
        (git_repo / "ignored_dir" / "model.pkl").write_bytes(b"should be excluded")

    tags = _get_run(git_repo, run_id).data.tags
    assert "model_sha256__0" not in tags


def test_failed_run_marks_status_and_releases_lock(git_repo):
    with pytest.raises(RuntimeError):
        with track_run(workspace_dir=git_repo) as run:
            run_id = run.run_id
            raise RuntimeError("boom")

    assert _get_run(git_repo, run_id).info.status == "FAILED"
    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["status"] == "FAILED"
    assert RunLock(git_repo).is_locked() is False


def test_setup_failure_after_start_run_releases_lock_and_fails_the_run(git_repo):
    # The reported bug: a params.yaml typo (or any failure past start_run())
    # left the lock held and the orphaned run looking FINISHED.
    (git_repo / "params.yaml").write_text("bad: [unterminated\n")
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "params"], cwd=git_repo, check=True)

    with pytest.raises(Exception):
        with track_run(workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False

    mlflow.set_tracking_uri(f"file:{git_repo / 'mlruns'}")
    experiment = mlflow.get_experiment_by_name("test-ns-test-model")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="list")
    assert runs[0].info.status == "FAILED"


def test_setup_failure_before_start_run_releases_lock_without_touching_mlflow(git_repo, monkeypatch):
    # No run exists yet at this point in __enter__ - end_run() must not be
    # called (there is nothing to fail), only the lock matters.
    def boom(*args, **kwargs):
        raise RuntimeError("tracking server unreachable")

    monkeypatch.setattr(mlflow, "set_experiment", boom)

    with pytest.raises(RuntimeError):
        with track_run(workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False


def test_yaml_syntax_error_in_mldna_yaml_releases_lock_and_next_run_works(git_repo):
    # #11: a non-TrackRunError from the very first gate used to leak the lock
    # for the rest of the process.
    good = (git_repo / "mldna.yaml").read_text()
    (git_repo / "mldna.yaml").write_text("project: [unterminated\n")

    with pytest.raises(Exception):
        with track_run(workspace_dir=git_repo, allow_dirty=True):
            pass

    assert RunLock(git_repo).is_locked() is False

    (git_repo / "mldna.yaml").write_text(good)
    with track_run(workspace_dir=git_repo, allow_dirty=True):
        pass


def test_end_run_failing_during_start_cleanup_still_releases_lock_and_warns(
    git_repo, monkeypatch, capsys
):
    # #34: marking the run FAILED must not be able to keep the lock.
    def boom(*args, **kwargs):
        raise RuntimeError("tracking server unreachable")

    monkeypatch.setattr(mlflow, "log_artifact", boom)
    monkeypatch.setattr(mlflow, "end_run", boom)

    with pytest.raises(RuntimeError, match="unreachable"):
        with track_run(workspace_dir=git_repo):
            pass

    assert RunLock(git_repo).is_locked() is False
    assert "may still show RUNNING" in capsys.readouterr().out


def test_params_file_logged_under_prefix_leaving_bare_name_free(git_repo):
    (git_repo / "params.yaml").write_text("seed: 8\n")
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "params"], cwd=git_repo, check=True)

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        mlflow.log_param("seed", 5)  # must not collide with the params.yaml copy

    run_data = _get_run(git_repo, run_id).data
    assert run_data.params["params_file.seed"] == "8"
    assert run_data.params["seed"] == "5"


def test_argv_tag_not_recorded_by_default(git_repo, monkeypatch):
    # D#75: argv can hold secrets, so capture is opt-in.
    monkeypatch.setattr(sys, "argv", ["train.py", "--api-key", "sk-secret"])

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags_data = _get_run(git_repo, run_id).data.tags
    assert "invocation_argv" not in tags_data


def test_argv_tag_recorded_when_opted_in(git_repo, monkeypatch):
    (git_repo / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  namespace: test-ns\n  repo_role: producer\n"
        "tracking:\n  record_argv: true\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "record argv"], cwd=git_repo, check=True)
    monkeypatch.setattr(sys, "argv", ["train.py", "--seed", "5"])

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags_data = _get_run(git_repo, run_id).data.tags
    assert json.loads(tags_data["invocation_argv"]) == ["train.py", "--seed", "5"]


def test_declared_env_vars_recorded(git_repo, monkeypatch):
    (git_repo / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  namespace: test-ns\n  repo_role: producer\n"
        "tracking:\n  env_allowlist: [\"SEED_OVERRIDE\"]\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "add env allowlist"], cwd=git_repo, check=True)
    monkeypatch.setenv("SEED_OVERRIDE", "5")

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags_data = _get_run(git_repo, run_id).data.tags
    assert json.loads(tags_data["invocation_env"]) == {"SEED_OVERRIDE": "5"}


def test_unset_env_allowlist_records_no_env_tag(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    tags_data = _get_run(git_repo, run_id).data.tags
    assert "invocation_env" not in tags_data


def test_lock_and_marker_survive_end_run_crashing(git_repo, monkeypatch, capsys):
    # Live-verified bug: mlflow.end_run() can raise mid-call against a real
    # remote tracking server (Windows console can't encode the emoji mlflow
    # writes for a real run URL) - the lock/marker must not depend on it
    # succeeding, or every crash there permanently locks the workspace.
    real_end_run = mlflow.end_run

    def boom(*args, **kwargs):
        raise UnicodeEncodeError("charmap", "x", 0, 1, "boom")

    # D#61 changed this from propagating to warning-and-continuing: losing the
    # whole run to a transport failure at exit was the harm, and `mldna attach`
    # can now finish it. The lock/marker guarantee this test exists for is
    # unchanged, and the failure is still reported, just not fatally.
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        monkeypatch.setattr(mlflow, "end_run", boom)

    monkeypatch.setattr(mlflow, "end_run", real_end_run)
    captured = capsys.readouterr()
    assert "could not report run status" in captured.out
    assert "mldna attach" in captured.out
    assert RunLock(git_repo).is_locked() is False
    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["run_id"] == run_id


# --- issue #10: model hashes must survive a network blip at run exit ---

def test_checksums_persisted_locally_when_server_tagging_fails(git_repo, monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("network blip")

    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "model.pkl").write_bytes(b"trained model bytes")
        monkeypatch.setattr(mlflow, "set_tag", boom)

    # The server never got the tags - the one thing that must not also be
    # true is that the hash is gone.
    tags_data = _get_run(git_repo, run_id).data.tags
    assert "model_sha256__0" not in tags_data
    assert "save_context" not in tags_data

    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["checksums"]["model.pkl"] == hashing.sha256_file(git_repo / "model.pkl")


def test_marker_records_empty_checksums_when_no_model_file_saved(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id

    marker = json.loads((git_repo / ".mldna" / "last_run.json").read_text())
    assert marker["run_id"] == run_id
    assert marker["checksums"] == {}


def test_successful_run_carries_no_attached_label(git_repo):
    with track_run(workspace_dir=git_repo) as run:
        run_id = run.run_id
        (git_repo / "model.pkl").write_bytes(b"trained model bytes")

    tags_data = _get_run(git_repo, run_id).data.tags
    assert "attached" not in tags_data
    assert "attached_at" not in tags_data
