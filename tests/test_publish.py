import subprocess
import sys
from types import SimpleNamespace

import mlflow
import pytest
import requests
from urllib3.exceptions import NewConnectionError

from mldna.core import dvc as dvc_module
from mldna.core import git
from mldna.core.commitcheck import CommitReachabilityError
from mldna.core.conversions import ConversionError
from mldna.core.hashing import sha256_file
from mldna.core.push import push as core_push
from mldna.core.resolve import ResolveError
from mldna.core.rungates import RunGateError
from mldna.publish import PublishError, _data_needs_push, publish
from mldna.tracking import track_convert, track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def _tracked_pkl(repo):
    _commit_scaffold(repo)
    core_push(repo)
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    return pkl, run.run_id


def test_publish_refuses_without_namespace(git_repo):
    # issue #16: no legacy fallback - a producer project with model_name but
    # no namespace must refuse rather than register under the bare name.
    (git_repo / "mldna.yaml").write_text("project:\n  model_name: test-model\n  repo_role: producer\n")
    pkl = git_repo / "models" / "model.pkl"
    pkl.parent.mkdir(exist_ok=True)
    pkl.write_bytes(b"trained model bytes")

    with pytest.raises(PublishError, match="project.namespace"):
        publish(pkl, workspace_dir=git_repo)


@pytest.fixture(scope="module")
def pkl_only_publish(producer_factory, tmp_path_factory):
    """One successful pkl-only publish, shared by the tests below that each
    assert a different facet of its outcome (a publish costs ~25s)."""
    repo = producer_factory(tmp_path_factory.mktemp("publish_once"))
    summaries, download_calls = [], []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["train.py", "--lr", "0.1"])
        with open(repo / "mldna.yaml", "a") as f:
            f.write("\ntracking:\n  record_argv: true\n")
        pkl, run_id = _tracked_pkl(repo)

        real_download = mlflow.artifacts.download_artifacts

        def spy_download(*args, **kwargs):
            download_calls.append(kwargs.get("artifact_path"))
            return real_download(*args, **kwargs)

        mp.setattr(mlflow.artifacts, "download_artifacts", spy_download)
        result = publish(pkl, workspace_dir=repo, confirm=lambda s: summaries.append(s) or True)

    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)
    return SimpleNamespace(repo=repo, run_id=run_id, result=result,
                           summaries=summaries, download_calls=download_calls)


def test_publish_registers_under_the_namespaced_model_name(pkl_only_publish):
    repo, result = pkl_only_publish.repo, pkl_only_publish.result

    assert result["model_name"] == "project-test-model"
    mlflow.set_tracking_uri(f"file:{repo / 'mlruns'}")
    client = mlflow.tracking.MlflowClient()
    assert client.get_model_version("project-test-model", result["version"])


def test_publish_pkl_only(pkl_only_publish):
    repo, result = pkl_only_publish.repo, pkl_only_publish.result

    assert result["run_id"] == pkl_only_publish.run_id
    mlflow.set_tracking_uri(f"file:{repo / 'mlruns'}")
    run = mlflow.get_run(pkl_only_publish.run_id)
    assert run.data.tags["registry_version"] == str(result["version"])
    assert "published_by" in run.data.tags
    mv = mlflow.tracking.MlflowClient().get_model_version(result["model_name"], result["version"])
    assert mv.tags["formats"] == "pkl"


def test_publish_unique_match_has_no_resolution_note(pkl_only_publish):
    import yaml
    repo, result = pkl_only_publish.repo, pkl_only_publish.result

    mlflow.set_tracking_uri(f"file:{repo / 'mlruns'}")
    mv = mlflow.tracking.MlflowClient().get_model_version(result["model_name"], result["version"])
    manifest = yaml.safe_load(open(mlflow.artifacts.download_artifacts(f"{mv.source}/manifest.yaml")).read())
    assert "resolution_note" not in manifest


def test_first_publish_stamps_model_id_on_registered_model(pkl_only_publish):
    import yaml
    repo, result = pkl_only_publish.repo, pkl_only_publish.result
    model_id = yaml.safe_load((repo / "mldna.yaml").read_text())["project"]["model_id"]

    assert result["model_id"] == model_id
    assert _registered_model_id(repo, result["model_name"]) == model_id


def test_first_publish_stamps_normalised_origin(pkl_only_publish):
    # D#77: the producer's origin, normalised so no credentials reach a tag.
    repo, result = pkl_only_publish.repo, pkl_only_publish.result
    mlflow.set_tracking_uri(f"file:{repo / 'mlruns'}")
    tags = mlflow.tracking.MlflowClient().get_registered_model(result["model_name"]).tags

    assert tags["mldna_origin"] == git.origin_url(repo)
    assert not tags["mldna_origin"].endswith(".git")


def test_publish_records_the_per_file_tag(pkl_only_publish):
    # D#78: one publish per file per run, keyed by the file's hash.
    repo, result = pkl_only_publish.repo, pkl_only_publish.result
    mlflow.set_tracking_uri(f"file:{repo / 'mlruns'}")
    tags = mlflow.get_run(result["run_id"]).data.tags
    sha = sha256_file(repo / "models" / "model.pkl")

    assert tags[f"published__{sha[:16]}"] == str(result["version"])


def test_publish_summary_omits_push_line_when_already_on_remote(pkl_only_publish):
    # _tracked_pkl's core_push already landed commit+data
    assert "Will push" not in pkl_only_publish.summaries[0]


def test_publish_pkl_only_does_not_download_conversions(pkl_only_publish):
    assert pkl_only_publish.result["run_id"] == pkl_only_publish.run_id
    assert pkl_only_publish.download_calls == []


def test_publish_confirmation_shows_recorded_invocation(pkl_only_publish):
    summary = pkl_only_publish.summaries[0]

    assert "recorded, not verified" in summary
    assert "--lr" in summary
    assert "not recorded" not in summary


def test_publish_summary_labels_absent_invocation():
    from mldna.publish import _summary
    run = SimpleNamespace(data=SimpleNamespace(tags={}))
    resolved = SimpleNamespace(run_id="abc", auto_resolved_note=None)

    summary = _summary("ns-model", run, resolved, "me", "id-1")

    assert "Invocation: not recorded (tracking.record_argv is off)" in summary


def test_publish_with_converted_format(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)
    with track_convert(pkl, workspace_dir=producer_repo):
        onnx_dir = producer_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted bytes")

    result = publish(pkl, workspace_dir=producer_repo,
                      format_dirs=[producer_repo / "models" / "onnx"])

    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    # Resolve from the registered version's source, not a fixed "models/" path:
    # each publish uploads to its own subdirectory (D#59).
    mv = mlflow.tracking.MlflowClient().get_model_version(
        result["model_name"], result["version"]
    )
    manifest_path = mlflow.artifacts.download_artifacts(f"{mv.source}/manifest.yaml")
    import yaml
    manifest = yaml.safe_load(open(manifest_path).read())
    assert "onnx" in manifest["formats"]
    assert manifest["conversions"]
    assert mv.tags["formats"] == "pkl,onnx"


def test_publish_same_run_twice_refuses_instead_of_reuploading(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    result = publish(pkl, workspace_dir=producer_repo)
    assert result["run_id"] == run_id

    # Same pkl content, no retrain in between (e.g. only unrelated code changed
    # and `mldna push` was re-run) - resolve_run matches the same already-
    # published run by content hash, which must be refused up front rather
    # than attempting to re-upload artifacts the registry already has.
    with pytest.raises(RunGateError, match="already published this exact file"):
        publish(pkl, workspace_dir=producer_repo)


def test_publish_untracked_pkl_refuses(git_repo):
    pkl = git_repo / "models" / "untracked.pkl"
    pkl.parent.mkdir(exist_ok=True)
    pkl.write_bytes(b"never tracked")
    with track_run(workspace_dir=git_repo):
        pass

    with pytest.raises(ResolveError):
        publish(pkl, workspace_dir=git_repo)


def test_publish_dirty_run_refuses(git_repo):
    (git_repo / "README.md").write_text("dirty\n")
    with track_run(workspace_dir=git_repo, allow_dirty=True) as run:
        pkl = git_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained on a dirty tree")

    # D#37 moved this refusal earlier: resolution drops dirty matches, so the
    # candidate set empties before rungates' dirty gate is ever reached.
    with pytest.raises(ResolveError, match="dirty=true"):
        publish(pkl, workspace_dir=git_repo)


def test_publish_unrecorded_conversion_refuses(producer_repo):
    pkl, _ = _tracked_pkl(producer_repo)
    fake_dir = producer_repo / "models" / "onnx"
    fake_dir.mkdir()
    (fake_dir / "model.onnx").write_bytes(b"never went through track_convert")

    with pytest.raises(ConversionError):
        publish(pkl, workspace_dir=producer_repo, format_dirs=[fake_dir])


def test_publish_aborts_when_confirm_declines(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    with pytest.raises(PublishError, match="Aborted"):
        publish(pkl, workspace_dir=producer_repo, confirm=lambda summary: False)

    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    run = mlflow.get_run(run_id)
    assert "registry_version" not in run.data.tags


def test_publish_enforces_require_formats_from_mldna_yaml(producer_repo):
    config_path = producer_repo / "mldna.yaml"
    import yaml
    data = yaml.safe_load(config_path.read_text())
    data["publish"] = {"require_formats": ["openvino"]}
    config_path.write_text(yaml.dump(data))

    pkl, run_id = _tracked_pkl(producer_repo)

    from mldna.core.structure import StructureError
    with pytest.raises(StructureError, match="openvino"):
        publish(pkl, workspace_dir=producer_repo)


def test_publish_missing_local_commit_refuses(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)
    client = mlflow.tracking.MlflowClient()
    client.set_tag(run_id, "git_commit", "f" * 40)

    with pytest.raises(CommitReachabilityError):
        publish(pkl, workspace_dir=producer_repo)


def test_publish_nonexistent_pkl_refuses(producer_repo):
    missing = producer_repo / "models" / "never-trained.pkl"

    with pytest.raises(PublishError, match="does not exist"):
        publish(missing, workspace_dir=producer_repo)


def test_publish_directory_path_refuses(producer_repo):
    models_dir = producer_repo / "models"
    models_dir.mkdir(exist_ok=True)

    with pytest.raises(PublishError, match="is a directory"):
        publish(models_dir, workspace_dir=producer_repo)


def test_publish_refuses_when_recorded_conversions_cannot_be_read(producer_repo, monkeypatch):
    pkl, _ = _tracked_pkl(producer_repo)
    onnx_dir = producer_repo / "models" / "onnx"
    with track_convert(pkl, workspace_dir=producer_repo):
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted bytes")

    def _boom(*args, **kwargs):
        raise RuntimeError("INTERNAL_ERROR: 500")

    monkeypatch.setattr(mlflow.artifacts, "download_artifacts", _boom)

    # A failed read of a conversions.json the listing says IS there must refuse,
    # not be reported as "this run has no recorded conversions" (invariant 6).
    with pytest.raises(PublishError, match="could not be read"):
        publish(pkl, workspace_dir=producer_repo, format_dirs=[onnx_dir])


def test_publish_summary_names_pending_branch_push(producer_repo):
    _commit_scaffold(producer_repo)
    core_push(producer_repo)

    (producer_repo / "README.md").write_text("second commit\n")
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=producer_repo, check=True)
    # Commit now exists locally but was never pushed to origin.

    with track_run(workspace_dir=producer_repo) as run:
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")

    captured = {}

    def confirm(summary):
        captured["summary"] = summary
        return True

    publish(pkl, workspace_dir=producer_repo, confirm=confirm)

    assert "Will push" in captured["summary"]
    assert "origin" in captured["summary"]
    sha = git.get_commit_sha(producer_repo)[:8]
    assert f"(commit {sha} not on the remote branch yet)" in captured["summary"]


def test_publish_summary_says_branch_not_on_remote_when_never_pushed(producer_repo):
    _commit_scaffold(producer_repo)

    with track_run(workspace_dir=producer_repo):
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")

    captured = {}

    def confirm(summary):
        captured["summary"] = summary
        return True

    publish(pkl, workspace_dir=producer_repo, confirm=confirm)

    assert "(branch not on the remote yet)" in captured["summary"]


def test_publish_refuses_without_dvc_remote_configured(producer_repo):
    subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "remove", "local-remote"],
        cwd=producer_repo, check=True, capture_output=True,
    )
    pkl, _ = _tracked_pkl(producer_repo)

    with pytest.raises(PublishError, match="No DVC remote is configured"):
        publish(pkl, workspace_dir=producer_repo)


def _dvc_config(url, remote="local-remote"):
    return f"[core]\n    remote = {remote}\n['remote \"{remote}\"']\n    url = {url}\n"


def _set_dvc_config(repo, text, commit):
    (repo / ".dvc" / "config").write_text(text)
    if commit:
        _commit_scaffold(repo)


def test_data_check_no_remote_fix_text_says_publish_uploads(producer_repo):
    _set_dvc_config(producer_repo, "[core]\n", commit=True)

    with pytest.raises(PublishError, match="publish uploads the data itself"):
        _data_needs_push(producer_repo)


def test_data_check_refuses_remote_only_in_uncommitted_config(producer_repo):
    _set_dvc_config(producer_repo, "[core]\n", commit=True)
    _set_dvc_config(producer_repo, _dvc_config("/tmp/store"), commit=False)

    with pytest.raises(PublishError, match="not in the committed .dvc/config"):
        _data_needs_push(producer_repo)


def test_data_check_refuses_remote_only_in_config_local(producer_repo):
    _set_dvc_config(producer_repo, "[core]\n", commit=True)
    (producer_repo / ".dvc" / "config.local").write_text(_dvc_config("/tmp/store"))

    with pytest.raises(PublishError, match="not in the committed .dvc/config"):
        _data_needs_push(producer_repo)


def test_data_check_refuses_url_mismatch_without_leaking_credentials(producer_repo):
    _set_dvc_config(producer_repo, _dvc_config("/tmp/store"), commit=True)
    _set_dvc_config(producer_repo, _dvc_config("https://user:hunter2@example.com/store"), commit=False)

    with pytest.raises(PublishError, match="not in the committed .dvc/config") as e:
        _data_needs_push(producer_repo)
    assert "hunter2" not in str(e.value)
    assert "example.com/store" in str(e.value)


def test_data_check_passes_with_committed_remote_and_local_credentials(producer_repo, monkeypatch):
    _set_dvc_config(producer_repo, _dvc_config("/tmp/store"), commit=True)
    (producer_repo / ".dvc" / "config.local").write_text(
        "['remote \"local-remote\"']\n    connection_string = hunter2\n"
    )
    monkeypatch.setattr(dvc_module, "cloud_status", lambda ws: {})

    assert _data_needs_push(producer_repo) is False


def test_publish_refuses_when_data_push_fails(producer_repo, monkeypatch):
    pkl, _ = _tracked_pkl(producer_repo)  # data already in sync via core_push
    real_run = dvc_module.run

    def fake_run(args, cwd, check=True):
        if args[:1] == ["push"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="remote unreachable")
        return real_run(args, cwd=cwd, check=check)

    monkeypatch.setattr(dvc_module, "run", fake_run)
    monkeypatch.setattr(dvc_module, "cloud_status", lambda ws: {"dataset.dvc": "new"})

    with pytest.raises(PublishError, match="dvc push failed"):
        publish(pkl, workspace_dir=producer_repo, confirm=lambda s: True)


def test_publish_pushes_data_when_absent_from_remote(producer_repo):
    _commit_scaffold(producer_repo)
    core_push(producer_repo)

    # New data arrives after the initial push: `dvc add` keeps the local
    # pointer honest (so track_run's clean-tree gate is unaffected), but
    # nothing has sent it to the remote yet.
    (producer_repo / "dataset" / "extra.bin").write_bytes(b"more data")
    subprocess.run([sys.executable, "-m", "dvc", "add", "dataset", "-q"],
                   cwd=producer_repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "more data"], cwd=producer_repo, check=True)

    with track_run(workspace_dir=producer_repo) as run:
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained on newer data")

    captured = {}

    def confirm(summary):
        captured["summary"] = summary
        return True

    publish(pkl, workspace_dir=producer_repo, confirm=confirm)

    assert "dataset" in captured["summary"].lower()
    assert dvc_module.cloud_status(producer_repo) == {}


# --- issue #8: concurrent publishes of one run ---

def test_publish_uploads_each_attempt_to_a_distinct_artifact_path(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(producer_repo)
    real_log_artifacts = mlflow.tracking.MlflowClient.log_artifacts
    paths = []

    def spy(self, run_id_arg, local_dir, artifact_path=None):
        paths.append(artifact_path)
        return real_log_artifacts(self, run_id_arg, local_dir, artifact_path=artifact_path)

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_artifacts", spy)

    publish(pkl, workspace_dir=producer_repo)

    # A second publish attempt on the same run is normally refused by the
    # already-published gate (D#33) - clearing the tag simulates the race
    # window where that gate ran for both attempts before either had
    # registered, which is exactly the scenario the unique path protects.
    client = mlflow.tracking.MlflowClient()
    client.delete_tag(run_id, "registry_version")
    client.delete_tag(run_id, f"published__{sha256_file(pkl)[:16]}")
    publish(pkl, workspace_dir=producer_repo)

    assert len(paths) == 2
    assert paths[0] != paths[1]
    assert all(p.startswith("models/") for p in paths)


def test_publish_surfaces_concurrent_registration_of_the_same_run(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(producer_repo)
    real_register_model = mlflow.register_model

    def fake_register_model(source, name):
        registered = real_register_model(source, name)
        # Simulate a second, concurrent publish of this file winning the
        # registration race and writing its own tag in the gap before this
        # call's post-register re-read.
        mlflow.tracking.MlflowClient().set_tag(run_id, f"published__{sha256_file(pkl)[:16]}", "999")
        captured["version"] = registered.version
        return registered

    captured = {}
    monkeypatch.setattr(mlflow, "register_model", fake_register_model)

    with pytest.raises(PublishError, match="concurrent publish") as excinfo:
        publish(pkl, workspace_dir=producer_repo)

    # D#76: the loser tags its own version so fetch/pull refuse it.
    assert "tagged abandoned=true" in str(excinfo.value)
    mv = mlflow.tracking.MlflowClient().get_model_version("project-test-model", captured["version"])
    assert mv.tags["abandoned"] == "true"
    assert mv.tags["abandoned_reason"] == "lost concurrent publish to v999"
    assert "registry_version" not in mlflow.get_run(run_id).data.tags


# --- issue #1c: publish must verify the run's own data, not just current data ---

def test_publish_refuses_when_dataset_changed_since_the_run_trained(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    (producer_repo / "dataset" / "extra.bin").write_bytes(b"new data, never seen by the run")
    subprocess.run([sys.executable, "-m", "dvc", "add", "dataset", "-q"],
                   cwd=producer_repo, check=True)

    sha = git.get_commit_sha(producer_repo)
    with pytest.raises(PublishError, match="dataset was changed and re-added") as exc:
        publish(pkl, workspace_dir=producer_repo)
    assert f"Fix: `git checkout {sha}` (the commit this run trained at) then `dvc pull`" in str(exc.value)

    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    run = mlflow.get_run(run_id)
    assert "registry_version" not in run.data.tags


def test_publish_refuses_when_one_of_several_outs_changed_since_the_run(producer_repo):
    # D#63: the comparison is per out, so a second out moving on is caught even
    # though the first still matches exactly what the run cites.
    second = producer_repo / "dataset2"
    second.mkdir()
    (second / "rows.csv").write_text("a,b\n1,2\n")
    subprocess.run([sys.executable, "-m", "dvc", "add", "dataset2", "-q"],
                   cwd=producer_repo, check=True)
    pkl, run_id = _tracked_pkl(producer_repo)

    (second / "extra.bin").write_bytes(b"new data, never seen by the run")
    subprocess.run([sys.executable, "-m", "dvc", "add", "dataset2", "-q"],
                   cwd=producer_repo, check=True)

    with pytest.raises(PublishError, match="dataset was changed and re-added") as excinfo:
        publish(pkl, workspace_dir=producer_repo)
    assert "dataset2:" in str(excinfo.value)

    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    assert "registry_version" not in mlflow.get_run(run_id).data.tags


# --- issue #33: an oversized upload must explain itself, not traceback ---

def test_publish_oversized_upload_reports_size_limit(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(producer_repo)

    def _boom(self, run_id_arg, local_dir, artifact_path=None):
        raise RuntimeError("413 Request Entity Too Large")

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_artifacts", _boom)

    with pytest.raises(PublishError, match="request-size limit"):
        publish(pkl, workspace_dir=producer_repo)


def test_publish_unrelated_upload_failure_is_not_reported_as_size(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(producer_repo)

    def _boom(self, run_id_arg, local_dir, artifact_path=None):
        raise ValueError("bad artifact path")

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_artifacts", _boom)

    with pytest.raises(ValueError, match="bad artifact path"):
        publish(pkl, workspace_dir=producer_repo)


def test_publish_dropped_connection_says_may_be_size_limit(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(producer_repo)

    def _boom(self, run_id_arg, local_dir, artifact_path=None):
        raise ConnectionResetError("connection reset by peer")

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_artifacts", _boom)

    with pytest.raises(PublishError, match="often means") as exc:
        publish(pkl, workspace_dir=producer_repo)
    assert "max_request_body_size" in str(exc.value)
    assert "can also be a network problem" in str(exc.value)


def test_publish_chained_connection_error_is_recognised(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(producer_repo)

    def _boom(self, run_id_arg, local_dir, artifact_path=None):
        try:
            raise requests.exceptions.ConnectionError("Connection aborted")
        except requests.exceptions.ConnectionError as inner:
            raise mlflow.exceptions.MlflowException("upload failed") from inner

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_artifacts", _boom)

    with pytest.raises(PublishError, match="can also be a network problem"):
        publish(pkl, workspace_dir=producer_repo)


def test_publish_refused_connection_is_not_reported_as_size_limit(producer_repo, monkeypatch):
    pkl, run_id = _tracked_pkl(producer_repo)

    def _boom(self, run_id_arg, local_dir, artifact_path=None):
        refused = NewConnectionError(None, "connection refused")
        raise requests.exceptions.ConnectionError("refused") from refused

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "log_artifacts", _boom)

    with pytest.raises(requests.exceptions.ConnectionError):
        publish(pkl, workspace_dir=producer_repo)


# --- issue #44: no mldna.yaml here, but one in a parent ---

def test_publish_from_subfolder_names_the_parent_project(producer_repo):
    sub = producer_repo / "notebooks"
    sub.mkdir()
    pkl = sub / "model.pkl"
    pkl.write_bytes(b"x")

    with pytest.raises(PublishError, match="but there is one at") as exc:
        publish(pkl, workspace_dir=sub)
    assert "project.model_name" not in str(exc.value)
    assert "mldna init" not in str(exc.value)


def test_publish_unreachable_remote_refuses_cleanly(producer_repo, tmp_path):
    pkl, run_id = _tracked_pkl(producer_repo)
    subprocess.run(["git", "remote", "set-url", "origin", str(tmp_path / "gone.git")],
                   cwd=producer_repo, check=True)

    with pytest.raises(CommitReachabilityError, match="git ls-remote origin"):
        publish(pkl, workspace_dir=producer_repo)


def _two_clean_runs_identical_bytes(repo):
    # D#64: fixed payload, two commits -> same hash, differing lineage.
    _commit_scaffold(repo)
    with track_run(workspace_dir=repo):
        pkl = repo / "models" / "model_a.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"deterministic model bytes")
    (repo / "README.md").write_text("docs only\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "docs only"], cwd=repo, check=True)
    core_push(repo)
    with track_run(workspace_dir=repo) as run_b:
        pkl = repo / "models" / "model_b.pkl"
        pkl.write_bytes(b"deterministic model bytes")
    commit_b = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return pkl, run_b.run_id, commit_b


def test_publish_differing_lineage_refuses_naming_commit(producer_repo):
    from mldna.core.resolve import AmbiguousRunError
    pkl, _, _ = _two_clean_runs_identical_bytes(producer_repo)

    with pytest.raises(AmbiguousRunError, match="--commit <sha>"):
        publish(pkl, workspace_dir=producer_repo, confirm=lambda s: True)


def test_publish_commit_selects_and_labels_summary_and_manifest(producer_repo):
    import yaml
    pkl, run_b, commit_b = _two_clean_runs_identical_bytes(producer_repo)
    captured = {}

    def confirm(summary):
        captured["summary"] = summary
        return True

    result = publish(pkl, workspace_dir=producer_repo, confirm=confirm, commit=commit_b)

    assert result["run_id"] == run_b
    assert "HUMAN-SELECTED" in captured["summary"]
    assert f"--commit {commit_b}" in captured["summary"]

    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    mv = mlflow.tracking.MlflowClient().get_model_version(result["model_name"], result["version"])
    manifest = yaml.safe_load(open(mlflow.artifacts.download_artifacts(f"{mv.source}/manifest.yaml")).read())
    assert "HUMAN-SELECTED" in manifest["resolution_note"]
    assert manifest["source_run_id"] == run_b


def _set_model_id(repo, value):
    import yaml
    path = repo / "mldna.yaml"
    data = yaml.safe_load(path.read_text())
    if value is None:
        data["project"].pop("model_id", None)
    else:
        data["project"]["model_id"] = value
    path.write_text(yaml.dump(data))


def _registered_model_id(repo, model_name):
    mlflow.set_tracking_uri(f"file:{repo / 'mlruns'}")
    return mlflow.tracking.MlflowClient().get_registered_model(model_name).tags.get("mldna_model_id")


def test_publish_without_model_id_refuses(producer_repo):
    _set_model_id(producer_repo, None)
    pkl, _ = _tracked_pkl(producer_repo)

    with pytest.raises(PublishError, match="mldna id"):
        publish(pkl, workspace_dir=producer_repo)


def test_publish_refuses_registry_name_owned_by_another_id(producer_repo):
    pkl, _ = _tracked_pkl(producer_repo)
    first = publish(pkl, workspace_dir=producer_repo)
    _set_model_id(producer_repo, "99999999-2222-4333-8444-555555555555")
    _commit_scaffold(producer_repo)
    with track_run(workspace_dir=producer_repo):
        pkl.write_bytes(b"second model")

    with pytest.raises(PublishError, match="belongs to another project") as excinfo:
        publish(pkl, workspace_dir=producer_repo)
    # D#77: a lost or regenerated id is the likelier cause - offer the restore first.
    assert (f"restore it: set project.model_id: {first['model_id']} in mldna.yaml "
            "and commit. Otherwise change project.namespace") in str(excinfo.value)


def test_publish_stamps_legacy_registered_model_without_id(producer_repo):
    pkl, _ = _tracked_pkl(producer_repo)
    result = publish(pkl, workspace_dir=producer_repo)
    mlflow.set_tracking_uri(f"file:{producer_repo / 'mlruns'}")
    client = mlflow.tracking.MlflowClient()
    client.delete_registered_model_tag(result["model_name"], "mldna_model_id")
    client.delete_registered_model_tag(result["model_name"], "mldna_origin")
    with track_run(workspace_dir=producer_repo):
        pkl.write_bytes(b"second model")

    result = publish(pkl, workspace_dir=producer_repo)

    assert _registered_model_id(producer_repo, result["model_name"]) == result["model_id"]
    tags = mlflow.tracking.MlflowClient().get_registered_model(result["model_name"]).tags
    assert tags["mldna_origin"] == git.origin_url(producer_repo)


# --- D#77: a copied mldna.yaml carries its model_id; the origin tells copies apart ---

ORIGINAL = "https://example.com/original/repo"


def _retrain(repo, files):
    with track_run(workspace_dir=repo):
        for name, content in files.items():
            (repo / "models" / name).write_bytes(content)
    return {name: repo / "models" / name for name in files}


def test_publish_differing_origin_refuses_without_a_person(published):
    repo, _, result = published
    mlflow.tracking.MlflowClient().set_registered_model_tag(
        result["model_name"], "mldna_origin", ORIGINAL)
    pkl = _retrain(repo, {"model.pkl": b"retrained in a copy"})["model.pkl"]

    with pytest.raises(PublishError, match="was published from") as excinfo:
        publish(pkl, workspace_dir=repo, confirm=lambda s: True)
    message = str(excinfo.value)
    assert f"{ORIGINAL}, but this repo's origin is {git.origin_url(repo)}" in message
    assert "Fix: if this is a different project, run `mldna id --new`" in message

    with pytest.raises(PublishError, match="Aborted"):
        publish(pkl, workspace_dir=repo, confirm=lambda s: True, confirm_origin=lambda w: False)


def test_publish_differing_origin_warns_then_records_the_confirmed_origin(published):
    repo, _, result = published
    client = mlflow.tracking.MlflowClient()
    client.set_registered_model_tag(result["model_name"], "mldna_origin", ORIGINAL)
    pkl = _retrain(repo, {"model.pkl": b"retrained after a move"})["model.pkl"]
    summaries, warnings = [], []

    publish(pkl, workspace_dir=repo, confirm=lambda s: summaries.append(s) or True,
            confirm_origin=lambda w: warnings.append(w) or True)

    assert f"WARNING: this registry model was published from {ORIGINAL}" in summaries[0]
    assert "`mldna id --new`" in warnings[0]
    assert client.get_registered_model(result["model_name"]).tags["mldna_origin"] == git.origin_url(repo)


def test_publish_same_remote_spelled_differently_needs_no_person(published):
    # A teammate's clone records the same remote as a relative path (or SSH vs
    # HTTPS): not a copied project, so CI must publish without a prompt.
    import os

    repo, _, result = published
    client = mlflow.tracking.MlflowClient()
    relative = os.path.relpath(git.origin_url(repo), repo)
    client.set_registered_model_tag(result["model_name"], "mldna_origin", relative)
    pkl = _retrain(repo, {"model.pkl": b"retrained by a teammate"})["model.pkl"]
    summaries = []

    publish(pkl, workspace_dir=repo, confirm=lambda s: summaries.append(s) or True)

    assert "WARNING" not in summaries[0]
    assert client.get_registered_model(result["model_name"]).tags["mldna_origin"] == relative


# --- D#78: one publish per model file per run ---

def test_one_run_publishes_each_model_file_once(published):
    repo, _, _ = published
    pkls = _retrain(repo, {"a.pkl": b"model a", "b.pkl": b"model b"})
    summaries = []

    first = publish(pkls["a.pkl"], workspace_dir=repo, confirm=lambda s: True)
    second = publish(pkls["b.pkl"], workspace_dir=repo,
                     confirm=lambda s: summaries.append(s) or True)

    assert second["run_id"] == first["run_id"]
    assert second["version"] != first["version"]
    assert "WARNING" not in summaries[0]  # same id, same origin: silent
    assert mlflow.get_run(first["run_id"]).data.tags["registry_version"] == str(second["version"])

    with pytest.raises(RunGateError, match="already published this exact file") as excinfo:
        publish(pkls["a.pkl"], workspace_dir=repo)
    assert f"as version {first['version']}" in str(excinfo.value)
