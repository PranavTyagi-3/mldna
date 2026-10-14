import os
import socket
import subprocess

import pytest

from mldna.core import blobauth, dvc
from mldna.core.lock import RunLock
from mldna.doctor import (
    check_blob_auth,
    check_git_remote,
    check_gitignore,
    check_large_tracked_files,
    check_run_lock,
    check_tools,
    check_tracking,
    parse_timeout,
    run_doctor,
)


def _write_filesystem_config(repo, path):
    (repo / "mldna.yaml").write_text(
        f"project:\n  model_name: test-model\nfilesystem:\n  path: {path}\n"
    )


def _commit(repo, *files):
    subprocess.run(["git", "add", *files], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "commit"], cwd=repo, check=True)


def test_check_tools_passes():
    assert check_tools().passed is True


def test_check_gitignore_passes_for_scaffolded_project(producer_repo):
    assert check_gitignore(producer_repo).passed is True


def test_check_gitignore_fix_is_runnable_in_either_repo_role(producer_repo):
    (producer_repo / ".gitignore").unlink()

    result = check_gitignore(producer_repo)

    assert result.passed is False
    # `mldna init` alone refuses without a model name; both roles need naming
    assert "mldna init <model-name>" in result.message
    assert "--consumer" in result.message


def test_check_gitignore_fails_when_entry_missing(producer_repo):
    (producer_repo / ".gitignore").write_text("models/\n")

    result = check_gitignore(producer_repo)

    assert result.passed is False
    assert "outputs/" in result.message
    assert "secrets, model binaries and local state" in result.message


def test_check_run_lock_passes_when_unlocked(producer_repo):
    assert check_run_lock(producer_repo).passed is True


def test_check_run_lock_passes_when_held_by_live_process(producer_repo):
    RunLock(producer_repo).acquire()

    assert check_run_lock(producer_repo).passed is True


def test_check_run_lock_passes_when_stale(producer_repo):
    import json
    lock_path = producer_repo / ".mldna" / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": 999999999, "run_id": None, "started_at": ""}))

    result = check_run_lock(producer_repo)

    assert result.passed is True
    assert "Stale lock" in result.message
    assert "reclaim" in result.message


def test_check_git_remote_passes_for_reachable_origin(producer_repo):
    _commit(producer_repo, "-A")

    result = check_git_remote(producer_repo)

    assert result.passed is True


def test_check_git_remote_fails_without_origin(git_repo):
    result = check_git_remote(git_repo)

    assert result.passed is False
    assert "No 'origin' remote" in result.message


def test_check_tracking_does_not_use_the_azure_name_for_a_plain_mlflow_server(
    producer_repo, monkeypatch
):
    # The bug this check was renamed for: a project with no Azure in it at all
    # was reporting its MLflow server under the Azure ML check name.
    from mldna.core.scaffold import write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")

    def boom(self, max_results=None):
        raise ConnectionError("refused")

    monkeypatch.setattr("mlflow.tracking.MlflowClient.search_experiments", boom)

    result = check_tracking(producer_repo)

    assert result.name == "mlflow_server"
    assert "aml" not in result.name
    assert "Azure" not in result.message


def test_check_tracking_fails_when_mlflow_server_unreachable(producer_repo, monkeypatch):
    from mldna.core.scaffold import write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")

    def boom(self, max_results=None):
        raise ConnectionError("refused")

    monkeypatch.setattr("mlflow.tracking.MlflowClient.search_experiments", boom)

    result = check_tracking(producer_repo)

    assert result.passed is False
    assert "MLflow tracking server" in result.message
    assert "refused" in result.message


def test_check_tracking_fails_when_filesystem_store_missing(producer_repo):
    _write_filesystem_config(producer_repo, "nope/mlruns")

    result = check_tracking(producer_repo)

    assert result.name == "tracking_store"
    assert result.passed is False
    assert "does not exist" in result.message


def test_check_tracking_fails_when_incomplete_azure_config(producer_repo):
    from mldna.core.scaffold import write_azure_config
    write_azure_config(producer_repo, "sub", "rg", None)

    result = check_tracking(producer_repo)

    assert result.name == "aml_auth"
    assert result.passed is False
    assert "workspace" in result.message


def test_check_tracking_fails_when_both_mlflow_and_azure_configured(producer_repo):
    from mldna.core.scaffold import write_azure_config, write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")
    write_azure_config(producer_repo, "sub", "rg", "ws")

    result = check_tracking(producer_repo)

    assert result.passed is False
    assert "only one tracking" in result.message


def test_check_tracking_passes_with_reachable_mlflow_server(producer_repo, monkeypatch):
    from mldna.core.scaffold import write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")
    monkeypatch.setattr(
        "mlflow.tracking.MlflowClient.search_experiments",
        lambda self, max_results=None: [],
    )

    result = check_tracking(producer_repo)

    assert result.name == "mlflow_server"
    assert result.passed is True
    assert "remote mlflow server" in result.message


def test_check_tracking_reports_the_local_store(producer_repo):
    result = check_tracking(producer_repo)

    assert result.name == "tracking_store"
    assert result.passed is True
    assert "tracking mode: local" in result.message
    assert "mlruns" in result.message


def test_check_tracking_reports_a_filesystem_store(producer_repo):
    store = producer_repo / "shared-mlruns"
    store.mkdir()
    _write_filesystem_config(producer_repo, "shared-mlruns")

    result = check_tracking(producer_repo)

    assert result.name == "tracking_store"
    assert result.passed is True
    assert "filesystem store" in result.message
    assert "shared-mlruns" in result.message


def test_check_tracking_reports_an_mlflow_file_uri_as_a_store(producer_repo):
    store = producer_repo / "shared-mlruns"
    store.mkdir()
    from mldna.core.scaffold import write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, f"file:{store}")

    result = check_tracking(producer_repo)

    assert result.name == "tracking_store"
    assert result.passed is True
    assert "filesystem store" in result.message


def test_check_tracking_keeps_the_azure_name_and_wording(producer_repo, monkeypatch):
    from mldna.core.scaffold import write_azure_config
    write_azure_config(producer_repo, "sub", "rg", "ws")
    monkeypatch.setattr(
        "mldna.doctor.resolve_and_set_tracking_uri",
        lambda _: ("azureml://fake", "AAD fallback"),
    )

    result = check_tracking(producer_repo)

    assert result.name == "aml_auth"
    assert result.passed is True
    assert result.message == "Reachable, tracking mode: AAD fallback"


def test_check_tracking_surfaces_the_azure_error_verbatim(producer_repo, monkeypatch):
    from mldna.core.scaffold import write_azure_config
    write_azure_config(producer_repo, "sub", "rg", "ws")

    def boom(_):
        raise RuntimeError("(ResourceNotFound) workspace not found")

    monkeypatch.setattr("mldna.doctor.resolve_and_set_tracking_uri", boom)

    result = check_tracking(producer_repo)

    assert result.name == "aml_auth"
    assert result.passed is False
    assert result.message == (
        "Could not reach Azure ML workspace: (ResourceNotFound) workspace not found"
    )


def test_check_blob_auth_reports_mode(producer_repo):
    result = check_blob_auth(producer_repo)

    assert result.passed is True
    assert "local-remote" in result.message


def test_check_blob_auth_passes_without_remote(git_repo):
    result = check_blob_auth(git_repo)

    assert result.passed is True
    assert "No DVC remote" in result.message


def test_run_doctor_returns_all_checks(producer_repo):
    _commit(producer_repo, "-A")

    results = run_doctor(producer_repo)

    assert len(results) == 7
    # producer_repo has no tracking section, so the tracking check is named for
    # the local store - "aml_auth" would name a plane this project never uses.
    assert {r.name for r in results} == {
        "tools", "gitignore", "run_lock", "git_remote",
        "large_tracked_files", "tracking_store", "blob_auth",
    }


def test_check_blob_auth_local_remote_is_not_azure_aad(producer_repo):
    result = check_blob_auth(producer_repo)

    assert result.passed is True
    assert "AAD" not in result.message
    assert "local/non-Azure" in result.message


def test_check_blob_auth_reports_aad_fallback_for_azure_remote(producer_repo, monkeypatch):
    monkeypatch.delenv("MLDNA_BLOB_CONNECTION_STRING", raising=False)
    dvc.run(["remote", "add", "-d", "azure-remote", "azure://fake-container/prefix"], cwd=producer_repo)

    result = check_blob_auth(producer_repo)

    assert result.passed is True
    assert result.message == "Remote 'azure-remote', auth mode: AAD fallback"


def test_check_blob_auth_fails_when_remote_url_unreadable(producer_repo, monkeypatch):
    monkeypatch.setattr(blobauth, "get_remote_url", lambda *_: None)

    result = check_blob_auth(producer_repo)

    assert result.passed is False
    assert "Fix: dvc remote modify" in result.message


@pytest.mark.parametrize("text", ["abc", "0", "-5", "1.5x", "", "10m"])
def test_parse_timeout_refuses_anything_but_whole_seconds(text):
    with pytest.raises(ValueError) as e:
        parse_timeout(text)

    assert "Fix: mldna doctor --timeout 60s" in str(e.value)


@pytest.mark.parametrize("text, expected", [("10", "10"), ("10s", "10"), ("100s", "100")])
def test_parse_timeout_accepts_seconds_with_optional_s(text, expected):
    assert parse_timeout(text) == expected


def _seen_env_during_probe(producer_repo, monkeypatch, timeout=None):
    from mldna.core.scaffold import write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")
    seen = {}

    def spy(self, max_results=None):
        seen["timeout"] = os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"]
        seen["retries"] = os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"]
        return []

    monkeypatch.setattr("mlflow.tracking.MlflowClient.search_experiments", spy)
    check_tracking(producer_repo, timeout)
    return seen


def test_probe_defaults_to_ten_seconds_and_no_retries(producer_repo, monkeypatch):
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)

    seen = _seen_env_during_probe(producer_repo, monkeypatch)

    assert seen == {"timeout": "10", "retries": "0"}


def test_ambient_timeout_overrides_the_default(producer_repo, monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "45")

    seen = _seen_env_during_probe(producer_repo, monkeypatch)

    assert seen["timeout"] == "45"


def test_timeout_flag_overrides_the_ambient_timeout(producer_repo, monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "45")

    seen = _seen_env_during_probe(producer_repo, monkeypatch, timeout="100")

    assert seen["timeout"] == "100"


def test_run_doctor_restores_the_environment(producer_repo, monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "45")
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", raising=False)

    run_doctor(producer_repo, "5")

    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "45"
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ


def test_environment_is_restored_when_the_probe_raises(producer_repo, monkeypatch):
    monkeypatch.delenv("MLFLOW_HTTP_REQUEST_TIMEOUT", raising=False)

    def boom(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("mldna.doctor.resolve_and_set_tracking_uri", boom)

    with pytest.raises(KeyboardInterrupt):
        run_doctor(producer_repo, "5")

    assert "MLFLOW_HTTP_REQUEST_TIMEOUT" not in os.environ
    assert "MLFLOW_HTTP_REQUEST_MAX_RETRIES" not in os.environ


def test_refused_connection_is_not_reported_as_no_response(producer_repo):
    from mldna.core.scaffold import write_mlflow_tracking_uri
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    write_mlflow_tracking_uri(producer_repo, f"http://127.0.0.1:{port}")

    result = check_tracking(producer_repo, "5")

    assert result.passed is False
    assert "no response within" not in result.message


def test_hung_server_is_reported_as_no_response_within_the_limit(producer_repo):
    from mldna.core.scaffold import write_mlflow_tracking_uri
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(5)
        write_mlflow_tracking_uri(producer_repo, f"http://127.0.0.1:{server.getsockname()[1]}")

        result = check_tracking(producer_repo, "1")

    assert result.name == "mlflow_server"
    assert result.passed is False
    assert "no response within 1 second." in result.message
    assert "mldna doctor --timeout 60s" in result.message


def test_azure_failure_never_claims_a_timeout(producer_repo, monkeypatch):
    from mldna.core.scaffold import write_azure_config
    write_azure_config(producer_repo, "sub", "rg", "ws")

    def boom(_):
        raise RuntimeError("(ResourceNotFound) workspace not found")

    monkeypatch.setattr("mldna.doctor.resolve_and_set_tracking_uri", boom)

    result = check_tracking(producer_repo, "5")

    assert result.message == (
        "Could not reach Azure ML workspace: (ResourceNotFound) workspace not found"
    )


def test_check_large_tracked_files_fails_outside_a_repo(tmp_path):
    result = check_large_tracked_files(tmp_path)

    assert result.passed is False
    assert "not checked" in result.message


def test_check_large_tracked_files_fails_on_a_tracked_large_file(producer_repo):
    (producer_repo / "big.csv").write_bytes(b"x" * (6 * 1024 * 1024))
    _commit(producer_repo, "-A")

    result = check_large_tracked_files(producer_repo)

    assert result.passed is False
    assert "big.csv" in result.message
    assert "git rm --cached" in result.message
    assert "history is not examined" in result.message.lower()


def test_check_large_tracked_files_caps_the_list_at_ten(producer_repo):
    for i in range(12):
        (producer_repo / f"m{i}.pkl").write_bytes(b"x" * 200 * 1024)
    _commit(producer_repo, "-A")

    result = check_large_tracked_files(producer_repo)

    assert result.passed is False
    assert "and 2 more" in result.message


def test_check_large_tracked_files_passes_on_a_clean_repo_and_states_its_bound(producer_repo):
    _commit(producer_repo, "-A")

    result = check_large_tracked_files(producer_repo)

    assert result.passed is True
    assert "tracked files checked; history not examined" in result.message
