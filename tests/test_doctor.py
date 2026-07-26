import subprocess

from mldna.core.lock import RunLock
from mldna.doctor import (
    check_aml_auth,
    check_blob_auth,
    check_committed_secrets,
    check_git_remote,
    check_gitignore,
    check_run_lock,
    check_tools,
    run_doctor,
)


def _commit(repo, *files):
    subprocess.run(["git", "add", *files], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "commit"], cwd=repo, check=True)


def test_check_tools_passes():
    assert check_tools().passed is True


def test_check_gitignore_passes_for_scaffolded_project(producer_repo):
    assert check_gitignore(producer_repo).passed is True


def test_check_gitignore_fails_when_entry_missing(producer_repo):
    (producer_repo / ".gitignore").write_text("models/\n")

    result = check_gitignore(producer_repo)

    assert result.passed is False
    assert "outputs/" in result.message


def test_check_run_lock_passes_when_unlocked(producer_repo):
    assert check_run_lock(producer_repo).passed is True


def test_check_run_lock_passes_when_held_by_live_process(producer_repo):
    RunLock(producer_repo).acquire()

    assert check_run_lock(producer_repo).passed is True


def test_check_run_lock_fails_when_stale(producer_repo):
    import json
    lock_path = producer_repo / ".mldna" / "run.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps({"pid": 999999999, "run_id": None, "started_at": ""}))

    result = check_run_lock(producer_repo)

    assert result.passed is False
    assert "Stale lock" in result.message


def test_check_git_remote_passes_for_reachable_origin(producer_repo):
    _commit(producer_repo, "-A")

    result = check_git_remote(producer_repo)

    assert result.passed is True


def test_check_git_remote_fails_without_origin(git_repo):
    result = check_git_remote(git_repo)

    assert result.passed is False
    assert "No 'origin' remote" in result.message


def test_check_committed_secrets_passes_when_clean(producer_repo):
    _commit(producer_repo, "-A")

    assert check_committed_secrets(producer_repo).passed is True


def test_check_committed_secrets_fails_on_account_key(producer_repo):
    (producer_repo / "notes.txt").write_text("AccountKey=abc123==\n")
    _commit(producer_repo, "-A")

    result = check_committed_secrets(producer_repo)

    assert result.passed is False
    assert "notes.txt" in result.message


def test_check_aml_auth_passes_when_not_configured(producer_repo):
    result = check_aml_auth(producer_repo)

    assert result.passed is True
    assert "local tracking mode" in result.message


def test_check_aml_auth_fails_when_incomplete(producer_repo):
    from mldna.core.scaffold import write_azure_config
    write_azure_config(producer_repo, "sub", "rg", None)

    result = check_aml_auth(producer_repo)

    assert result.passed is False
    assert "workspace" in result.message


def test_check_aml_auth_passes_with_reachable_mlflow_server(producer_repo, monkeypatch):
    from mldna.core.scaffold import write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")
    monkeypatch.setattr(
        "mlflow.tracking.MlflowClient.search_experiments",
        lambda self, max_results=None: [],
    )

    result = check_aml_auth(producer_repo)

    assert result.passed is True
    assert "remote mlflow server" in result.message


def test_check_aml_auth_fails_when_mlflow_server_unreachable(producer_repo, monkeypatch):
    from mldna.core.scaffold import write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")

    def boom(self, max_results=None):
        raise ConnectionError("refused")

    monkeypatch.setattr("mlflow.tracking.MlflowClient.search_experiments", boom)

    result = check_aml_auth(producer_repo)

    assert result.passed is False
    assert "MLflow tracking server" in result.message


def test_check_aml_auth_fails_when_both_mlflow_and_azure_configured(producer_repo):
    from mldna.core.scaffold import write_azure_config, write_mlflow_tracking_uri
    write_mlflow_tracking_uri(producer_repo, "http://my-mlflow-host:5000")
    write_azure_config(producer_repo, "sub", "rg", "ws")

    result = check_aml_auth(producer_repo)

    assert result.passed is False
    assert "only one tracking" in result.message


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
    assert {r.name for r in results} == {
        "tools", "gitignore", "run_lock", "git_remote",
        "committed_secrets", "aml_auth", "blob_auth",
    }
