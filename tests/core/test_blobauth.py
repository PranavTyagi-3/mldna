from mldna.core import blobauth, dvc
from mldna.core.scaffold import scaffold_producer


def _config_local(repo):
    return (repo / ".dvc" / "config.local").read_text() if (repo / ".dvc" / "config.local").exists() else ""


def _azure_remote_repo(tmp_path):
    scaffold_producer(tmp_path, "test-model")
    dvc.run(["remote", "add", "-d", "azure-remote", "azure://fake-container/prefix"], cwd=tmp_path)
    return tmp_path


def test_get_default_remote(producer_repo):
    assert blobauth.get_default_remote(producer_repo) == "local-remote"


def test_set_and_clear_connection_string(tmp_path):
    repo = _azure_remote_repo(tmp_path)

    blobauth.set_connection_string(repo, "azure-remote", "DefaultEndpointsProtocol=https;fake=1")
    assert "fake=1" in _config_local(repo)

    blobauth.clear(repo, "azure-remote")
    assert "fake=1" not in _config_local(repo)


def test_resolve_env_override_sets_connection_string(tmp_path, monkeypatch):
    repo = _azure_remote_repo(tmp_path)
    monkeypatch.setenv("MLDNA_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;fromenv=1")

    used = blobauth.resolve_env_override(repo, "azure-remote")

    assert used is True
    assert "fromenv=1" in _config_local(repo)


def test_resolve_env_override_noop_without_env_var(tmp_path, monkeypatch):
    repo = _azure_remote_repo(tmp_path)
    monkeypatch.delenv("MLDNA_BLOB_CONNECTION_STRING", raising=False)

    used = blobauth.resolve_env_override(repo, "azure-remote")

    assert used is False
    assert "fromenv" not in _config_local(repo)


def test_resolve_mode_reports_aad_fallback_for_azure_remote(tmp_path, monkeypatch):
    repo = _azure_remote_repo(tmp_path)
    monkeypatch.delenv("MLDNA_BLOB_CONNECTION_STRING", raising=False)

    assert blobauth.resolve_mode(repo, "azure-remote") == "AAD fallback"


def test_resolve_mode_prefers_env_var(tmp_path, monkeypatch):
    repo = _azure_remote_repo(tmp_path)
    blobauth.set_connection_string(repo, "azure-remote", "DefaultEndpointsProtocol=https;fromfile=1")
    monkeypatch.setenv("MLDNA_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;fromenv=1")

    # The label names the variable: "env var" alone leaves a user whose auth is
    # being overridden with no way to find the override.
    assert blobauth.resolve_mode(repo, "azure-remote") == "env var MLDNA_BLOB_CONNECTION_STRING"
    assert blobauth.ENV_VAR in blobauth.resolve_mode(repo, "azure-remote")


def test_resolve_mode_uses_config_local_without_env_var(tmp_path, monkeypatch):
    repo = _azure_remote_repo(tmp_path)
    blobauth.set_connection_string(repo, "azure-remote", "DefaultEndpointsProtocol=https;fromfile=1")
    monkeypatch.delenv("MLDNA_BLOB_CONNECTION_STRING", raising=False)

    assert blobauth.resolve_mode(repo, "azure-remote") == ".dvc/config.local"


def test_get_remote_url_for_local_remote(producer_repo):
    url = blobauth.get_remote_url(producer_repo, "local-remote")

    assert url
    assert not blobauth.is_azure_remote(url)


def test_get_remote_url_for_azure_remote(tmp_path):
    repo = _azure_remote_repo(tmp_path)

    url = blobauth.get_remote_url(repo, "azure-remote")

    assert url == "azure://fake-container/prefix"
    assert blobauth.is_azure_remote(url) is True


def test_get_remote_url_none_for_unknown_remote(producer_repo):
    assert blobauth.get_remote_url(producer_repo, "no-such-remote") is None
