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
