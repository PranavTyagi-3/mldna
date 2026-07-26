from azure.identity import ClientSecretCredential, DefaultAzureCredential

from mldna.azure.auth import (
    clear_local_auth,
    read_local_auth,
    resolve_aml_credential,
    write_local_auth,
)

ENV_VARS = ("AZURE_TENANT_ID", "AZURE_CLIENT_ID", "AZURE_CLIENT_SECRET")


def _clear_env(monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def test_env_var_leg_wins_when_set(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("AZURE_TENANT_ID", "t")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s")

    credential, mode = resolve_aml_credential(tmp_path)

    assert isinstance(credential, ClientSecretCredential)
    assert "env var" in mode


def test_local_file_leg_used_when_no_env_vars(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    write_local_auth(tmp_path, "t", "c", "s")

    credential, mode = resolve_aml_credential(tmp_path)

    assert isinstance(credential, ClientSecretCredential)
    assert "aml_auth.json" in mode


def test_default_azure_credential_fallback(tmp_path, monkeypatch):
    _clear_env(monkeypatch)

    credential, mode = resolve_aml_credential(tmp_path)

    assert isinstance(credential, DefaultAzureCredential)
    assert "DefaultAzureCredential" in mode


def test_env_var_takes_precedence_over_local_file(tmp_path, monkeypatch):
    _clear_env(monkeypatch)
    monkeypatch.setenv("AZURE_TENANT_ID", "t")
    monkeypatch.setenv("AZURE_CLIENT_ID", "c")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "s")
    write_local_auth(tmp_path, "other-t", "other-c", "other-s")

    _, mode = resolve_aml_credential(tmp_path)

    assert "env var" in mode


def test_local_file_leg_sets_env_vars_for_real_api_calls(tmp_path, monkeypatch):
    # azureml-mlflow builds its own DefaultAzureCredential() for every real
    # run/log/publish call - it never sees the ClientSecretCredential object
    # this function returns. Only setting these env vars makes the file leg
    # actually authenticate, not just resolve the tracking URI.
    _clear_env(monkeypatch)
    write_local_auth(tmp_path, "t", "c", "s")

    resolve_aml_credential(tmp_path)

    import os
    assert os.environ["AZURE_TENANT_ID"] == "t"
    assert os.environ["AZURE_CLIENT_ID"] == "c"
    assert os.environ["AZURE_CLIENT_SECRET"] == "s"


def test_write_and_clear_local_auth(tmp_path):
    write_local_auth(tmp_path, "t", "c", "s")
    assert read_local_auth(tmp_path) == {"tenant_id": "t", "client_id": "c", "client_secret": "s"}

    clear_local_auth(tmp_path)

    assert read_local_auth(tmp_path) is None
