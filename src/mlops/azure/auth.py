import json
import os
from pathlib import Path

from azure.identity import ClientSecretCredential, DefaultAzureCredential

AUTH_FILE = "aml_auth.json"


def _auth_file_path(workspace_dir):
    return Path(workspace_dir) / ".mlops" / AUTH_FILE


def write_local_auth(workspace_dir, tenant_id, client_id, client_secret):
    path = _auth_file_path(workspace_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "tenant_id": tenant_id, "client_id": client_id, "client_secret": client_secret,
    }))


def clear_local_auth(workspace_dir):
    _auth_file_path(workspace_dir).unlink(missing_ok=True)


def read_local_auth(workspace_dir):
    path = _auth_file_path(workspace_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def resolve_aml_credential(workspace_dir):
    tenant_id = os.environ.get("AZURE_TENANT_ID")
    client_id = os.environ.get("AZURE_CLIENT_ID")
    client_secret = os.environ.get("AZURE_CLIENT_SECRET")
    if tenant_id and client_id and client_secret:
        return ClientSecretCredential(tenant_id, client_id, client_secret), "env var (service principal)"

    creds = read_local_auth(workspace_dir)
    if creds:
        # azureml-mlflow builds its own DefaultAzureCredential() internally for
        # every real API call beyond this function's own workspace lookup - it
        # never sees the ClientSecretCredential object below. Setting the same
        # env vars its EnvironmentCredential leg reads is what actually makes
        # this leg authenticate real run/log/publish calls, not just resolve a
        # tracking URI.
        os.environ["AZURE_TENANT_ID"] = creds["tenant_id"]
        os.environ["AZURE_CLIENT_ID"] = creds["client_id"]
        os.environ["AZURE_CLIENT_SECRET"] = creds["client_secret"]
        credential = ClientSecretCredential(
            creds["tenant_id"], creds["client_id"], creds["client_secret"]
        )
        return credential, f".mlops/{AUTH_FILE} (service principal)"

    return DefaultAzureCredential(), "DefaultAzureCredential (az login / managed identity)"
