import os

from mlops.core import dvc

ENV_VAR = "MLOPS_BLOB_CONNECTION_STRING"


def get_default_remote(workspace_dir):
    result = dvc.run(["config", "core.remote"], cwd=workspace_dir, check=False)
    name = result.stdout.strip()
    return name or None


def set_connection_string(workspace_dir, remote_name, connection_string):
    dvc.run(
        ["remote", "modify", "--local", remote_name, "connection_string", connection_string],
        cwd=workspace_dir,
    )


def clear(workspace_dir, remote_name):
    dvc.run(
        ["remote", "modify", "--local", remote_name, "--unset", "connection_string"],
        cwd=workspace_dir, check=False,
    )


def resolve_env_override(workspace_dir, remote_name):
    connection_string = os.environ.get(ENV_VAR)
    if connection_string:
        set_connection_string(workspace_dir, remote_name, connection_string)
        return True
    return False
