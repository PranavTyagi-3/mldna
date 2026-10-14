import configparser
import os
import subprocess
from pathlib import Path

from mldna.core import dvc

ENV_VAR = "MLDNA_BLOB_CONNECTION_STRING"


def get_default_remote(workspace_dir):
    result = dvc.run(["config", "core.remote"], cwd=workspace_dir, check=False)
    name = result.stdout.strip()
    return name or None


def get_remote_url(workspace_dir, remote_name):
    result = dvc.run(["config", f"remote.{remote_name}.url"], cwd=workspace_dir, check=False)
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def get_committed_remote(workspace_dir):
    """(default remote name, its url) as committed at HEAD in .dvc/config;
    (None, None) when there is no committed config or no default remote."""
    result = subprocess.run(
        ["git", "show", "HEAD:.dvc/config"], cwd=workspace_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None, None
    parser = configparser.ConfigParser(interpolation=None)
    try:
        parser.read_string(result.stdout)
    except configparser.Error:
        return None, None
    # DVC writes remote sections as ['remote "name"'], quotes included.
    sections = {s.strip("'"): s for s in parser.sections()}
    name = parser.get("core", "remote", fallback=None)
    section = sections.get(f'remote "{name}"')
    return name, (parser.get(section, "url", fallback=None) if section else None)


def is_azure_remote(remote_url):
    return bool(remote_url) and remote_url.startswith("azure://")


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


def resolve_mode(workspace_dir, remote_name):
    if os.environ.get(ENV_VAR):
        # Named, not just "env var": an override the user cannot see the source
        # of is undebuggable (D#22 makes this var mldna's own concept).
        return f"env var {ENV_VAR}"

    config_local = Path(workspace_dir) / ".dvc" / "config.local"
    if config_local.exists():
        text = config_local.read_text()
        if f'"{remote_name}"' in text and "connection_string" in text:
            return ".dvc/config.local"

    return "AAD fallback"
