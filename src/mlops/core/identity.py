import subprocess
from pathlib import Path


def _from_user_file(workspace_dir):
    path = Path(workspace_dir) / ".mlops" / "user"
    if path.exists():
        value = path.read_text().strip()
        return value or None
    return None


def _from_git_config(workspace_dir):
    result = subprocess.run(
        ["git", "config", "user.email"], cwd=workspace_dir,
        capture_output=True, text=True,
    )
    email = result.stdout.strip()
    if email:
        return email
    result = subprocess.run(
        ["git", "config", "user.name"], cwd=workspace_dir,
        capture_output=True, text=True,
    )
    name = result.stdout.strip()
    return name or None


def resolve_identity(workspace_dir, extra_resolver=None):
    for resolver in (_from_user_file, _from_git_config):
        identity = resolver(workspace_dir)
        if identity:
            return identity
    if extra_resolver:
        return extra_resolver()
    return None
