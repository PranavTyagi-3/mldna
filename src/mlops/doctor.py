import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from mlops.core import blobauth, dvc
from mlops.core.lock import RunLock
from mlops.core.scaffold import GITIGNORE, has_azure_config
from mlops.trackinguri import TrackingUriError, resolve_and_set_tracking_uri

SECRET_PATTERNS = (
    "AccountKey=", "SharedAccessSignature=", "sig=", "client_secret", "AZURE_CLIENT_SECRET=",
)
MANDATORY_GITIGNORE_ENTRIES = [line for line in GITIGNORE.splitlines() if line and "cache" not in line]


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def _check(name, passed, message):
    return CheckResult(name, passed, message)


def check_tools():
    if shutil.which("git") is None:
        return _check("tools", False, "git not found on PATH.")
    dvc_result = dvc.run(["--version"], cwd=".", check=False)
    if dvc_result.returncode != 0:
        return _check("tools", False, "dvc not runnable via python -m dvc.")
    return _check("tools", True, "git and dvc available.")


def check_gitignore(workspace_dir):
    path = Path(workspace_dir) / ".gitignore"
    if not path.exists():
        return _check("gitignore", False, ".gitignore missing. Fix: mlops init")
    text = path.read_text()
    missing = [entry for entry in MANDATORY_GITIGNORE_ENTRIES if entry not in text]
    if missing:
        return _check("gitignore", False, f".gitignore missing entries: {', '.join(missing)}")
    return _check("gitignore", True, "All mandatory entries present.")


def check_run_lock(workspace_dir):
    lock = RunLock(workspace_dir)
    if not lock.path.exists():
        return _check("run_lock", True, "No lock held.")
    if lock.is_locked():
        return _check("run_lock", True, "Lock held by a live process (normal during a run).")
    return _check(
        "run_lock", False,
        f"Stale lock at {lock.path} (owning process is dead). "
        f"Fix: delete the file or start a new run (auto-reclaimed)."
    )


def check_git_remote(workspace_dir):
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], cwd=workspace_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        return _check("git_remote", False, "No 'origin' remote configured. Fix: git remote add origin <url>")
    # No --exit-code: a reachable-but-empty remote (nothing pushed yet) still
    # returns 0 without it; --exit-code would misreport that as unreachable.
    reachable = subprocess.run(
        ["git", "ls-remote", "origin"], cwd=workspace_dir, capture_output=True, text=True
    )
    if reachable.returncode != 0:
        return _check("git_remote", False, f"origin ({result.stdout.strip()}) is not reachable.")
    return _check("git_remote", True, f"origin ({result.stdout.strip()}) reachable.")


def check_committed_secrets(workspace_dir):
    result = subprocess.run(
        ["git", "ls-files"], cwd=workspace_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        return _check("committed_secrets", True, "Not a git repo yet - nothing committed.")

    hits = []
    for relpath in result.stdout.splitlines():
        path = Path(workspace_dir) / relpath
        try:
            text = path.read_text(errors="ignore")
        except (OSError, UnicodeDecodeError):
            continue
        for pattern in SECRET_PATTERNS:
            if pattern in text:
                hits.append(f"{relpath} (matches '{pattern}')")
                break

    if hits:
        return _check(
            "committed_secrets", False,
            f"Possible secrets in committed files: {', '.join(hits)}. "
            f"Fix: remove them, rotate the secret, and use env vars / "
            f"mlops configure aml-auth / mlops configure blob-auth instead."
        )
    return _check("committed_secrets", True, "No secret patterns found in committed files.")


def check_aml_auth(workspace_dir):
    if not has_azure_config(workspace_dir):
        return _check("aml_auth", True, "Not configured - local tracking mode.")
    try:
        _, mode = resolve_and_set_tracking_uri(workspace_dir)
    except TrackingUriError as e:
        return _check("aml_auth", False, str(e))
    except Exception as e:
        return _check("aml_auth", False, f"Could not reach Azure ML workspace: {e}")
    return _check("aml_auth", True, f"Reachable, auth mode: {mode}")


def check_blob_auth(workspace_dir):
    remote_name = blobauth.get_default_remote(workspace_dir)
    if not remote_name:
        return _check("blob_auth", True, "No DVC remote configured yet.")
    mode = blobauth.resolve_mode(workspace_dir, remote_name)
    return _check("blob_auth", True, f"Remote '{remote_name}', auth mode: {mode}")


def run_doctor(workspace_dir="."):
    return [
        check_tools(),
        check_gitignore(workspace_dir),
        check_run_lock(workspace_dir),
        check_git_remote(workspace_dir),
        check_committed_secrets(workspace_dir),
        check_aml_auth(workspace_dir),
        check_blob_auth(workspace_dir),
    ]
