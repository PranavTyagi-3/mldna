import os
import re
import shutil
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import url2pathname

from urllib3.exceptions import NewConnectionError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from mldna.core import bigfiles, blobauth, dvc, git
from mldna.core.config import ProjectConfig
from mldna.core.lock import RunLock
from mldna.core.scaffold import GITIGNORE, has_azure_config, has_mlflow_config
from mldna.trackinguri import FILESYSTEM_MODE, TrackingUriError, resolve_and_set_tracking_uri

MANDATORY_GITIGNORE_ENTRIES = [line for line in GITIGNORE.splitlines() if line and "cache" not in line]
TRACKING_TARGETS = {
    "aml_auth": "Azure ML workspace",
    "mlflow_server": "MLflow tracking server",
    "tracking_store": "tracking store",
}
DEFAULT_TIMEOUT_SECONDS = "10"
TIMEOUT_ENV = "MLFLOW_HTTP_REQUEST_TIMEOUT"
RETRIES_ENV = "MLFLOW_HTTP_REQUEST_MAX_RETRIES"


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def _check(name, passed, message):
    return CheckResult(name, passed, message)


def check_tools():
    if shutil.which("git") is None:
        return _check(
            "tools", False,
            "git not found on PATH, so no run can record the commit it trained from. "
            "Fix: install git from https://git-scm.com/downloads, then reopen this shell."
        )
    dvc_result = dvc.run(["--version"], cwd=".", check=False)
    if dvc_result.returncode != 0:
        return _check(
            "tools", False,
            "dvc not runnable via python -m dvc, so dataset hashes cannot be read or "
            "pushed. Fix: pip install dvc"
        )
    return _check("tools", True, "git and dvc available.")


def check_gitignore(workspace_dir):
    path = Path(workspace_dir) / ".gitignore"
    if not path.exists():
        return _check(
            "gitignore", False,
            ".gitignore missing, so secrets, model binaries and local state are not "
            "excluded from git. Fix: mldna init <model-name> (or mldna init --consumer in a "
            "consumer repo)"
        )
    text = path.read_text()
    # Whole-line comparison, for the same reason scaffold.py does it: `.env.example`
    # must not satisfy the `.env` requirement.
    present = {line.strip() for line in text.splitlines()}
    missing = [entry for entry in MANDATORY_GITIGNORE_ENTRIES if entry not in present]
    if missing:
        return _check(
            "gitignore", False,
            f".gitignore missing entries: {', '.join(missing)}. Without them secrets, "
            f"model binaries and local state can be committed to git. Fix: add those lines "
            f"to .gitignore, then re-run mldna doctor."
        )
    return _check("gitignore", True, "All mandatory entries present.")


def check_run_lock(workspace_dir):
    lock = RunLock(workspace_dir)
    if not lock.path.exists():
        return _check("run_lock", True, "No lock held.")
    if lock.is_locked():
        return _check("run_lock", True, "Lock held by a live process (normal during a run).")
    return _check(
        "run_lock", True,
        f"Stale lock found at {lock.path} (owning process is dead); the next run will reclaim it."
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
        return _check(
            "git_remote", False,
            f"origin ({result.stdout.strip()}) is not reachable, so publish cannot "
            f"verify a run's commit on the remote. Fix: check the url with `git "
            f"remote -v` and your network/credentials, then re-run mldna doctor."
        )
    return _check("git_remote", True, f"origin ({result.stdout.strip()}) reachable.")


def check_large_tracked_files(workspace_dir):
    try:
        tracked = git.tracked_files(workspace_dir)
    except git.GitStatusError as e:
        return _check(
            "large_tracked_files", False,
            f"Could not list tracked files, so large or model-shaped ones were not checked: {e}. "
            f"Fix: run this in a git repo (git init), then re-run mldna doctor."
        )
    config = ProjectConfig.load(workspace_dir)
    risky = bigfiles.risky_files(workspace_dir, tracked, config.model_globs, config.max_file_mb)
    if not risky:
        return _check(
            "large_tracked_files", True,
            f"No large or model-shaped files among {len(tracked)} tracked files checked; "
            f"history not examined."
        )
    shown = [f"{path} ({reason})" for path, reason in risky[:10]]
    if len(risky) > 10:
        shown.append(f"and {len(risky) - 10} more")
    return _check(
        "large_tracked_files", False,
        f"Large or model-shaped files are tracked by git: {', '.join(shown)}. Model binaries "
        f"must never enter git. Fix: `git rm --cached <path>`, then put it under DVC "
        f"(`dvc add <path>`) or add it to .gitignore. History is not examined, and removing "
        f"a file already committed from history needs a rewrite mldna does not do."
    )


def _tracking_check_name(workspace_dir):
    # The check name is a label like any other (invariant 10): it names the
    # plane actually configured, so an Azure-ML name never appears for a
    # project with no Azure in it (D#42).
    if has_azure_config(workspace_dir):
        return "aml_auth"
    if has_mlflow_config(workspace_dir):
        return "mlflow_server"
    return "tracking_store"


def _store_path(uri):
    parsed = urlparse(uri)
    return Path(url2pathname(parsed.netloc + parsed.path))


def _check_tracking_store(name, mode, uri):
    path = _store_path(uri)
    if not path.exists():
        # Only reachable for local mode: the filesystem: branch of
        # resolve_and_set_tracking_uri already refuses a missing path (D#39).
        return _check(
            name, True,
            f"Store at {path} not created yet (the first run creates it), tracking mode: {mode}"
        )
    if not os.access(path, os.R_OK):
        return _check(
            name, False,
            f"Tracking store at {path} is not readable, so no run can be recorded or resolved. "
            f"Fix: grant read access to {path}, or point `filesystem: path:` in mldna.yaml at a readable store."
        )
    return _check(name, True, f"Readable store at {path}, tracking mode: {mode}")


def parse_timeout(text):
    match = re.fullmatch(r"(\d+)s?", text.strip())
    if not match or int(match.group(1)) == 0:
        raise ValueError(
            f"--timeout must be a whole number of seconds above zero, such as 60 or 60s "
            f"(got '{text}'). Fix: mldna doctor --timeout 60s"
        )
    return match.group(1)


@contextmanager
def _probe_limits(timeout):
    # doctor is its own process and uploads nothing, so it alone can shorten the
    # variable that also governs proxied artifact uploads (D#65). The flag wins,
    # then a deliberately set variable, then the default.
    seconds = timeout or os.environ.get(TIMEOUT_ENV) or DEFAULT_TIMEOUT_SECONDS
    saved = {key: os.environ.get(key) for key in (TIMEOUT_ENV, RETRIES_ENV)}
    os.environ[TIMEOUT_ENV] = seconds
    os.environ[RETRIES_ENV] = "0"
    try:
        yield seconds
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _timed_out(error):
    # MLflow wraps the requests error in an MlflowException, but the urllib3
    # timeout stays reachable through the cause chain (checked against mlflow
    # 2.22.5), so no message text is matched (D#41).
    while error is not None:
        if isinstance(error, NewConnectionError):
            # urllib3 makes this a ConnectTimeoutError subclass, but it is what a
            # REFUSED connection raises: the server is down, not slow, and
            # "raise the limit" would be the wrong advice.
            return False
        if isinstance(error, Urllib3TimeoutError):
            return True
        error = error.__cause__ or error.__context__
    return False


def check_tracking(workspace_dir, timeout=None):
    name = _tracking_check_name(workspace_dir)
    target = TRACKING_TARGETS[name]
    with _probe_limits(timeout) as seconds:
        try:
            uri, mode = resolve_and_set_tracking_uri(workspace_dir)
            if mode == "remote mlflow server":
                # resolve_and_set_tracking_uri only sets the URI for a plain MLflow
                # server - unlike the azure branch, it never actually calls the
                # server, so doctor's "reachable" claim would otherwise be a lie.
                from mlflow.tracking import MlflowClient
                try:
                    MlflowClient().search_experiments(max_results=1)
                except Exception as e:
                    if _timed_out(e):
                        return _check(
                            name, False,
                            f"Could not reach {target}: no response within {seconds} second{'' if seconds == '1' else 's'}. "
                            f"Fix: if the server is slow rather than down, raise the limit with "
                            f"`mldna doctor --timeout 60s`."
                        )
                    raise
        except TrackingUriError as e:
            return _check(name, False, str(e))
        except Exception as e:
            return _check(name, False, f"Could not reach {target}: {e}")
    if mode in (FILESYSTEM_MODE, "local"):
        # A file: URI under mlflow: resolves to the filesystem mode, so the
        # name follows what resolved, not which section was written.
        return _check_tracking_store("tracking_store", mode, uri)
    return _check(name, True, f"Reachable, tracking mode: {mode}")


def check_blob_auth(workspace_dir):
    remote_name = blobauth.get_default_remote(workspace_dir)
    if not remote_name:
        return _check("blob_auth", True, "No DVC remote configured yet.")
    remote_url = blobauth.get_remote_url(workspace_dir, remote_name)
    if not remote_url:
        return _check(
            "blob_auth", False,
            f"Remote '{remote_name}' has no readable url, so its auth mode is undetermined. "
            f"Fix: dvc remote modify {remote_name} url <url>"
        )
    if not blobauth.is_azure_remote(remote_url):
        return _check(
            "blob_auth", True,
            f"Remote '{remote_name}' is a local/non-Azure remote ({remote_url}); "
            f"blob auth not applicable."
        )
    mode = blobauth.resolve_mode(workspace_dir, remote_name)
    return _check("blob_auth", True, f"Remote '{remote_name}', auth mode: {mode}")


def run_doctor(workspace_dir=".", timeout=None):
    return [
        check_tools(),
        check_gitignore(workspace_dir),
        check_run_lock(workspace_dir),
        check_git_remote(workspace_dir),
        check_large_tracked_files(workspace_dir),
        check_tracking(workspace_dir, timeout),
        check_blob_auth(workspace_dir),
    ]
