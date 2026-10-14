import subprocess
from pathlib import Path

from mldna.core import blobauth, dvc, git
from mldna.core.config import ProjectConfig
from mldna.core.lock import RunLock
from mldna.core.push import failure_text


class PullError(Exception):
    pass


def _run_dvc(args, cwd, hint):
    result = dvc.run(args, cwd=cwd, check=False)
    if result.returncode != 0:
        raise PullError(f"{hint}: {failure_text(result)}")
    return result


def pull(workspace_dir):
    workspace_dir = Path(workspace_dir)

    config = ProjectConfig.load(workspace_dir)
    if config.repo_role == "consumer":
        raise PullError(
            "This is a consumer repo - `mldna pull` only syncs producer "
            "code+data (git+dvc). Fix: use `mldna fetch` to pull pinned "
            "model versions instead."
        )

    lock = RunLock(workspace_dir)
    if lock.is_locked():
        raise PullError(
            "A training run is active in this working copy "
            f"({lock.path}). Pulling now could change dataset/ or code files "
            "a live run depends on mid-run. Fix: wait for the run to finish, "
            "or use `mldna workspace new` to pull in a separate worktree."
        )

    branch = git.get_branch(workspace_dir)
    try:
        git.pull_branch(workspace_dir, branch)
    except subprocess.CalledProcessError as e:
        # git writes CONFLICT lines to stdout and progress noise to
        # stderr, so both are surfaced - on separate lines, since a
        # colon-joined chain of them is unreadable.
        msg_parts = ["git pull failed:"]
        if e.stdout and e.stdout.strip():
            msg_parts.append(e.stdout.strip())
        if e.stderr and e.stderr.strip():
            msg_parts.append(e.stderr.strip())
        raise PullError("\n".join(msg_parts))

    remote_name = blobauth.get_default_remote(workspace_dir)
    if remote_name:
        remote_url = blobauth.get_remote_url(workspace_dir, remote_name)
        if remote_url and blobauth.is_azure_remote(remote_url):
            blobauth.resolve_env_override(workspace_dir, remote_name)

    _run_dvc(
        ["pull", "-q"], workspace_dir,
        "dvc pull failed - is a remote configured? run `dvc remote add -d <name> <url>`",
    )
