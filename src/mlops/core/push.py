import subprocess
from pathlib import Path

from mlops.core import blobauth, dvc, git


class PushError(Exception):
    pass


def _run_git(args, cwd, hint):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PushError(f"{hint}: {result.stderr.strip()}")
    return result


def _run_dvc(args, cwd, hint):
    result = dvc.run(args, cwd=cwd, check=False)
    if result.returncode != 0:
        raise PushError(f"{hint}: {result.stderr.strip()}")
    return result


def push(workspace_dir, message="mlops push"):
    workspace_dir = Path(workspace_dir)

    _run_dvc(["add", "dataset", "-q"], workspace_dir, "dvc add failed")

    _run_git(["add", "-A"], workspace_dir, "git add failed")
    diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=workspace_dir)
    committed = diff.returncode != 0
    if committed:
        _run_git(["commit", "-q", "-m", message], workspace_dir, "git commit failed")

    remote_name = blobauth.get_default_remote(workspace_dir)
    if remote_name:
        blobauth.resolve_env_override(workspace_dir, remote_name)

    _run_dvc(["push", "-q"], workspace_dir,
              "dvc push failed - is a remote configured? run `dvc remote add -d <name> <url>`")

    branch = git.get_branch(workspace_dir)
    try:
        git.push_branch(workspace_dir, branch)
    except subprocess.CalledProcessError as e:
        raise PushError(
            f"git push failed - is a remote configured? run `git remote add origin <url>`: "
            f"{e.stderr.strip()}"
        )

    return {"committed": committed}
