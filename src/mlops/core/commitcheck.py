import subprocess

import mlflow

from mlops.core import git


class CommitReachabilityError(Exception):
    pass


def check_commit_reachable(workspace_dir, run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    sha = run.data.tags.get("git_commit")
    branch = run.data.tags.get("git_branch")

    if not git.commit_exists_locally(workspace_dir, sha):
        raise CommitReachabilityError(
            f"Commit {sha} (recorded on run {run_id}) does not exist in this local "
            f"repo. Publish must run from the same repo the run trained in."
        )

    git.fetch(workspace_dir)
    remote_ref = f"origin/{branch}"
    if git.is_ancestor(workspace_dir, sha, remote_ref):
        return

    try:
        git.push_branch(workspace_dir, branch)
    except subprocess.CalledProcessError as e:
        raise CommitReachabilityError(
            f"Commit {sha} exists locally but is missing on the remote, and "
            f"pushing it failed: {e.stderr.strip()}"
        )

    git.fetch(workspace_dir)
    if not git.is_ancestor(workspace_dir, sha, remote_ref):
        raise CommitReachabilityError(
            f"Commit {sha} still not reachable on {remote_ref} after pushing. "
            f"Refusing to publish - lineage cannot be verified on the remote."
        )
