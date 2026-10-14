import subprocess

import mlflow

from mldna.core import git


class CommitReachabilityError(Exception):
    pass


def _commit_sha_and_branch(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    return run.data.tags.get("git_commit"), run.data.tags.get("git_branch")


def _assert_exists_locally(workspace_dir, sha, run_id):
    if not git.commit_exists_locally(workspace_dir, sha):
        raise CommitReachabilityError(
            f"Commit {sha} (recorded on run {run_id}) does not exist in this local "
            f"repo. Publish must run from the same repo the run trained in, or the "
            f"lineage it records cannot be verified. Fix: run publish from that "
            f"repo, or `git fetch origin` if the commit is already on the remote."
        )


def _assert_remote_configured(workspace_dir, remote="origin"):
    result = subprocess.run(
        ["git", "remote"], cwd=workspace_dir, capture_output=True, text=True
    )
    if remote not in result.stdout.split():
        raise CommitReachabilityError(
            f"No git remote named '{remote}' is configured. Publish needs the "
            f"training commit to be reachable on a remote so anyone verifying "
            f"the lineage can fetch it, and there is no remote to check against. "
            f"Fix: git remote add {remote} <url>."
        )


def _fetch(workspace_dir):
    try:
        git.fetch(workspace_dir)
    except subprocess.CalledProcessError as e:
        raise CommitReachabilityError(
            f"Could not fetch from the git remote: {e.stderr.decode(errors='replace').strip()}. "
            f"Publish needs the training commit to be reachable on the remote, "
            f"and that cannot be checked while the remote is unreachable. "
            f"Fix: run `git ls-remote origin` to check connectivity and "
            f"credentials, then publish again."
        ) from e


def commit_needs_push(workspace_dir, run_id):
    """Whether publishing this run will have to push its branch - checked ahead
    of the confirmation summary so the pending push can be named there (D#56)
    instead of only being discovered after it already happened. Raises the
    same refusal as check_commit_reachable when the commit isn't even local;
    never pushes itself.
    """
    sha, branch = _commit_sha_and_branch(run_id)
    _assert_exists_locally(workspace_dir, sha, run_id)
    _assert_remote_configured(workspace_dir)
    _fetch(workspace_dir)
    return branch, not git.is_ancestor(workspace_dir, sha, f"origin/{branch}")


def check_commit_reachable(workspace_dir, run_id):
    sha, branch = _commit_sha_and_branch(run_id)
    _assert_exists_locally(workspace_dir, sha, run_id)
    _assert_remote_configured(workspace_dir)

    _fetch(workspace_dir)
    remote_ref = f"origin/{branch}"
    if git.is_ancestor(workspace_dir, sha, remote_ref):
        return

    try:
        git.push_branch(workspace_dir, branch)
    except subprocess.CalledProcessError as e:
        raise CommitReachabilityError(
            f"Commit {sha} exists locally but is missing on the remote, and "
            f"pushing it failed: {e.stderr.strip()}. A published model's commit "
            f"must be fetchable by anyone verifying its lineage. Fix: resolve that "
            f"push error, run `git push origin {branch}`, then publish again."
        )

    _fetch(workspace_dir)
    if not git.is_ancestor(workspace_dir, sha, remote_ref):
        raise CommitReachabilityError(
            f"Commit {sha} still not reachable on {remote_ref} after pushing. "
            f"Refusing to publish - lineage cannot be verified on the remote. "
            f"Fix: confirm `git push origin {branch}` lands on {remote_ref} (a "
            f"branch policy or server-side hook can reject it), then publish again."
        )
