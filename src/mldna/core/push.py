import subprocess
from pathlib import Path
from typing import NamedTuple

from mldna.core import bigfiles, blobauth, dvc, git
from mldna.core.config import ProjectConfig


class PushError(Exception):
    pass


def failure_text(result):
    return (
        result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}"
    )


def _run_git(args, cwd, hint):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise PushError(f"{hint}: {failure_text(result)}")
    return result


def _run_dvc(args, cwd, hint):
    result = dvc.run(args, cwd=cwd, check=False)
    if result.returncode != 0:
        raise PushError(f"{hint}: {failure_text(result)}")
    return result


def _risky_staged(workspace_dir):
    config = ProjectConfig.load(workspace_dir)
    try:
        staged = git.staged_files(workspace_dir)
    except git.GitStatusError as e:
        raise PushError(f"git could not list the staged files: {e}")
    return bigfiles.risky_files(workspace_dir, staged, config.model_globs, config.max_file_mb)


class StageResult(NamedTuple):
    lines: list      # `git diff --cached --name-status`
    unstaged: list   # newly staged paths removed from the index because they are ignored
    in_history: list # ignored paths already committed; .gitignore cannot remove them


def _nul_list(workspace_dir, args, hint):
    out = _run_git(args, workspace_dir, hint).stdout
    return [p for p in out.split("\0") if p]


def _unstage_ignored(workspace_dir):
    """`git add -A` never drops an indexed file that was since ignored; do it here."""
    ignored = _nul_list(workspace_dir, ["ls-files", "-ci", "--exclude-standard", "-z"], "git ls-files failed")
    if not ignored:
        return [], []
    head = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "-z", "HEAD"],
        cwd=workspace_dir, capture_output=True, text=True,
    )
    committed = set(head.stdout.split("\0")) if head.returncode == 0 else set()
    new = [p for p in ignored if p not in committed]
    for i in range(0, len(new), 100):
        _run_git(["rm", "--cached", "-q", "--", *new[i:i + 100]], workspace_dir, "git rm --cached failed")
    return new, [p for p in ignored if p in committed]


def stage(workspace_dir):
    """dvc add the data, then git add what push would commit, then unstage
    anything now ignored. Idempotent."""
    workspace_dir = Path(workspace_dir)
    _run_dvc(["add", "dataset", "-q"], workspace_dir, "dvc add failed")
    _run_git(["add", "-A"], workspace_dir, "git add failed")
    unstaged, in_history = _unstage_ignored(workspace_dir)
    result = _run_git(["diff", "--cached", "--name-status"], workspace_dir, "git diff failed")
    return StageResult(result.stdout.splitlines(), unstaged, in_history)


def _shrink_notes(workspace_dir):
    notes = []
    for path in git.staged_files(workspace_dir):
        if not path.endswith(".dvc"):
            continue
        old = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=workspace_dir, capture_output=True, text=True)
        if old.returncode != 0:
            continue
        new = _run_git(["show", f":{path}"], workspace_dir, "git show failed")
        note = bigfiles.shrink_note(path, old.stdout, new.stdout)
        if note:
            notes.append(note)
    return notes


def _has_unpushed(workspace_dir, branch):
    ahead = subprocess.run(
        ["git", "rev-list", "--count", f"origin/{branch}..HEAD"],
        cwd=workspace_dir, capture_output=True, text=True,
    )
    if ahead.returncode == 0:
        return int(ahead.stdout.strip()) > 0
    head = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=workspace_dir, capture_output=True)
    return head.returncode == 0


def check_something_to_push(workspace_dir):
    """Raises PushError when nothing is staged and no commits await pushing;
    returns whether anything is staged."""
    workspace_dir = Path(workspace_dir)
    branch = git.get_branch(workspace_dir)
    has_staged = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=workspace_dir).returncode != 0
    if not has_staged and not _has_unpushed(workspace_dir, branch):
        raise PushError(
            "Nothing is staged and no commits are waiting to be pushed. "
            "push commits only what you staged, so there is nothing to send. "
            "Fix: run `mldna stage`, review the files, then `mldna push -m \"<message>\"`."
        )
    return has_staged


def push(workspace_dir, message="mldna push", confirm=lambda msg: True):
    """Commits what is already staged (it never stages), then dvc push and git push.
    When something is staged, confirm(msg) is always called with the full list of
    files (plus large/model-shaped and dataset-shrink warnings); the last line of
    msg is the question. Declining refuses and leaves the index untouched."""
    workspace_dir = Path(workspace_dir)
    branch = git.get_branch(workspace_dir)
    has_staged = check_something_to_push(workspace_dir)

    if has_staged:
        lines = _run_git(["diff", "--cached", "--name-status"], workspace_dir, "git diff failed").stdout.splitlines()
        parts = [f"Files to be committed ({len(lines)}):\n" + "\n".join(f"  {line}" for line in lines)]
        risky = _risky_staged(workspace_dir)
        if risky:
            listing = "\n".join(f"  {path} ({reason})" for path, reason in risky)
            parts.append(
                f"These are large or model-shaped and would enter git history permanently:\n{listing}\n"
                f"A pushed commit cannot be undone without rewriting history, and model binaries "
                f"must never be committed. To keep them out: put data under dataset/ (tracked by DVC), "
                f"run `dvc add <path>`, or add the path to .gitignore, then re-run `mldna stage`."
            )
        notes = _shrink_notes(workspace_dir)
        if notes:
            parts.append("Dataset shrink:\n" + "\n".join(f"  {note}" for note in notes))
        parts.append(f"Commit these {len(lines)} files?")
        if not confirm("\n\n".join(parts)):
            raise PushError(
                "Not committed: you declined, and the staged files are unchanged. "
                "Fix: edit .gitignore and re-run `mldna stage`, or run `mldna push --yes` to commit them."
            )

    committed = has_staged
    if committed:
        _run_git(["commit", "-q", "-m", message], workspace_dir, "git commit failed")

    remote_name = blobauth.get_default_remote(workspace_dir)
    if remote_name:
        remote_url = blobauth.get_remote_url(workspace_dir, remote_name)
        if remote_url and blobauth.is_azure_remote(remote_url):
            blobauth.resolve_env_override(workspace_dir, remote_name)
        _run_dvc(["push", "-q"], workspace_dir,
                  "dvc push failed - is a remote configured? run `dvc remote add -d <name> <url>`")
    else:
        print("Skipped dvc push: no DVC remote configured")

    try:
        git.push_branch(workspace_dir, branch)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.strip()
        message = f"git push failed:\n{stderr}"
        if "non-fast-forward" in stderr or "fetch first" in stderr:
            message += "\nFix: run `mldna pull` to merge the remote changes, then push again."
        elif "No configured push destination" in stderr or "does not appear to be a git repository" in stderr:
            message += "\nFix: git remote add origin <url>"
        raise PushError(message)

    return {"committed": committed}
