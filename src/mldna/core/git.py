import os
import re
import subprocess
from pathlib import Path


def _run(repo_dir, args):
    result = subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True
    )
    return result.stdout.strip(), result.returncode


class GitStatusError(Exception):
    pass


def is_clean(repo_dir) -> bool:
    """True iff `git status --porcelain` succeeded and reported nothing.

    Raises GitStatusError when the answer cannot be determined (not a repo,
    permissions, the `dubious ownership` error common on shared machines).
    Callers must treat that as "not known to be clean", never as clean - see
    D#45, which fixed the mirror-image bug on the dvc plane and flagged this
    exact one (`out == ""` with the exit code discarded) as the shape to avoid.
    Empty stdout only means "clean" when the command actually ran; empty stdout
    from a failed command means nothing.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo_dir, capture_output=True, text=True
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise GitStatusError(detail)
    return result.stdout.strip() == ""


def ignored_files(repo_dir) -> list:
    """Untracked-and-ignored files: the ones git cannot restore once deleted.

    Raises GitStatusError when git cannot answer; an empty list only ever means
    "none", never "failed".
    """
    result = subprocess.run(
        ["git", "ls-files", "--others", "--ignored", "--exclude-standard"],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise GitStatusError(detail)
    return result.stdout.splitlines()


def _lines(repo_dir, args):
    result = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise GitStatusError(detail)
    return [p for p in result.stdout.split("\0") if p]


def staged_files(repo_dir) -> list:
    """Added/copied/modified/renamed paths in the index: what a commit would add.

    Raises GitStatusError when git cannot answer; an empty list only ever means "none"."""
    return _lines(repo_dir, ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"])


def tracked_files(repo_dir) -> list:
    """Paths git tracks. Raises GitStatusError when git cannot answer."""
    return _lines(repo_dir, ["ls-files", "-z"])


def get_commit_sha(repo_dir) -> str:
    out, _ = _run(repo_dir, ["rev-parse", "HEAD"])
    return out


def get_branch(repo_dir) -> str:
    out, _ = _run(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    return out


def normalise_remote_url(url):
    """Drop userinfo (credentials included) and a trailing `.git` or `/` (D#77)."""
    scheme, sep, rest = url.strip().partition("://")
    if not sep:
        scheme, rest = "", url.strip()
    host = rest.split("/", 1)[0]
    if "@" in host:
        rest = rest[host.rindex("@") + 1:]
    rest = rest.rstrip("/")
    if rest.endswith(".git"):
        rest = rest[:-4].rstrip("/")
    return f"{scheme}{sep}{rest}"


def origin_url(repo_dir):
    """The normalised `origin` URL, or None when there is no origin. A relative
    local path is made absolute, since it means nothing outside this clone."""
    out, code = _run(repo_dir, ["remote", "get-url", "origin"])
    if code != 0 or not out:
        return None
    url = normalise_remote_url(out)
    path = _local_path(url)
    if path is not None and not Path(path).expanduser().is_absolute():
        url = str((Path(repo_dir) / path).resolve())
    return url


def _local_path(url):
    """The filesystem path a remote URL names, or None when it names a host."""
    scheme, sep, rest = url.partition("://")
    if sep:
        if scheme != "file":
            return None
        return rest[1:] if re.match(r"^/[A-Za-z]:", rest) else rest  # file:///C:/x -> C:/x
    if re.match(r"^[A-Za-z]:[\\/]", url) or url.startswith(("/", "\\", ".", "~")):
        return url
    # scp-like `host:path` names a host; anything else is a relative local path.
    return None if re.match(r"^[^/\\]+:", url) else url


def canonical_remote(url, repo_dir="."):
    """One spelling per repository, for comparing remotes (D#77): `host/path`
    for hosted remotes (scheme, user, port, case of host and `.git` ignored, so
    SSH and HTTPS match), an absolute normalised path for local ones."""
    url = normalise_remote_url(url)
    path = _local_path(url)
    if path is not None:
        resolved = (Path(repo_dir) / Path(path).expanduser()).resolve()
        return "file:" + os.path.normcase(str(resolved)).replace("\\", "/").rstrip("/")
    scheme, sep, rest = url.partition("://")
    hostport, _, tail = rest.partition("/") if sep else url.partition(":")
    host = hostport.rsplit("@", 1)[-1].split(":", 1)[0].lower()
    return f"{host}/{tail.strip('/')}"


def same_remote(a, b, repo_dir="."):
    if not a or not b:
        return a == b
    return canonical_remote(a, repo_dir) == canonical_remote(b, repo_dir)


def fetch(repo_dir, remote="origin"):
    subprocess.run(["git", "fetch", remote, "-q"], cwd=repo_dir, check=True, capture_output=True)


def commit_exists_locally(repo_dir, sha) -> bool:
    _, code = _run(repo_dir, ["cat-file", "-e", f"{sha}^{{commit}}"])
    return code == 0


def is_ancestor(repo_dir, sha, ref) -> bool:
    _, code = _run(repo_dir, ["merge-base", "--is-ancestor", sha, ref])
    return code == 0


def remote_branch_exists(repo_dir, branch, remote="origin") -> bool:
    _, code = _run(repo_dir, ["rev-parse", "--verify", "-q", f"refs/remotes/{remote}/{branch}"])
    return code == 0


def push_branch(repo_dir, branch, remote="origin"):
    subprocess.run(["git", "push", remote, branch], cwd=repo_dir, check=True, capture_output=True, text=True)


def pull_branch(repo_dir, branch, remote="origin"):
    # Explicit merge: without it, git 2.27+ refuses diverged branches unless
    # pull.rebase/pull.ff is configured (Git for Windows sets it; Linux git does not).
    subprocess.run(["git", "pull", "--no-rebase", remote, branch], cwd=repo_dir, check=True, capture_output=True, text=True)


class GitWorktreeError(Exception):
    pass


def _run_checked(repo_dir, args):
    result = subprocess.run(["git", *args], cwd=repo_dir, capture_output=True, text=True)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise GitWorktreeError(detail)
    return result.stdout


def worktree_list(repo_dir) -> list:
    """Paths of every worktree of this repo, main worktree first.

    Raises GitWorktreeError when git cannot answer (e.g. not a repo).
    """
    out = _run_checked(repo_dir, ["worktree", "list", "--porcelain"])
    return [Path(line[len("worktree "):]) for line in out.splitlines() if line.startswith("worktree ")]


def worktree_add(repo_dir, path):
    _run_checked(repo_dir, ["worktree", "add", str(path)])


def worktree_remove(repo_dir, path):
    _run_checked(repo_dir, ["worktree", "remove", str(path)])
