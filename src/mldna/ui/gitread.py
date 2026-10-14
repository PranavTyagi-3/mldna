"""Read-only git and dvc access for the UI.

Only the subcommands in READ_ONLY_GIT can run, and every commit or path comes
from a validator below before it reaches the command line, so a request can
never smuggle in an option or a write.
"""
import json
import re
import subprocess
import sys

READ_ONLY_GIT = {"log", "show", "diff", "blame", "ls-tree", "cat-file", "remote"}

_SHA = re.compile(r"^[0-9a-f]{4,40}$")


class GitReadError(Exception):
    pass


def check_sha(sha):
    if not isinstance(sha, str) or not _SHA.match(sha):
        raise GitReadError(f"{sha!r} is not a commit hash.")
    return sha


def git(repo_dir, *args):
    if args[0] not in READ_ONLY_GIT:
        raise GitReadError(f"git {args[0]} is not allowed in the UI.")
    result = subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise GitReadError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout


def commit_exists(repo_dir, sha):
    try:
        git(repo_dir, "cat-file", "-e", f"{check_sha(sha)}^{{commit}}")
        return True
    except GitReadError:
        return False


def commit_info(repo_dir, sha):
    """{sha, short, author, date, subject, body}, or None when the commit is not in this clone."""
    if not commit_exists(repo_dir, sha):
        return None
    out = git(repo_dir, "show", "-s", "--format=%H%x00%h%x00%an%x00%aI%x00%s%x00%b", sha)
    full, short, author, date, subject, body = out.split("\x00", 5)
    return {"sha": full, "short": short, "author": author, "date": date,
            "subject": subject, "body": body.strip()}


def commits_between(repo_dir, old, new):
    """Commits reachable from `new` but not `old`, newest first - the reasons for a diff."""
    out = git(repo_dir, "log", "--format=%H%x00%h%x00%an%x00%aI%x00%s",
              f"{check_sha(old)}..{check_sha(new)}")
    commits = []
    for line in out.splitlines():
        full, short, author, date, subject = line.split("\x00", 4)
        commits.append({"sha": full, "short": short, "author": author,
                        "date": date, "subject": subject})
    return commits


def changed_files(repo_dir, old, new):
    return git(repo_dir, "diff", "--name-only", check_sha(old), check_sha(new)).splitlines()


def is_pointer(path):
    """A DVC pointer file records data, not code."""
    return path.endswith(".dvc")


def diff(repo_dir, old, new):
    """The code diff: every changed file except DVC pointers (those are shown as data)."""
    return git(repo_dir, "diff", "--no-color", "--no-ext-diff", check_sha(old), check_sha(new),
               "--", ".", ":(exclude)*.dvc")


def files_at(repo_dir, sha):
    return git(repo_dir, "ls-tree", "-r", "--name-only", check_sha(sha)).splitlines()


def blame(repo_dir, sha, path):
    """One dict per line: the commit that last changed it and that commit's message."""
    if path not in files_at(repo_dir, sha):
        raise GitReadError(f"{path} is not a file in commit {sha[:8]}.")
    out = git(repo_dir, "blame", "--line-porcelain", sha, "--", path)
    lines, commits, current = [], {}, None
    for raw in out.splitlines():
        if raw.startswith("\t"):
            lines.append({"commit": current, "text": raw[1:]})
            continue
        key, _, value = raw.partition(" ")
        if _SHA.match(key) and len(key) == 40:
            current = key
            commits.setdefault(key, {"sha": key, "short": key[:8]})
        elif key == "author":
            commits[current]["author"] = value
        elif key == "author-time":
            commits[current]["time"] = int(value)
        elif key == "summary":
            commits[current]["subject"] = value
    return {"lines": lines, "commits": commits}


def show_file(repo_dir, sha, path):
    """A file's text at a commit, or None when it does not exist there."""
    try:
        return git(repo_dir, "show", f"{check_sha(sha)}:{path}")
    except GitReadError:
        return None


def dvc_diff(repo_dir, old, new):
    """File-level data changes between two commits (`dvc diff --json`), or None when
    dvc cannot tell (for example the directory listings are not in the local cache)."""
    result = subprocess.run(
        [sys.executable, "-m", "dvc", "diff", "--json", check_sha(old), check_sha(new)],
        cwd=repo_dir, capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        return None
    try:
        changes = json.loads(result.stdout)
    except ValueError:
        return None
    # dvc also lists each changed directory itself (path ending in a separator); only files are useful.
    return {
        kind: [e for e in entries if not (isinstance(e.get("path"), str) and e["path"].endswith(("/", "\\")))]
        for kind, entries in changes.items()
    }
