import subprocess


def _run(repo_dir, args):
    result = subprocess.run(
        ["git", *args], cwd=repo_dir, capture_output=True, text=True
    )
    return result.stdout.strip(), result.returncode


def is_clean(repo_dir) -> bool:
    out, _ = _run(repo_dir, ["status", "--porcelain"])
    return out == ""


def get_commit_sha(repo_dir) -> str:
    out, _ = _run(repo_dir, ["rev-parse", "HEAD"])
    return out


def get_branch(repo_dir) -> str:
    out, _ = _run(repo_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    return out


def fetch(repo_dir, remote="origin"):
    subprocess.run(["git", "fetch", remote, "-q"], cwd=repo_dir, check=True, capture_output=True)


def commit_exists_locally(repo_dir, sha) -> bool:
    _, code = _run(repo_dir, ["cat-file", "-e", f"{sha}^{{commit}}"])
    return code == 0


def is_ancestor(repo_dir, sha, ref) -> bool:
    _, code = _run(repo_dir, ["merge-base", "--is-ancestor", sha, ref])
    return code == 0


def push_branch(repo_dir, branch, remote="origin"):
    subprocess.run(["git", "push", remote, branch], cwd=repo_dir, check=True, capture_output=True, text=True)


def pull_branch(repo_dir, branch, remote="origin"):
    subprocess.run(["git", "pull", remote, branch], cwd=repo_dir, check=True, capture_output=True, text=True)
