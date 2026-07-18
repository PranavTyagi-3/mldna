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
