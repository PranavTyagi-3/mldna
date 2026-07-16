"""Phase 0 spike: git worktree + shared DVC cache (ROADMAP.md Phase 0, item 5).

SPEC.md §11.2 requires that `mlops workspace new` create a git worktree whose DVC
cache is pointed at a shared directory, so datasets are never duplicated across
parallel workspaces (content-addressed cache = instant pull, no re-download).

This spike is fully local — no Azure needed. It builds a throwaway git+DVC repo
under a scratch directory, adds a worktree, points both copies' DVC cache at one
shared directory (via `dvc cache dir --local`, which writes the gitignored
`.dvc/config.local` rather than the committed `.dvc/config` — each worktree sets
its own local pointer at the same absolute path), and confirms the dataset object
is stored exactly once in the shared cache while being checked out (not copied) in
both working trees.

Run:  python spikes/05_worktree_dvc_cache.py [--workdir PATH]
Cleans up everything it creates unless --keep is passed.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd, cwd, **kwargs):
    print(f"  $ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False, **kwargs
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"command failed: {' '.join(cmd)}")
    return result


def dvc_cmd():
    # dvc's own console script may not be on PATH; invoke it as a module of the
    # same interpreter running this spike so it resolves the venv install.
    return [sys.executable, "-m", "dvc"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", default=None, help="scratch dir to use (created if absent)")
    parser.add_argument("--keep", action="store_true", help="don't delete scratch dir afterward")
    args = parser.parse_args()

    base = Path(args.workdir) if args.workdir else Path(tempfile.mkdtemp(prefix="mlops-spike05-"))
    base.mkdir(parents=True, exist_ok=True)
    repo = base / "producer-repo"
    shared_cache = base / "shared-dvc-cache"
    worktree = base / "producer-repo-experiment"

    try:
        print(f"Scratch dir: {base}")
        repo.mkdir()
        run(["git", "init", "-q"], cwd=repo)
        run(["git", "config", "user.email", "spike@example.com"], cwd=repo)
        run(["git", "config", "user.name", "spike"], cwd=repo)

        run(dvc_cmd() + ["init", "-q"], cwd=repo)
        run(dvc_cmd() + ["cache", "dir", "--local", str(shared_cache)], cwd=repo)

        dataset_dir = repo / "dataset"
        dataset_dir.mkdir()
        data_file = dataset_dir / "data.bin"
        data_file.write_bytes(b"spike-05-payload" * 1024)

        run(dvc_cmd() + ["add", "dataset/data.bin", "-q"], cwd=repo)
        run(["git", "add", "dataset/data.bin.dvc", "dataset/.gitignore"], cwd=repo)
        run(["git", "commit", "-q", "-m", "add dataset via dvc"], cwd=repo)

        cache_files_before = list(shared_cache.rglob("*"))
        cache_object_count_before = sum(1 for p in cache_files_before if p.is_file())
        print(f"Shared cache holds {cache_object_count_before} object file(s) after first add.")

        run(["git", "worktree", "add", str(worktree), "-q", "HEAD"], cwd=repo)
        run(dvc_cmd() + ["cache", "dir", "--local", str(shared_cache)], cwd=worktree)
        run(dvc_cmd() + ["checkout", "-q"], cwd=worktree)

        worktree_data = (worktree / "dataset" / "data.bin").read_bytes()
        original_data = data_file.read_bytes()
        assert worktree_data == original_data, "worktree checkout did not match source data"

        cache_files_after = list(shared_cache.rglob("*"))
        cache_object_count_after = sum(1 for p in cache_files_after if p.is_file())
        print(f"Shared cache holds {cache_object_count_after} object file(s) after worktree checkout.")

        assert cache_object_count_after == cache_object_count_before, (
            "checking out the same content in a second worktree should not add new "
            "cache objects — content-addressed cache should have deduped it"
        )

        print("PASS: worktree checked out identical bytes with zero new cache objects.")
        print("PASS: git worktree + shared DVC cache dedup confirmed locally.")

    finally:
        if not args.keep:
            # release the worktree registration before deleting its directory
            try:
                run(["git", "worktree", "remove", "--force", str(worktree)], cwd=repo)
            except Exception:
                pass
            shutil.rmtree(base, ignore_errors=True)
            print(f"Cleaned up scratch dir: {base}")
        else:
            print(f"Left scratch dir in place: {base}")


if __name__ == "__main__":
    main()
