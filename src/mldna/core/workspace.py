from pathlib import Path

from mldna.core import discovery, dvc, git
from mldna.core.config import ProjectConfig
from mldna.core.lock import RunLock


class WorkspaceError(Exception):
    pass


def _main_worktree(dir):
    try:
        return git.worktree_list(dir)[0].resolve()
    except git.GitWorktreeError as e:
        raise WorkspaceError(
            f"`mldna workspace` needs a git repository, and {Path(dir).resolve()} is not "
            f"inside one (git said: {e}). Workspaces are git worktrees of the repo. "
            f"Fix: run it from inside your project, or pass --dir <project>."
        )


def _path_for(main, name):
    if not name or name in (".", "..") or name != Path(name).name or "\\" in name:
        raise WorkspaceError(
            f"Workspace name '{name}' is not a plain name. The worktree is created as a "
            f"sibling directory '{main.name}-<name>', so a name with path parts could "
            f"land outside it. Fix: use a simple name such as `mldna workspace new lr-sweep`."
        )
    return main.parent / f"{main.name}-{name}"


def new(dir, name) -> Path:
    main = _main_worktree(dir)
    path = _path_for(main, name)
    if path.exists() or path in [p.resolve() for p in git.worktree_list(main)]:
        raise WorkspaceError(
            f"Workspace '{name}' already exists at {path}. Each parallel run needs its own "
            f"directory, so a name cannot be reused. Fix: pick another name, or "
            f"`mldna workspace remove {name}` first."
        )
    try:
        git.worktree_add(main, path)
    except git.GitWorktreeError as e:
        raise WorkspaceError(
            f"git could not create the worktree at {path}: {e}. "
            f"Fix: resolve the git error above, then re-run `mldna workspace new {name}`."
        )
    if (path / ".dvc").is_dir():
        # Same cache dir as the main worktree: the cache is content-addressed, so
        # sharing it means no dataset duplication and an instant `dvc pull`.
        cache = dvc.run(["cache", "dir"], cwd=main).stdout.strip()
        dvc.run(["cache", "dir", "--local", str(Path(cache).resolve())], cwd=path)
        # A fresh worktree has the pointers but not the data (D#74).
        result = dvc.run(["checkout"], cwd=path, check=False)
        if result.returncode == 0:
            print(f"data checked out in {path}")
        else:
            detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
            print(f"data not checked out: {detail}. Fix: run `dvc pull` in {path}")
    return path


def list_workspaces(dir) -> list:
    """(name, path) per worktree; the main worktree is named '(main)'."""
    main = _main_worktree(dir)
    prefix = f"{main.name}-"
    return [
        ("(main)" if p.resolve() == main else p.name.removeprefix(prefix), p.resolve())
        for p in git.worktree_list(main)
    ]


def _unrestorable(path) -> list:
    """Model-shaped ignored files in the worktree (the .dvc/cache is excluded), plus its mlruns note."""
    try:
        ignored = git.ignored_files(path)
    except git.GitStatusError as e:
        raise WorkspaceError(
            f"Could not check {path} for unpublished models: {e}. Removing it could "
            f"delete work git cannot restore. Fix: resolve the git error above, then retry."
        )
    ignored = [f for f in ignored if not f.startswith(".dvc/cache/")]
    lost = discovery.filter_by_globs(ignored, ProjectConfig.load(path).model_globs)
    if (path / "mlruns").is_dir():
        lost.append("mlruns/ (this workspace's local MLflow store: its runs and registered versions)")
    return lost


def remove(dir, name, confirm=lambda msg: False) -> bool:
    """Removes the worktree. Returns True if it carried a stale (dead-PID) lock.

    Gitignored models and a local mlruns/ die with the worktree, so confirm(msg)
    must approve before anything is deleted; nothing to lose means no prompt."""
    main = _main_worktree(dir)
    if name == main.name:
        raise WorkspaceError(
            f"'{name}' is the main working copy, not a workspace; removing it would delete "
            f"the repo itself. Fix: `mldna workspace list` shows what can be removed."
        )
    path = _path_for(main, name)
    if path not in [p.resolve() for p in git.worktree_list(main)]:
        raise WorkspaceError(
            f"'{name}' is not a workspace of this repo (no worktree at {path}). "
            f"remove only deletes worktrees. Fix: `mldna workspace list` shows the names."
        )
    lock = RunLock(path)
    if lock.is_locked():
        raise WorkspaceError(
            f"Workspace '{name}' has a live run lock ({lock.path}). Deleting a directory "
            f"mid-run destroys the run's data and lineage (one directory, one execution). "
            f"Fix: wait for the run to finish, then `mldna workspace remove {name}`."
        )
    lost = _unrestorable(path)
    if lost:
        listing = "\n".join(f"  {f}" for f in lost)
        msg = (
            f"Removing workspace '{name}' would delete files git cannot restore "
            f"(they are gitignored, so they are in no commit):\n{listing}\n"
            f"Fix: copy them out first, or `mldna workspace remove {name} --yes` to delete them."
        )
        if not confirm(msg):
            raise WorkspaceError(msg)
    stale = lock.path.exists()
    try:
        git.worktree_remove(main, path)
    except git.GitWorktreeError as e:
        raise WorkspaceError(
            f"git refused to remove {path}: {e}. git only removes clean worktrees, so "
            f"uncommitted work is never lost silently. Fix: commit or discard the "
            f"changes in {path}, then `mldna workspace remove {name}`."
        )
    return stale
