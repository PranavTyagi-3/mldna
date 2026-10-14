import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def run(args, cwd, check=True):
    return subprocess.run(
        [sys.executable, "-m", "dvc", *args], cwd=cwd,
        check=check, capture_output=True, text=True,
    )


POINTER_NAMES = ("dvc.lock", "dvc.yaml")

# Directories a pointer file is never inside, pruned so the walk below stays
# cheap. Tracked outs are pruned separately, by their own pointer (see
# find_pointer_files).
_PRUNE_DIRS = {
    ".git", ".dvc", ".mldna", "mlruns", "outputs", "models",
    ".venv", "venv", "node_modules", "__pycache__",
}
_MAX_DEPTH = 3


def find_pointer_files(workspace_dir, max_depth=_MAX_DEPTH):
    """Every DVC pointer file (`*.dvc`, `dvc.lock`, `dvc.yaml`) at or below the
    workspace root - the full scope of what this workspace claims about data.

    Bounded so it never walks the data itself: `dvc add` writes the pointer
    *beside* the out (`dataset/` -> `dataset.dvc`) and a pointer is never
    inside a tracked out, so any directory with a sibling `<name>.dvc` is
    pruned. That keeps a multi-GB `dataset/` out of the walk entirely; the
    depth bound is a second floor for untracked junk directories.
    """
    root = Path(workspace_dir)
    found = []

    def walk(directory, depth):
        try:
            entries = sorted(directory.iterdir())
        except OSError:
            return
        names = {e.name for e in entries}
        for entry in entries:
            if entry.is_dir():
                pruned = entry.name in _PRUNE_DIRS or f"{entry.name}.dvc" in names
                if not pruned and depth < max_depth:
                    walk(entry, depth + 1)
            elif entry.name.endswith(".dvc") or entry.name in POINTER_NAMES:
                found.append(entry)

    walk(root, 0)
    return found


def _pointer_outs(path):
    data = yaml.safe_load(path.read_text()) or {}
    if path.name == "dvc.lock":
        stages = data.get("stages") or {}
        outs = [o for stage in stages.values() for o in (stage.get("outs") or [])]
    else:
        outs = data.get("outs") or []
    return [o for o in outs if isinstance(o, dict)]


def read_data_hashes(workspace_dir):
    """`{out path relative to the workspace: md5}` across every pointer file,
    sorted by path so the per-out tag indices track_run writes are stable.

    Outs with no recorded md5 (a `dvc.yaml` stage declaration, which carries
    no hash until `dvc repro` writes `dvc.lock`) are absent rather than null:
    there is no hash to record, and D#45's gate refuses on that drift anyway.
    """
    root = Path(workspace_dir)
    hashes = {}
    for pointer in find_pointer_files(root):
        for out in _pointer_outs(pointer):
            md5 = out.get("md5")
            if not md5 or not out.get("path"):
                continue
            relpath = os.path.relpath(pointer.parent / out["path"], root)
            hashes[Path(relpath).as_posix()] = md5
    return dict(sorted(hashes.items()))


class DvcStatusError(Exception):
    pass


def _tracks_data(path):
    # MUST stay at least as wide as read_data_hashes' discovery (D#63): if a
    # pointer deeper than the root can contribute a recorded hash while this
    # skip returns False, the run records data the gate never checked - the
    # exact fail-open D#45 exists to close. find_pointer_files is therefore
    # the shared scope, and `.dvc/` only widens it further.
    return bool(find_pointer_files(path)) or (Path(path) / ".dvc").exists()


def data_status(workspace_dir):
    """Workspace-vs-pointer status: {} when every tracked out matches its record.

    Raises DvcStatusError when the answer cannot be determined - callers must
    treat that as "not known to be clean", never as clean.
    """
    # Skipped only when this workspace tracks nothing of its own. The test is
    # local pointers, not a local .dvc, because `dvc status` walks UP to the
    # governing repo: a project nested in a DVC repo keeps its pointers beside
    # its data while .dvc stays at the parent, and skipping on a missing .dvc
    # there would read a stale pointer as clean - a fail-open inside the fix
    # for a fail-open. The walk-up is also why "tracks nothing locally" has to
    # skip rather than delegate: with no pointers of its own there is nothing
    # to check, and asking dvc would report on whatever unrelated ancestor
    # repository happens to sit above this directory.
    if not _tracks_data(Path(workspace_dir)):
        return {}

    # `dvc status` is informational: it exits 0 whether or not anything
    # changed (only `--quiet` encodes the answer in the exit code), so the
    # exit code here means strictly "did the command work". --json carries
    # the answer - {} is in-sync, any entry is drift - and a real failure
    # (not a repo, permissions, git/dvc ownership) prints nothing parseable
    # and exits non-zero. Both are raised, never read as clean.
    result = run(["status", "--json"], cwd=workspace_dir, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise DvcStatusError(detail)
    try:
        return json.loads(result.stdout or "{}")
    except ValueError as e:
        raise DvcStatusError(f"unparseable `dvc status --json` output: {e}") from e


def _empty_on_disk(path):
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def split_status(status, workspace_dir=None):
    """(missing, changed) target names from a data_status result. A target is
    missing when every difference is an out deleted from the working tree (restore
    with `dvc checkout`); anything else - modified, new, changed deps - is changed
    (the pointers need re-recording) (D#74). dvc reports an out whose folder
    exists but is empty as "modified", so with workspace_dir an out that is
    absent or empty on disk also counts as deleted."""
    missing, changed = [], []
    for target, entries in status.items():
        states = []
        for entry in entries if isinstance(entries, list) else [entries]:
            if isinstance(entry, dict):
                for kind, outs in entry.items():
                    if kind == "changed outs" and isinstance(outs, dict):
                        for out, state in outs.items():
                            if (state == "modified" and workspace_dir is not None
                                    and _empty_on_disk((Path(workspace_dir) / target).parent / out)):
                                state = "deleted"
                            states.append(state)
                    else:
                        states.append(kind)
            else:
                states.append(str(entry))
        (missing if states and all(s == "deleted" for s in states) else changed).append(target)
    return sorted(missing), sorted(changed)


def cloud_status(workspace_dir):
    """Local-cache-vs-remote presence: {} when every tracked out is already on
    the configured remote (`dvc status --cloud --json`). Mirrors data_status's
    workspace-vs-pointer check but looks the other direction - together they
    answer "is the pointer honest" and "can anyone but me retrieve what it
    points to" (D#56).

    Raises DvcStatusError when presence can't be determined (no remote
    configured, unreachable, unparseable output) - callers must treat that as
    "not known to be present", never as present.
    """
    result = run(["status", "--cloud", "--json"], cwd=workspace_dir, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip() or f"exit code {result.returncode}"
        raise DvcStatusError(detail)
    try:
        return json.loads(result.stdout or "{}")
    except ValueError as e:
        raise DvcStatusError(f"unparseable `dvc status --cloud --json` output: {e}") from e
