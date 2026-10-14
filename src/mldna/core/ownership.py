"""Guards for a directory mldna is about to replace wholesale.

fetch() and pull() both swap a staged directory into a destination the user
named in config (a lockfile dest, a models/ path). Both must refuse to
delete anything mldna did not create itself. See D#46.

The marker also doubles as fetch's idempotency token (D#38) and, per D#53,
as its *identity* token: a version number alone cannot tell "same model,
same version" apart from "lockfile repointed to a different model that
happens to share a version number", so the marker carries model name,
version, format and a manifest digest. Ownership, below, only ever checks
the marker's *presence* - its contents are identity's business, not
ownership's - so an old bare-version marker still proves ownership even
though it proves nothing about identity.
"""
import json
import shutil
import uuid
from pathlib import Path

MARKER_NAME = ".mldna-version"


class OwnershipError(Exception):
    pass


def read_marker(dest):
    """Returns the marker's identity dict, or None if dest carries none.

    A marker written before D#53 is a bare version string, not JSON. It
    parses here as {"version": <string>, "legacy": True}: enough to report
    a version, but flagged so callers never mistake it for a full identity
    match - it carries no model name, format or digest to compare.
    """
    marker = Path(dest) / MARKER_NAME
    if not marker.exists():
        return None
    raw = marker.read_text().strip()
    try:
        data = json.loads(raw)
    except ValueError:
        return {"version": raw, "legacy": True}
    if not isinstance(data, dict):
        return {"version": raw, "legacy": True}
    return data


def write_marker(dest, model_name, version, format=None, manifest_digest=None, files=None):
    data = {"model_name": model_name, "version": str(version)}
    if format is not None:
        data["format"] = format
    if manifest_digest is not None:
        data["manifest_digest"] = manifest_digest
    if files is not None:
        data["files"] = sorted(files)
    (Path(dest) / MARKER_NAME).write_text(json.dumps(data))


def require_no_foreign_files(dest, known_files, label):
    """D#79: refuse to replace a marked dest holding files mldna did not put
    there. known_files are the bundle's relpaths; the marker is mldna's too."""
    dest = Path(dest)
    if not dest.is_dir():
        return
    known = {*known_files, MARKER_NAME}
    foreign = sorted(
        p.relative_to(dest).as_posix() for p in dest.rglob("*")
        if p.is_file() and p.relative_to(dest).as_posix() not in known
    )
    if not foreign:
        return
    listed = ", ".join(foreign[:10])
    if len(foreign) > 10:
        listed += f", ... and {len(foreign) - 10} more"
    raise OwnershipError(
        f"{label} dest '{dest}' contains files mldna did not put there: {listed}. "
        f"Replacing the folder would delete them. "
        f"Fix: move them out of '{dest}' (keep app config next to, not inside, "
        f"the model folder), then re-run mldna fetch."
    )


def validate_contained(workspace_dir, dest, label):
    """Refuse a dest that is absolute, or that resolves outside - or exactly
    onto - workspace_dir. Returns the resolved absolute dest Path.

    Resolving both sides before comparing (rather than a string prefix check)
    is what catches '..', symlink tricks, and 'dest: .' alike.
    """
    workspace_dir = Path(workspace_dir).resolve()
    raw = Path(dest)
    if raw.is_absolute():
        raise OwnershipError(
            f"{label} has dest '{dest}', which is an absolute path. mldna "
            f"only ever replaces a directory it can prove sits inside the "
            f"workspace, so an absolute dest is refused before touching "
            f"anything. Fix: change dest to a path relative to the workspace "
            f"root, e.g. 'app/models/<name>'."
        )

    resolved = (workspace_dir / raw).resolve()
    if resolved == workspace_dir or not resolved.is_relative_to(workspace_dir):
        raise OwnershipError(
            f"{label} has dest '{dest}', which resolves to {resolved} - "
            f"outside the workspace root {workspace_dir}, or the workspace "
            f"root itself. mldna refuses to delete anything there. "
            f"Fix: point dest at a path below the workspace root, not '.' "
            f"and not a '..'-escaping path."
        )
    return resolved


def is_safe_to_replace(dest):
    """True if dest is absent, empty, or carries the marker mldna itself
    writes - i.e. nothing is lost by replacing it."""
    dest = Path(dest)
    if not dest.exists():
        return True
    if not any(dest.iterdir()):
        return True
    return (dest / MARKER_NAME).exists()


def require_owned_or_absent(dest, label):
    if is_safe_to_replace(dest):
        return
    raise OwnershipError(
        f"{label} dest '{dest}' already exists, is not empty, and has no "
        f"{MARKER_NAME} marker - mldna did not create this directory and "
        f"refuses to delete files it did not put there. "
        f"Fix: point dest at an empty directory or one mldna already owns, "
        f"or move your own files out of '{dest}' first."
    )


def swap_into_dest(staged_dir, dest, label):
    """Replace dest with staged_dir, refusing unless dest is ours to delete,
    and never leaving dest observably absent or half-written.

    rmtree-then-move has a window where a crash leaves dest empty or
    partially written - an app loading it then sees nothing, or a truncated
    model. Instead: move the existing dest aside (a rename, not a copy, so
    it cannot itself be interrupted mid-write), move the staged directory
    into place, and only then delete the old one. If the move-in fails,
    the old directory is put back, so dest is never left missing.
    """
    staged_dir = Path(staged_dir)
    dest = Path(dest)
    require_owned_or_absent(dest, label)
    dest.parent.mkdir(parents=True, exist_ok=True)

    old_aside = None
    if dest.exists():
        old_aside = dest.with_name(f"{dest.name}.mldna-old-{uuid.uuid4().hex}")
        dest.rename(old_aside)

    try:
        shutil.move(str(staged_dir), str(dest))
    except Exception:
        if old_aside is not None:
            old_aside.rename(dest)
        raise

    if old_aside is not None:
        shutil.rmtree(old_aside)
