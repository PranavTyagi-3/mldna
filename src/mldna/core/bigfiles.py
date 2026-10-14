from pathlib import Path

import yaml

from mldna.core import discovery

# Below this, a model-shaped name is almost surely a config or asset file
# (pom.xml, logging.xml, a protobuf descriptor), not a model. The `large`
# signal has no floor.
MODEL_SIZE_FLOOR = 100 * 1024
MB = 1024 * 1024


def _mb(size):
    return f"{size / MB:.1f} MB"


def human_size(size):
    if size < 1024:
        return f"{size} B"
    if size < MB:
        return f"{size / 1024:.0f} KB"
    return _mb(size)


def pointer_totals(text):
    """(nfiles, size) summed over a .dvc pointer's outs; a single-file out has no nfiles."""
    outs = (yaml.safe_load(text) or {}).get("outs") or []
    return (
        sum(out.get("nfiles", 1) for out in outs),
        sum(out.get("size", 0) for out in outs),
    )


def is_sharp_drop(old, new):
    """True when new fell to zero from something, or by more than half."""
    return old > 0 and (new == 0 or new < old / 2)


def shrink_note(path, old_text, new_text):
    """A warning line when a pointer's file count or size dropped sharply, else None."""
    try:
        old_files, old_size = pointer_totals(old_text)
        new_files, new_size = pointer_totals(new_text)
    except (yaml.YAMLError, AttributeError, TypeError):
        return None
    if not (is_sharp_drop(old_files, new_files) or is_sharp_drop(old_size, new_size)):
        return None
    return (
        f"{path}: files {old_files} -> {new_files}, size {human_size(old_size)} -> "
        f"{human_size(new_size)} - the dataset would shrink sharply for everyone who pulls this"
    )


def risky_files(repo_dir, paths, model_globs, max_file_mb):
    """(path, reason) for each path that is model-shaped or large.

    Skips paths that no longer exist."""
    repo_dir = Path(repo_dir)
    model_shaped = set(discovery.filter_by_globs(paths, model_globs))
    risky = []
    for path in paths:
        full = repo_dir / path
        if not full.is_file():
            continue
        size = full.stat().st_size
        reasons = []
        if path in model_shaped and size >= MODEL_SIZE_FLOOR:
            reasons.append(f"model-shaped, {_mb(size)}")
        if size > max_file_mb * MB:
            reasons.append(f"larger than {max_file_mb} MB, {_mb(size)}")
        if reasons:
            risky.append((path, "; ".join(reasons)))
    return risky
