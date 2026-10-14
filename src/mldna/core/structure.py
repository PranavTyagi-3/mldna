from pathlib import Path


class StructureError(Exception):
    pass


_CHECKPOINT_DIR_NAMES = {"checkpoint", "checkpoints"}


def _looks_like_junk(path):
    if "__pycache__" in path.parts:
        return True
    if path.suffix == ".ipynb":
        return True
    # Directory component, not a filename substring - "best_checkpoint.pkl" is
    # a legitimate model name, but anything under a checkpoints/ dir is scratch.
    if any(part.lower() in _CHECKPOINT_DIR_NAMES for part in path.parts[:-1]):
        return True
    return False


def check_structure(pkl_path, format_dirs, require_formats=()):
    pkl_path = Path(pkl_path)
    format_dirs = [Path(d) for d in format_dirs]

    provided = {d.name for d in format_dirs}
    missing = set(require_formats) - provided
    if missing:
        raise StructureError(
            f"Missing required format(s): {', '.join(sorted(missing))}. "
            f"mldna.yaml requires {sorted(require_formats)} so every consumer can "
            f"pin a format this model always publishes. Fix: re-run the conversion "
            f"inside `mldna.track_convert()` to produce the missing format(s), or "
            f"drop them from publish.require_formats in mldna.yaml."
        )

    all_files = [pkl_path] + [f for d in format_dirs for f in d.rglob("*") if f.is_file()]
    junk = [str(f) for f in all_files if _looks_like_junk(f)]
    if junk:
        raise StructureError(
            f"Junk files in bundle: {junk}. A published bundle must contain only "
            f"model files, so consumers never receive notebooks or caches. "
            f"Fix: delete these files, then re-run mldna publish."
        )
