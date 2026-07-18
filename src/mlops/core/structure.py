from pathlib import Path


class StructureError(Exception):
    pass


def _looks_like_junk(path):
    if "__pycache__" in path.parts:
        return True
    if path.suffix == ".ipynb":
        return True
    if "checkpoint" in path.name.lower():
        return True
    return False


def check_structure(pkl_path, format_dirs, require_formats=()):
    pkl_path = Path(pkl_path)
    format_dirs = [Path(d) for d in format_dirs]

    if not format_dirs:
        raise StructureError(
            "pkl-only bundles are refused - consumers deploy converted formats, "
            "not pickles. Provide at least one --formats <dir>."
        )

    provided = {d.name for d in format_dirs}
    missing = set(require_formats) - provided
    if missing:
        raise StructureError(
            f"Missing required format(s): {', '.join(sorted(missing))}. "
            f"mlops.yaml requires {sorted(require_formats)}."
        )

    all_files = [pkl_path] + [f for d in format_dirs for f in d.rglob("*") if f.is_file()]
    junk = [str(f) for f in all_files if _looks_like_junk(f)]
    if junk:
        raise StructureError(f"Junk files in bundle: {junk}. Remove before publishing.")
