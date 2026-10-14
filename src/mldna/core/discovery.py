import fnmatch
from pathlib import Path

DEFAULT_EXCLUDES = ["dataset/", ".git/", ".dvc/cache/", ".mldna/", "outputs/", "mlruns/"]

DEFAULT_MODEL_GLOBS = [
    "*.pkl", "*.pt", "*.pth", "*.h5", "*.joblib",
    "*.safetensors", "*.onnx", "*.bin", "*.xml", "*.keras",
    "*.cbm", "*.ubj", "*.ckpt", "*.pb",
]


def _is_excluded(relpath, excludes):
    parts = Path(relpath).as_posix()
    return any(parts.startswith(prefix) for prefix in excludes)


def snapshot(root, excludes=DEFAULT_EXCLUDES):
    root = Path(root)
    result = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relpath = path.relative_to(root).as_posix()
        if _is_excluded(relpath, excludes):
            continue
        stat = path.stat()
        result[relpath] = (stat.st_size, stat.st_mtime)
    return result


def diff(before, after):
    changed = []
    for relpath, meta in after.items():
        if before.get(relpath) != meta:
            changed.append(relpath)
    return changed


def filter_by_globs(paths, globs=DEFAULT_MODEL_GLOBS):
    return [p for p in paths if any(fnmatch.fnmatch(p, g) for g in globs)]


def find_excluded_models(root, excludes=DEFAULT_EXCLUDES, globs=DEFAULT_MODEL_GLOBS):
    """Model-shaped files sitting under an excluded directory (e.g. outputs/) -
    snapshot() never sees these, so a caller finding no model should explain
    why instead of letting it read as user error."""
    root = Path(root)
    found = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        relpath = path.relative_to(root).as_posix()
        if _is_excluded(relpath, excludes) and filter_by_globs([relpath], globs):
            found.append(relpath)
    return found


def excluded_model_hint(root, excludes=DEFAULT_EXCLUDES, globs=DEFAULT_MODEL_GLOBS):
    found = find_excluded_models(root, excludes, globs)
    if not found:
        return None
    relpath = found[0]
    excluded_dir = relpath.split("/")[0] + "/"
    return (
        f"found {relpath} in {excluded_dir}, which is excluded from scanning. "
        f"Fix: save models under models/ and re-train, so the run sees the "
        f"file saved inside a tracked run."
    )
