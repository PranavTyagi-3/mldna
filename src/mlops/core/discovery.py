import fnmatch
from pathlib import Path

DEFAULT_EXCLUDES = ["dataset/", ".git/", ".dvc/cache/", ".mlops/", "outputs/"]

DEFAULT_MODEL_GLOBS = [
    "*.pkl", "*.pt", "*.pth", "*.h5", "*.joblib",
    "*.safetensors", "*.onnx", "*.bin", "*.xml",
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
