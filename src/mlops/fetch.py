import shutil
import tempfile
from pathlib import Path

import mlflow
import yaml

from mlops.core.lockfile import load_lockfile
from mlops.core.manifest import resolve_format, verify_format_files
from mlops.trackinguri import resolve_and_set_tracking_uri

MARKER_NAME = ".mlops-version"


class FetchError(Exception):
    pass


def _read_marker(dest):
    marker = Path(dest) / MARKER_NAME
    if not marker.exists():
        return None
    return marker.read_text().strip()


def _write_marker(dest, version):
    (Path(dest) / MARKER_NAME).write_text(str(version))


def _stage_download(mv_source, format_entry, tmp_dir):
    path = format_entry["path"].rstrip("/")
    downloaded = mlflow.artifacts.download_artifacts(f"{mv_source}/{path}", dst_path=tmp_dir)
    if "files" in format_entry:
        # a directory subtree already lands with its files at the root - stage as-is
        return Path(downloaded)
    # a single file: stage it inside its own dir so dest/<name> is what lands
    staged = Path(tempfile.mkdtemp(dir=tmp_dir))
    shutil.copy2(downloaded, staged / Path(path).name)
    return staged


def _swap_into_dest(staged_dir, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.move(str(staged_dir), str(dest))


def fetch(workspace_dir="."):
    workspace_dir = Path(workspace_dir)
    _, mode = resolve_and_set_tracking_uri(workspace_dir)
    print(f"Tracking mode: {mode}")
    entries = load_lockfile(workspace_dir)
    client = mlflow.tracking.MlflowClient()

    results = []
    for entry in entries:
        try:
            mv = client.get_model_version(entry.model_name, entry.version)
        except Exception as e:
            raise FetchError(
                f"{entry.model_name} v{entry.version} not found in the registry. "
                f"Fix: check models.lock.yaml, or confirm the version was published. ({e})"
            ) from e

        if mv.tags.get("recalled") == "true":
            raise FetchError(
                f"{entry.model_name} v{entry.version} has been recalled - refusing to fetch. "
                f"Fix: pin a different version in models.lock.yaml; contact the model owner."
            )

        with tempfile.TemporaryDirectory() as tmp:
            try:
                manifest_path = mlflow.artifacts.download_artifacts(
                    f"{mv.source}/manifest.yaml", dst_path=tmp
                )
            except Exception as e:
                raise FetchError(
                    f"could not download manifest.yaml for {entry.model_name} v{entry.version}: {e}"
                ) from e
            manifest_dict = yaml.safe_load(Path(manifest_path).read_text())
            format_entry = resolve_format(manifest_dict, entry.format)

            dest = workspace_dir / entry.dest
            if _read_marker(dest) == entry.version:
                results.append({"model_name": entry.model_name, "version": entry.version, "skipped": True})
                continue

            staged = _stage_download(mv.source, format_entry, tmp)
            verify_format_files(staged, format_entry)
            _swap_into_dest(staged, dest)

        _write_marker(dest, entry.version)
        results.append({"model_name": entry.model_name, "version": entry.version, "skipped": False})

    return results
