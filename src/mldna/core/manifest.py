from pathlib import Path

import yaml

from mldna.core import hashing


class ManifestError(Exception):
    pass


def _format_files(format_dir):
    format_dir = Path(format_dir)
    return {
        str(f.relative_to(format_dir).as_posix()): hashing.sha256_file(f)
        for f in format_dir.rglob("*") if f.is_file()
    }


def build_manifest(model_name, run, resolved, pkl_path, format_dirs,
                    conversions, published_by, published_at):
    pkl_path = Path(pkl_path)
    formats = {
        "pkl": {"path": pkl_path.name, "sha256": hashing.sha256_file(pkl_path)},
    }
    bundle_hashes = set()
    for format_dir in format_dirs:
        format_dir = Path(format_dir)
        files = _format_files(format_dir)
        formats[format_dir.name] = {"path": f"{format_dir.name}/", "files": files}
        bundle_hashes.update(files.values())

    used_conversions = {
        sha: entry for sha, entry in conversions.items() if sha in bundle_hashes
    }

    tags = run.data.tags
    return {
        "manifest_version": 1,
        "model_name": model_name,
        "source_run_id": resolved.run_id,
        "resolved_by": "content_hash",
        "save_context": tags.get("save_context", "in_run"),
        "published_by": published_by,
        "published_at": published_at,
        "formats": formats,
        "conversions": used_conversions,
        "lineage": {
            "git_commit": tags.get("git_commit"),
            "dvc_data_hash": tags.get("dvc_data_hash"),
        },
    }


def write_manifest(manifest, path):
    Path(path).write_text(yaml.dump(manifest, sort_keys=False))


def resolve_format(manifest, format_name):
    formats = manifest.get("formats", {})
    if format_name not in formats:
        available = ", ".join(sorted(formats)) or "none"
        raise ManifestError(
            f"{manifest.get('model_name', 'this version')} has formats [{available}]; "
            f"you requested '{format_name}'. "
            f"Fix: pin a format in models.lock.yaml that this version actually published."
        )
    return formats[format_name]


def _verify_one(file_path, expected_sha256, label):
    if not file_path.exists():
        raise ManifestError(
            f"expected file '{label}' missing from the downloaded bundle. "
            f"Fix: re-run mldna fetch; if this persists, contact the model publisher."
        )
    actual = hashing.sha256_file(file_path)
    if actual != expected_sha256:
        raise ManifestError(
            f"checksum mismatch for '{label}': expected {expected_sha256}, got {actual}. "
            f"Fix: re-run mldna fetch; if this persists, the registry artifact may be corrupted."
        )


def verify_format_files(local_dir, format_entry):
    local_dir = Path(local_dir)
    if "sha256" in format_entry:
        file_path = local_dir / Path(format_entry["path"]).name
        _verify_one(file_path, format_entry["sha256"], file_path.name)
        return
    for relpath, expected in format_entry.get("files", {}).items():
        _verify_one(local_dir / relpath, expected, relpath)
