from pathlib import Path

import yaml

from mlops.core import hashing


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
