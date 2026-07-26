import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mldna.core import commitcheck, hashing, identity, manifest, rungates, structure
from mldna.core.config import ProjectConfig
from mldna.core.conversions import verify_bundle
from mldna.core.resolve import resolve_run
from mldna.trackinguri import resolve_and_set_tracking_uri


class PublishError(Exception):
    pass


def _existing_conversions(run_id):
    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="conversions.json"
        )
        return json.loads(Path(path).read_text())
    except Exception:
        return {}


def _summary(model_name, run, resolved, published_by):
    tags = run.data.tags
    lines = [
        f"Matched run {resolved.run_id} (commit {tags.get('git_commit', '?')[:8]}, "
        f"user {tags.get('user', '?')}, save_context {tags.get('save_context', '?')})",
    ]
    if resolved.auto_resolved_note:
        lines.append(resolved.auto_resolved_note)
    lines.append(f"Publishing as {model_name}, published by {published_by}.")
    return "\n".join(lines)


def publish(pkl_path, workspace_dir=".", format_dirs=(), confirm=None):
    workspace_dir = Path(workspace_dir)
    pkl_path = Path(pkl_path)
    format_dirs = [Path(d) for d in format_dirs]

    config = ProjectConfig.load(workspace_dir)
    if not config.model_name:
        raise PublishError(
            "mldna.yaml has no project.model_name set, so there's no experiment "
            "to resolve the run from. Fix: set project.model_name in mldna.yaml."
        )
    model_name = config.model_name

    _, mode = resolve_and_set_tracking_uri(workspace_dir)
    print(f"Tracking mode: {mode}")

    resolved = resolve_run(model_name, pkl_path)
    run = rungates.check_run_gates(resolved.run_id)
    commitcheck.check_commit_reachable(workspace_dir, resolved.run_id)
    structure.check_structure(pkl_path, format_dirs, require_formats=config.require_formats)

    existing_conversions = _existing_conversions(resolved.run_id)
    verify_bundle(existing_conversions, format_dirs, hashing.sha256_file(pkl_path))

    published_by = identity.resolve_identity(workspace_dir)
    published_at = datetime.now(timezone.utc).isoformat()

    summary = _summary(model_name, run, resolved, published_by)
    print(summary)
    if confirm is not None and not confirm(summary):
        raise PublishError("Aborted.")

    manifest_dict = manifest.build_manifest(
        model_name, run, resolved, pkl_path, format_dirs,
        existing_conversions, published_by, published_at,
    )

    client = mlflow.tracking.MlflowClient()

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp) / "models"
        bundle_dir.mkdir()
        shutil.copy2(pkl_path, bundle_dir / pkl_path.name)
        for format_dir in format_dirs:
            shutil.copytree(format_dir, bundle_dir / format_dir.name)
        manifest.write_manifest(manifest_dict, bundle_dir / "manifest.yaml")

        # explicit run_id target, not the fluent API - mlflow.log_artifacts()
        # auto-starts (and leaks) an untracked run if none is active
        client.log_artifacts(resolved.run_id, str(bundle_dir), artifact_path="models")

    registered = mlflow.register_model(f"runs:/{resolved.run_id}/models", model_name)

    client.set_tag(resolved.run_id, "published_by", published_by)
    client.set_tag(resolved.run_id, "published_at", published_at)
    client.set_tag(resolved.run_id, "registry_version", registered.version)
    client.set_model_version_tag(model_name, registered.version, "formats",
                                  ",".join(d.name for d in format_dirs) or "pkl-only")
    client.set_model_version_tag(model_name, registered.version, "source_run", resolved.run_id)
    client.set_model_version_tag(model_name, registered.version, "published_by", published_by)

    return {"version": registered.version, "run_id": resolved.run_id, "model_name": model_name}
