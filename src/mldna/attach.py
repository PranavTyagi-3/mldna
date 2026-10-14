import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mldna.core import tags
from mldna.core.config import ProjectConfig, missing_namespace_hint, missing_project_hint
from mldna.trackinguri import resolve_and_set_tracking_uri


class AttachError(Exception):
    pass


def _find_marker(workspace_dir, run_id):
    """The record for `run_id` (any unambiguous prefix) from .mldna/runs/."""
    runs_dir = workspace_dir / ".mldna" / "runs"
    saved = list(runs_dir.glob("*.json")) if runs_dir.is_dir() else []
    matches = [p for p in saved if p.stem.startswith(run_id)]
    if len(matches) != 1:
        # Ambiguous: show the matches. No match: show what is saved. Newest
        # first and capped, since a busy project saves hundreds of records.
        shown = sorted(matches or saved, key=lambda p: p.stat().st_mtime, reverse=True)
        lines = [
            f"  {p.stem}  saved {datetime.fromtimestamp(p.stat().st_mtime):%Y-%m-%d %H:%M}"
            for p in shown[:10]
        ]
        if len(shown) > 10:
            lines.append(f"  ... and {len(shown) - 10} more in {runs_dir}")
        listing = "\n".join(lines) if lines else "  (none saved)"
        heading = "matching records" if matches else f"newest of {len(saved)} saved records"
        raise AttachError(
            f"--run-id {run_id} matches {len(matches)} saved records in {runs_dir}. "
            f"The {heading}:\n{listing}\n"
            f"Fix: pass enough characters of one saved run id."
        )
    return matches[0]


def attach(workspace_dir=".", run_id=None):
    """Re-binds a local .mldna/last_run.json record to its run (issue #10): a
    network blip at run exit can leave a run RUNNING forever and its model
    hashes un-tagged, even though track_run already saved them locally. This
    writes the missing tags and terminates the run, labeling it `attached`
    so it is never mistaken for a run that reported itself cleanly.
    """
    workspace_dir = Path(workspace_dir)

    config_path = workspace_dir / "mldna.yaml"
    if not config_path.exists():
        raise AttachError(missing_project_hint(workspace_dir))
    config = ProjectConfig.from_yaml(config_path)
    if not config.model_name:
        raise AttachError(
            "mldna.yaml has no project.model_name set, so there's nothing to "
            "attribute this run to. Fix: set project.model_name in mldna.yaml."
        )
    if not config.namespace:
        raise AttachError(missing_namespace_hint(workspace_dir))

    if run_id:
        marker_path = _find_marker(workspace_dir, run_id)
    else:
        marker_path = workspace_dir / ".mldna" / "last_run.json"
    if not marker_path.exists():
        raise AttachError(
            f"No {marker_path} found, so there is no local record to attach. "
            f"attach only recovers a run that already ran inside `with "
            f"mldna.track_run():` and saved a record there - there is nothing "
            f"to recover without it."
        )
    marker = json.loads(marker_path.read_text())
    run_id = marker.get("run_id")
    if not run_id:
        raise AttachError(
            f"{marker_path} has no run_id recorded, so attach cannot tell "
            f"which run to re-bind to. Fix: re-train inside `with "
            f"mldna.track_run():` to produce a fresh record."
        )

    _, mode = resolve_and_set_tracking_uri(workspace_dir)
    print(f"attach: tracking mode {mode}")
    client = mlflow.tracking.MlflowClient()

    try:
        run = client.get_run(run_id)
    except Exception as e:
        raise AttachError(
            f"Run {run_id} (from {marker_path}) could not be found on the "
            f"tracking server ({type(e).__name__}: {e}). attach only re-binds "
            f"a local record to a run the server already knows about. Fix: "
            f"check the tracking server is reachable with `mldna doctor`; if "
            f"the run genuinely never started there, there is nothing to "
            f"recover."
        ) from e

    if run.data.tags.get(tags.GIT_COMMIT) != marker.get("git_commit"):
        raise AttachError(
            f"Run {run_id}'s recorded commit ({run.data.tags.get(tags.GIT_COMMIT)!r}) "
            f"does not match the local record's commit ({marker.get('git_commit')!r}). "
            f"attach refuses to bind a record to a run it cannot confirm is "
            f"the same one - doing so would mislabel that run's lineage. Fix: "
            f"check that {marker_path} is the record for run {run_id}; if it "
            f"isn't, there is nothing attach can safely do here."
        )

    checksums = marker.get("checksums") or {}
    tagged = sum(1 for k in run.data.tags if k.startswith("model_sha256__"))
    if run.info.status == "FINISHED" and tagged >= len(checksums):
        raise AttachError(
            f"Run {run_id} is already FINISHED and already carries its model "
            f"checksum tags - there is nothing for attach to do."
        )

    for i, (relpath, sha256) in enumerate(checksums.items()):
        client.set_tag(run_id, tags.checksum_tag_name(i), tags.checksum_tag_value(sha256, relpath))
    if checksums:
        client.set_tag(run_id, tags.SAVE_CONTEXT, tags.IN_RUN)

    # Permanent label, same pattern as dirty=true/save_context=post_run
    # (D#10): this run's FINISHED status and/or checksum tags were written
    # after the fact, not when they became true, so it must never look like
    # a run that reported itself cleanly (invariant 9).
    client.set_tag(run_id, tags.ATTACHED, "true")
    client.set_tag(run_id, tags.ATTACHED_AT, datetime.now(timezone.utc).isoformat())

    if run.info.status not in ("FINISHED", "FAILED"):
        client.set_terminated(run_id, status=marker.get("status", "FINISHED"))

    return {"run_id": run_id, "checksums_written": len(checksums)}
