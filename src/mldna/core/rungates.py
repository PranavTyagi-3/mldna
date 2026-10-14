import mlflow

from mldna.core import tags as tagschema

REQUIRED_TAGS = ["git_commit", "user"]


class RunGateError(Exception):
    pass


def check_run_gates(run_id, pkl_sha256=None):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise RunGateError(
            f"Run {run_id} did not complete successfully (status={run.info.status}). "
            f"Only completed runs can be published - a run that failed or was "
            f"killed may have half-written artifacts, so its lineage does not "
            f"describe a finished model. Fix: re-train inside "
            f"`with mldna.track_run():` and publish the run that completes."
        )

    if run.data.tags.get("dirty") == "true":
        raise RunGateError(
            f"Run {run_id} was trained from a dirty working tree and can never be "
            f"published (dirty=true is permanent). A dirty run's recorded commit "
            f"does not describe the code that produced the file, so its lineage can "
            f"never be made true. Fix: commit or stash your changes, re-train "
            f"inside `with mldna.track_run():` on a clean tree, then publish the "
            f"model that run saves."
        )

    # One publish per model file per run, keyed by the file's hash (D#78).
    tags = run.data.tags
    per_file = [t for t in tags if t.startswith(tagschema.PUBLISHED_PREFIX)]
    if "registry_version" in tags and not per_file:
        raise RunGateError(
            f"Run {run_id} was already published as version "
            f"{tags['registry_version']} (by {tags.get('published_by', '?')} at "
            f"{tags.get('published_at', '?')}) by an older mldna that recorded no "
            f"per-file publish tag, so mldna cannot tell which file that version "
            f"holds and refuses rather than risk publishing the same bytes twice. "
            f"Per-file publishing applies to runs published from now on. Fix: "
            f"retrain inside a fresh `with mldna.track_run():` block, then publish "
            f"that new run."
        )
    published_version = tags.get(tagschema.published_tag_name(pkl_sha256)) if pkl_sha256 else None
    if published_version:
        raise RunGateError(
            f"Run {run_id} already published this exact file (SHA-256 "
            f"{pkl_sha256}) as version {published_version}. Publishing identical "
            f"bytes again would only mint a duplicate version. Fix: publish a "
            f"different model file, or retrain inside `with mldna.track_run():` "
            f"and publish that new run."
        )

    missing = [t for t in REQUIRED_TAGS if t not in run.data.tags]
    # The data hash is now one tag per DVC out, so "present" means at least one
    # of them - or the single legacy tag a pre-D#63 run carries (D#63).
    if not tagschema.data_hashes_from_tags(run.data.tags):
        missing.append("dvc_data_hash__<n>")
    if missing:
        raise RunGateError(
            f"Run {run_id} is missing required lineage tag(s): {', '.join(missing)}. "
            f"A published model must carry the commit, data hash and author that "
            f"produced it, or its registry entry cannot be traced back. Fix: "
            f"re-train inside `with mldna.track_run():`, which writes these tags, "
            f"then publish that run."
        )

    artifact_paths = {a.path for a in client.list_artifacts(run_id, "env")}
    if "env/requirements.frozen.txt" not in artifact_paths:
        raise RunGateError(
            f"Run {run_id} has no captured environment "
            f"(env/requirements.frozen.txt missing). A published model whose "
            f"environment was never recorded cannot be reproduced. Fix: re-train "
            f"inside `with mldna.track_run():`, which captures the environment, "
            f"then publish that run."
        )

    return run
