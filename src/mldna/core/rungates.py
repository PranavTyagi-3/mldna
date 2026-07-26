import mlflow

REQUIRED_TAGS = ["git_commit", "dvc_data_hash", "user"]


class RunGateError(Exception):
    pass


def check_run_gates(run_id):
    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)

    if run.info.status != "FINISHED":
        raise RunGateError(
            f"Run {run_id} did not complete successfully (status={run.info.status}). "
            f"Only completed runs can be published."
        )

    if run.data.tags.get("dirty") == "true":
        raise RunGateError(
            f"Run {run_id} was trained from a dirty working tree and can never be "
            f"published (dirty=true is permanent). Retrain from a committed state."
        )

    if "registry_version" in run.data.tags:
        raise RunGateError(
            f"Run {run_id} was already published as version "
            f"{run.data.tags['registry_version']} (by "
            f"{run.data.tags.get('published_by', '?')} at "
            f"{run.data.tags.get('published_at', '?')}). Publishing the same run "
            f"twice is refused - the pkl you pointed at has identical content to "
            f"what was already published, so there's nothing new to publish. "
            f"Fix: retrain inside a fresh `mldna.track_run()` block so the saved "
            f"model has new content, then publish that new run."
        )

    missing = [t for t in REQUIRED_TAGS if t not in run.data.tags]
    if missing:
        raise RunGateError(
            f"Run {run_id} is missing required lineage tag(s): {', '.join(missing)}."
        )

    artifact_paths = {a.path for a in client.list_artifacts(run_id, "env")}
    if "env/requirements.frozen.txt" not in artifact_paths:
        raise RunGateError(
            f"Run {run_id} has no captured environment "
            f"(env/requirements.frozen.txt missing) - was this run created by "
            f"mldna.track_run()?"
        )

    return run
