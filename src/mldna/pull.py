import tempfile
from pathlib import Path

import mlflow

from mldna.core import git, ownership
from mldna.core import tags as tagschema
from mldna.core.config import ProjectConfig, missing_namespace_hint, missing_project_hint
from mldna.core.pull import PullError, pull as pull_code_and_data
from mldna.trackinguri import resolve_and_set_tracking_uri


def _published_runs_for_commit(client, experiment_id, commit):
    runs = client.search_runs([experiment_id], order_by=["attribute.start_time DESC"])
    return [
        run for run in runs
        if run.data.tags.get("git_commit") == commit and run.data.tags.get("registry_version")
    ]


def _download_bundle(workspace_dir, mv_source, model_name, version):
    # Never touch models/ itself - it holds the developer's own unpublished
    # experiments. Only the version subdirectory is ours to replace. See D#47.
    rel_dest = Path("models") / f"{model_name}-v{version}"
    label = f"pull of {model_name} v{version}"
    try:
        dest = ownership.validate_contained(workspace_dir, rel_dest, label)
    except ownership.OwnershipError as e:
        raise PullError(str(e)) from e

    with tempfile.TemporaryDirectory() as tmp:
        # Resolve from the registered version's own source URI, never a fixed
        # "runs:/<id>/models" path: publish uploads each attempt to a unique
        # subdirectory (D#59), so the fixed path is now that folder's PARENT.
        downloaded = mlflow.artifacts.download_artifacts(mv_source, dst_path=tmp)
        try:
            ownership.swap_into_dest(downloaded, dest, label)
        except ownership.OwnershipError as e:
            raise PullError(str(e)) from e

    ownership.write_marker(dest, model_name, version)
    return rel_dest


def pull(workspace_dir=".", select=None):
    workspace_dir = Path(workspace_dir)
    if not (workspace_dir / "mldna.yaml").exists():
        raise PullError(missing_project_hint(workspace_dir))

    pull_code_and_data(workspace_dir)

    result = {"model_pulled": False}

    config = ProjectConfig.load(workspace_dir)
    if not config.model_name:
        return result
    # The experiment is namespaced (D#58); looking it up by the bare name would
    # silently never find it, so refuse rather than report "no published model".
    if not config.namespace:
        raise PullError(missing_namespace_hint(workspace_dir))

    _, mode = resolve_and_set_tracking_uri(workspace_dir)
    print(f"Tracking mode: {mode}")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.full_model_name())
    if experiment is None:
        return result

    commit = git.get_commit_sha(workspace_dir)
    candidates = _published_runs_for_commit(client, experiment.experiment_id, commit)
    if not candidates:
        return result

    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        if select is None:
            raise PullError(
                f"{len(candidates)} published runs match commit {commit[:8]}: "
                f"{', '.join(r.info.run_id for r in candidates)}. "
                f"Cannot guess which one to pull - pass a `select` callback "
                f"(the CLI prompts interactively)."
            )
        chosen = select(candidates)

    version = chosen.data.tags.get("registry_version")
    try:
        mv = client.get_model_version(config.full_model_name(), version)
    except Exception as e:
        raise PullError(
            f"run {chosen.info.run_id} is tagged registry_version={version} but "
            f"that version is not in the registry ({e}). Fix: confirm the version "
            f"still exists, or re-publish this run."
        ) from e
    if tagschema.is_abandoned(mv.tags):
        raise PullError(
            f"{config.full_model_name()} v{version}, which run {chosen.info.run_id} "
            f"points at, is tagged abandoned "
            f"({mv.tags.get(tagschema.ABANDONED_REASON, 'no reason recorded')}) - it "
            f"was registered by a publish that lost a concurrent race and failed, so "
            f"it does not back this run. Fix: fetch the winning version "
            f"instead - pin it in a consumer's models.lock.yaml and run `mldna fetch`."
        )
    registered_id = client.get_registered_model(config.full_model_name()).tags.get(tagschema.MODEL_ID)
    if config.model_id and registered_id and registered_id != config.model_id:
        raise PullError(
            f"Registry model {config.full_model_name()} belongs to another project "
            f"(its model id is {registered_id}, this project's is {config.model_id}). "
            f"Names are user-chosen and can collide. Fix: if this is your project "
            f"and its model_id was lost or regenerated, restore it: set "
            f"project.model_id: {registered_id} in mldna.yaml and commit. Otherwise "
            f"change project.namespace or project.model_name in mldna.yaml."
        )
    dest = _download_bundle(workspace_dir, mv.source, config.full_model_name(), version)
    result["model_pulled"] = True
    result["run_id"] = chosen.info.run_id
    result["registry_version"] = version
    result["dest"] = str(dest)
    return result
