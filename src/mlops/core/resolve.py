from dataclasses import dataclass

import mlflow

from mlops.core import hashing


class ResolveError(Exception):
    pass


@dataclass
class ResolvedRun:
    run_id: str
    relpath: str
    auto_resolved_note: str = None


def _matching_relpath(run, target_hash):
    for tag_name, tag_value in run.data.tags.items():
        if not tag_name.startswith("model_sha256__"):
            continue
        sha256, _, relpath = tag_value.partition(":")
        if sha256 == target_hash:
            return relpath
    return None


def resolve_run(experiment_name, pkl_path, scan_limit=200):
    target_hash = hashing.sha256_file(pkl_path)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ResolveError(f"Experiment '{experiment_name}' not found.")

    runs = client.search_runs(
        [experiment.experiment_id], order_by=["start_time DESC"], max_results=scan_limit
    )

    matches = []
    for run in runs:
        relpath = _matching_relpath(run, target_hash)
        if relpath:
            matches.append((run, relpath))

    if not matches:
        raise ResolveError(
            f"No tracked run matches the SHA-256 of {pkl_path}. The file was "
            f"never saved inside a tracked context (track_run/track_save/snapshot), "
            f"or was modified after saving. Fix: publish a file that was saved "
            f"inside a `with mlops.track_run():` block."
        )

    if len(matches) == 1:
        run, relpath = matches[0]
        return ResolvedRun(run_id=run.info.run_id, relpath=relpath)

    lineages = {(r.data.tags.get("git_commit"), r.data.tags.get("dvc_data_hash")) for r, _ in matches}
    if len(lineages) == 1:
        run, relpath = min(matches, key=lambda pair: pair[0].info.start_time)
        return ResolvedRun(
            run_id=run.info.run_id,
            relpath=relpath,
            auto_resolved_note=(
                f"{len(matches)} runs matched this file's hash, all with identical "
                f"lineage; auto-picked the earliest ({run.info.run_id})."
            ),
        )

    candidate_ids = ", ".join(r.info.run_id for r, _ in matches)
    raise ResolveError(
        f"Multiple runs match this file's hash with DIFFERING lineage: "
        f"{candidate_ids}. Cannot guess which is correct - resolve manually."
    )
