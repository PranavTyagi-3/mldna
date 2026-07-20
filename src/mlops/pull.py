import shutil
import tempfile
from pathlib import Path

import mlflow

from mlops.core import git
from mlops.core.config import ProjectConfig
from mlops.core.pull import PullError, pull as pull_code_and_data
from mlops.trackinguri import resolve_and_set_tracking_uri


def _published_runs_for_commit(client, experiment_id, commit):
    runs = client.search_runs([experiment_id], order_by=["attribute.start_time DESC"])
    return [
        run for run in runs
        if run.data.tags.get("git_commit") == commit and run.data.tags.get("registry_version")
    ]


def _download_bundle(run_id, dest):
    dest = Path(dest)
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/models", dst_path=tmp)
        if dest.exists():
            shutil.rmtree(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(downloaded, str(dest))


def pull(workspace_dir=".", select=None):
    workspace_dir = Path(workspace_dir)

    pull_code_and_data(workspace_dir)

    result = {"model_pulled": False}

    config = ProjectConfig.load(workspace_dir)
    if not config.model_name:
        return result

    _, mode = resolve_and_set_tracking_uri(workspace_dir)
    print(f"Tracking mode: {mode}")

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.model_name)
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

    _download_bundle(chosen.info.run_id, workspace_dir / "models")
    result["model_pulled"] = True
    result["run_id"] = chosen.info.run_id
    result["registry_version"] = chosen.data.tags.get("registry_version")
    return result
