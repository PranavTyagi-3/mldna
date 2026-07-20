from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mlops.core.config import ProjectConfig
from mlops.trackinguri import resolve_and_set_tracking_uri

DEFAULT_LAST = 20


class RunsError(Exception):
    pass


def _format_started(start_time_ms):
    if start_time_ms is None:
        return "?"
    return datetime.fromtimestamp(start_time_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def list_runs(workspace_dir=".", last=DEFAULT_LAST):
    workspace_dir = Path(workspace_dir)
    config_path = workspace_dir / "mlops.yaml"
    if not config_path.exists():
        raise RunsError(
            "mlops.yaml not found. Fix: run `mlops init <model-name>` in a producer project first."
        )
    config = ProjectConfig.from_yaml(config_path)
    if not config.model_name:
        raise RunsError(
            "mlops.yaml has no project.model_name - mlops runs only works in a producer "
            "project. Fix: set project.model_name in mlops.yaml."
        )

    resolve_and_set_tracking_uri(workspace_dir)
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(config.model_name)
    if experiment is None:
        return []

    rows = []
    for run in client.search_runs(
        [experiment.experiment_id], order_by=["attribute.start_time DESC"], max_results=last
    ):
        tags = run.data.tags
        key_metric_value = "-"
        if config.key_metric and config.key_metric in run.data.metrics:
            key_metric_value = run.data.metrics[config.key_metric]
        rows.append({
            "run_id": run.info.run_id[:8],
            "started": _format_started(run.info.start_time),
            "git_commit": tags.get("git_commit", "?")[:8],
            "key_metric": key_metric_value,
            "dirty": tags.get("dirty", "false"),
            "save_context": tags.get("save_context", "-"),
        })
    return rows
