"""Phase 0 spike: client-side run scan performance (ROADMAP.md Phase 0, item 4).

SPEC.md §7.4 resolves publish's content-hash lookup by scanning an experiment's runs
client-side, newest first, bounded to the last 200 by default. This spike measures
how long that scan actually takes against a real workspace, so the default bound is
a measured number and not a guess.

Prerequisites (not yet provisioned as of writing this script):
  - Same as spike 01 (workspace reachable via MLOPS_SPIKE_* env vars)
  - `az login` refreshed
  - An experiment with a realistic number of existing runs (this spike does not
    create runs itself — pass MLOPS_SPIKE_EXPERIMENT to point at one)
  - Optional: MLOPS_SPIKE_SCAN_LIMIT (default 200)

Run:  python spikes/04_run_scan_perf.py
"""

import os
import statistics
import sys
import time

REQUIRED_ENV = [
    "MLOPS_SPIKE_SUBSCRIPTION_ID",
    "MLOPS_SPIKE_RESOURCE_GROUP",
    "MLOPS_SPIKE_WORKSPACE",
    "MLOPS_SPIKE_EXPERIMENT",
]


def main() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        print(
            "This spike needs a real Azure ML workspace with an experiment that "
            "already has a realistic number of runs in it."
        )
        sys.exit(1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    import mlflow

    subscription_id = os.environ["MLOPS_SPIKE_SUBSCRIPTION_ID"]
    resource_group = os.environ["MLOPS_SPIKE_RESOURCE_GROUP"]
    workspace_name = os.environ["MLOPS_SPIKE_WORKSPACE"]
    experiment_name = os.environ["MLOPS_SPIKE_EXPERIMENT"]
    scan_limit = int(os.environ.get("MLOPS_SPIKE_SCAN_LIMIT", "200"))

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    workspace = ml_client.workspaces.get(workspace_name)
    mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        sys.exit(1)

    t0 = time.perf_counter()
    runs = client.search_runs(
        [experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=scan_limit,
    )
    list_elapsed = time.perf_counter() - t0
    print(f"Listed {len(runs)} runs (newest-first, limit {scan_limit}) in {list_elapsed:.2f}s")

    per_run_times = []
    for run in runs:
        t1 = time.perf_counter()
        tags = run.data.tags
        _ = tags.get("git_commit"), tags.get("dvc_data_hash")
        per_run_times.append(time.perf_counter() - t1)

    if per_run_times:
        sorted_times = sorted(per_run_times)
        p50 = statistics.median(sorted_times)
        p95_index = min(len(sorted_times) - 1, int(len(sorted_times) * 0.95))
        p95 = sorted_times[p95_index]
        print(f"Per-run tag-read time: p50={p50 * 1000:.2f}ms p95={p95 * 1000:.2f}ms")

    print("Run-scan perf spike complete.")


if __name__ == "__main__":
    main()
