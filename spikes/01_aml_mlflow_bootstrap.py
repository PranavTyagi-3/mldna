"""Phase 0 spike: AML MLflow bootstrap (ROADMAP.md Phase 0, item 1).

Confirms that `MLClient` -> workspace tracking URI -> a plain `mlflow` run with
tags/artifacts works end-to-end from a bare VM using `DefaultAzureCredential`
(no service principal, no connection string).

Prerequisites (not yet provisioned as of writing this script):
  - `az login` (the CLI's cached token has expired; interactive login is required)
  - An Azure ML workspace + its resource group, in a subscription you control
  - Env vars:
      MLOPS_SPIKE_SUBSCRIPTION_ID
      MLOPS_SPIKE_RESOURCE_GROUP
      MLOPS_SPIKE_WORKSPACE

Run:  python spikes/01_aml_mlflow_bootstrap.py
"""

import os
import sys

REQUIRED_ENV = [
    "MLOPS_SPIKE_SUBSCRIPTION_ID",
    "MLOPS_SPIKE_RESOURCE_GROUP",
    "MLOPS_SPIKE_WORKSPACE",
]


def main() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        print("This spike needs a real Azure ML workspace to bootstrap against.")
        sys.exit(1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    import mlflow

    subscription_id = os.environ["MLOPS_SPIKE_SUBSCRIPTION_ID"]
    resource_group = os.environ["MLOPS_SPIKE_RESOURCE_GROUP"]
    workspace_name = os.environ["MLOPS_SPIKE_WORKSPACE"]

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    workspace = ml_client.workspaces.get(workspace_name)
    tracking_uri = workspace.mlflow_tracking_uri
    print(f"Resolved tracking URI: {tracking_uri}")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("mlops-spike-01")

    with mlflow.start_run() as run:
        mlflow.set_tag("spike", "01_aml_mlflow_bootstrap")
        mlflow.log_param("example_param", 1)
        mlflow.log_metric("example_metric", 0.5)
        with open("spike_artifact.txt", "w") as f:
            f.write("hello from spike 01\n")
        mlflow.log_artifact("spike_artifact.txt")
        os.remove("spike_artifact.txt")
        print(f"Run id: {run.info.run_id}")
        print(f"Run status: {run.info.status}")

    print("Bootstrap spike succeeded.")


if __name__ == "__main__":
    main()
