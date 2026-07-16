"""Phase 0 spike: register a folder bundle as a model version (ROADMAP.md Phase 0,
item 2). Confirms the folder-bundle convention from SPEC.md D#7 actually round-trips
through Azure ML: log a small local folder as run artifacts, register it (not a
single pkl), then download only a sub-path and verify the bytes match.

Prerequisites (not yet provisioned as of writing this script):
  - Same as spike 01 (workspace reachable via MLOPS_SPIKE_* env vars)
  - `az login` refreshed

Run:  python spikes/02_register_folder_model.py
"""

import hashlib
import os
import shutil
import sys
import tempfile

REQUIRED_ENV = [
    "MLOPS_SPIKE_SUBSCRIPTION_ID",
    "MLOPS_SPIKE_RESOURCE_GROUP",
    "MLOPS_SPIKE_WORKSPACE",
]


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        print("This spike needs a real Azure ML workspace to register a model in.")
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
    mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)
    mlflow.set_experiment("mlops-spike-02")

    bundle_dir = tempfile.mkdtemp(prefix="mlops-spike02-bundle-")
    sub_dir = os.path.join(bundle_dir, "subdir")
    os.makedirs(sub_dir)
    pkl_path = os.path.join(bundle_dir, "model.pkl")
    sub_file_path = os.path.join(sub_dir, "file.bin")
    with open(pkl_path, "wb") as f:
        f.write(os.urandom(1024))
    with open(sub_file_path, "wb") as f:
        f.write(os.urandom(4096))
    expected_sub_hash = sha256_of(sub_file_path)

    with mlflow.start_run() as run:
        mlflow.set_tag("spike", "02_register_folder_model")
        mlflow.log_artifacts(bundle_dir, artifact_path="models")
        run_id = run.info.run_id

    model_uri = f"runs:/{run_id}/models"
    registered = mlflow.register_model(model_uri, "mlops-spike-02-model")
    print(f"Registered version: {registered.version}")

    download_dir = tempfile.mkdtemp(prefix="mlops-spike02-download-")
    downloaded_path = mlflow.artifacts.download_artifacts(
        f"{model_uri}/subdir", dst_path=download_dir
    )
    downloaded_file = os.path.join(downloaded_path, "file.bin")
    actual_hash = sha256_of(downloaded_file)

    shutil.rmtree(bundle_dir, ignore_errors=True)
    shutil.rmtree(download_dir, ignore_errors=True)

    assert actual_hash == expected_sub_hash, "sub-path download did not match source bytes"
    print("Sub-path download verified byte-for-byte.")
    print("Folder-model registration spike succeeded.")


if __name__ == "__main__":
    main()
