"""Phase 0 spike: AML tag limits in practice (ROADMAP.md Phase 0, item 3).

SPEC.md §6/§13 records one `model_sha256__<n>` tag per checksummed file, plus a
`checksums.json` artifact as the unbounded source of truth. This spike writes tags
at a realistic file count and size, then reads both back, to find where (if
anywhere) Azure ML's tag key/value/count limits bite in practice.

Prerequisites (not yet provisioned as of writing this script):
  - Same as spike 01 (workspace reachable via MLOPS_SPIKE_* env vars)
  - `az login` refreshed
  - Optional: MLOPS_SPIKE_FILE_COUNT (default 50) to size the experiment

Run:  python spikes/03_tag_limits.py
"""

import json
import os
import sys
import tempfile

REQUIRED_ENV = [
    "MLOPS_SPIKE_SUBSCRIPTION_ID",
    "MLOPS_SPIKE_RESOURCE_GROUP",
    "MLOPS_SPIKE_WORKSPACE",
]


def main() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        print("This spike needs a real Azure ML workspace to test tag limits against.")
        sys.exit(1)

    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    import mlflow

    subscription_id = os.environ["MLOPS_SPIKE_SUBSCRIPTION_ID"]
    resource_group = os.environ["MLOPS_SPIKE_RESOURCE_GROUP"]
    workspace_name = os.environ["MLOPS_SPIKE_WORKSPACE"]
    file_count = int(os.environ.get("MLOPS_SPIKE_FILE_COUNT", "50"))

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    workspace = ml_client.workspaces.get(workspace_name)
    mlflow.set_tracking_uri(workspace.mlflow_tracking_uri)
    mlflow.set_experiment("mlops-spike-03")

    checksums = {
        f"models/checkpoint_{i}.pkl": {
            "sha256": os.urandom(32).hex(),
            "context": "in_run",
        }
        for i in range(file_count)
    }

    errors = []
    with mlflow.start_run() as run:
        mlflow.set_tag("spike", "03_tag_limits")
        for i, (relpath, meta) in enumerate(checksums.items()):
            tag_value = f"{meta['sha256']}:{relpath}"
            try:
                mlflow.set_tag(f"model_sha256__{i}", tag_value)
            except Exception as exc:  # noqa: BLE001 — spike wants to see every failure mode
                errors.append((f"model_sha256__{i}", len(tag_value), str(exc)))

        with tempfile.TemporaryDirectory() as tmp:
            checksums_path = os.path.join(tmp, "checksums.json")
            with open(checksums_path, "w") as f:
                json.dump(checksums, f)
            mlflow.log_artifact(checksums_path)

        run_id = run.info.run_id

    fetched_run = mlflow.get_run(run_id)
    tags_read_back = {
        k: v for k, v in fetched_run.data.tags.items() if k.startswith("model_sha256__")
    }
    downloaded = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/checksums.json")
    with open(downloaded) as f:
        checksums_read_back = json.load(f)

    print(f"Wrote {file_count} per-file tags, {len(errors)} errors during set_tag.")
    for tag_name, value_len, err in errors:
        print(f"  {tag_name} (value len {value_len}): {err}")
    print(f"Tags read back: {len(tags_read_back)} / {file_count}")
    print(f"checksums.json read back: {len(checksums_read_back)} / {file_count} entries")
    print("Tag-limits spike complete.")


if __name__ == "__main__":
    main()
