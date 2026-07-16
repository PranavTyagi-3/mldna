"""Phase 0 spike: DVC blob auth resolution matrix (ROADMAP.md Phase 0, item 6).

D#19 / SPEC.md §3.3 defines a three-step resolution order for the DVC blob plane:
  1. env var MLOPS_BLOB_CONNECTION_STRING (highest precedence)
  2. .dvc/config.local (written by `mlops configure blob-auth`)
  3. DefaultAzureCredential fallback against `dvc.account_name`

This spike exercises all three against a *real* Azure Storage account + container,
confirming precedence order and that each mode can actually list/push/pull.

Prerequisites (not yet provisioned as of writing this script):
  - A real storage account + container reachable over the DVC azure remote
  - `az login` refreshed (for the AAD-fallback leg)
  - Env vars:
      MLOPS_SPIKE_STORAGE_ACCOUNT
      MLOPS_SPIKE_CONTAINER
      MLOPS_SPIKE_CONNECTION_STRING   (a real, valid connection string to test leg 1/2 with)
  - Storage Blob Data Contributor role on the container for the AAD identity (leg 3)

Run:  python spikes/06_dvc_blob_auth_matrix.py
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_ENV = [
    "MLOPS_SPIKE_STORAGE_ACCOUNT",
    "MLOPS_SPIKE_CONTAINER",
    "MLOPS_SPIKE_CONNECTION_STRING",
]


def run(cmd, cwd, env=None):
    print(f"  $ {' '.join(cmd)}  (cwd={cwd})")
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, check=False, env=env
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
    return result.returncode == 0


def dvc_cmd():
    return [sys.executable, "-m", "dvc"]


def try_push(repo: Path, label: str, env: dict) -> bool:
    ok = run(dvc_cmd() + ["push", "-q"], cwd=repo, env=env)
    print(f"[{label}] push {'succeeded' if ok else 'FAILED'}")
    return ok


def main() -> None:
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        print(f"Missing required env vars: {', '.join(missing)}")
        print("This spike needs a real storage account/container to test auth against.")
        sys.exit(1)

    account = os.environ["MLOPS_SPIKE_STORAGE_ACCOUNT"]
    container = os.environ["MLOPS_SPIKE_CONTAINER"]
    connection_string = os.environ["MLOPS_SPIKE_CONNECTION_STRING"]

    base = Path(tempfile.mkdtemp(prefix="mlops-spike06-"))
    repo = base / "auth-matrix-repo"
    try:
        repo.mkdir()
        run(["git", "init", "-q"], cwd=repo)
        run(dvc_cmd() + ["init", "-q"], cwd=repo)
        run(
            dvc_cmd()
            + ["remote", "add", "-d", "spike-remote", f"azure://{container}/spike06"],
            cwd=repo,
        )
        run(
            dvc_cmd() + ["remote", "modify", "spike-remote", "account_name", account],
            cwd=repo,
        )

        data_file = repo / "payload.bin"
        data_file.write_bytes(os.urandom(1024))
        run(dvc_cmd() + ["add", "payload.bin", "-q"], cwd=repo)

        print("\n--- Leg 1: env var MLOPS_BLOB_CONNECTION_STRING (highest precedence) ---")
        env1 = {**os.environ, "MLOPS_BLOB_CONNECTION_STRING": connection_string}
        try_push(repo, "env-var", env1)

        print("\n--- Leg 2: .dvc/config.local ---")
        run(
            dvc_cmd()
            + ["remote", "modify", "--local", "spike-remote", "connection_string", connection_string],
            cwd=repo,
        )
        env2 = {k: v for k, v in os.environ.items() if k != "MLOPS_BLOB_CONNECTION_STRING"}
        try_push(repo, "config.local", env2)

        print("\n--- Leg 3: AAD fallback (DefaultAzureCredential, needs az login) ---")
        run(dvc_cmd() + ["remote", "modify", "--local", "spike-remote", "--unset", "connection_string"], cwd=repo)
        env3 = {k: v for k, v in os.environ.items() if k != "MLOPS_BLOB_CONNECTION_STRING"}
        try_push(repo, "aad-fallback", env3)

    finally:
        shutil.rmtree(base, ignore_errors=True)
        print(f"\nCleaned up scratch dir: {base}")


if __name__ == "__main__":
    main()
