import json
import subprocess
import sys
import tempfile
from contextlib import ContextDecorator
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mlops import __version__
from mlops.core import conversions, discovery, dvc, git, hashing, identity, tags
from mlops.core.lock import RunLock
from mlops.core.resolve import resolve_run


class TrackRunError(Exception):
    pass


class track_run(ContextDecorator):
    def __init__(self, model_name, workspace_dir=".", allow_dirty=False):
        self.workspace_dir = Path(workspace_dir)
        self.model_name = model_name
        self.allow_dirty = allow_dirty
        self.lock = RunLock(self.workspace_dir)

    def __enter__(self):
        self.lock.acquire()

        dirty = not git.is_clean(self.workspace_dir)
        if dirty and not self.allow_dirty:
            self.lock.release()
            raise TrackRunError(
                "Working tree has uncommitted changes. Runs must start from a "
                "committed state so the recorded commit is provably the code "
                "that trained the model. Fix: mlops push "
                "(exploring? pass allow_dirty=True — the run will be unpublishable)"
            )

        user = identity.resolve_identity(self.workspace_dir)
        if not user:
            self.lock.release()
            raise TrackRunError(
                "No identity could be resolved. Set .mlops/user, or configure "
                "`git config user.email`, to attribute this run."
            )

        mlflow.set_tracking_uri(f"file:{self.workspace_dir / 'mlruns'}")
        mlflow.set_experiment(self.model_name)

        self._before_snapshot = discovery.snapshot(self.workspace_dir)

        run = mlflow.start_run()
        self.run_id = run.info.run_id
        mlflow.set_tag(tags.GIT_COMMIT, git.get_commit_sha(self.workspace_dir))
        mlflow.set_tag(tags.GIT_BRANCH, git.get_branch(self.workspace_dir))
        mlflow.set_tag(tags.USER, user)
        mlflow.set_tag(tags.MLOPS_VERSION, __version__)
        if dirty:
            mlflow.set_tag(tags.DIRTY, "true")
        data_hash = dvc.read_data_hash(self.workspace_dir)
        if data_hash:
            mlflow.set_tag(tags.DVC_DATA_HASH, data_hash)

        params_path = self.workspace_dir / "params.yaml"
        if params_path.exists():
            import yaml
            mlflow.log_params(yaml.safe_load(params_path.read_text()) or {})

        freeze = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
        ).stdout
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "requirements.frozen.txt"
            path.write_text(freeze)
            mlflow.log_artifact(str(path), artifact_path="env")

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        try:
            self._record_checksums()
        finally:
            mlflow.end_run(status=status)
            self._write_last_run(status)
            self.lock.release()
        return False

    def _record_checksums(self):
        after_snapshot = discovery.snapshot(self.workspace_dir)
        changed = discovery.diff(self._before_snapshot, after_snapshot)
        model_files = discovery.filter_by_globs(changed)
        if not model_files:
            return

        checksums = {}
        for i, relpath in enumerate(model_files):
            sha256 = hashing.sha256_file(self.workspace_dir / relpath)
            mlflow.set_tag(tags.checksum_tag_name(i), tags.checksum_tag_value(sha256, relpath))
            checksums[relpath] = {
                "sha256": sha256,
                "context": tags.IN_RUN,
                "at": datetime.now(timezone.utc).isoformat(),
            }
        mlflow.set_tag(tags.SAVE_CONTEXT, tags.IN_RUN)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checksums.json"
            path.write_text(json.dumps(checksums))
            mlflow.log_artifact(str(path))

    def _write_last_run(self, status):
        marker = {
            "run_id": self.run_id,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git.get_commit_sha(self.workspace_dir),
            "status": status,
        }
        marker_path = self.workspace_dir / ".mlops" / "last_run.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker))


class track_convert(ContextDecorator):
    def __init__(self, pkl_path, model_name, workspace_dir="."):
        self.workspace_dir = Path(workspace_dir)
        self.pkl_path = Path(pkl_path)
        self.model_name = model_name
        self.lock = RunLock(self.workspace_dir)

    def __enter__(self):
        self.lock.acquire()
        try:
            self.source_pkl_sha = hashing.sha256_file(self.pkl_path)
            mlflow.set_tracking_uri(f"file:{self.workspace_dir / 'mlruns'}")
            resolved = resolve_run(self.model_name, self.pkl_path)
        except Exception:
            self.lock.release()
            raise
        self.run_id = resolved.run_id
        print(f"track_convert: attaching conversions of {self.pkl_path} to run {self.run_id}")
        self._before_snapshot = discovery.snapshot(self.workspace_dir)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self._record_conversions()
        finally:
            self.lock.release()
        return False

    def _record_conversions(self):
        after_snapshot = discovery.snapshot(self.workspace_dir)
        changed = discovery.diff(self._before_snapshot, after_snapshot)
        if not changed:
            print("track_convert: warning - no new files were produced in this block.")
            return

        hashed = {
            relpath: hashing.sha256_file(self.workspace_dir / relpath)
            for relpath in changed
        }
        at = datetime.now(timezone.utc).isoformat()
        new_entries = conversions.build_entries(hashed, self.source_pkl_sha, at)

        merged = self._existing_conversions()
        merged.update(new_entries)

        client = mlflow.tracking.MlflowClient()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversions.json"
            path.write_text(json.dumps(merged))
            client.log_artifact(self.run_id, str(path))

        for relpath in hashed:
            print(f"track_convert: recorded {relpath}")

    def _existing_conversions(self):
        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id, artifact_path="conversions.json"
            )
            return json.loads(Path(path).read_text())
        except Exception:
            return {}
