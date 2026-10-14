import functools
import json
import os
import subprocess
import sys
import tempfile
from contextlib import ContextDecorator
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from mldna import __version__
from mldna.core import conversions, discovery, dvc, git, hashing, identity, tags
from mldna.core.config import ProjectConfig, missing_namespace_hint, missing_project_hint
from mldna.core.conversions import ConversionError
from mldna.core.lock import RunLock
from mldna.core.resolve import resolve_run
from mldna.trackinguri import resolve_and_set_tracking_uri


class TrackRunError(Exception):
    pass


@functools.lru_cache(maxsize=1)
def _pip_freeze():
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True
    ).stdout


def _require_model_name(workspace_dir):
    config = ProjectConfig.load(workspace_dir)
    if not config.model_name:
        if not (Path(workspace_dir) / "mldna.yaml").exists():
            # No project here at all. If one sits further up, say so rather
            # than suggesting init, which would scaffold a second project
            # on top of it (D#57).
            raise TrackRunError(missing_project_hint(workspace_dir))
        raise TrackRunError(
            "mldna.yaml has no project.model_name set, so there's nothing to "
            "attribute this run to. Fix: set project.model_name in "
            f"{workspace_dir}/mldna.yaml."
        )
    if not config.namespace:
        raise TrackRunError(missing_namespace_hint(workspace_dir))
    return config


_GIT_PLANE = (
    "Working tree has uncommitted changes. Runs must start from a committed "
    "state so the recorded commit is provably the code that trained the model. "
    "Fix: run `mldna stage`, review the files, then `mldna push -m \"<message>\"`"
)

_DVC_PLANE = (
    "DVC-tracked data has changes its pointers do not record ({targets}). Runs "
    "must start from recorded data so the recorded dvc_data_hash is provably the "
    "data that trained the model. Fix: `mldna stage` (which runs `dvc add "
    "dataset`), or `dvc add <path>` for the path above, then "
    "`mldna push -m \"<message>\"`"
)

_DVC_MISSING = (
    "DVC-tracked data is missing from the working tree ({targets}). The pointers "
    "are fine but the files are not on disk, so the run would not see the data it "
    "records. Fix: `dvc checkout` (or `dvc pull` if the blobs are not in the local cache)"
)

_ESCAPE = " (exploring? pass allow_dirty=True - the run will be unpublishable)"

# Bare key would collide with a same-named mlflow.log_param() the user makes at
# runtime (params are write-once); the prefix keeps the whole file visible in
# the UI without ever claiming a name the user might want for themselves.
_PARAMS_FILE_PREFIX = "params_file."

# Names chosen to not imply more than what's captured: argv and an opt-in list
# of env vars, never "the inputs" or "the config" - see invocation tags below.
_ARGV_TAG = "invocation_argv"
_ENV_TAG = "invocation_env"


def _dirty_message(git_dirty, data_changes, workspace_dir=None):
    parts = []
    if git_dirty:
        parts.append(_GIT_PLANE)
    missing, changed = dvc.split_status(data_changes, workspace_dir)
    if missing:
        parts.append(_DVC_MISSING.format(targets=", ".join(missing)))
    if changed:
        parts.append(_DVC_PLANE.format(targets=", ".join(changed)))
    return " ".join(parts) + _ESCAPE


class track_run(ContextDecorator):
    def __init__(self, workspace_dir=".", allow_dirty=False):
        self.workspace_dir = Path(workspace_dir)
        self.allow_dirty = allow_dirty
        self.lock = RunLock(self.workspace_dir)

    def __enter__(self):
        self.lock.acquire()
        try:
            return self._start()
        except BaseException:
            # Any failure while starting must free the lock, whatever its type.
            self.lock.release()
            raise

    def _start(self):
        try:
            config = _require_model_name(self.workspace_dir)
        except TrackRunError:
            self.lock.release()
            raise
        self.model_name = config.full_model_name()
        self._scan_exclude = config.scan_exclude
        self._model_globs = config.model_globs

        try:
            git_dirty = not git.is_clean(self.workspace_dir)
        except git.GitStatusError as e:
            self.lock.release()
            raise TrackRunError(
                f"Could not determine whether the git working tree is clean ({e}). "
                "A run may only start when the tree is known to match the commit it "
                "will record, and an unanswered check is not an answer of clean. "
                f"Fix: run `git status` in {self.workspace_dir} and resolve what it "
                "reports, then re-run."
            ) from e
        print("track_run: checking DVC-tracked data against its pointers")
        try:
            data_changes = dvc.data_status(self.workspace_dir)
        except dvc.DvcStatusError as e:
            self.lock.release()
            raise TrackRunError(
                f"Could not determine whether the DVC-tracked data is clean ({e}). "
                "A run may only start when the data on disk is known to match the "
                "dvc_data_hash it will record, and an unanswered check is not an "
                "answer of clean. Fix: run `dvc status` in "
                f"{self.workspace_dir} and resolve what it reports, then re-run."
            ) from e
        dirty = git_dirty or bool(data_changes)
        if dirty and not self.allow_dirty:
            self.lock.release()
            raise TrackRunError(_dirty_message(git_dirty, data_changes, self.workspace_dir))

        user = identity.resolve_identity(self.workspace_dir)
        if not user:
            self.lock.release()
            raise TrackRunError(
                "No identity could be resolved, so this run cannot be attributed "
                "to anyone. Fix: run `git config user.email you@example.com` "
                "(add --global to set it for every repo). Alternatively, write the "
                f"identity you want recorded as a single line of plain text to "
                f"{self.workspace_dir / '.mldna' / 'user'}, which takes precedence "
                "over git."
            )

        try:
            _, mode = resolve_and_set_tracking_uri(self.workspace_dir)
        except Exception:
            self.lock.release()
            raise
        print(f"track_run: tracking mode {mode}")

        # From here, a run may already exist server-side when something fails
        # (e.g. start_run() succeeds but a later tag/log call doesn't) - the
        # same release-on-error idiom as the gates above, plus ending the run
        # as FAILED so it doesn't sit there looking FINISHED (mlflow's own
        # exit hook would mark it FINISHED, which is a lie for a setup that
        # never completed).
        try:
            mlflow.set_experiment(self.model_name)

            self._before_snapshot = discovery.snapshot(self.workspace_dir, excludes=self._scan_exclude)

            run = mlflow.start_run()
            self.run_id = run.info.run_id
            mlflow.set_tag(tags.GIT_COMMIT, git.get_commit_sha(self.workspace_dir))
            mlflow.set_tag(tags.GIT_BRANCH, git.get_branch(self.workspace_dir))
            mlflow.set_tag(tags.USER, user)
            mlflow.set_tag(tags.MLDNA_VERSION, __version__)
            if dirty:
                mlflow.set_tag(tags.DIRTY, "true")
            # One tag per DVC out, same idiom as model_sha256__<n> (D#63) -
            # scoped to exactly what D#45's gate checks.
            for index, (relpath, md5) in enumerate(
                dvc.read_data_hashes(self.workspace_dir).items()
            ):
                mlflow.set_tag(
                    tags.data_hash_tag_name(index),
                    tags.data_hash_tag_value(md5, relpath),
                )

            self._log_invocation(config.env_allowlist, config.record_argv)

            params_path = self.workspace_dir / "params.yaml"
            if params_path.exists():
                import yaml
                raw_params = yaml.safe_load(params_path.read_text()) or {}
                mlflow.log_params({f"{_PARAMS_FILE_PREFIX}{k}": v for k, v in raw_params.items()})

            freeze = _pip_freeze()
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "requirements.frozen.txt"
                path.write_text(freeze)
                mlflow.log_artifact(str(path), artifact_path="env")
        except Exception:
            if getattr(self, "run_id", None):
                try:
                    mlflow.end_run(status="FAILED")
                except Exception:
                    print(
                        "track_run: warning - could not mark the failed run as FAILED; "
                        "it may still show RUNNING on the server."
                    )
            raise

        return self

    def _log_invocation(self, env_allowlist, record_argv=False):
        # Partial by construction: argv and env vars are both opt-in, never
        # "everything that affected this run". argv can hold secrets and the
        # tag is readable by every server user (D#75).
        if record_argv:
            mlflow.set_tag(_ARGV_TAG, json.dumps(sys.argv))
        if env_allowlist:
            captured = {name: os.environ[name] for name in env_allowlist if name in os.environ}
            mlflow.set_tag(_ENV_TAG, json.dumps(captured))

    def __exit__(self, exc_type, exc_val, exc_tb):
        status = "FAILED" if exc_type else "FINISHED"
        try:
            self._record_checksums()
        finally:
            try:
                mlflow.end_run(status=status)
            except Exception as e:
                # issue #10: a network blip here used to leave the run RUNNING
                # on the server forever, with no local trace of what to fix.
                # Caught rather than re-raised so the marker below still gets
                # written - `mldna attach` reports this status once the
                # server is reachable again.
                print(
                    f"track_run: warning - could not report run status to the "
                    f"tracking server ({type(e).__name__}: {e}). The run may "
                    f"still show RUNNING there; once reachable, run `mldna "
                    f"attach` to report {status} for it."
                )
            finally:
                self._write_last_run(status)
                self.lock.release()
        return False

    def _record_checksums(self):
        after_snapshot = discovery.snapshot(self.workspace_dir, excludes=self._scan_exclude)
        changed = discovery.diff(self._before_snapshot, after_snapshot)
        model_files = discovery.filter_by_globs(changed, globs=self._model_globs)
        self._checksums = {}
        if not model_files:
            # Nothing tracked. If a model-shaped file does exist but sits in a
            # directory the scan skips, say so now - otherwise the user only
            # finds out at publish time, from a refusal that reads as though
            # they never tracked it at all (D#54).
            hint = discovery.excluded_model_hint(
                self.workspace_dir, globs=self._model_globs
            )
            if hint:
                print(f"track_run: warning - no model files recorded; {hint}")
            return

        # Hashing is local disk I/O - it cannot fail from a network blip, and
        # happens before anything that can. _write_last_run persists these
        # regardless of whether the tagging below reaches the server.
        for relpath in model_files:
            self._checksums[relpath] = hashing.sha256_file(self.workspace_dir / relpath)

        try:
            for i, (relpath, sha256) in enumerate(self._checksums.items()):
                mlflow.set_tag(tags.checksum_tag_name(i), tags.checksum_tag_value(sha256, relpath))
            mlflow.set_tag(tags.SAVE_CONTEXT, tags.IN_RUN)

            checksums_artifact = {
                relpath: {"sha256": sha256, "context": tags.IN_RUN,
                          "at": datetime.now(timezone.utc).isoformat()}
                for relpath, sha256 in self._checksums.items()
            }
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "checksums.json"
                path.write_text(json.dumps(checksums_artifact))
                mlflow.log_artifact(str(path))
        except Exception as e:
            # issue #10: these hashes are the one thing a failed publish can
            # never recover without - they're saved locally below either way.
            print(
                f"track_run: warning - could not report model checksums to "
                f"the tracking server ({type(e).__name__}: {e}). They were "
                f"saved locally in .mldna/last_run.json; once reachable, run "
                f"`mldna attach` to finish this run."
            )

    def _write_last_run(self, status):
        marker = {
            "run_id": self.run_id,
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": git.get_commit_sha(self.workspace_dir),
            "status": status,
            "checksums": getattr(self, "_checksums", {}),
        }
        marker_path = self.workspace_dir / ".mldna" / "last_run.json"
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(json.dumps(marker))
        # One file per run too, so a later run never overwrites the record
        # of an earlier run that still needs `mldna attach --run-id`.
        per_run_path = marker_path.parent / "runs" / f"{self.run_id}.json"
        per_run_path.parent.mkdir(exist_ok=True)
        per_run_path.write_text(json.dumps(marker))


class track_convert(ContextDecorator):
    def __init__(self, pkl_path, workspace_dir=".", commit=None):
        self.workspace_dir = Path(workspace_dir)
        self.pkl_path = Path(pkl_path)
        self.commit = commit
        self.lock = RunLock(self.workspace_dir)

    def __enter__(self):
        self.lock.acquire()
        try:
            config = _require_model_name(self.workspace_dir)
            self.model_name = config.full_model_name()
            self._scan_exclude = config.scan_exclude
            self.source_pkl_sha = hashing.sha256_file(self.pkl_path)
            resolve_and_set_tracking_uri(self.workspace_dir)
            resolved = resolve_run(
                self.model_name, self.pkl_path, scan_limit=config.scan_limit, commit=self.commit,
                workspace_dir=self.workspace_dir,
            )
        except Exception:
            self.lock.release()
            raise
        self.run_id = resolved.run_id
        print(f"track_convert: attaching conversions of {self.pkl_path} to run {self.run_id}")
        self._before_snapshot = discovery.snapshot(self.workspace_dir, excludes=self._scan_exclude)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if exc_type is None:
                self._record_conversions()
        finally:
            self.lock.release()
        return False

    def _record_conversions(self):
        after_snapshot = discovery.snapshot(self.workspace_dir, excludes=self._scan_exclude)
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

        client = mlflow.tracking.MlflowClient()
        merged = self._existing_conversions(client)
        merged.update(new_entries)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "conversions.json"
            path.write_text(json.dumps(merged))
            client.log_artifact(self.run_id, str(path))

        for relpath in hashed:
            print(f"track_convert: recorded {relpath}")

    def _existing_conversions(self, client):
        # Probe before downloading: some MLflow servers answer a request for a
        # missing artifact with HTTP 500 rather than 404, and the mlflow client
        # retries 500s for minutes - so a first conversion (where the artifact
        # legitimately does not exist) paid a ~4 minute stall for a file we
        # already know is absent. A listing is one cheap call.
        # Absence is also only ever concluded from the listing, never from a
        # failed read: this result is the base of the merged record written back
        # over conversions.json, so returning {} for an existing-but-unreadable
        # file would erase every conversion recorded in an earlier session.
        try:
            present = any(f.path == "conversions.json" for f in client.list_artifacts(self.run_id))
        except Exception as e:
            raise ConversionError(
                f"Could not list the artifacts of run {self.run_id}, so "
                f"track_convert cannot tell whether conversions were already "
                f"recorded ({type(e).__name__}: {e}). Writing now could overwrite "
                f"a provenance record mldna was unable to read. Fix: check the "
                f"tracking server is reachable with `mldna doctor`, then re-run "
                f"the conversion inside a track_convert block."
            ) from e

        if not present:
            return {}

        try:
            path = mlflow.artifacts.download_artifacts(
                run_id=self.run_id, artifact_path="conversions.json"
            )
            return json.loads(Path(path).read_text())
        except Exception as e:
            raise ConversionError(
                f"Run {self.run_id} has a conversions.json artifact but it could "
                f"not be read ({type(e).__name__}: {e}). Recording this session on "
                f"top of it would overwrite the conversions already recorded there "
                f"with only the files from this block, destroying their provenance. "
                f"Fix: check the tracking server is reachable with `mldna doctor`, "
                f"then re-run the conversion inside a track_convert block."
            ) from e
