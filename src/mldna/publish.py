import json
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import mlflow
from mlflow.exceptions import MlflowException
from requests.exceptions import ConnectionError as RequestsConnectionError
from requests.exceptions import Timeout as RequestsTimeout
from urllib3.exceptions import NewConnectionError, ProtocolError
from urllib3.exceptions import TimeoutError as Urllib3TimeoutError

from mldna.core import blobauth, commitcheck, dvc, git, hashing, identity, manifest, rungates, structure
from mldna.core import tags as tagschema
from mldna.core.config import ProjectConfig, missing_namespace_hint, missing_project_hint
from mldna.core.conversions import verify_bundle
from mldna.core.manifest import INVOCATION_LABEL
from mldna.core.resolve import resolve_run
from mldna.trackinguri import resolve_and_set_tracking_uri


class PublishError(Exception):
    pass


def _dropped_connection(error):
    # The cause chain is walked, never the message text (D#41). A refused or
    # timed-out connection is not a drop mid-upload, and requests'
    # ConnectTimeout subclasses ConnectionError, so those are ruled out first.
    chain = []
    while error is not None:
        chain.append(error)
        error = error.__cause__ or error.__context__
    if any(isinstance(e, (NewConnectionError, Urllib3TimeoutError, RequestsTimeout)) for e in chain):
        return False
    dropped = (RequestsConnectionError, ConnectionResetError, ConnectionAbortedError, ProtocolError)
    return any(isinstance(e, dropped) for e in chain)


def _existing_conversions(client, run_id):
    # Probe before downloading: some MLflow servers answer a request for a
    # missing artifact with HTTP 500 rather than 404, and the mlflow client
    # retries 500s for minutes - which made every pkl-only publish (the common
    # case, where conversions.json legitimately does not exist) pay a ~4 minute
    # stall for a file we already know is absent. A listing is one cheap call.
    # Absence is also only ever concluded from the listing, never from a failed
    # read: under invariant 6 a provenance record's absence must be proven, so
    # a read that fails refuses instead of reporting "no conversions recorded".
    try:
        present = any(f.path == "conversions.json" for f in client.list_artifacts(run_id))
    except Exception as e:
        raise PublishError(
            f"Could not list the artifacts of run {run_id}, so publish cannot "
            f"tell whether conversion provenance was recorded ({type(e).__name__}: "
            f"{e}). Conversion provenance must be proven, never assumed absent. "
            f"Fix: check the tracking server is reachable with `mldna doctor`, "
            f"then re-run publish."
        ) from e

    if not present:
        return {}

    try:
        path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="conversions.json"
        )
        return json.loads(Path(path).read_text())
    except Exception as e:
        raise PublishError(
            f"Run {run_id} has a conversions.json artifact but it could not be "
            f"read ({type(e).__name__}: {e}). Publishing without it would verify "
            f"the bundle against an empty provenance record, which is not the "
            f"same as there being none. Fix: check the tracking server is "
            f"reachable with `mldna doctor`, then re-run publish."
        ) from e


def _check_data_hash_current(workspace_dir, run):
    """The run's per-out dvc_data_hash__<n> tags were recorded against this
    workspace's DVC pointers at train time (tracking.py's track_run). D#56's
    cloud_status check answers "are the current pointers present on the
    remote" - that only verifies THIS run's data if no pointer has moved since
    training. Compares the hashes directly rather than introspecting the DVC
    cache/remote for specific past hashes: this reuses read_data_hashes, which
    already exists and is already how the tags got written, instead of adding a
    second way to read DVC state (D#60, per-out since D#63).
    """
    recorded = tagschema.data_hashes_from_tags(run.data.tags)
    current = dvc.read_data_hashes(workspace_dir)
    drifted = sorted(set(recorded) | set(current))
    drifted = [out for out in drifted if recorded.get(out) != current.get(out)]
    if drifted:
        detail = "; ".join(
            f"{out}: run cites {recorded.get(out, 'nothing')}, "
            f"workspace points at {current.get(out, 'nothing')}"
            for out in drifted
        )
        raise PublishError(
            f"The data this run cites is not what the workspace's DVC pointers "
            f"currently point at ({detail}). The dataset was changed and "
            f"re-added after this run trained, so publish cannot attest that "
            f"run's exact data snapshot. Fix: `git checkout "
            f"{run.data.tags.get('git_commit', '<commit>')}` (the commit this run "
            f"trained at) then `dvc pull`, or retrain "
            f"inside `with mldna.track_run():` against the current dataset and "
            f"publish that new run instead."
        )


def _without_userinfo(url):
    parts = urlsplit(url or "")
    if "@" not in parts.netloc:
        return url
    return urlunsplit(parts._replace(netloc=parts.netloc.rsplit("@", 1)[1]))


def _data_needs_push(workspace_dir):
    """Whether the dataset this run cites still needs to reach the DVC remote -
    checked ahead of the confirmation summary, symmetric with
    commitcheck.commit_needs_push (D#56). Refuses outright when no remote is
    configured at all: a dvc_data_hash citing bytes that can never leave this
    machine isn't a weaker guarantee, it's no guarantee, so this is not a
    labeled escape hatch like allow_dirty/post_run - see D#56.
    """
    remote_name = blobauth.get_default_remote(workspace_dir)
    if not remote_name:
        raise PublishError(
            "No DVC remote is configured, so the dataset this run cites could "
            "never be retrieved by anyone else - publishing it would cite an "
            "unreachable snapshot. Fix: `dvc remote add -d <name> <url>` (or "
            "`mldna configure blob-auth` for Azure Blob), commit .dvc/config "
            "(`mldna stage`, then `mldna push -m \"add dvc remote\"`), then "
            "publish again - publish uploads the data itself."
        )
    # The commit this model records must say where its data lives (D#80).
    url = blobauth.get_remote_url(workspace_dir, remote_name)
    if blobauth.get_committed_remote(workspace_dir) != (remote_name, url):
        raise PublishError(
            f"DVC remote '{remote_name}' ({_without_userinfo(url)}) is not in the "
            f"committed .dvc/config, so the commit this model records does not "
            f"say where its data lives. Fix: commit .dvc/config (`mldna stage`, "
            f"then `mldna push -m \"add dvc remote\"`), then publish again. "
            f"Credentials can stay in .dvc/config.local."
        )
    blobauth.resolve_env_override(workspace_dir, remote_name)
    try:
        status = dvc.cloud_status(workspace_dir)
    except dvc.DvcStatusError as e:
        raise PublishError(
            f"Could not determine whether the dataset is present on the DVC "
            f"remote ({e}). An unanswered check is not an answer of present. "
            f"Fix: check the remote is reachable, then publish again."
        ) from e
    return bool(status)


def _push_data(workspace_dir):
    result = dvc.run(["push", "-q"], cwd=workspace_dir, check=False)
    if result.returncode != 0:
        raise PublishError(
            f"dvc push failed: {result.stderr.strip()}. The dataset a published "
            f"model cites must be retrievable by anyone verifying its lineage. "
            f"Fix: resolve that push error, then publish again."
        )
    try:
        status = dvc.cloud_status(workspace_dir)
    except dvc.DvcStatusError as e:
        raise PublishError(
            f"dvc push ran but presence on the remote could not be re-verified "
            f"({e}). Fix: check the remote is reachable, then publish again."
        ) from e
    if status:
        raise PublishError(
            f"Dataset still not present on the remote after pushing "
            f"({', '.join(status)}). Refusing to publish - lineage cannot be "
            f"verified on the remote. Fix: confirm the push actually reached "
            f"the remote, then publish again."
        )


def _check_model_id(client, model_name, model_id):
    """D#69: the registry name is user-chosen, so the id on the registered model
    says which project owns it. Returns the registered model's tags ({} for a
    new model); missing id/origin tags are stamped after registering."""
    try:
        existing = client.get_registered_model(model_name)
    except MlflowException as e:
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            return {}
        raise
    registered_id = existing.tags.get(tagschema.MODEL_ID)
    if registered_id and registered_id != model_id:
        raise PublishError(
            f"Registry model {model_name} belongs to another project (its "
            f"model id is {registered_id}, this project's is {model_id}). Names "
            f"are user-chosen and can collide, and publishing here would mix two "
            f"projects' versions. Fix: if this is your project and its model_id "
            f"was lost or regenerated, restore it: set project.model_id: "
            f"{registered_id} in mldna.yaml and commit. Otherwise change "
            f"project.namespace or project.model_name in mldna.yaml, then "
            f"publish again."
        )
    return existing.tags


def _origin_warning(registered_origin, origin, workspace_dir):
    """D#77: a copied mldna.yaml carries its model_id, so the id check alone
    cannot tell a template copy from the original project. Remotes compare by
    canonical form, so SSH vs HTTPS or a relative vs absolute path is no warning."""
    if not registered_origin or git.same_remote(registered_origin, origin, workspace_dir):
        return None
    return (
        f"WARNING: this registry model was published from {registered_origin}; "
        f"this repo's origin is {origin or 'not set'}. A copied mldna.yaml carries "
        f"its model_id - if this is a different project, cancel and run `mldna id "
        f"--new`, commit mldna.yaml, then publish again. If it is the same project "
        f"moved to a new remote, confirm and the new origin is recorded."
    )


def _abandon(client, model_name, version, winner, run_id):
    """D#76: the losing publish tags its own version so fetch and pull refuse
    it, then fails loudly. Returns the error to raise."""
    lost = (
        f"Run {run_id} already records this file as published as version "
        f"{winner}, set by a concurrent publish of the same file while this "
        f"publish was uploading - this call independently registered version "
        f"{version} in {model_name}. Only one version should back this file."
    )
    try:
        client.set_model_version_tag(model_name, version, tagschema.ABANDONED, "true")
        client.set_model_version_tag(model_name, version, tagschema.ABANDONED_REASON,
                                      f"lost concurrent publish to v{winner}")
    except Exception as e:
        return PublishError(
            f"{lost} Tagging version {version} abandoned also failed "
            f"({type(e).__name__}: {e}), so it is UNTAGGED and still fetchable. "
            f"Fix: tag it by hand - set tag abandoned=true on {model_name} "
            f"version {version} in the MLflow UI - and pin version {winner}."
        )
    return PublishError(
        f"{lost} Version {version} is now tagged abandoned=true, so `mldna fetch` "
        f"and `mldna pull` refuse it. Fix: nothing to undo - pin version {winner}."
    )


def _summary(model_name, run, resolved, published_by, model_id, push_branch=None, data_pending=False,
             branch_on_remote=False):
    tags = run.data.tags
    lines = [
        f"Matched run {resolved.run_id} (commit {tags.get('git_commit', '?')[:8]}, "
        f"user {tags.get('user', '?')}, save_context {tags.get('save_context', '?')})",
    ]
    if resolved.auto_resolved_note:
        lines.append(resolved.auto_resolved_note)
    if tags.get("attached") == "true":
        # D#61: this run's completion was replayed by `mldna attach` after its
        # exit-time report failed. Surfaced here with the other weak labels
        # (dirty, save_context) so a reviewer minting a version sees it.
        lines.append(
            f"Run was attached retroactively at {tags.get('attached_at', '?')} - its "
            "FINISHED status and/or model checksum tags were recorded after the "
            "exit-time report failed, not when they became true."
        )
    if push_branch:
        why = (f"commit {tags.get('git_commit', '?')[:8]} not on the remote branch yet"
               if branch_on_remote else "branch not on the remote yet")
        lines.append(f"Will push {push_branch} to origin ({why}).")
    if data_pending:
        lines.append("Will push the dataset to the DVC remote (not yet present there).")
    argv = tags.get(tagschema.INVOCATION_ARGV)
    if argv:
        lines.append(f"Invocation argv ({INVOCATION_LABEL}): {argv}")
        env = tags.get(tagschema.INVOCATION_ENV)
        if env:
            lines.append(f"Invocation env ({INVOCATION_LABEL}): {env}")
    else:
        lines.append("Invocation: not recorded (tracking.record_argv is off).")
    lines.append(f"Model id {model_id} - pin it in models.lock.yaml with `id: {model_id}`.")
    lines.append(f"Publishing as {model_name}, published by {published_by}.")
    return "\n".join(lines)


def publish(pkl_path, workspace_dir=".", format_dirs=(), confirm=None,
            commit=None, select=None, confirm_origin=None):
    workspace_dir = Path(workspace_dir)
    pkl_path = Path(pkl_path)
    format_dirs = [Path(d) for d in format_dirs]

    if not pkl_path.exists():
        raise PublishError(
            f"{pkl_path} does not exist, and publish resolves the run to publish "
            f"from the content hash of that exact file. Fix: check the path, or "
            f"re-run training to produce it, then re-run `mldna publish {pkl_path}`."
        )
    if not pkl_path.is_file():
        raise PublishError(
            f"{pkl_path} is a directory, but publish takes the single anchor pkl "
            f"file (converted formats are passed with --formats). Fix: re-run as "
            f"`mldna publish {pkl_path}/<model>.pkl`."
        )

    if not (workspace_dir / "mldna.yaml").exists():
        # Never suggest init over an existing parent project (D#57).
        raise PublishError(missing_project_hint(workspace_dir))
    config = ProjectConfig.load(workspace_dir)
    if not config.model_name:
        raise PublishError(
            "mldna.yaml has no project.model_name set, so there's no experiment "
            "to resolve the run from. Fix: set project.model_name in mldna.yaml."
        )
    if not config.namespace:
        raise PublishError(missing_namespace_hint(workspace_dir))
    if not config.model_id:
        raise PublishError(
            "mldna.yaml has no project.model_id, the unique id that stops another "
            "project's same-named registry model being mixed with this one. "
            "Fix: run `mldna id`, commit mldna.yaml, then publish again."
        )
    model_name = config.full_model_name()

    _, mode = resolve_and_set_tracking_uri(workspace_dir)
    print(f"Tracking mode: {mode}")

    resolved = resolve_run(model_name, pkl_path, scan_limit=config.scan_limit,
                           commit=commit, select=select, workspace_dir=workspace_dir)
    pkl_sha256 = hashing.sha256_file(pkl_path)
    run = rungates.check_run_gates(resolved.run_id, pkl_sha256)
    push_branch, needs_commit_push = commitcheck.commit_needs_push(workspace_dir, resolved.run_id)
    _check_data_hash_current(workspace_dir, run)
    needs_data_push = _data_needs_push(workspace_dir)
    structure.check_structure(pkl_path, format_dirs, require_formats=config.require_formats)

    client = mlflow.tracking.MlflowClient()
    existing_conversions = _existing_conversions(client, resolved.run_id)
    verify_bundle(existing_conversions, format_dirs, pkl_sha256)

    model_tags = _check_model_id(client, model_name, config.model_id)
    origin = git.origin_url(workspace_dir)
    origin_warning = _origin_warning(model_tags.get(tagschema.MODEL_ORIGIN), origin, workspace_dir)

    published_by = identity.resolve_identity(workspace_dir)
    published_at = datetime.now(timezone.utc).isoformat()

    summary = _summary(
        model_name, run, resolved, published_by, config.model_id,
        push_branch=push_branch if needs_commit_push else None,
        data_pending=needs_data_push,
        branch_on_remote=git.remote_branch_exists(workspace_dir, push_branch),
    )
    if origin_warning:
        summary += "\n" + origin_warning
    print(summary)
    if origin_warning and confirm_origin is None:
        raise PublishError(
            f"Registry model {model_name} was published from "
            f"{model_tags[tagschema.MODEL_ORIGIN]}, but this repo's origin is "
            f"{origin or 'not set'}. A copied mldna.yaml carries the original "
            f"project's model_id, so only a person can tell a moved repo from a "
            f"copy, and none can confirm here. Fix: if this is a different "
            f"project, run `mldna id --new` and commit mldna.yaml; if it is the "
            f"same project on a new remote, re-run `mldna publish {pkl_path}` in "
            f"an interactive terminal and confirm."
        )
    if confirm is not None and not confirm(summary):
        raise PublishError("Aborted.")
    if origin_warning and not confirm_origin(origin_warning):
        raise PublishError("Aborted.")

    # Gates above only determined whether a push is pending (for the summary
    # the user just confirmed) - the actual transmissions happen here, after
    # consent, same D#4 shape as publishing an existing commit: never a new
    # commit, just moving already-frozen bytes (code and data alike) to where
    # lineage can be verified.
    commitcheck.check_commit_reachable(workspace_dir, resolved.run_id)
    if needs_data_push:
        _push_data(workspace_dir)

    manifest_dict = manifest.build_manifest(
        model_name, run, resolved, pkl_path, format_dirs,
        existing_conversions, published_by, published_at,
        model_id=config.model_id,
    )

    # Unique per publish attempt (not per registered version - the registry
    # version isn't known until after registration), so two concurrent
    # publishes of the same run upload to two different artifact paths and
    # neither can overwrite the bytes the other just wrote (D#59).
    artifact_path = f"models/{uuid.uuid4().hex}"

    with tempfile.TemporaryDirectory() as tmp:
        bundle_dir = Path(tmp) / "models"
        bundle_dir.mkdir()
        shutil.copy2(pkl_path, bundle_dir / pkl_path.name)
        for format_dir in format_dirs:
            shutil.copytree(format_dir, bundle_dir / format_dir.name)
        manifest.write_manifest(manifest_dict, bundle_dir / "manifest.yaml")

        try:
            # explicit run_id target, not the fluent API - mlflow.log_artifacts()
            # auto-starts (and leaks) an untracked run if none is active
            client.log_artifacts(resolved.run_id, str(bundle_dir), artifact_path=artifact_path)
        except Exception as e:
            message = str(e)
            size = sum(f.stat().st_size for f in bundle_dir.rglob("*") if f.is_file())
            if "413" in message or "entity too large" in message.lower():
                raise PublishError(
                    f"Uploading the bundle ({size / (1024 * 1024):.0f} MiB) was "
                    f"rejected by the tracking server as too large "
                    f"({type(e).__name__}: {message}). This is the server's "
                    f"configured request-size limit (e.g. waitress's "
                    f"max_request_body_size), not something mldna can raise. "
                    f"Fix: raise that limit on the tracking server, then "
                    f"publish again."
                ) from e
            if _dropped_connection(e):
                raise PublishError(
                    f"Uploading the bundle ({size / (1024 * 1024):.0f} MiB) failed "
                    f"({type(e).__name__}: {message}). A dropped connection during "
                    f"a large upload often means the tracking server's request-size "
                    f"limit was exceeded (waitress: `max_request_body_size`), but it "
                    f"can also be a network problem. Fix: raise that limit on the "
                    f"server, or retry on a stable connection."
                ) from e
            raise

    registered = mlflow.register_model(f"runs:/{resolved.run_id}/{artifact_path}", model_name)

    if not model_tags.get(tagschema.MODEL_ID):
        client.set_registered_model_tag(model_name, tagschema.MODEL_ID, config.model_id)
    if origin and not git.same_remote(model_tags.get(tagschema.MODEL_ORIGIN), origin, workspace_dir):
        # First publish, a legacy model, or a move the user just confirmed (D#77).
        client.set_registered_model_tag(model_name, tagschema.MODEL_ORIGIN, origin)

    # MLflow's registry has no transaction: another publish of this same file
    # could register its own version in the gap between the call above and
    # this re-read. This narrows that window - it does not close it (D#59).
    published_tag = tagschema.published_tag_name(pkl_sha256)
    current_version = client.get_run(resolved.run_id).data.tags.get(published_tag)
    if current_version and current_version != str(registered.version):
        raise _abandon(client, model_name, registered.version, current_version, resolved.run_id)

    client.set_tag(resolved.run_id, "published_by", published_by)
    client.set_tag(resolved.run_id, "published_at", published_at)
    client.set_tag(resolved.run_id, published_tag, str(registered.version))
    client.set_tag(resolved.run_id, "registry_version", registered.version)
    client.set_model_version_tag(model_name, registered.version, "formats",
                                  ",".join(["pkl", *(d.name for d in format_dirs)]))
    client.set_model_version_tag(model_name, registered.version, "source_run", resolved.run_id)
    client.set_model_version_tag(model_name, registered.version, "published_by", published_by)

    return {"version": registered.version, "run_id": resolved.run_id, "model_name": model_name,
            "model_id": config.model_id}
