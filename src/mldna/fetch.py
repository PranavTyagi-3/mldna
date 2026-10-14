import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
import yaml
from mlflow.exceptions import MlflowException

from mldna.core import ownership
from mldna.core import tags as tagschema
from mldna.core.lockfile import load_lockfile
from mldna.core.manifest import ManifestError, resolve_format, verify_format_files
from mldna.trackinguri import resolve_and_set_tracking_uri

MARKER_NAME = ownership.MARKER_NAME

# D#50: the `recalled` tag is the one admin-only safety switch in the whole
# design (D#8/D#9) - an equality check against the literal string "true" made
# it trivial to defeat by accident: an admin typing `True`, `yes`, or `1` into
# the MLflow UI (the realistic path - nobody is scripting this) produced a tag
# that was silently ignored, and the recalled model fetched anyway. The two
# failure modes are not symmetric: serving a recalled model is far worse than
# refusing a healthy one, so this must fail SAFE, not fail exact. The rule is
# therefore inverted from a normal flag parse: if the tag is PRESENT at all, the
# version is treated as recalled UNLESS its value - compared case-insensitively
# and stripped of surrounding whitespace - is in this small, explicit false set.
# Do not "simplify" this back to `== "true"`; that is the exact bug this fixes.
_RECALL_FALSE_VALUES = {"false", "no", "0", "off", ""}


def _is_recalled(tag_value):
    return tag_value is not None and tag_value.strip().lower() not in _RECALL_FALSE_VALUES


class FetchError(Exception):
    pass


def _validated_dest(workspace_dir, entry):
    label = f"{entry.model_name} v{entry.version} (models.lock.yaml dest '{entry.dest}')"
    try:
        return ownership.validate_contained(workspace_dir, entry.dest, label)
    except ownership.OwnershipError as e:
        raise FetchError(str(e)) from e


def _stage_download(mv_source, format_entry, tmp_dir):
    path = format_entry["path"].rstrip("/")
    downloaded = mlflow.artifacts.download_artifacts(f"{mv_source}/{path}", dst_path=tmp_dir)
    if "files" in format_entry:
        # a directory subtree already lands with its files at the root - stage as-is
        return Path(downloaded)
    # a single file: stage it inside its own dir so dest/<name> is what lands
    staged = Path(tempfile.mkdtemp(dir=tmp_dir))
    shutil.copy2(downloaded, staged / Path(path).name)
    return staged


def _format_digest(format_entry):
    # Identifies this exact format entry (its path, and its sha256 or its
    # per-file hashes) - the same model+version+format with different bytes
    # published under it digests differently. sort_keys makes it independent
    # of the dict's insertion order.
    return hashlib.sha256(json.dumps(format_entry, sort_keys=True).encode()).hexdigest()


def _bundle_files(format_entry):
    if "sha256" in format_entry:
        return [Path(format_entry["path"]).name]
    return list(format_entry.get("files", {}))


def _marker_matches(marker, entry, digest):
    # D#53: an old bare-version marker (flagged "legacy" by read_marker) is
    # ownership evidence but never identity evidence - it carries no model
    # name, format or digest to compare, so it can never produce a skip.
    if marker is None or marker.get("legacy"):
        return False
    return (
        marker.get("model_name") == entry.model_name
        and marker.get("version") == entry.version
        and marker.get("format") == entry.format
        and marker.get("manifest_digest") == digest
    )


def _namespaced_hint(client, bare_name):
    """issue #16 migration aid: a pin written before namespacing exists is a
    bare name that will never resolve again. Best-effort only - a search
    failure here must never mask the real not-found error, so any exception
    is swallowed and no hint is offered."""
    try:
        candidates = client.search_registered_models(filter_string=f"name LIKE '%-{bare_name}'")
    except Exception:
        return None
    return candidates[0].name if candidates else None


def _registry_lookup_error(entry, uri, mode, e, client=None):
    # Absent-vs-unreachable is discriminated on mlflow's structured error_code,
    # never on message text. Verified per backend: a genuine absence is the only
    # thing that carries RESOURCE_DOES_NOT_EXIST - the file store raises it
    # directly for a missing model/version, and the REST stores (plain `mlflow:`
    # server, D#34, and Azure ML) map the registry's 404 onto it through
    # RestException. Infrastructure failures never reach that code: an
    # unreachable host, DNS failure or timeout surfaces as
    # MlflowException/INTERNAL_ERROR wrapping the requests error, auth as
    # UNAUTHENTICATED/PERMISSION_DENIED, a server fault as INTERNAL_ERROR. An
    # exception that is not an MlflowException at all is undeterminable, and is
    # labeled as such rather than attributed to either cause.
    if isinstance(e, MlflowException):
        if e.error_code == "RESOURCE_DOES_NOT_EXIST":
            hint = _namespaced_hint(client, entry.model_name) if client else None
            if hint:
                return FetchError(
                    f"no {entry.model_name} in the registry; did you mean {hint}? Models "
                    f"are now namespaced and the old bare name no longer "
                    f"resolves. Fix: update the pin in models.lock.yaml - change "
                    f"'{entry.alias}: model: {entry.model_name}' (or the key itself, if "
                    f"it has no explicit 'model:') to {hint}."
                )
            return FetchError(
                f"{entry.model_name} v{entry.version} not found in the registry. "
                f"Fix: check models.lock.yaml, or confirm the version was published. ({e})"
            )
        return FetchError(
            f"could not reach the model registry at {uri} ({mode}) to look up "
            f"{entry.model_name} v{entry.version}. The lookup failed with "
            f"{e.error_code}, which reports an unreachable or failing registry - "
            f"the pinned version is not reported missing, so the pin is not the "
            f"problem. Fix: confirm the tracking server is up and reachable with "
            f"working credentials, then re-run: mldna doctor. If your server "
            f"rate-limits, raise MLFLOW_HTTP_REQUEST_MAX_RETRIES - mldna caps "
            f"it to fail fast. ({e})"
        )
    return FetchError(
        f"the registry lookup for {entry.model_name} v{entry.version} at {uri} "
        f"({mode}) failed with an error mldna does not recognize, so it cannot tell "
        f"whether the registry is unreachable or the version is genuinely absent. "
        f"Fix: check the registry is reachable with `mldna doctor`; if it is, "
        f"confirm the version was published. ({type(e).__name__}: {e})"
    )


def _check_model_id(client, entry):
    """D#69: the pinned id must equal the id stamped on the registered model.
    Without an id the pin is unverified (back-compat) and a note says so."""
    registered = client.get_registered_model(entry.model_name)
    registered_id = registered.tags.get(tagschema.MODEL_ID)
    if entry.id is None:
        hint = f" Add `id: {registered_id}` under '{entry.alias}' to verify it." if registered_id else ""
        print(f"{entry.model_name}: pin has no id, so it is not verified to be the model you meant.{hint}")
    elif registered_id != entry.id:
        raise FetchError(
            f"{entry.model_name} in the registry has model id {registered_id or 'none'}, "
            f"but models.lock.yaml pins id {entry.id} for '{entry.alias}'. Names are "
            f"user-chosen and can collide, so this may be a different project's model. "
            f"Fix: confirm the right model name with its publisher, then correct "
            f"'model:' or 'id:' under '{entry.alias}' in models.lock.yaml."
        )


def fetch(workspace_dir="."):
    workspace_dir = Path(workspace_dir)
    uri, mode = resolve_and_set_tracking_uri(workspace_dir)
    print(f"Tracking mode: {mode}")
    entries = load_lockfile(workspace_dir)
    client = mlflow.tracking.MlflowClient()

    results = []
    for entry in entries:
        dest = _validated_dest(workspace_dir, entry)

        try:
            mv = client.get_model_version(entry.model_name, entry.version)
        except Exception as e:
            raise _registry_lookup_error(entry, uri, mode, e, client=client) from e

        _check_model_id(client, entry)

        if tagschema.is_abandoned(mv.tags):
            raise FetchError(
                f"{entry.model_name} v{entry.version} is tagged abandoned "
                f"({mv.tags.get(tagschema.ABANDONED_REASON, 'no reason recorded')}) - "
                f"it was registered by a publish that lost a concurrent race and "
                f"failed, so it is not the version its run published. Fix: "
                f"pin a different version under '{entry.alias}' in models.lock.yaml "
                f"(the reason names the version that won)."
            )

        recalled_tag = mv.tags.get("recalled")
        if _is_recalled(recalled_tag):
            raise FetchError(
                f"{entry.model_name} v{entry.version} carries a 'recalled' tag "
                f"(value {recalled_tag!r}) - refusing to fetch, since a recalled "
                f"version carrying this tag is the one thing this registry must "
                f"never serve. Fix: pin a different version in models.lock.yaml; "
                f"contact the model owner. If this tag was set by mistake, clear "
                f"it or set it to one of: {sorted(_RECALL_FALSE_VALUES - {''})}."
            )

        with tempfile.TemporaryDirectory() as tmp:
            try:
                manifest_path = mlflow.artifacts.download_artifacts(
                    f"{mv.source}/manifest.yaml", dst_path=tmp
                )
            except Exception as e:
                raise FetchError(
                    f"could not download manifest.yaml for {entry.model_name} "
                    f"v{entry.version}: {e}. The manifest names and checksums every "
                    f"file in the bundle, so fetch cannot verify anything without "
                    f"it. Fix: check the tracking store is reachable with `mldna "
                    f"doctor`, then re-run mldna fetch."
                ) from e
            manifest_dict = yaml.safe_load(Path(manifest_path).read_text())
            format_entry = resolve_format(manifest_dict, entry.format)
            digest = _format_digest(format_entry)

            repaired = False
            marker = ownership.read_marker(dest)
            if _marker_matches(marker, entry, digest):
                # identity matches: the marker alone is still not evidence
                # on its own - re-hash what is on disk
                try:
                    verify_format_files(dest, format_entry)
                except ManifestError as e:
                    print(
                        f"{entry.model_name} v{entry.version}: on-disk copy failed verification "
                        f"- re-fetching. ({e})"
                    )
                    repaired = True
                else:
                    results.append({"model_name": entry.model_name, "version": entry.version,
                                    "skipped": True, "repaired": False})
                    continue

            label = f"{entry.model_name} v{entry.version} (models.lock.yaml dest '{entry.dest}')"
            new_files = _bundle_files(format_entry)
            if marker is not None:
                # D#79: the installed bundle's own file list; a marker without
                # one (legacy) falls back to the new list, so unknown files refuse
                try:
                    ownership.require_no_foreign_files(dest, marker.get("files", new_files), label)
                except ownership.OwnershipError as e:
                    raise FetchError(str(e)) from e

            staged = _stage_download(mv.source, format_entry, tmp)
            verify_format_files(staged, format_entry)
            try:
                ownership.swap_into_dest(staged, dest, label)
            except ownership.OwnershipError as e:
                raise FetchError(str(e)) from e

        ownership.write_marker(dest, entry.model_name, entry.version, entry.format, digest, new_files)
        results.append({"model_name": entry.model_name, "version": entry.version,
                        "skipped": False, "repaired": repaired})

    return results
