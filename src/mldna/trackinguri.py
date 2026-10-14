import os
import re
import sys
from pathlib import Path

import mlflow

from mldna.core.config import ProjectConfig

AZURE_KEYS = ("subscription_id", "resource_group", "workspace")
TRACKING_SECTIONS = ("azure", "mlflow", "filesystem")
FILESYSTEM_MODE = "filesystem store"


class TrackingUriError(Exception):
    pass


def _load_config(workspace_dir):
    config_path = Path(workspace_dir) / "mldna.yaml"
    if not config_path.exists():
        return None
    return ProjectConfig.from_yaml(config_path)


def _make_stdout_unicode_safe():
    # mlflow prints emoji ("View run ... at: ...") when it talks to a real
    # REST tracking server (never for the local file store) - on Windows'
    # default console codepage that write raises UnicodeEncodeError, and it
    # happens *inside* end_run()/set_terminated() *before* the run's terminal
    # status REST call, so an unhandled crash here left real runs stuck
    # RUNNING forever (confirmed live against mlops-spike-ws). errors=
    # "replace" degrades to "?" instead of crashing; it never affects the
    # plain-ASCII output mldna itself prints.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")


def _bound_http_retries():
    # mlflow's shipped retry policy is tuned for a managed cloud endpoint that
    # rate-limits for minutes: 7 retries at backoff_factor 2 sleeps
    # 0+4+9+16+32+64+120 (urllib3 skips the first backoff and caps each sleep at
    # 120) = ~246s of pure waiting before a down server is ever reported. That
    # is the whole of the ~4min publish / ~4.5min fetch + doctor stalls a user
    # measured against a self-hosted server. 3 retries at factor 1 sleeps
    # 0+2+4 = 6s (~9s worst case with mlflow's 1.0s jitter), still enough to
    # ride out a dropped connection or a brief restart. Only the *number and
    # spacing of retries* is bounded - MLFLOW_HTTP_REQUEST_TIMEOUT and the
    # artifact-plane timeouts stay at mlflow's generous defaults, because a
    # multi-GB bundle transfer legitimately needs them. setdefault, so an
    # operator who needs mlflow's original patience just sets the env var.
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "3")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")


def _resolve_filesystem_store(workspace_dir, configured_path):
    if not configured_path:
        raise TrackingUriError(
            "mldna.yaml has a filesystem: section but no path key. "
            "Fix: add `path: \"/shared/mlruns\"` under filesystem: (absolute, "
            "repo-relative, or containing any environment variable your shell "
            "defines, such as $SHARED_DRIVE), or remove the filesystem: section "
            "entirely to use local tracking."
        )

    # expandvars/expanduser so one committed value resolves per machine:
    # mldna.yaml is committed, and an absolute path is correct on exactly one
    # developer's disk.
    expanded = os.path.expanduser(os.path.expandvars(str(configured_path)))
    unset = re.findall(r"\$(?:\{(\w+)\}|(\w+))", expanded)
    if unset:
        names = ", ".join(a or b for a, b in unset)
        raise TrackingUriError(
            f"{names} is not set (filesystem.path in mldna.yaml uses it). "
            f"Fix: set {names}, or use a literal path."
        )
    path = Path(expanded)
    if not path.is_absolute():
        path = Path(workspace_dir) / path
    path = path.resolve()

    if not path.exists():
        raise TrackingUriError(
            f"mldna.yaml's filesystem: path does not exist (resolved to {path}). "
            "mldna will not create a tracking store it was told to read, because "
            "an empty one would report every pinned version as not found in the "
            "registry. Fix: correct `path:` under filesystem: in mldna.yaml, or "
            f"create the store: mkdir -p {path}"
        )

    return f"file:{path}"


def resolve_and_set_tracking_uri(workspace_dir):
    config = _load_config(workspace_dir)

    sections = [s for s in TRACKING_SECTIONS if config is not None and s in config.raw]

    if len(sections) > 1:
        named = ", ".join(f"{s}:" for s in sections)
        raise TrackingUriError(
            f"mldna.yaml has {named} sections - only one tracking source may be "
            "configured, so the resolved tracking URI is never ambiguous. "
            "Fix: delete all but one of those sections from mldna.yaml."
        )

    if "mlflow" in sections:
        if not config.mlflow_tracking_uri:
            raise TrackingUriError(
                "mldna.yaml has an mlflow: section but no tracking_uri key. "
                "Fix: add `tracking_uri: \"http://host:port\"` under mlflow: for an "
                "MLflow server, or use a `filesystem:` section with a `path:` key to "
                "read an existing file-based store, or remove the mlflow: "
                "section entirely to use local tracking."
            )
        if config.mlflow_tracking_uri.startswith("file:"):
            # A file: URI under mlflow: is a file-based store on disk, not a
            # server - mlflow accepts it either way, but reporting it as
            # "remote mlflow server" would misdescribe what is in use, so it
            # gets the filesystem label (D#39).
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            return config.mlflow_tracking_uri, FILESYSTEM_MODE
        _make_stdout_unicode_safe()
        _bound_http_retries()
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        return config.mlflow_tracking_uri, "remote mlflow server"

    if "filesystem" in sections:
        uri = _resolve_filesystem_store(workspace_dir, config.filesystem_path)
        mlflow.set_tracking_uri(uri)
        return uri, FILESYSTEM_MODE

    if "azure" not in sections:
        uri = f"file:{Path(workspace_dir) / 'mlruns'}"
        mlflow.set_tracking_uri(uri)
        return uri, "local"

    _make_stdout_unicode_safe()
    _bound_http_retries()

    try:
        import azureml.mlflow  # noqa: F401 - registers the "azureml" mlflow store scheme
    except ImportError as e:
        # azure-ai-ml (needed below for discovery/bootstrap) can be present
        # without azureml-mlflow, or vice versa - and unlike a missing
        # azure-ai-ml import, a missing azureml-mlflow doesn't fail until the
        # *first real mlflow call* (mlflow.start_run(), etc.), deep inside
        # mlflow's own store-registry code, as an
        # UnsupportedModelRegistryStoreURIException that never mentions mldna
        # or the fix at all. Fail here instead, at config-resolution time.
        raise TrackingUriError(
            f'Azure ML tracking needs the `azure` extra. Fix: pip install "mldna-lineage[azure]" ({e})'
        )

    from mldna.azure.auth import resolve_aml_credential

    if config.tracking_uri:
        # Service Principal flow: the endpoint was pasted directly, so there's
        # nothing to look up via MLClient - resolve_aml_credential is still
        # called for its env-var-setting side effect (see azure/auth.py), even
        # though the returned credential object itself isn't needed here.
        _, mode = resolve_aml_credential(workspace_dir)
        mlflow.set_tracking_uri(config.tracking_uri)
        return config.tracking_uri, mode

    missing = [key for key in AZURE_KEYS if not getattr(config, key)]
    if missing:
        raise TrackingUriError(
            f"mldna.yaml has an azure: section but is missing {', '.join(missing)} "
            f"(or a single tracking_uri key). Add the missing key(s), or remove the "
            f"azure: section entirely to use local tracking."
        )

    try:
        from mldna.azure.tracking import bootstrap_tracking_uri
    except ImportError as e:
        raise TrackingUriError(
            f'Azure ML tracking needs the `azure` extra. Fix: pip install "mldna-lineage[azure]" ({e})'
        )

    credential, mode = resolve_aml_credential(workspace_dir)
    uri = bootstrap_tracking_uri(
        config.subscription_id, config.resource_group, config.workspace, credential
    )
    mlflow.set_tracking_uri(uri)
    return uri, mode
