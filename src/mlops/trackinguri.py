import sys
from pathlib import Path

import mlflow

from mlops.core.config import ProjectConfig

AZURE_KEYS = ("subscription_id", "resource_group", "workspace")


class TrackingUriError(Exception):
    pass


def _load_config(workspace_dir):
    config_path = Path(workspace_dir) / "mlops.yaml"
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
    # plain-ASCII output mlops itself prints.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")


def resolve_and_set_tracking_uri(workspace_dir):
    config = _load_config(workspace_dir)

    if config is None or "azure" not in config.raw:
        uri = f"file:{Path(workspace_dir) / 'mlruns'}"
        mlflow.set_tracking_uri(uri)
        return uri, "local"

    _make_stdout_unicode_safe()

    try:
        import azureml.mlflow  # noqa: F401 - registers the "azureml" mlflow store scheme
    except ImportError as e:
        # azure-ai-ml (needed below for discovery/bootstrap) can be present
        # without azureml-mlflow, or vice versa - and unlike a missing
        # azure-ai-ml import, a missing azureml-mlflow doesn't fail until the
        # *first real mlflow call* (mlflow.start_run(), etc.), deep inside
        # mlflow's own store-registry code, as an
        # UnsupportedModelRegistryStoreURIException that never mentions mlops
        # or the fix at all. Fail here instead, at config-resolution time.
        raise TrackingUriError(
            f'Azure ML tracking needs the `azure` extra. Fix: pip install "mlops[azure]" ({e})'
        )

    from mlops.azure.auth import resolve_aml_credential

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
            f"mlops.yaml has an azure: section but is missing {', '.join(missing)} "
            f"(or a single tracking_uri key). Add the missing key(s), or remove the "
            f"azure: section entirely to use local tracking."
        )

    try:
        from mlops.azure.tracking import bootstrap_tracking_uri
    except ImportError as e:
        raise TrackingUriError(
            f'Azure ML tracking needs the `azure` extra. Fix: pip install "mlops[azure]" ({e})'
        )

    credential, mode = resolve_aml_credential(workspace_dir)
    uri = bootstrap_tracking_uri(
        config.subscription_id, config.resource_group, config.workspace, credential
    )
    mlflow.set_tracking_uri(uri)
    return uri, mode
