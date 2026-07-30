import sys
from pathlib import Path

import mlflow

from mldna.core.config import ProjectConfig

AZURE_KEYS = ("subscription_id", "resource_group", "workspace")


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


def resolve_and_set_tracking_uri(workspace_dir):
    config = _load_config(workspace_dir)

    has_mlflow_section = config is not None and "mlflow" in config.raw
    has_azure_section = config is not None and "azure" in config.raw

    if has_mlflow_section and has_azure_section:
        raise TrackingUriError(
            "mldna.yaml has both mlflow: and azure: sections - only one tracking "
            "source may be configured. Fix: remove the mlflow: section to use "
            "Azure ML, or remove the azure: section to use the remote MLflow server."
        )

    if has_mlflow_section:
        if not config.mlflow_tracking_uri:
            raise TrackingUriError(
                "mldna.yaml has an mlflow: section but no tracking_uri key. "
                "Fix: add `tracking_uri: \"http://host:port\"` under mlflow:, or "
                "remove the mlflow: section entirely to use local tracking."
            )
        _make_stdout_unicode_safe()
        mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        return config.mlflow_tracking_uri, "remote mlflow server"

    if not has_azure_section:
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
