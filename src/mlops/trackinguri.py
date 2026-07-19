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


def resolve_and_set_tracking_uri(workspace_dir):
    config = _load_config(workspace_dir)

    if config is None or "azure" not in config.raw:
        uri = f"file:{Path(workspace_dir) / 'mlruns'}"
        mlflow.set_tracking_uri(uri)
        return uri, "local"

    missing = [key for key in AZURE_KEYS if not getattr(config, key)]
    if missing:
        raise TrackingUriError(
            f"mlops.yaml has an azure: section but is missing {', '.join(missing)}. "
            f"Add the missing key(s), or remove the azure: section entirely to use "
            f"local tracking."
        )

    from mlops.azure.auth import resolve_aml_credential
    from mlops.azure.tracking import bootstrap_tracking_uri

    credential, mode = resolve_aml_credential(workspace_dir)
    uri = bootstrap_tracking_uri(
        config.subscription_id, config.resource_group, config.workspace, credential
    )
    mlflow.set_tracking_uri(uri)
    return uri, mode
