from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mldna.core.discovery import DEFAULT_EXCLUDES, DEFAULT_MODEL_GLOBS


@dataclass
class ProjectConfig:
    model_name: str = None
    repo_role: str = None
    model_globs: list = field(default_factory=lambda: list(DEFAULT_MODEL_GLOBS))
    scan_exclude: list = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    subscription_id: str = None
    resource_group: str = None
    workspace: str = None
    tracking_uri: str = None
    mlflow_tracking_uri: str = None
    key_metric: str = None
    require_formats: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        data = yaml.safe_load(Path(path).read_text())
        project = data.get("project", {})
        tracking = data.get("tracking", {})
        azure = data.get("azure", {})
        mlflow_section = data.get("mlflow", {})
        publish = data.get("publish", {})
        return cls(
            model_name=project.get("model_name"),
            repo_role=project.get("repo_role"),
            model_globs=[*DEFAULT_MODEL_GLOBS, *tracking.get("model_globs", [])],
            scan_exclude=[*DEFAULT_EXCLUDES, *tracking.get("scan_exclude", [])],
            subscription_id=azure.get("subscription_id"),
            resource_group=azure.get("resource_group"),
            workspace=azure.get("workspace"),
            tracking_uri=azure.get("tracking_uri"),
            mlflow_tracking_uri=mlflow_section.get("tracking_uri"),
            key_metric=tracking.get("key_metric"),
            require_formats=publish.get("require_formats", []),
            raw=data,
        )

    @classmethod
    def load(cls, workspace_dir):
        path = Path(workspace_dir) / "mldna.yaml"
        if not path.exists():
            return cls()
        return cls.from_yaml(path)
