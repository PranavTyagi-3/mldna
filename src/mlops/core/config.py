from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mlops.core.discovery import DEFAULT_EXCLUDES, DEFAULT_MODEL_GLOBS


@dataclass
class ProjectConfig:
    model_name: str
    repo_role: str
    model_globs: list = field(default_factory=lambda: list(DEFAULT_MODEL_GLOBS))
    scan_exclude: list = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    subscription_id: str = None
    resource_group: str = None
    workspace: str = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        data = yaml.safe_load(Path(path).read_text())
        project = data.get("project", {})
        tracking = data.get("tracking", {})
        azure = data.get("azure", {})
        return cls(
            model_name=project["model_name"],
            repo_role=project["repo_role"],
            model_globs=[*DEFAULT_MODEL_GLOBS, *tracking.get("model_globs", [])],
            scan_exclude=[*DEFAULT_EXCLUDES, *tracking.get("scan_exclude", [])],
            subscription_id=azure.get("subscription_id"),
            resource_group=azure.get("resource_group"),
            workspace=azure.get("workspace"),
            raw=data,
        )
