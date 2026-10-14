from dataclasses import dataclass, field
from pathlib import Path

import yaml

from mldna.core.discovery import DEFAULT_EXCLUDES, DEFAULT_MODEL_GLOBS


@dataclass
class ProjectConfig:
    model_name: str = None
    namespace: str = None
    model_id: str = None
    repo_role: str = None
    model_globs: list = field(default_factory=lambda: list(DEFAULT_MODEL_GLOBS))
    scan_exclude: list = field(default_factory=lambda: list(DEFAULT_EXCLUDES))
    subscription_id: str = None
    resource_group: str = None
    workspace: str = None
    tracking_uri: str = None
    mlflow_tracking_uri: str = None
    filesystem_path: str = None
    key_metric: str = None
    require_formats: list = field(default_factory=list)
    scan_limit: int = 200
    env_allowlist: list = field(default_factory=list)
    record_argv: bool = False
    max_file_mb: float = 5
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path):
        data = yaml.safe_load(Path(path).read_text())
        project = data.get("project", {})
        tracking = data.get("tracking", {})
        azure = data.get("azure", {})
        mlflow_section = data.get("mlflow", {})
        filesystem = data.get("filesystem", {})
        publish = data.get("publish", {})
        push = data.get("push", {})
        record_argv = tracking.get("record_argv", False)
        if not isinstance(record_argv, bool):
            raise ValueError(
                f"tracking.record_argv must be true or false, got {record_argv!r}. "
                "Command-line arguments can contain secrets, so recording them is "
                "an explicit boolean opt-in. "
                f"Fix: set tracking.record_argv: true or false in {path}."
            )
        return cls(
            model_name=project.get("model_name"),
            namespace=project.get("namespace"),
            model_id=project.get("model_id"),
            repo_role=project.get("repo_role"),
            model_globs=[*DEFAULT_MODEL_GLOBS, *tracking.get("model_globs", [])],
            scan_exclude=[*DEFAULT_EXCLUDES, *tracking.get("scan_exclude", [])],
            subscription_id=azure.get("subscription_id"),
            resource_group=azure.get("resource_group"),
            workspace=azure.get("workspace"),
            tracking_uri=azure.get("tracking_uri"),
            mlflow_tracking_uri=mlflow_section.get("tracking_uri"),
            filesystem_path=filesystem.get("path"),
            key_metric=tracking.get("key_metric"),
            scan_limit=tracking.get("scan_limit", 200),
            env_allowlist=tracking.get("env_allowlist", []),
            record_argv=record_argv,
            max_file_mb=push.get("max_file_mb", 5),
            require_formats=publish.get("require_formats", []),
            raw=data,
        )

    @classmethod
    def load(cls, workspace_dir):
        path = Path(workspace_dir) / "mldna.yaml"
        if not path.exists():
            return cls()
        return cls.from_yaml(path)

    def full_model_name(self):
        """The namespaced name used as both the experiment and the registry
        name (issue #16) - never the bare model_name alone, so two teams'
        same-named models can never collide."""
        if not self.namespace or not self.model_name:
            return None
        return f"{self.namespace}-{self.model_name}"


def find_parent_project(workspace_dir):
    """Walk up from workspace_dir's parent looking for an ancestor mldna.yaml.
    Never checks workspace_dir itself - callers already know whether it has
    one. Returns the found path, or None if no ancestor project exists."""
    current = Path(workspace_dir).resolve().parent
    while True:
        candidate = current / "mldna.yaml"
        if candidate.exists():
            return candidate
        parent = current.parent
        if parent == current:
            return None
        current = parent


def missing_project_hint(workspace_dir):
    """Message for a command that found no usable mldna.yaml in
    workspace_dir. Only ever suggests `mldna init` when no ancestor project
    exists - suggesting it over an existing parent project would scaffold a
    confusing second project on top of it (D#57)."""
    parent = find_parent_project(workspace_dir)
    if parent:
        return (
            f"No mldna.yaml in {workspace_dir}, but there is one at {parent} - "
            f"run mldna from there, or pass --dir {parent.parent}."
        )
    return (
        f"No mldna.yaml found in {workspace_dir} or any parent directory. "
        "Fix: mldna init <model-name>."
    )


def missing_namespace_hint(workspace_dir):
    """Message for a producer project with a model_name but no namespace to
    prefix it with. No legacy fallback to the bare name exists (issue #16) -
    that would leave two naming regimes in the registry forever, so this
    refuses rather than guessing."""
    return (
        f"mldna.yaml has project.model_name but no project.namespace, so the "
        f"experiment and registry name would be unprefixed and could collide "
        f"with another team's same-named model. Fix: add project.namespace: "
        f"<your-namespace> to {workspace_dir}/mldna.yaml."
    )
