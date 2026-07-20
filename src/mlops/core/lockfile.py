from dataclasses import dataclass
from pathlib import Path

import yaml

LOCKFILE_NAME = "models.lock.yaml"
REQUIRED_KEYS = ("version", "format", "dest")


class LockfileError(Exception):
    pass


@dataclass
class LockEntry:
    model_name: str
    version: str
    format: str
    dest: str


def load_lockfile(workspace_dir):
    path = Path(workspace_dir) / LOCKFILE_NAME
    if not path.exists():
        raise LockfileError(
            f"{LOCKFILE_NAME} not found in {workspace_dir}. "
            f"Fix: create it with entries like:\n"
            f"  <model-name>:\n    version: 1\n    format: onnx\n    dest: ./app/models/<name>"
        )

    data = yaml.safe_load(path.read_text()) or {}
    if not data:
        raise LockfileError(f"{LOCKFILE_NAME} has no entries. Fix: add at least one model pin.")

    entries = []
    for model_name, spec in data.items():
        missing = [key for key in REQUIRED_KEYS if key not in spec]
        if missing:
            raise LockfileError(
                f"{LOCKFILE_NAME} entry '{model_name}' is missing {', '.join(missing)}. "
                f"Fix: add the missing key(s) under '{model_name}:'."
            )
        entries.append(LockEntry(
            model_name=model_name,
            version=str(spec["version"]),
            format=spec["format"],
            dest=spec["dest"],
        ))
    return entries
