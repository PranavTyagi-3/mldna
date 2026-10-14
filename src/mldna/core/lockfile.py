from dataclasses import dataclass
from pathlib import Path

import yaml

LOCKFILE_NAME = "models.lock.yaml"
REQUIRED_KEYS = ("version", "format", "dest")


class LockfileError(Exception):
    pass


@dataclass
class LockEntry:
    alias: str
    model_name: str
    version: str
    format: str
    dest: str
    id: str = None


class _DuplicateKeyCheckingLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys instead of yaml.safe_load's
    silent last-one-wins behavior, which would drop a duplicated model pin with
    no warning before mldna ever sees the data."""

    def construct_mapping(self, node, deep=False):
        seen_at_line = {}
        for key_node, _ in node.value:
            key = key_node.value
            line = key_node.start_mark.line + 1
            if key in seen_at_line:
                raise LockfileError(
                    f"{LOCKFILE_NAME} has duplicate key '{key}' (lines {seen_at_line[key]} and {line}). "
                    f"A duplicate key silently discards the earlier entry, which would drop a model "
                    f"pin without warning. Fix: remove or rename one of the '{key}:' entries."
                )
            seen_at_line[key] = line
        return super().construct_mapping(node, deep=deep)


def load_lockfile(workspace_dir):
    path = Path(workspace_dir) / LOCKFILE_NAME
    if not path.exists():
        raise LockfileError(
            f"{LOCKFILE_NAME} not found in {workspace_dir}. "
            f"Fix: create it with entries like:\n"
            f"  <model-name>:\n    version: 1\n    format: onnx\n    dest: ./app/models/<name>"
        )

    try:
        data = yaml.load(path.read_text(), Loader=_DuplicateKeyCheckingLoader) or {}
    except yaml.YAMLError as e:
        mark = getattr(e, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        problem = getattr(e, "problem", None) or str(e)
        raise LockfileError(
            f"{path} has a YAML syntax error{where}: {problem}. "
            f"Fix: correct the syntax at that location and re-run."
        )
    if not data:
        raise LockfileError(f"{LOCKFILE_NAME} has no entries. Fix: add at least one model pin.")

    if not isinstance(data, dict):
        raise LockfileError(
            f"{LOCKFILE_NAME} is {data!r}, not a mapping of model pins. "
            f"Fix: give it the shape:\n"
            f"  <model-name>:\n    version: 1\n    format: onnx\n    dest: ./app/models/<name>"
        )

    entries = []
    seen_dests = {}
    for alias, spec in data.items():
        if not isinstance(spec, dict):
            raise LockfileError(
                f"{LOCKFILE_NAME} entry '{alias}' is {spec!r}, not a mapping. "
                f"Fix: give it the shape:\n"
                f"  {alias}:\n    version: 1\n    format: onnx\n    dest: ./app/models/<name>"
            )

        missing = [key for key in REQUIRED_KEYS if key not in spec]
        if missing:
            raise LockfileError(
                f"{LOCKFILE_NAME} entry '{alias}' is missing {', '.join(missing)}. "
                f"Fix: add the missing key(s) under '{alias}:'."
            )

        for key in ("format", "dest", "model", "id"):
            if key in spec and (not isinstance(spec[key], str) or not spec[key]):
                raise LockfileError(
                    f"{LOCKFILE_NAME} entry '{alias}' has {key}: {spec[key]!r}, which is not text. "
                    f"Fix: give it the shape:\n"
                    f"  {alias}:\n    version: 1\n    format: onnx\n    dest: ./app/models/<name>"
                )

        version = spec["version"]
        if isinstance(version, bool) or not (
            (isinstance(version, int) and version > 0)
            or (isinstance(version, str) and version.isascii() and version.isdigit() and int(version) > 0)
        ):
            raise LockfileError(
                f"{LOCKFILE_NAME} entry '{alias}' has version: {version!r}, which is not a positive whole number. "
                f"Fix: give it the shape:\n"
                f"  {alias}:\n    version: 1\n    format: onnx\n    dest: ./app/models/<name>"
            )

        dest = spec["dest"]
        resolved_dest = (Path(workspace_dir) / dest).resolve()
        if resolved_dest.is_file():
            raise LockfileError(
                f"{LOCKFILE_NAME} entry '{alias}' has dest {dest}, which is an existing file. "
                f"dest is the folder mldna fetches the model into. "
                f"Fix: point dest at a folder (new or mldna-fetched), e.g. ./models/{alias}."
            )
        if resolved_dest in seen_dests:
            other = seen_dests[resolved_dest]
            raise LockfileError(
                f"{LOCKFILE_NAME} entries '{other}' and '{alias}' both resolve dest to "
                f"{resolved_dest}. Two pins sharing a dest would overwrite each other. "
                f"Fix: give '{alias}' its own dest."
            )
        for other_dest, other in seen_dests.items():
            if resolved_dest.is_relative_to(other_dest) or other_dest.is_relative_to(resolved_dest):
                raise LockfileError(
                    f"{LOCKFILE_NAME} entries '{other}' and '{alias}' have nested dests "
                    f"({other_dest} and {resolved_dest}). Fetching the outer one replaces "
                    f"everything inside it, deleting the inner model. "
                    f"Fix: give '{alias}' a dest outside '{other}''s folder."
                )
        seen_dests[resolved_dest] = alias

        # model: defaults to the alias (the YAML key), so an unmodified
        # entry still names its own model - see candidate B in D#55.
        entries.append(LockEntry(
            alias=alias,
            model_name=spec.get("model", alias),
            version=str(spec["version"]),
            format=spec["format"],
            dest=dest,
            id=spec.get("id"),
        ))
    return entries
