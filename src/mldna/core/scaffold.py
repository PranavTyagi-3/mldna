import re
import subprocess
import uuid
from pathlib import Path

import yaml

from mldna import __version__
from mldna.core import blobauth, dvc

GITIGNORE = "models/\noutputs/\n.mldna/\nmlruns/\n.dvc/cache/\n.env\n.env.*\n*.pem\n*.key\n"

MAIN_PY = "import mldna\n\nwith mldna.track_run():\n    pass\n"

CONSUMER_GITIGNORE_ENTRY = ".mldna/\n"

# Commented out, so an unedited scaffold behaves exactly as a bare one: these
# keys are parsed by core/config.py but were otherwise undiscoverable. Shape
# follows SPEC 3.1's own commented-optional-section style.
OPTIONAL_SECTIONS_COMMENT = (
    "# Optional settings - uncomment what you need:\n"
    "#\n"
    "# tracking:\n"
    "#   key_metric: accuracy        # metric `mldna runs` shows per run;\n"
    "#                               #   never guessed when unset\n"
    "#   model_globs:                # extends the built-in model-file patterns\n"
    '#     - "*.tflite"\n'
    "#   scan_exclude:               # extra paths the model scan skips\n"
    '#     - "checkpoints_tmp/"\n'
    "#   scan_limit: 200             # how many recent runs publish searches for your file\n"
    "#   env_allowlist: []           # env vars to record per run, e.g. [SEED]; only these,\n"
    "#                               #   never \"all inputs\"\n"
    "#   record_argv: false          # WARNING: command-line arguments may contain secrets and\n"
    "#                               #   are readable by everyone with access to the tracking server\n"
    "#\n"
    "# publish:\n"
    "#   require_formats: []         # e.g. [openvino] to refuse pkl-only bundles\n"
    "#\n"
    "# push:\n"
    "#   max_file_mb: 5              # `mldna push` asks before committing a larger file\n"
)

# The dvc: section is written by `mldna init` and read by nothing - blob auth
# and every dvc invocation read .dvc/config instead. Labeled rather than
# parsed, so there stays exactly one source of truth (D#44).
DVC_MIRROR_COMMENT = (
    "# Informational mirror of the DVC remote `mldna init` configured - mldna\n"
    "# reads .dvc/config, never these values. Repoint storage with\n"
    "# `dvc remote modify <remote> url <url>` (names: `dvc remote list`).\n"
)

LOCKFILE_STUB = (
    "# Pin exact registry versions here - `mldna fetch` resolves ONLY these\n"
    "# exact versions, never latest/stage. Example:\n"
    "#\n"
    "# fraud-detector:\n"
    "#   version: 9\n"
    "#   format: openvino\n"
    "#   dest: ./app/models/fraud\n"
    "#\n"
    "# For A/B, shadow or canary: give the second pin its own alias key and\n"
    "# state the model it pins explicitly - 'model' defaults to the key\n"
    "# otherwise, so every entry above keeps working unchanged.\n"
    "#\n"
    "# fraud-detector-canary:\n"
    "#   model: fraud-detector\n"
    "#   version: 10\n"
    "#   format: openvino\n"
    "#   dest: ./app/models/fraud-canary\n"
)

# Copy-paste starting point only - the validated CI gate (pin-exists,
# format+provenance present, recalled fails, staleness warns) is a separate,
# later piece. Supports the shared backends (a local store sits on the
# producer's disk, so CI cannot read it): filesystem: (path via mldna.yaml),
# mlflow: (self-run server), or azure (service-principal secrets). Uncomment the section for your backend.
CI_SNIPPET = (
    "# mldna fetch - pick ONE backend section below and uncomment it.\n"
    "# (The install step works for all backends.)\n"
    "#\n"
    "- name: Install mldna\n"
    f"  run: pip install 'mldna-lineage=={__version__}'\n"
    "#\n"
    "# BACKEND OPTION 1: Filesystem store (mldna.yaml has `filesystem: path:`)\n"
    "# Ensure the path is reachable from this CI runner (e.g., mounted NFS).\n"
    "# Uncomment to use:\n"
    "#- name: Fetch pinned models\n"
    "#  run: mldna fetch\n"
    "#\n"
    "# BACKEND OPTION 2: MLflow server (mldna.yaml has `mlflow: tracking_uri:`)\n"
    "# Add auth if the server requires it (see MLflow's docs for\n"
    "# MLFLOW_TRACKING_USERNAME, MLFLOW_TRACKING_PASSWORD, MLFLOW_TRACKING_TOKEN).\n"
    "# Example with username/password:\n"
    "#- name: Fetch pinned models\n"
    "#  env:\n"
    "#    MLFLOW_TRACKING_USERNAME: ${{ secrets.MLFLOW_USER }}\n"
    "#    MLFLOW_TRACKING_PASSWORD: ${{ secrets.MLFLOW_PASSWORD }}\n"
    "#  run: mldna fetch\n"
    "#\n"
    "# BACKEND OPTION 3: Azure ML (mldna.yaml has `azure:` or `tracking_uri:`)\n"
    "# Requires service-principal secrets. Add these to your CI secrets:\n"
    "# AZURE_TENANT_ID, AZURE_CLIENT_ID, AZURE_CLIENT_SECRET.\n"
    "# Uncomment to use:\n"
    "#- name: Fetch pinned models\n"
    "#  env:\n"
    "#    AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}\n"
    "#    AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}\n"
    "#    AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}\n"
    f"#  run: pip install 'mldna-lineage[azure]=={__version__}' && mldna fetch\n"
)


def _default_namespace(project_dir):
    """Zero-friction default (issue #16) - the project directory name,
    slugified. Written once by init; stays editable in mldna.yaml afterward."""
    slug = re.sub(r"[^a-z0-9]+", "-", Path(project_dir).resolve().name.lower()).strip("-")
    return slug or "project"


def _write_config(config_path, data):
    """Single writer for mldna.yaml: yaml.dump plus the comments yaml.dump
    cannot carry (it drops any comment already in the file)."""
    lines = yaml.dump(data, sort_keys=False).splitlines(keepends=True)
    if "dvc" in data and "dvc:\n" in lines:
        lines.insert(lines.index("dvc:\n"), DVC_MIRROR_COMMENT)
    text = "".join(lines)
    if "tracking" not in data and "publish" not in data and "push" not in data:
        text += "\n" + OPTIONAL_SECTIONS_COMMENT
    config_path.write_text(text)


def _load_config(project_dir):
    config_path = Path(project_dir) / "mldna.yaml"
    return config_path, yaml.safe_load(config_path.read_text()) or {}


def ensure_model_id(project_dir, new=False):
    """D#69: the immutable model id. Idempotent - an existing id is returned
    untouched unless `new` (a project copied from another, D#77). Returns
    (model_id, created)."""
    config_path, data = _load_config(project_dir)
    project = data.setdefault("project", {})
    if project.get("model_id") and not new:
        return project["model_id"], False
    project["model_id"] = str(uuid.uuid4())
    _write_config(config_path, data)
    return project["model_id"], True


def _write_if_absent(path, content):
    if path.exists():
        return False
    path.write_text(content)
    return True


def scaffold_producer(project_dir, model_name, namespace=None):
    project_dir = Path(project_dir)
    namespace = namespace or _default_namespace(project_dir)
    created = []

    for name in ("dataset", "models", "outputs"):
        path = project_dir / name
        if not path.exists():
            path.mkdir(parents=True)
            created.append(str(path))

    gitignore_path = project_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(GITIGNORE)
        created.append(str(gitignore_path))
    else:
        existing = gitignore_path.read_text()
        # Compare whole lines, not substrings: a repo with `.env.example` or
        # `trained_models/` already in .gitignore would otherwise look as though
        # it covered `.env` and `models/`, and those entries would be skipped.
        present = {line.strip() for line in existing.splitlines()}
        missing = [line for line in GITIGNORE.splitlines() if line and line not in present]
        if missing:
            with gitignore_path.open("a") as f:
                for line in missing:
                    f.write(line + "\n")

    for name, content in (
        ("main.py", MAIN_PY),
        ("params.yaml", ""),
        ("requirements.txt", f"mldna-lineage=={__version__}\n"),
        ("mldna.yaml", yaml.dump(
            {"project": {"model_name": model_name, "repo_role": "producer", "namespace": namespace,
                         "model_id": str(uuid.uuid4())}},
            sort_keys=False,
        ) + "\n" + OPTIONAL_SECTIONS_COMMENT),
    ):
        if _write_if_absent(project_dir / name, content):
            created.append(str(project_dir / name))

    if not (project_dir / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
        created.append(str(project_dir / ".git"))

    if not (project_dir / ".dvc").exists():
        dvc.run(["init", "-q"], cwd=project_dir)
        created.append(str(project_dir / ".dvc"))

    dataset_dir = project_dir / "dataset"
    if not any(dataset_dir.iterdir()):
        (dataset_dir / ".gitkeep").write_text("")

    if not (project_dir / "dataset.dvc").exists():
        dvc.run(["add", "dataset", "-q"], cwd=project_dir)
        created.append(str(project_dir / "dataset.dvc"))

    return created


def scaffold_consumer(project_dir):
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    created = []

    gitignore_path = project_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(CONSUMER_GITIGNORE_ENTRY)
        created.append(str(gitignore_path))
    elif ".mldna/" not in gitignore_path.read_text():
        with gitignore_path.open("a") as f:
            f.write(CONSUMER_GITIGNORE_ENTRY)

    for name, content in (
        ("mldna.yaml", yaml.dump({"project": {"repo_role": "consumer"}}, sort_keys=False)),
        ("models.lock.yaml", LOCKFILE_STUB),
        ("ci/mldna-fetch.snippet.yml", CI_SNIPPET),
    ):
        path = project_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if _write_if_absent(path, content):
            created.append(str(path))

    if not (project_dir / ".git").exists():
        subprocess.run(["git", "init", "-q"], cwd=project_dir, check=True)
        created.append(str(project_dir / ".git"))

    return created


def has_azure_config(project_dir):
    config_path = Path(project_dir) / "mldna.yaml"
    if not config_path.exists():
        return False
    data = yaml.safe_load(config_path.read_text()) or {}
    return "azure" in data


def write_azure_config(project_dir, subscription_id, resource_group, workspace):
    config_path, data = _load_config(project_dir)
    data["azure"] = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace": workspace,
    }
    _write_config(config_path, data)


def write_azure_tracking_uri(project_dir, tracking_uri):
    config_path, data = _load_config(project_dir)
    data["azure"] = {"tracking_uri": tracking_uri}
    _write_config(config_path, data)


def has_mlflow_config(project_dir):
    config_path = Path(project_dir) / "mldna.yaml"
    if not config_path.exists():
        return False
    data = yaml.safe_load(config_path.read_text()) or {}
    return "mlflow" in data


def write_mlflow_tracking_uri(project_dir, tracking_uri):
    config_path, data = _load_config(project_dir)
    data["mlflow"] = {"tracking_uri": tracking_uri}
    _write_config(config_path, data)


def has_filesystem_config(project_dir):
    config_path = Path(project_dir) / "mldna.yaml"
    if not config_path.exists():
        return False
    data = yaml.safe_load(config_path.read_text()) or {}
    return "filesystem" in data


def write_filesystem_path(project_dir, path):
    config_path, data = _load_config(project_dir)
    data["filesystem"] = {"path": str(path)}
    _write_config(config_path, data)


def has_dvc_config(project_dir):
    config_path = Path(project_dir) / "mldna.yaml"
    if not config_path.exists():
        return False
    data = yaml.safe_load(config_path.read_text()) or {}
    return "dvc" in data


def configure_dvc_remote(project_dir, account_name, container, prefix, remote_name="azure"):
    config_path, data = _load_config(project_dir)
    remote_url = f"azure://{container}/{prefix}"
    data["dvc"] = {"remote_url": remote_url, "account_name": account_name}
    _write_config(config_path, data)

    if not blobauth.get_default_remote(project_dir):
        dvc.run(["remote", "add", "-d", remote_name, remote_url], cwd=project_dir)
        dvc.run(["remote", "modify", remote_name, "account_name", account_name], cwd=project_dir)


def configure_local_dvc_remote(project_dir, path, remote_name="local"):
    config_path, data = _load_config(project_dir)
    data["dvc"] = {"remote_url": str(path)}
    _write_config(config_path, data)

    if not blobauth.get_default_remote(project_dir):
        (Path(project_dir) / path).mkdir(parents=True, exist_ok=True)
        dvc.run(["remote", "add", "-d", remote_name, str(path)], cwd=project_dir)
