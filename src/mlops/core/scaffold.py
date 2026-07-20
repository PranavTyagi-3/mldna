import subprocess
from pathlib import Path

import yaml

from mlops import __version__
from mlops.core import blobauth, dvc

GITIGNORE = "models/\noutputs/\n.mlops/\nmlruns/\n.dvc/cache/\n"

MAIN_PY = "import mlops\n\nwith mlops.track_run():\n    pass\n"

CONSUMER_GITIGNORE_ENTRY = ".mlops/\n"

LOCKFILE_STUB = (
    "# Pin exact registry versions here - `mlops fetch` resolves ONLY these\n"
    "# exact versions, never latest/stage. Example:\n"
    "#\n"
    "# fraud-detector:\n"
    "#   version: 9\n"
    "#   format: openvino\n"
    "#   dest: ./app/models/fraud\n"
)

# Copy-paste starting point only - the validated CI gate (pin-exists,
# format+provenance present, recalled fails, staleness warns) is a separate,
# later piece. This just shows the service-principal auth leg so `mlops
# fetch` needs zero personal az login in CI.
CI_SNIPPET = (
    "- name: Fetch pinned models\n"
    "  env:\n"
    "    AZURE_TENANT_ID: ${{ secrets.AZURE_TENANT_ID }}\n"
    "    AZURE_CLIENT_ID: ${{ secrets.AZURE_CLIENT_ID }}\n"
    "    AZURE_CLIENT_SECRET: ${{ secrets.AZURE_CLIENT_SECRET }}\n"
    "  run: mlops fetch\n"
)


def _write_if_absent(path, content):
    if path.exists():
        return False
    path.write_text(content)
    return True


def scaffold_producer(project_dir, model_name):
    project_dir = Path(project_dir)
    created = []

    for name in ("dataset", "models", "outputs"):
        path = project_dir / name
        if not path.exists():
            path.mkdir(parents=True)
            created.append(str(path))

    for name, content in (
        ("main.py", MAIN_PY),
        ("params.yaml", ""),
        ("requirements.txt", f"mlops=={__version__}\n"),
        (".gitignore", GITIGNORE),
        ("mlops.yaml", yaml.dump(
            {"project": {"model_name": model_name, "repo_role": "producer"}},
            sort_keys=False,
        )),
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
    created = []

    gitignore_path = project_dir / ".gitignore"
    if not gitignore_path.exists():
        gitignore_path.write_text(CONSUMER_GITIGNORE_ENTRY)
        created.append(str(gitignore_path))
    elif ".mlops/" not in gitignore_path.read_text():
        with gitignore_path.open("a") as f:
            f.write(CONSUMER_GITIGNORE_ENTRY)

    for name, content in (
        ("mlops.yaml", yaml.dump({"project": {"repo_role": "consumer"}}, sort_keys=False)),
        ("models.lock.yaml", LOCKFILE_STUB),
        ("ci/mlops-fetch.snippet.yml", CI_SNIPPET),
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
    config_path = Path(project_dir) / "mlops.yaml"
    if not config_path.exists():
        return False
    data = yaml.safe_load(config_path.read_text()) or {}
    return "azure" in data


def write_azure_config(project_dir, subscription_id, resource_group, workspace):
    config_path = Path(project_dir) / "mlops.yaml"
    data = yaml.safe_load(config_path.read_text()) or {}
    data["azure"] = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace": workspace,
    }
    config_path.write_text(yaml.dump(data, sort_keys=False))


def write_azure_tracking_uri(project_dir, tracking_uri):
    config_path = Path(project_dir) / "mlops.yaml"
    data = yaml.safe_load(config_path.read_text()) or {}
    data["azure"] = {"tracking_uri": tracking_uri}
    config_path.write_text(yaml.dump(data, sort_keys=False))


def has_dvc_config(project_dir):
    config_path = Path(project_dir) / "mlops.yaml"
    if not config_path.exists():
        return False
    data = yaml.safe_load(config_path.read_text()) or {}
    return "dvc" in data


def configure_dvc_remote(project_dir, account_name, container, prefix, remote_name="azure"):
    config_path = Path(project_dir) / "mlops.yaml"
    data = yaml.safe_load(config_path.read_text()) or {}
    remote_url = f"azure://{container}/{prefix}"
    data["dvc"] = {"remote_url": remote_url, "account_name": account_name}
    config_path.write_text(yaml.dump(data, sort_keys=False))

    if not blobauth.get_default_remote(project_dir):
        dvc.run(["remote", "add", "-d", remote_name, remote_url], cwd=project_dir)
        dvc.run(["remote", "modify", remote_name, "account_name", account_name], cwd=project_dir)


def configure_local_dvc_remote(project_dir, path, remote_name="local"):
    config_path = Path(project_dir) / "mlops.yaml"
    data = yaml.safe_load(config_path.read_text()) or {}
    data["dvc"] = {"remote_url": str(path)}
    config_path.write_text(yaml.dump(data, sort_keys=False))

    if not blobauth.get_default_remote(project_dir):
        (Path(project_dir) / path).mkdir(parents=True, exist_ok=True)
        dvc.run(["remote", "add", "-d", remote_name, str(path)], cwd=project_dir)
