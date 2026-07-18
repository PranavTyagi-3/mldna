import subprocess
from pathlib import Path

import yaml

from mlops import __version__
from mlops.core import dvc

GITIGNORE = "models/\noutputs/\n.mlops/\nmlruns/\n.dvc/cache/\n"

MAIN_PY = "import mlops\n\nwith mlops.track_run():\n    pass\n"


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
