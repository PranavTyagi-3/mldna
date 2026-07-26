import subprocess
import sys
from pathlib import Path

import yaml


def run(args, cwd, check=True):
    return subprocess.run(
        [sys.executable, "-m", "dvc", *args], cwd=cwd,
        check=check, capture_output=True, text=True,
    )


def read_data_hash(workspace_dir, pointer_file="dataset.dvc"):
    path = Path(workspace_dir) / pointer_file
    if not path.exists():
        return None
    data = yaml.safe_load(path.read_text())
    outs = data.get("outs") or []
    if not outs:
        return None
    return outs[0].get("md5")
