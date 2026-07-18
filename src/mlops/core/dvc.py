import subprocess
import sys


def run(args, cwd, check=True):
    return subprocess.run(
        [sys.executable, "-m", "dvc", *args], cwd=cwd,
        check=check, capture_output=True, text=True,
    )
