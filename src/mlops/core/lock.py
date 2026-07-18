import json
import os
from datetime import datetime, timezone
from pathlib import Path

import psutil


class RunLockError(Exception):
    pass


class RunLock:
    def __init__(self, workspace_dir):
        self.path = Path(workspace_dir) / ".mlops" / "run.lock"

    def is_locked(self):
        if not self.path.exists():
            return False
        data = json.loads(self.path.read_text())
        return psutil.pid_exists(data["pid"])

    def acquire(self, run_id=None):
        if self.is_locked():
            raise RunLockError(
                f"Another run is already active in this working copy "
                f"({self.path}). Use `mlops workspace new` to run in parallel, "
                f"or wait for it to finish."
            )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(data))

    def release(self):
        self.path.unlink(missing_ok=True)
