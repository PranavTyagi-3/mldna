import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

_O_BINARY = getattr(os, "O_BINARY", 0)  # no-op on POSIX, required on Windows


class RunLockError(Exception):
    pass


class RunLock:
    def __init__(self, workspace_dir):
        self.path = Path(workspace_dir) / ".mldna" / "run.lock"
        # Mutex used only while reclaiming a stale lock (see acquire()).
        self._reclaim_path = self.path.with_name(self.path.name + ".reclaim")

    def is_locked(self):
        data = self._read()
        if data is None:
            return False
        return psutil.pid_exists(data["pid"])

    def _read(self):
        try:
            return json.loads(self.path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def _payload(self, run_id):
        return json.dumps({
            "pid": os.getpid(),
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }).encode()

    def _create_exclusive(self, path, payload):
        # O_CREAT|O_EXCL makes the *creation itself* the test for "is this
        # lock already held" - the OS guarantees open() either creates the
        # file or fails, with no gap in which a second process could also
        # observe "not present" and also create it. Do NOT "simplify" this
        # back into `if path.exists(): ... else: write(...)` - that is
        # exactly the check-then-write race this replaces: two processes can
        # both pass the exists() check before either has written, and both
        # end up believing they hold the lock.
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _O_BINARY)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)

    def acquire(self, run_id=None):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._payload(run_id)

        # Bounded retry: each iteration either wins outright, discovers a
        # live holder (and raises), or loses a reclaim race against another
        # process and loops to re-check the now-settled state. 50 iterations
        # of sub-millisecond work is far more than contention ever needs.
        # On Windows, a file that another process currently has open (even
        # just for a brief read_text()) cannot always be deleted or
        # recreated - the OS raises PermissionError (WinError 32) rather
        # than FileExistsError in that case. That is a timing collision, not
        # a real error, so it is treated exactly like "someone else is
        # touching this file right now" everywhere below: lose this
        # attempt, loop, and re-observe.
        contention = (FileExistsError, PermissionError)

        for _ in range(50):
            try:
                self._create_exclusive(self.path, payload)
                return
            except contention:
                pass

            data = self._read()
            if data is not None and psutil.pid_exists(data["pid"]):
                raise RunLockError(
                    f"Another run is already active in this working copy "
                    f"({self.path}). Use `mldna workspace new` to run in parallel, "
                    f"or wait for it to finish."
                )

            # The lock file is present but its owning process is dead (or it
            # vanished mid-read) - stale. Several processes may reach this
            # point for the SAME stale lock at once, so the reclaim itself
            # goes through a second O_CREAT|O_EXCL gate: only whoever creates
            # `.reclaim` first may replace the stale lock file. Everyone else
            # loses that gate and loops back to the top, where they will see
            # either the winner's fresh (live) lock and be refused, or - if
            # the winner hasn't written yet - another stale read and another
            # attempt at the gate.
            try:
                reclaim_fd = os.open(
                    self._reclaim_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | _O_BINARY
                )
            except contention:
                time.sleep(0.01)
                continue
            try:
                os.close(reclaim_fd)
                # Re-check under the reclaim gate: the lock could have been
                # released or genuinely re-acquired while we were racing in.
                data = self._read()
                if data is not None and psutil.pid_exists(data["pid"]):
                    continue
                try:
                    self.path.unlink(missing_ok=True)
                except PermissionError:
                    # Another racer has it open right now; try again later.
                    continue
                self._create_exclusive(self.path, payload)
                return
            except contention:
                # Someone created self.path between our unlink and our
                # create - they won fair and square; loop and re-observe.
                continue
            finally:
                try:
                    self._reclaim_path.unlink(missing_ok=True)
                except PermissionError:
                    pass

        raise RunLockError(
            f"Could not acquire run lock at {self.path} - contention did not "
            f"resolve. Use `mldna workspace new` to run in parallel, or retry."
        )

    def release(self):
        data = self._read()
        if data is None:
            return
        if data.get("pid") != os.getpid():
            # Not ours - never delete another process's lock.
            return
        for _ in range(50):
            try:
                self.path.unlink(missing_ok=True)
                return
            except PermissionError:
                # Windows: another process has it open (e.g. a concurrent
                # is_locked() read) right now - retry rather than fail.
                time.sleep(0.01)
        self.path.unlink(missing_ok=True)
