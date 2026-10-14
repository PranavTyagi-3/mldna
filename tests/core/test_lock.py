import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from mldna.core.lock import RunLock, RunLockError

N_RACERS = 8

_RACE_SCRIPT = """
import os, sys, time
from pathlib import Path
from mldna.core.lock import RunLock, RunLockError

workspace_dir, ready_dir, results_dir, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
ready_dir = Path(ready_dir)
results_dir = Path(results_dir)

# Barrier: announce readiness, then spin until every racer has announced, so
# all N processes call acquire() at genuinely the same moment instead of
# trickling in by process-launch order.
(ready_dir / str(os.getpid())).write_text("1")
while len(list(ready_dir.iterdir())) < n:
    time.sleep(0.001)

lock = RunLock(workspace_dir)
try:
    lock.acquire()
    (results_dir / f"{os.getpid()}.txt").write_text("acquired")
    time.sleep(0.4)
except RunLockError:
    (results_dir / f"{os.getpid()}.txt").write_text("refused")
"""


def _run_race(workspace_dir, tmp_path, n=N_RACERS, timeout=60):
    """Launch n real OS processes that all try to acquire the same lock at
    once. Returns the list of outcome strings ("acquired"/"refused")."""
    ready_dir = tmp_path / "ready"
    results_dir = tmp_path / "results"
    ready_dir.mkdir()
    results_dir.mkdir()

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _RACE_SCRIPT,
             str(workspace_dir), str(ready_dir), str(results_dir), str(n)]
        )
        for _ in range(n)
    ]
    for p in procs:
        assert p.wait(timeout=timeout) == 0

    outcomes = [f.read_text() for f in sorted(results_dir.iterdir())]
    assert len(outcomes) == n
    return outcomes


def test_acquire_and_release(tmp_path):
    lock = RunLock(tmp_path)
    assert lock.is_locked() is False

    lock.acquire(run_id="run-1")
    assert lock.is_locked() is True

    lock.release()
    assert lock.is_locked() is False


def test_acquire_refuses_when_already_held(tmp_path):
    lock = RunLock(tmp_path)
    lock.acquire()

    with pytest.raises(RunLockError):
        RunLock(tmp_path).acquire()


def test_refusal_message_still_names_workspace_new(tmp_path):
    # Deliberately deferred wording (SPEC amendment pending) - must not drift.
    lock = RunLock(tmp_path)
    lock.acquire()
    with pytest.raises(RunLockError, match="mldna workspace new"):
        RunLock(tmp_path).acquire()


def test_stale_lock_is_reclaimed(tmp_path):
    lock_path = tmp_path / ".mldna" / "run.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"pid": 999999999, "run_id": None, "started_at": ""}))

    lock = RunLock(tmp_path)
    assert lock.is_locked() is False
    lock.acquire()
    assert lock.is_locked() is True


def test_release_does_not_delete_a_lock_owned_by_another_pid(tmp_path):
    lock = RunLock(tmp_path)
    lock.path.parent.mkdir(parents=True, exist_ok=True)
    other_pid = os.getpid() + 1
    lock.path.write_text(json.dumps({"pid": other_pid, "run_id": None, "started_at": ""}))

    lock.release()

    assert lock.path.exists()
    assert json.loads(lock.path.read_text())["pid"] == other_pid


def test_release_deletes_a_lock_owned_by_this_pid(tmp_path):
    lock = RunLock(tmp_path)
    lock.acquire()
    assert lock.path.exists()

    lock.release()

    assert not lock.path.exists()


def test_concurrent_acquire_on_fresh_lock_has_exactly_one_winner(tmp_path):
    """Headline race test: N real processes hit an un-held lock at the same
    instant. Exactly one may acquire it; the rest must be refused."""
    workspace_dir = tmp_path
    outcomes = _run_race(workspace_dir, tmp_path)

    assert outcomes.count("acquired") == 1
    assert outcomes.count("refused") == N_RACERS - 1


def test_concurrent_reclaim_of_one_stale_lock_has_exactly_one_winner(tmp_path):
    """N real processes race to reclaim the SAME stale (dead-pid) lock.
    Exactly one may win the reclaim; the rest must be refused."""
    workspace_dir = tmp_path
    lock_path = workspace_dir / ".mldna" / "run.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"pid": 999999999, "run_id": None, "started_at": ""}))

    outcomes = _run_race(workspace_dir, tmp_path)

    assert outcomes.count("acquired") == 1
    assert outcomes.count("refused") == N_RACERS - 1


def test_is_locked_reflects_live_vs_absent(tmp_path):
    lock = RunLock(tmp_path)
    assert lock.is_locked() is False
    lock.acquire()
    assert lock.is_locked() is True
    lock.release()
    assert lock.is_locked() is False


_TRACK_RUN_RACE_SCRIPT = """
import os, sys, time
from pathlib import Path

workspace_dir = Path(sys.argv[1])
ready_dir = Path(sys.argv[2])
results_dir = Path(sys.argv[3])
n = int(sys.argv[4])

# Barrier: announce readiness, then spin until every racer has announced
(ready_dir / str(os.getpid())).write_text("1")
while len(list(ready_dir.iterdir())) < n:
    time.sleep(0.001)

# Set tracking URI to the workspace
import mlflow
mlflow.set_tracking_uri(f"file:{workspace_dir / 'mlruns'}")

# Race to enter track_run context
from mldna.tracking import track_run, TrackRunError
# RunLockError is raised by core.lock and is NOT a TrackRunError
# subclass, so it needs catching in its own right.
from mldna.core.lock import RunLockError

try:
    with track_run(workspace_dir=workspace_dir):
        # Hold the lock briefly so other racers get a clear "refused"
        time.sleep(0.4)
    (results_dir / f"{os.getpid()}.txt").write_text("acquired")
except (RunLockError, TrackRunError) as e:
    if "Another run is already active" in str(e):
        (results_dir / f"{os.getpid()}.txt").write_text("refused")
    else:
        # Some other TrackRunError (bad config, etc.) - treat as error not refusal
        (results_dir / f"{os.getpid()}.txt").write_text(f"error: {str(e)[:100]}")
except Exception as e:
    # Unexpected error
    (results_dir / f"{os.getpid()}.txt").write_text(f"crash: {type(e).__name__}: {str(e)[:50]}")
"""


def _run_track_run_race(workspace_dir, tmp_path, n=N_RACERS, timeout=60):
    """Like _run_race but races two concurrent track_run contexts in a single workspace.
    Returns list of outcome strings ("acquired"/"refused"/"error"/etc.)."""
    ready_dir = tmp_path / "track_ready"
    results_dir = tmp_path / "track_results"
    ready_dir.mkdir()
    results_dir.mkdir()

    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _TRACK_RUN_RACE_SCRIPT,
             str(workspace_dir), str(ready_dir), str(results_dir), str(n)]
        )
        for _ in range(n)
    ]
    for p in procs:
        assert p.wait(timeout=timeout) == 0

    outcomes = [f.read_text() for f in sorted(results_dir.iterdir())]
    assert len(outcomes) == n
    return outcomes


def test_concurrent_track_run_on_same_workspace_has_exactly_one_winner(
    tmp_path, producer_repo
):
    """Two concurrent track_run contexts in ONE workspace yield exactly one winner.
    This is the real-world test: integration of track_run.__enter__'s lock acquisition
    with discovery.snapshot. If the lock protection reaches the entry point, exactly
    one process should succeed; the other should get RunLockError refused."""
    # producer_repo is scaffolded but NOT committed, so commit first:
    # track_run's clean-tree gate would otherwise refuse the winner.
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo,
                   check=True)
    outcomes = _run_track_run_race(producer_repo, tmp_path)

    # Exactly one should acquire; rest refused
    acquired = outcomes.count("acquired")
    refused = outcomes.count("refused")

    # Verify: exactly one winner
    assert acquired == 1, f"Expected 1 acquisition, got {acquired}. Outcomes: {outcomes}"
    assert refused == N_RACERS - 1, f"Expected {N_RACERS - 1} refusals, got {refused}. Outcomes: {outcomes}"

    # No errors or crashes
    errors = [o for o in outcomes if o.startswith("error") or o.startswith("crash")]
    assert not errors, f"Unexpected errors/crashes: {errors}"
