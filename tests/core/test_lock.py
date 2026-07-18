import json

import pytest

from mlops.core.lock import RunLock, RunLockError


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


def test_stale_lock_is_reclaimed(tmp_path):
    lock_path = tmp_path / ".mlops" / "run.lock"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_text(json.dumps({"pid": 999999999, "run_id": None, "started_at": ""}))

    lock = RunLock(tmp_path)
    assert lock.is_locked() is False
    lock.acquire()
    assert lock.is_locked() is True
