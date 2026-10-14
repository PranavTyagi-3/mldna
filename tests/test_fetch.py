import subprocess

import mlflow
import pytest
from mlflow.exceptions import MlflowException, RestException

from mldna.core import ownership
from mldna.core.lockfile import LockfileError
from mldna.core.manifest import ManifestError
from mldna.core.push import push as core_push
from mldna.fetch import FetchError, _is_recalled, fetch
from mldna.publish import publish
from mldna.tracking import track_convert, track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def _publish_pkl_only(repo, pkl_bytes=b"trained model bytes"):
    _commit_scaffold(repo)
    core_push(repo)
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(pkl_bytes)
    result = publish(pkl, workspace_dir=repo, confirm=lambda s: True)
    return pkl, result


def _write_lockfile(repo, model_name="project-test-model", version=1, format="pkl", dest="app/models/test"):
    (repo / "models.lock.yaml").write_text(
        f"{model_name}:\n  version: {version}\n  format: {format}\n  dest: {dest}\n"
    )
    return repo / dest


def _publish_next_version(repo, pkl_bytes):
    # scaffold is already committed and the model file lives under the
    # gitignored models/, so the tree is still clean - no re-commit needed.
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.write_bytes(pkl_bytes)
    result = publish(pkl, workspace_dir=repo, confirm=lambda s: True)
    return pkl, result


def test_fetch_missing_lockfile_refuses(producer_repo):
    with pytest.raises(LockfileError):
        fetch(workspace_dir=producer_repo)


def _raise(exc):
    def _raiser(*args, **kwargs):
        raise exc
    return _raiser


def test_fetch_unreachable_registry_is_not_reported_as_a_missing_version(
    producer_repo, monkeypatch
):
    _write_lockfile(producer_repo, model_name="ghost-model", version=1)
    # the exact shape mlflow's REST store raises when the host is down:
    # MlflowException/INTERNAL_ERROR wrapping the requests connection error
    unreachable = MlflowException(
        "API request to http://localhost:5000/api/2.0/mlflow/model-versions/get failed "
        "with exception HTTPConnectionPool(host='localhost', port=5000): Max retries "
        "exceeded"
    )
    monkeypatch.setattr(
        mlflow.tracking.MlflowClient, "get_model_version", _raise(unreachable)
    )

    with pytest.raises(FetchError) as excinfo:
        fetch(workspace_dir=producer_repo)

    message = str(excinfo.value)
    assert "could not reach the model registry" in message
    assert "models.lock.yaml" not in message
    assert "not found in the registry" not in message
    assert "INTERNAL_ERROR" in message
    assert "mldna doctor" in message


def test_fetch_unauthenticated_registry_is_not_reported_as_a_missing_version(
    producer_repo, monkeypatch
):
    _write_lockfile(producer_repo, model_name="ghost-model", version=1)
    denied = RestException({"error_code": "PERMISSION_DENIED", "message": "forbidden"})
    monkeypatch.setattr(
        mlflow.tracking.MlflowClient, "get_model_version", _raise(denied)
    )

    with pytest.raises(FetchError) as excinfo:
        fetch(workspace_dir=producer_repo)

    message = str(excinfo.value)
    assert "could not reach the model registry" in message
    assert "PERMISSION_DENIED" in message
    assert "models.lock.yaml" not in message


def test_fetch_unrecognized_registry_error_states_both_possibilities(
    producer_repo, monkeypatch
):
    _write_lockfile(producer_repo, model_name="ghost-model", version=1)
    monkeypatch.setattr(
        mlflow.tracking.MlflowClient,
        "get_model_version",
        _raise(RuntimeError("something the store never documents")),
    )

    with pytest.raises(FetchError) as excinfo:
        fetch(workspace_dir=producer_repo)

    message = str(excinfo.value)
    assert "cannot tell" in message
    assert "unreachable" in message
    assert "something the store never documents" in message


def test_fetch_unknown_version_refuses(producer_repo):
    _write_lockfile(producer_repo, model_name="ghost-model", version=1)

    with pytest.raises(FetchError) as excinfo:
        fetch(workspace_dir=producer_repo)

    message = str(excinfo.value)
    assert "not found in the registry" in message
    assert "check models.lock.yaml" in message
    assert "could not reach" not in message


def test_fetch_bare_pin_of_a_namespaced_model_gets_did_you_mean_hint(published):
    # issue #16 migration: a lockfile pinning the old bare name (written
    # before namespacing) must not just say "not found" - that reads as an
    # outage rather than a known, fixable migration.
    producer_repo, _, result = published
    _write_lockfile(producer_repo, model_name="test-model", version=result["version"])

    with pytest.raises(FetchError) as excinfo:
        fetch(workspace_dir=producer_repo)

    message = str(excinfo.value)
    assert "no test-model in the registry" in message
    assert "did you mean project-test-model" in message
    assert "models.lock.yaml" in message


def test_fetch_recalled_version_refuses(published):
    repo, _, result = published
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag("project-test-model", result["version"], "recalled", "true")
    _write_lockfile(repo, version=result["version"])

    with pytest.raises(FetchError, match="recalled"):
        fetch(workspace_dir=repo)


def test_fetch_recall_after_a_successful_fetch_stops_the_next_fetch(published):
    repo, pkl, result = published
    dest = _write_lockfile(repo, version=result["version"])
    fetch(workspace_dir=repo)
    mlflow.tracking.MlflowClient().set_model_version_tag(
        "project-test-model", result["version"], "recalled", "true"
    )

    with pytest.raises(FetchError, match="recalled"):
        fetch(workspace_dir=repo)

    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_abandoned_version_refuses(published):
    # D#76: a version registered by the loser of a concurrent publish.
    repo, _, result = published
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag("project-test-model", result["version"], "abandoned", "true")
    client.set_model_version_tag("project-test-model", result["version"], "abandoned_reason",
                                 "lost concurrent publish to v7")
    dest = _write_lockfile(repo, version=result["version"])

    with pytest.raises(FetchError, match="tagged abandoned") as excinfo:
        fetch(workspace_dir=repo)
    message = str(excinfo.value)
    assert "lost concurrent publish to v7" in message
    assert "Fix: pin a different version" in message
    assert not dest.exists()


def test_fetch_not_recalled_on_explicit_falsy_value_fetches(published):
    repo, pkl, result = published
    client = mlflow.tracking.MlflowClient()
    client.set_model_version_tag("project-test-model", result["version"], "recalled", "false")
    dest = _write_lockfile(repo, version=result["version"])

    fetched = fetch(workspace_dir=repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


# --- recall kill switch must fail safe on any truthy-looking value (D#50) --
# The rule itself, over every value, without a publish per value.

@pytest.mark.parametrize("truthy_value", ["true", "True", "yes", "1", " TRUE ", "anything else"])
def test_is_recalled_true_for_any_present_value_outside_the_false_set(truthy_value):
    assert _is_recalled(truthy_value) is True


@pytest.mark.parametrize("falsy_value", ["false", "no", "0", "off", "", " False "])
def test_is_recalled_false_for_explicit_falsy_values(falsy_value):
    assert _is_recalled(falsy_value) is False


def test_is_recalled_false_when_the_tag_is_absent():
    assert _is_recalled(None) is False


def test_fetch_without_recalled_tag_fetches(published):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"])

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_from_a_consumer_repo_needs_no_namespace_of_its_own(published, tmp_path):
    # issue #16: namespacing is a producer-side concern. A consumer repo's
    # mldna.yaml never has project.namespace, and fetch must not require one -
    # the lockfile already carries the full namespaced name verbatim.
    from mldna.core.scaffold import scaffold_consumer, write_filesystem_path

    producer_repo, pkl, result = published

    consumer_dir = tmp_path / "consumer"
    scaffold_consumer(consumer_dir)
    write_filesystem_path(consumer_dir, producer_repo / "mlruns")
    assert "namespace" not in (consumer_dir / "mldna.yaml").read_text()
    dest = _write_lockfile(consumer_dir, version=result["version"], format="pkl")

    fetched = fetch(workspace_dir=consumer_dir)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_missing_format_refuses(published):
    producer_repo, _, result = published
    _write_lockfile(producer_repo, version=result["version"], format="onnx")

    with pytest.raises(ManifestError, match="onnx"):
        fetch(workspace_dir=producer_repo)


def test_fetch_pkl_only_happy_path(published):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched == [{"model_name": "project-test-model", "version": str(result["version"]),
                        "skipped": False, "repaired": False}]
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()
    marker = ownership.read_marker(dest)
    assert marker["model_name"] == "project-test-model"
    assert marker["version"] == str(result["version"])
    assert marker["format"] == "pkl"


def test_fetch_converted_format_happy_path(producer_repo):
    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    with track_convert(pkl, workspace_dir=producer_repo):
        onnx_dir = producer_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted bytes")
    result = publish(pkl, workspace_dir=producer_repo,
                      format_dirs=[onnx_dir], confirm=lambda s: True)
    dest = _write_lockfile(producer_repo, version=result["version"], format="onnx")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched == [{"model_name": "project-test-model", "version": str(result["version"]),
                        "skipped": False, "repaired": False}]
    assert (dest / "model.onnx").read_bytes() == b"converted bytes"


def test_fetch_repair_with_user_file_in_dest_refuses_and_keeps_files(published):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")
    fetch(workspace_dir=producer_repo)
    (dest / "serving_config.yaml").write_text("mine")
    with open(dest / "model.pkl", "ab") as f:
        f.write(b"x")
    tampered = (dest / "model.pkl").read_bytes()

    with pytest.raises(FetchError, match="serving_config.yaml.*Fix: move them out"):
        fetch(workspace_dir=producer_repo)

    assert (dest / "serving_config.yaml").read_text() == "mine"
    assert (dest / "model.pkl").read_bytes() == tampered


def test_fetch_repair_without_user_files_still_works(published):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")
    fetch(workspace_dir=producer_repo)
    with open(dest / "model.pkl", "ab") as f:
        f.write(b"x")

    assert fetch(workspace_dir=producer_repo)[0]["repaired"] is True
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_version_change_with_user_file_in_dest_refuses(published_twice):
    producer_repo, (r1, r2) = published_twice
    dest = _write_lockfile(producer_repo, version=r1["version"], format="pkl")
    fetch(workspace_dir=producer_repo)
    (dest / "serving_config.yaml").write_text("mine")
    before = (dest / "model.pkl").read_bytes()

    _write_lockfile(producer_repo, version=r2["version"], format="pkl")
    with pytest.raises(FetchError, match="serving_config.yaml"):
        fetch(workspace_dir=producer_repo)

    assert (dest / "serving_config.yaml").read_text() == "mine"
    assert (dest / "model.pkl").read_bytes() == before


def test_fetch_tampered_on_disk_file_repairs(published, capsys):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")

    fetch(workspace_dir=producer_repo)
    with open(dest / "model.pkl", "ab") as f:
        f.write(b"x")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert fetched[0]["repaired"] is True
    assert "failed verification" in capsys.readouterr().out
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_missing_on_disk_file_repairs(published):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")

    fetch(workspace_dir=producer_repo)
    (dest / "model.pkl").unlink()

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["repaired"] is True
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_idempotent_skip_avoids_redownload(published, monkeypatch):
    producer_repo, _, result = published
    _write_lockfile(producer_repo, version=result["version"], format="pkl")

    first = fetch(workspace_dir=producer_repo)
    assert first[0]["skipped"] is False

    import mldna.fetch as fetch_module

    def _boom(*args, **kwargs):
        raise AssertionError("subtree should not be re-downloaded on an idempotent fetch")

    monkeypatch.setattr(fetch_module, "_stage_download", _boom)

    second = fetch(workspace_dir=producer_repo)
    assert second[0]["skipped"] is True
    assert second[0]["repaired"] is False
    assert second[0] == {"model_name": "project-test-model", "version": str(result["version"]),
                         "skipped": True, "repaired": False}


def test_fetch_version_bump_refetches(producer_repo):
    # Both publishes happen before any lockfile/fetch writes touch the working
    # tree - fetch's own output files would otherwise dirty the tree that the
    # second track_run() needs clean.
    _, result1 = _publish_pkl_only(producer_repo, pkl_bytes=b"v1 bytes")
    with track_run(workspace_dir=producer_repo):
        pkl2 = producer_repo / "models" / "model.pkl"
        pkl2.write_bytes(b"v2 bytes")
    result2 = publish(pkl2, workspace_dir=producer_repo, confirm=lambda s: True)
    assert result2["version"] != result1["version"]

    dest = _write_lockfile(producer_repo, version=result1["version"], format="pkl")
    fetch(workspace_dir=producer_repo)
    assert (dest / "model.pkl").read_bytes() == b"v1 bytes"

    _write_lockfile(producer_repo, version=result2["version"], format="pkl")
    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == b"v2 bytes"
    assert ownership.read_marker(dest)["version"] == str(result2["version"])


# --- dest validation and ownership (D#46) ---------------------------------
# Failing-case tests first: each asserts that nothing mldna did not create
# was deleted.

def test_fetch_dest_equal_workspace_root_refuses(producer_repo):
    marker_file = producer_repo / "mldna.yaml"
    assert marker_file.exists()
    _write_lockfile(producer_repo, dest=".")

    with pytest.raises(FetchError) as excinfo:
        fetch(workspace_dir=producer_repo)

    assert "dest" in str(excinfo.value)
    assert marker_file.exists()


def test_fetch_dest_escaping_workspace_refuses(producer_repo, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("keep me")
    _write_lockfile(producer_repo, dest="../victim")

    with pytest.raises(FetchError):
        fetch(workspace_dir=producer_repo)

    assert (victim / "important.txt").read_text() == "keep me"


def test_fetch_absolute_dest_refuses(producer_repo, tmp_path):
    target = tmp_path / "abs_target"
    target.mkdir()
    (target / "keep.txt").write_text("keep me")
    _write_lockfile(producer_repo, dest=target.as_posix())

    with pytest.raises(FetchError, match="absolute"):
        fetch(workspace_dir=producer_repo)

    assert (target / "keep.txt").read_text() == "keep me"


def test_fetch_foreign_nonempty_dest_refuses(published):
    producer_repo, _, result = published
    dest = producer_repo / "app" / "src"
    dest.mkdir(parents=True)
    (dest / "server.py").write_text("print('do not delete me')")
    _write_lockfile(producer_repo, version=result["version"], format="pkl", dest="app/src")

    with pytest.raises(FetchError, match="did not create"):
        fetch(workspace_dir=producer_repo)

    assert (dest / "server.py").read_text() == "print('do not delete me')"


def test_fetch_owned_dest_is_replaced(published):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")
    dest.mkdir(parents=True)
    (dest / ".mldna-version").write_text("0")
    (dest / "model.pkl").write_bytes(b"stale")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_first_time_into_nonexistent_dest(published):
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")
    assert not dest.exists()

    fetch(workspace_dir=producer_repo)

    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


# --- marker identity, not just version (D#53, issue #17) ------------------
# Failing-case ("stale skip") shape first per CLAUDE.md: each of these sets
# up a scenario where version alone would wrongly say "up to date".

def test_fetch_repointed_model_name_same_version_forces_refetch(producer_repo):
    # The owner's exact repro: a lockfile entry repointed from one model to
    # a completely different one that happens to land on the same version
    # number (both are the first version ever registered under their name).
    pkl1, result1 = _publish_pkl_only(producer_repo, pkl_bytes=b"churn model bytes")
    dest = _write_lockfile(producer_repo, model_name="project-test-model",
                           version=result1["version"], format="pkl")
    fetch(workspace_dir=producer_repo)
    assert (dest / "model.pkl").read_bytes() == pkl1.read_bytes()

    from mldna.core.config import ProjectConfig

    model_id = ProjectConfig.load(producer_repo).model_id
    (producer_repo / "mldna.yaml").write_text(
        "project:\n  model_name: other-model\n  namespace: project\n  repo_role: producer\n"
        f"  model_id: {model_id}\n"
    )
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "repoint model"], cwd=producer_repo, check=True)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        pkl2 = producer_repo / "models" / "model.pkl"
        pkl2.write_bytes(b"iris model bytes")
    result2 = publish(pkl2, workspace_dir=producer_repo, confirm=lambda s: True)
    assert result2["version"] == result1["version"]  # same version, different model

    _write_lockfile(producer_repo, model_name="project-other-model",
                    version=result2["version"], format="pkl", dest="app/models/test")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == pkl2.read_bytes()


def test_fetch_repointed_format_same_version_forces_refetch(producer_repo):
    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        pkl = producer_repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    with track_convert(pkl, workspace_dir=producer_repo):
        onnx_dir = producer_repo / "models" / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"converted bytes")
    result = publish(pkl, workspace_dir=producer_repo,
                      format_dirs=[onnx_dir], confirm=lambda s: True)
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")

    fetch(workspace_dir=producer_repo)
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()

    _write_lockfile(producer_repo, version=result["version"], format="onnx")
    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.onnx").read_bytes() == b"converted bytes"


def test_fetch_legacy_bare_version_marker_forces_refetch_even_when_version_matches(
    published
):
    # A marker from before D#53 is ownership evidence, never identity
    # evidence - even a version string that happens to match must not skip.
    producer_repo, pkl, result = published
    dest = _write_lockfile(producer_repo, version=result["version"], format="pkl")
    dest.mkdir(parents=True)
    (dest / ".mldna-version").write_text(str(result["version"]))
    (dest / "model.pkl").write_bytes(b"stale bytes under an old-format marker")

    fetched = fetch(workspace_dir=producer_repo)

    assert fetched[0]["skipped"] is False
    assert (dest / "model.pkl").read_bytes() == pkl.read_bytes()


def test_fetch_two_aliases_pin_same_model_at_different_versions_fetch_independently(
    published
):
    # The capability D#55 adds: an A/B-style lockfile with a second alias
    # pinning the same model at a different version, each landing in its
    # own dest without colliding.
    producer_repo, _, result_v1 = published
    _, result_v2 = _publish_next_version(producer_repo, b"canary bytes")

    (producer_repo / "models.lock.yaml").write_text(
        f"project-test-model:\n"
        f"  version: {result_v1['version']}\n"
        f"  format: pkl\n"
        f"  dest: app/models/stable\n"
        f"test-model-canary:\n"
        f"  model: project-test-model\n"
        f"  version: {result_v2['version']}\n"
        f"  format: pkl\n"
        f"  dest: app/models/canary\n"
    )

    fetched = fetch(workspace_dir=producer_repo)

    assert len(fetched) == 2
    stable_dest = producer_repo / "app" / "models" / "stable"
    canary_dest = producer_repo / "app" / "models" / "canary"
    assert (stable_dest / "model.pkl").read_bytes() == b"trained model bytes"
    assert (canary_dest / "model.pkl").read_bytes() == b"canary bytes"

    stable_marker = ownership.read_marker(stable_dest)
    canary_marker = ownership.read_marker(canary_dest)
    assert stable_marker["model_name"] == "project-test-model"
    assert stable_marker["version"] == str(result_v1["version"])
    assert canary_marker["model_name"] == "project-test-model"
    assert canary_marker["version"] == str(result_v2["version"])


def _lock_with_id(repo, version, model_id):
    id_line = f"  id: {model_id}\n" if model_id else ""
    (repo / "models.lock.yaml").write_text(
        f"project-test-model:\n{id_line}  version: {version}\n  format: pkl\n  dest: app/models/test\n"
    )


def test_fetch_with_wrong_model_id_refuses(published):
    producer_repo, _, result = published
    _lock_with_id(producer_repo, result["version"], "99999999-2222-4333-8444-555555555555")

    with pytest.raises(FetchError, match="has model id"):
        fetch(workspace_dir=producer_repo)


def test_fetch_with_matching_model_id_works(published):
    producer_repo, _, result = published
    _lock_with_id(producer_repo, result["version"], result["model_id"])

    assert len(fetch(workspace_dir=producer_repo)) == 1


def test_fetch_without_model_id_works_and_notes_it(published, capsys):
    producer_repo, _, result = published
    _lock_with_id(producer_repo, result["version"], None)

    fetch(workspace_dir=producer_repo)

    assert f"id: {result['model_id']}" in capsys.readouterr().out
