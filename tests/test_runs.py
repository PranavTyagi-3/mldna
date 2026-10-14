import subprocess

import pytest
import yaml

from mldna.core.push import push as core_push
from mldna.runs import RunsError, list_runs
from mldna.tracking import track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def test_runs_missing_mldna_yaml_refuses(tmp_path):
    # D#57 reworded this: it now names the searched dir and only suggests
    # init when no ancestor project exists.
    with pytest.raises(RunsError, match="No mldna.yaml found"):
        list_runs(workspace_dir=tmp_path)


def test_runs_consumer_repo_without_model_name_refuses(tmp_path):
    (tmp_path / "mldna.yaml").write_text("project:\n  repo_role: consumer\n")

    with pytest.raises(RunsError, match="producer"):
        list_runs(workspace_dir=tmp_path)


def test_runs_producer_without_namespace_refuses(tmp_path):
    # issue #16: no legacy fallback to the bare model_name.
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "mldna.yaml").write_text("project:\n  model_name: test-model\n  repo_role: producer\n")

    with pytest.raises(RunsError, match="project.namespace"):
        list_runs(workspace_dir=tmp_path)


def test_runs_only_shows_this_projects_runs(tmp_path):
    # issue #16's exact repro: two teams publish a same-named model
    # ("churn-model") to one shared tracking store. Without namespacing,
    # alice's `mldna runs` would also list bob's runs under the same
    # experiment. Both projects point at one shared filesystem store so the
    # collision is actually exercised, not avoided by isolated mlruns dirs.
    from mldna.core.scaffold import scaffold_producer, write_filesystem_path

    shared_store = tmp_path / "shared-mlruns"
    shared_store.mkdir()

    alice_dir = tmp_path / "alice-project"
    scaffold_producer(alice_dir, "churn-model", namespace="alice")
    write_filesystem_path(alice_dir, shared_store)
    subprocess.run(["git", "add", "-A"], cwd=alice_dir, check=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=alice_dir, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=alice_dir, check=True)
    _commit_scaffold(alice_dir)
    with track_run(workspace_dir=alice_dir):
        pass

    bob_dir = tmp_path / "bob-project"
    scaffold_producer(bob_dir, "churn-model", namespace="bob")
    write_filesystem_path(bob_dir, shared_store)
    subprocess.run(["git", "add", "-A"], cwd=bob_dir, check=True)
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=bob_dir, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=bob_dir, check=True)
    _commit_scaffold(bob_dir)
    with track_run(workspace_dir=bob_dir):
        pass

    alice_rows = list_runs(workspace_dir=alice_dir)
    bob_rows = list_runs(workspace_dir=bob_dir)

    assert len(alice_rows) == 1
    assert len(bob_rows) == 1
    assert alice_rows[0]["run_id"] != bob_rows[0]["run_id"]


def test_runs_no_runs_yet_returns_empty(producer_repo):
    assert list_runs(workspace_dir=producer_repo) == []


def test_runs_lists_recent_runs(producer_repo):
    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        pass

    rows = list_runs(workspace_dir=producer_repo)

    assert len(rows) == 1
    assert rows[0]["run_id"] == run.run_id[:8]
    assert rows[0]["dirty"] == "false"


def test_runs_last_limits_results(producer_repo):
    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    for _ in range(3):
        with track_run(workspace_dir=producer_repo):
            pass

    rows = list_runs(workspace_dir=producer_repo, last=2)

    assert len(rows) == 2


def test_runs_key_metric_shown_when_configured(producer_repo):
    config_path = producer_repo / "mldna.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["tracking"] = {"key_metric": "accuracy"}
    config_path.write_text(yaml.dump(config, sort_keys=False))

    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        import mlflow
        mlflow.log_metric("accuracy", 0.95)

    rows = list_runs(workspace_dir=producer_repo)

    assert rows[0]["key_metric"] == 0.95


def test_runs_key_metric_missing_on_run_shows_placeholder(producer_repo):
    config_path = producer_repo / "mldna.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["tracking"] = {"key_metric": "accuracy"}
    config_path.write_text(yaml.dump(config, sort_keys=False))

    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        pass

    rows = list_runs(workspace_dir=producer_repo)

    assert rows[0]["key_metric"] == "-"


def test_runs_key_metric_value_is_returned_unrounded(producer_repo):
    config_path = producer_repo / "mldna.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["tracking"] = {"key_metric": "accuracy"}
    config_path.write_text(yaml.dump(config, sort_keys=False))

    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        import mlflow
        mlflow.log_metric("accuracy", 0.8734918273645)

    rows = list_runs(workspace_dir=producer_repo)

    assert rows[0]["key_metric"] == 0.8734918273645
