import subprocess

import pytest
import yaml

from mlops.core.push import push as core_push
from mlops.runs import RunsError, list_runs
from mlops.tracking import track_run


def _commit_scaffold(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def test_runs_missing_mlops_yaml_refuses(tmp_path):
    with pytest.raises(RunsError, match="mlops.yaml not found"):
        list_runs(workspace_dir=tmp_path)


def test_runs_consumer_repo_without_model_name_refuses(tmp_path):
    (tmp_path / "mlops.yaml").write_text("project:\n  repo_role: consumer\n")

    with pytest.raises(RunsError, match="producer"):
        list_runs(workspace_dir=tmp_path)


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
    config_path = producer_repo / "mlops.yaml"
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
    config_path = producer_repo / "mlops.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["tracking"] = {"key_metric": "accuracy"}
    config_path.write_text(yaml.dump(config, sort_keys=False))

    _commit_scaffold(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        pass

    rows = list_runs(workspace_dir=producer_repo)

    assert rows[0]["key_metric"] == "-"
