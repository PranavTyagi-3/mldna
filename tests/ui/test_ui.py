import sys

import mlflow
import mlflow.tracking.fluent as mlflow_fluent
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from typer.testing import CliRunner  # noqa: E402

from mldna.cli.main import app as cli  # noqa: E402
from mldna.core import dvc  # noqa: E402
from mldna.tracking import track_run  # noqa: E402
from mldna.trackinguri import resolve_and_set_tracking_uri  # noqa: E402
from mldna.ui import gitread  # noqa: E402
from mldna.ui.data import UiData, UiDataError, web_url  # noqa: E402
from mldna.ui.server import create_app  # noqa: E402
from tests.conftest import run as git  # noqa: E402


def _train(repo, model_bytes):
    with track_run(workspace_dir=repo) as tracked:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(model_bytes)
    return tracked.run_id


@pytest.fixture(scope="module")
def two_runs_source(producer_factory, tmp_path_factory):
    """Run 1 on the scaffold; then a commit changing code AND data; then run 2."""
    repo = producer_factory(tmp_path_factory.mktemp("ui"))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "scaffold")
    first = _train(repo, b"first model")

    (repo / "train.py").write_text("learning_rate = 0.1\n")
    (repo / "dataset" / "extra.csv").write_text("a,b\n1,2\n")
    dvc.run(["add", "dataset", "-q"], cwd=repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "raise the learning rate and add extra rows")
    second = _train(repo, b"second model")

    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)
    mlflow_fluent._active_experiment_id = None
    return repo, first, second


@pytest.fixture
def two_runs(two_runs_source):
    repo, first, second = two_runs_source
    resolve_and_set_tracking_uri(repo)
    return repo, first, second


# --- read-only guards first ---

@pytest.mark.parametrize("bad", ["--output=/tmp/x", "HEAD", "abc", "../etc", "a" * 41, None])
def test_check_sha_rejects_anything_but_a_hash(bad):
    with pytest.raises(gitread.GitReadError):
        gitread.check_sha(bad)


@pytest.mark.parametrize("subcommand", ["push", "commit", "checkout", "reset", "fetch", "config"])
def test_git_refuses_write_subcommands(tmp_path, subcommand):
    with pytest.raises(gitread.GitReadError, match="not allowed"):
        gitread.git(tmp_path, subcommand)


def test_blame_refuses_a_path_not_in_the_commit(two_runs):
    repo, _, second = two_runs
    sha = UiData(repo).run(second)["commit"]
    with pytest.raises(gitread.GitReadError, match="not a file"):
        gitread.blame(repo, sha, "../../outside.txt")


def test_server_has_no_write_routes(two_runs):
    client = TestClient(create_app(two_runs[0]))
    for path in ("/api/runs", "/api/models", "/api/summary"):
        assert client.post(path).status_code == 405
        assert client.delete(path).status_code == 405


def test_ui_refuses_outside_a_project(tmp_path):
    with pytest.raises(UiDataError, match="mldna.yaml"):
        UiData(tmp_path)


def test_cli_without_the_extra_names_the_install_fix(monkeypatch):
    monkeypatch.delitem(sys.modules, "mldna.ui.server", raising=False)
    monkeypatch.setitem(sys.modules, "fastapi", None)

    result = CliRunner().invoke(cli, ["ui", "--no-browser"])

    assert result.exit_code == 1
    assert 'pip install "mldna-lineage[ui]"' in result.output


# --- links ---

@pytest.mark.parametrize("origin, expected", [
    ("https://github.com/acme/churn", "https://github.com/acme/churn"),
    ("github.com:acme/churn", "https://github.com/acme/churn"),
    ("https://gitlab.com/acme/churn", "https://gitlab.com/acme/churn"),
    ("https://git.internal.example/acme/churn", None),
    (None, None),
])
def test_web_url_only_for_known_hosts(origin, expected):
    assert web_url(origin) == expected


# --- what the pages show ---

def test_runs_newest_first_with_change_flags(two_runs):
    repo, first, second = two_runs
    runs = UiData(repo).runs()

    assert [r["run_id"] for r in runs] == [second, first]
    assert runs[0]["subject"] == "raise the learning rate and add extra rows"
    assert runs[0]["code_changed"] and runs[0]["data_changed"]
    assert runs[0]["previous_run"] == first
    assert not runs[1]["code_changed"] and runs[1]["previous_run"] is None


def test_code_diff_shows_the_change_and_the_commit_behind_it(two_runs):
    repo, _, second = two_runs
    data = UiData(repo)
    detail = data.run(second)

    result = data.code_diff(detail["previous_commit"], detail["commit"])

    assert "+learning_rate = 0.1" in result["diff"]
    assert [c["subject"] for c in result["commits"]] == ["raise the learning rate and add extra rows"]


def test_line_history_names_the_commit_that_changed_each_line(two_runs):
    repo, _, second = two_runs
    data = UiData(repo)
    sha = data.run(second)["commit"]

    result = data.blame(sha, "train.py")

    (line,) = result["lines"]
    assert line["text"] == "learning_rate = 0.1"
    assert result["commits"][line["commit"]]["subject"] == "raise the learning rate and add extra rows"


def test_data_diff_reads_the_pointers_without_the_data(two_runs):
    repo, _, second = two_runs
    data = UiData(repo)
    detail = data.run(second)

    result = data.data_diff(detail["previous_commit"], detail["commit"])

    (out,) = result["outs"]
    assert out["path"] == "dataset"
    assert out["after"]["nfiles"] == out["before"]["nfiles"] + 1


def test_models_page_before_any_publish(two_runs):
    assert UiData(two_runs[0]).models()["registered"] is False


def test_api_routes_answer_and_errors_are_clean(two_runs):
    repo, _, second = two_runs
    client = TestClient(create_app(repo))

    assert client.get("/").status_code == 200
    assert client.get("/static/app.js").status_code == 200
    assert len(client.get("/api/runs").json()) == 2
    assert client.get(f"/api/runs/{second}").json()["run_id"] == second

    bad = client.get("/api/blame", params={"commit": "--output=x", "path": "train.py"})
    assert bad.status_code == 400
    assert "not a commit hash" in bad.json()["detail"]
    assert client.get("/api/runs/0123456789abcdef0123456789abcdef").status_code == 404


def test_a_commit_changing_only_a_dvc_pointer_is_data_not_code(two_runs_source, tmp_path):
    import shutil

    source, _, _ = two_runs_source
    repo = tmp_path / "project"
    shutil.copytree(source, repo, symlinks=True)
    resolve_and_set_tracking_uri(repo)
    (repo / "dataset" / "more.csv").write_text("c,d\n3,4\n")
    dvc.run(["add", "dataset", "-q"], cwd=repo)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "add more rows")
    third = _train(repo, b"third model")

    data = UiData(repo)
    latest = data.runs()[0]
    assert latest["run_id"] == third
    assert latest["data_changed"] and not latest["code_changed"]

    detail = data.run(third)
    assert detail["code_changed"] is False
    result = data.code_diff(detail["previous_commit"], detail["commit"])
    assert result["diff"] == ""
    assert result["pointers"] == ["dataset.dvc"]
