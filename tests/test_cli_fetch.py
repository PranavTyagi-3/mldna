from typer.testing import CliRunner

from mldna.cli.main import app

runner = CliRunner()


def test_fetch_cli_happy_path(published):
    producer_repo, _, result = published
    (producer_repo / "models.lock.yaml").write_text(
        f"project-test-model:\n  version: {result['version']}\n  format: pkl\n  dest: app/models/test\n"
    )

    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 0
    assert "project-test-model v" in invocation.stdout
    assert (producer_repo / "app" / "models" / "test" / "model.pkl").exists()


def test_fetch_cli_missing_lockfile_fails_cleanly(producer_repo):
    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 1
    assert "fetch failed" in invocation.output
    assert "models.lock.yaml" in invocation.output


def test_fetch_cli_dest_escaping_workspace_fails_cleanly(producer_repo, tmp_path):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("keep me")
    (producer_repo / "models.lock.yaml").write_text(
        "test-model:\n  version: 1\n  format: pkl\n  dest: ../victim\n"
    )

    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 1
    assert "fetch failed" in invocation.output
    assert (victim / "important.txt").read_text() == "keep me"


def test_fetch_cli_foreign_nonempty_dest_fails_cleanly(published):
    producer_repo, _, result = published
    dest = producer_repo / "app" / "src"
    dest.mkdir(parents=True)
    (dest / "server.py").write_text("print('do not delete me')")
    (producer_repo / "models.lock.yaml").write_text(
        f"project-test-model:\n  version: {result['version']}\n  format: pkl\n  dest: app/src\n"
    )

    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 1
    assert "fetch failed" in invocation.output
    assert "did not create" in invocation.output
    assert (dest / "server.py").read_text() == "print('do not delete me')"


def test_fetch_cli_unreachable_registry_does_not_blame_the_lockfile(
    producer_repo, monkeypatch
):
    import mlflow
    from mlflow.exceptions import MlflowException

    (producer_repo / "models.lock.yaml").write_text(
        "test-model:\n  version: 1\n  format: pkl\n  dest: app/models/test\n"
    )

    def _unreachable(*args, **kwargs):
        raise MlflowException(
            "API request to http://localhost:5000/api/2.0/mlflow/model-versions/get "
            "failed with exception HTTPConnectionPool(host='localhost', port=5000)"
        )

    monkeypatch.setattr(mlflow.tracking.MlflowClient, "get_model_version", _unreachable)

    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 1
    assert "could not reach the model registry" in invocation.output
    assert "models.lock.yaml" not in invocation.output


def test_fetch_cli_shared_dest_fails_cleanly(producer_repo):
    (producer_repo / "models.lock.yaml").write_text(
        "test-model:\n  version: 1\n  format: pkl\n  dest: app/models/x\n"
        "test-model-canary:\n  model: test-model\n  version: 2\n  format: pkl\n  dest: ./app/models/x\n"
    )

    invocation = runner.invoke(app, ["fetch", "--dir", str(producer_repo)])

    assert invocation.exit_code == 1
    assert "fetch failed" in invocation.output
    assert "dest" in invocation.output
