import json
import subprocess

from typer.testing import CliRunner

from mlops.cli.main import app
from mlops.core import dvc
from mlops.core.scaffold import scaffold_producer

runner = CliRunner()


def _azure_remote_repo(tmp_path):
    scaffold_producer(tmp_path, "test-model")
    dvc.run(["remote", "add", "-d", "azure-remote", "azure://fake-container/prefix"], cwd=tmp_path)
    return tmp_path


def test_configure_aml_auth_writes_file(tmp_path):
    result = runner.invoke(app, [
        "configure", "aml-auth",
        "--tenant-id", "t", "--client-id", "c", "--client-secret", "s",
        "--dir", str(tmp_path),
    ])

    assert result.exit_code == 0
    assert "Active AML auth mode" in result.output
    data = json.loads((tmp_path / ".mlops" / "aml_auth.json").read_text())
    assert data == {"tenant_id": "t", "client_id": "c", "client_secret": "s"}


def test_configure_aml_auth_clear(tmp_path):
    runner.invoke(app, [
        "configure", "aml-auth",
        "--tenant-id", "t", "--client-id", "c", "--client-secret", "s",
        "--dir", str(tmp_path),
    ])

    result = runner.invoke(app, ["configure", "aml-auth", "--clear", "--dir", str(tmp_path)])

    assert result.exit_code == 0
    assert not (tmp_path / ".mlops" / "aml_auth.json").exists()


def test_configure_blob_auth_writes_connection_string(tmp_path):
    repo = _azure_remote_repo(tmp_path)

    result = runner.invoke(app, [
        "configure", "blob-auth",
        "--connection-string", "DefaultEndpointsProtocol=https;fake=1",
        "--dir", str(repo),
    ])

    assert result.exit_code == 0
    config_local = (repo / ".dvc" / "config.local").read_text()
    assert "fake=1" in config_local


def test_configure_blob_auth_no_remote_configured(tmp_path):
    scaffold_producer(tmp_path, "test-model")

    result = runner.invoke(app, [
        "configure", "blob-auth",
        "--connection-string", "x",
        "--dir", str(tmp_path),
    ])

    assert result.exit_code == 1
    assert "No DVC remote configured" in result.output


def test_secrets_never_appear_in_git_status(tmp_path):
    repo = _azure_remote_repo(tmp_path)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)

    result1 = runner.invoke(app, [
        "configure", "aml-auth",
        "--tenant-id", "t", "--client-id", "c", "--client-secret", "s",
        "--dir", str(repo),
    ])
    result2 = runner.invoke(app, [
        "configure", "blob-auth",
        "--connection-string", "DefaultEndpointsProtocol=https;secret=1",
        "--dir", str(repo),
    ])
    assert result1.exit_code == 0
    assert result2.exit_code == 0

    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True
    ).stdout

    assert status.strip() == ""
