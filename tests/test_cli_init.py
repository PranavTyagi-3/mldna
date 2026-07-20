import yaml
from typer.testing import CliRunner

from mlops.cli.main import app

runner = CliRunner()


def test_init_scaffolds_project(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    assert (tmp_path / "mlops.yaml").exists()
    assert "Initialized producer project 'fraud-detector'" in result.stdout


def test_init_local_tracking_source_has_no_azure_section(tmp_path):
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert "azure" not in config


def test_init_bare_prompts_for_tracking_source_and_dvc_backend(tmp_path):
    # No flags at all: numbered menus for tracking source then DVC backend.
    # "1" = Local, then "3" = Skip for now.
    result = runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path)], input="1\n3\n")

    assert result.exit_code == 0
    assert "Select MLflow tracking source:" in result.output
    assert "Select DVC storage backend:" in result.output
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert "azure" not in config
    assert "dvc" not in config


def test_init_dvc_backend_local_configures_real_remote(tmp_path):
    dvc_storage = tmp_path.parent / "fraud-detector-dvc-storage"
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "local",
        "--dvc-local-path", str(dvc_storage),
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["dvc"] == {"remote_url": str(dvc_storage)}
    from mlops.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "local"


def test_init_azure_flag_with_options_skips_prompts(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path), "--azure",
        "--subscription-id", "sub", "--resource-group", "rg", "--workspace", "ws",
        "--dvc-account", "acct", "--dvc-container", "cont", "--dvc-prefix", "prefix",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["azure"] == {"subscription_id": "sub", "resource_group": "rg", "workspace": "ws"}
    assert config["dvc"] == {"remote_url": "azure://cont/prefix", "account_name": "acct"}
    assert "configure aml-auth" in result.output


def test_init_service_principal_flow_pastes_tracking_uri(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--auth-mode", "service-principal",
        "--tenant-id", "t", "--client-id", "c", "--client-secret", "s",
        "--tracking-uri", "azureml://pasted-endpoint",
        "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["azure"] == {"tracking_uri": "azureml://pasted-endpoint"}
    assert (tmp_path / ".mlops" / "aml_auth.json").exists()


def test_init_azure_ad_flow_discovers_and_selects_workspace(tmp_path, monkeypatch):
    # list_workspaces is imported lazily inside _configure_azure_tracking (so
    # a base install without the [azure] extra never needs it at all) -
    # patch it at its source module, not at mlops.cli.main, which no longer
    # holds a module-level reference to it.
    monkeypatch.setattr(
        "mlops.azure.tracking.list_workspaces",
        lambda subscription_id, credential: [
            ("ml-workspace-dev", "rg-dev"), ("ml-workspace-prod", "rg-prod"),
        ],
    )
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: "fake-credential")

    # auth-mode=aad (menu choice "1"), then subscription ID, then pick workspace #2.
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path), "--dvc-backend", "skip",
    ], input="2\n1\nsub-123\n2\n")

    assert result.exit_code == 0
    assert "Select AML workspace:" in result.output
    assert "ml-workspace-dev" in result.output
    assert "ml-workspace-prod" in result.output
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["azure"] == {
        "subscription_id": "sub-123", "resource_group": "rg-prod", "workspace": "ml-workspace-prod",
    }


def test_init_azure_flag_does_not_reprompt_when_already_configured(tmp_path):
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path), "--azure",
        "--subscription-id", "sub", "--resource-group", "rg", "--workspace", "ws",
        "--dvc-account", "acct", "--dvc-container", "cont", "--dvc-prefix", "prefix",
    ])

    result = runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path), "--azure"])

    assert result.exit_code == 0
    assert "already has an azure: section" in result.output
    assert "already has a dvc: section" in result.output


def test_init_without_model_name_requires_consumer_flag(tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "--consumer" in result.output


def test_init_consumer_scaffolds_lockfile_and_ci_snippet(tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--consumer"])

    assert result.exit_code == 0
    assert (tmp_path / "models.lock.yaml").exists()
    assert (tmp_path / "ci" / "mlops-fetch.snippet.yml").exists()
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["project"]["repo_role"] == "consumer"


def test_init_dvc_connection_string_derives_account_and_lists_containers(tmp_path, monkeypatch):
    # Account name must come from the connection string itself, never asked
    # separately; the container is picked from a real listing instead of typed.
    monkeypatch.setattr(
        "mlops.azure.blob.list_containers_via_connection_string",
        lambda connection_string: ["alpha", "bravo"],
    )

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "azure",
        "--blob-auth-mode", "connection-string",
        "--connection-string", "DefaultEndpointsProtocol=https;AccountName=myacct;AccountKey=xxx;EndpointSuffix=core.windows.net",
        "--dvc-prefix", "fraud-detector",
    ], input="2\n")

    assert result.exit_code == 0
    assert "Select DVC container:" in result.output
    assert "alpha" in result.output and "bravo" in result.output
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["dvc"] == {
        "remote_url": "azure://bravo/fraud-detector", "account_name": "myacct",
    }


def test_init_dvc_connection_string_without_account_name_fails_clearly(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "azure",
        "--blob-auth-mode", "connection-string",
        "--connection-string", "SharedAccessSignature=sv=2020-01-01&sig=xxx",
    ])

    assert result.exit_code == 1
    assert "Could not find AccountName=" in result.output


def test_init_dvc_aad_lists_containers_for_given_account(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mlops.azure.blob.list_containers_via_credential",
        lambda account_name, credential: ["datasets"],
    )
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: "fake-credential")

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "azure",
        "--blob-auth-mode", "aad", "--dvc-account", "myacct",
        "--dvc-prefix", "fraud-detector",
    ], input="1\n")

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["dvc"] == {
        "remote_url": "azure://datasets/fraud-detector", "account_name": "myacct",
    }


def test_init_dvc_container_listing_can_create_new_container(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mlops.azure.blob.list_containers_via_credential",
        lambda account_name, credential: ["datasets"],
    )
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: "fake-credential")

    # "2" picks the trailing "Enter a new container name" menu option.
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "azure",
        "--blob-auth-mode", "aad", "--dvc-account", "myacct",
        "--dvc-prefix", "fraud-detector",
    ], input="2\nbrand-new\n")

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["dvc"]["remote_url"] == "azure://brand-new/fraud-detector"


def test_init_dvc_container_listing_failure_falls_back_to_manual_prompt(tmp_path, monkeypatch):
    def _boom(account_name, credential):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr("mlops.azure.blob.list_containers_via_credential", _boom)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: "fake-credential")

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "azure",
        "--blob-auth-mode", "aad", "--dvc-account", "myacct",
        "--dvc-prefix", "fraud-detector",
    ], input="typed-container\n")

    assert result.exit_code == 0
    assert "Could not list containers" in result.output
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["dvc"]["remote_url"] == "azure://typed-container/fraud-detector"


def test_init_consumer_azure_flag_writes_azure_section_only(tmp_path):
    result = runner.invoke(app, [
        "init", "--dir", str(tmp_path), "--consumer", "--azure",
        "--subscription-id", "sub", "--resource-group", "rg", "--workspace", "ws",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["azure"] == {"subscription_id": "sub", "resource_group": "rg", "workspace": "ws"}
    assert "dvc" not in config
