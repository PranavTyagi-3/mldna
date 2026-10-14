import yaml
from typer.testing import CliRunner

from mldna.cli.main import app

runner = CliRunner()


def test_init_scaffolds_project(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    assert (tmp_path / "mldna.yaml").exists()
    assert "Initialized producer project 'fraud-detector'" in result.stdout


def test_init_writes_namespace_automatically(tmp_path):
    # issue #16: zero friction - nobody has to type a namespace, init derives
    # one from the project directory name.
    project = tmp_path / "alice-fraud-detector"
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(project),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((project / "mldna.yaml").read_text())
    assert config["project"]["namespace"] == "alice-fraud-detector"


def test_init_namespace_flag_overrides_the_default(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path), "--namespace", "alice",
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["namespace"] == "alice"


def test_init_local_tracking_source_has_no_azure_section(tmp_path):
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert "azure" not in config


def test_init_bare_prompts_for_tracking_source_and_dvc_backend(tmp_path):
    # No flags at all: numbered menus for tracking source then DVC backend.
    # "1" = Local, then "3" = Skip for now.
    result = runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path)], input="1\n3\n")

    assert result.exit_code == 0
    assert "Select MLflow tracking source:" in result.output
    assert "Select DVC storage backend:" in result.output
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
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
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["dvc"] == {"remote_url": str(dvc_storage)}
    from mldna.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "local"


def test_init_azure_flag_with_options_skips_prompts(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path), "--azure",
        "--subscription-id", "sub", "--resource-group", "rg", "--workspace", "ws",
        "--dvc-account", "acct", "--dvc-container", "cont", "--dvc-prefix", "prefix",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
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
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["azure"] == {"tracking_uri": "azureml://pasted-endpoint"}
    assert (tmp_path / ".mldna" / "aml_auth.json").exists()


def test_init_azure_ad_flow_discovers_and_selects_workspace(tmp_path, monkeypatch):
    # list_workspaces is imported lazily inside _configure_azure_tracking (so
    # a base install without the [azure] extra never needs it at all) -
    # patch it at its source module, not at mldna.cli.main, which no longer
    # holds a module-level reference to it.
    monkeypatch.setattr(
        "mldna.azure.tracking.list_workspaces",
        lambda subscription_id, credential: [
            ("ml-workspace-dev", "rg-dev"), ("ml-workspace-prod", "rg-prod"),
        ],
    )
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: "fake-credential")

    # tracking-source=Azure ML (menu choice "3"), then auth-mode=aad ("1"),
    # then subscription ID, then pick workspace #2.
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path), "--dvc-backend", "skip",
    ], input="3\n1\nsub-123\n2\n")

    assert result.exit_code == 0
    assert "Select AML workspace:" in result.output
    assert "ml-workspace-dev" in result.output
    assert "ml-workspace-prod" in result.output
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
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
    assert "already has a tracking source configured" in result.output
    assert "already has a dvc: section" in result.output


def test_init_mlflow_uri_flag_skips_menu(tmp_path):
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--mlflow-uri", "http://my-mlflow-host:5000", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["mlflow"] == {"tracking_uri": "http://my-mlflow-host:5000"}
    assert "azure" not in config


def test_init_bare_prompts_offer_mlflow_menu_option(tmp_path):
    # "2" = Remote MLflow server (URL), then the URL prompt, then "3" = Skip
    # DVC backend.
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
    ], input="2\nhttp://my-mlflow-host:5000\n3\n")

    assert result.exit_code == 0
    assert "Remote MLflow server (URL)" in result.output
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["mlflow"] == {"tracking_uri": "http://my-mlflow-host:5000"}


def test_init_mlflow_flag_does_not_reprompt_when_already_configured(tmp_path):
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--mlflow-uri", "http://my-mlflow-host:5000", "--dvc-backend", "skip",
    ])

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--mlflow-uri", "http://other-host:5000", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    assert "already has a tracking source configured" in result.output
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["mlflow"] == {"tracking_uri": "http://my-mlflow-host:5000"}


def test_init_unknown_tracking_source_refuses(tmp_path):
    # Used to configure nothing at all, silently.
    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "databricks", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 1
    assert "Unknown --tracking-source 'databricks'" in result.output
    assert "local | mlflow | azure | filesystem" in result.output
    assert not (tmp_path / "mldna.yaml").exists()


def test_init_filesystem_tracking_source_writes_resolvable_section(tmp_path):
    store = tmp_path.parent / "shared-mlruns"
    store.mkdir(exist_ok=True)
    project = tmp_path / "project"

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(project),
        "--tracking-source", "filesystem", "--filesystem-path", str(store),
        "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((project / "mldna.yaml").read_text())
    assert config["filesystem"] == {"path": str(store)}
    assert "mlflow" not in config and "azure" not in config

    from mldna.trackinguri import resolve_and_set_tracking_uri
    uri, mode = resolve_and_set_tracking_uri(project)
    assert uri == f"file:{store.resolve()}"
    assert mode == "filesystem store"


def test_init_filesystem_path_prompted_when_flag_omitted(tmp_path):
    store = tmp_path.parent / "prompted-mlruns"
    store.mkdir(exist_ok=True)
    project = tmp_path / "project"

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(project),
        "--tracking-source", "filesystem", "--dvc-backend", "skip",
    ], input=f"{store}\n")

    assert result.exit_code == 0
    assert "Path to an existing MLflow file store" in result.output
    config = yaml.safe_load((project / "mldna.yaml").read_text())
    assert config["filesystem"] == {"path": str(store)}


def test_init_bare_menu_offers_filesystem_option(tmp_path):
    store = tmp_path.parent / "menu-mlruns"
    store.mkdir(exist_ok=True)
    project = tmp_path / "project"

    # "4" = Shared filesystem store, then the path, then "3" = skip DVC.
    result = runner.invoke(
        app, ["init", "fraud-detector", "--dir", str(project)],
        input=f"4\n{store}\n3\n",
    )

    assert result.exit_code == 0
    assert "Shared filesystem store (existing path)" in result.output
    config = yaml.safe_load((project / "mldna.yaml").read_text())
    assert config["filesystem"] == {"path": str(store)}


def test_init_consumer_filesystem_path_writes_section(tmp_path):
    store = tmp_path.parent / "consumer-mlruns"
    store.mkdir(exist_ok=True)

    result = runner.invoke(app, [
        "init", "--dir", str(tmp_path), "--consumer", "--filesystem-path", str(store),
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["filesystem"] == {"path": str(store)}


def test_init_filesystem_flag_does_not_reprompt_when_already_configured(tmp_path):
    store = tmp_path.parent / "reprompt-mlruns"
    store.mkdir(exist_ok=True)
    project = tmp_path / "project"
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(project),
        "--filesystem-path", str(store), "--dvc-backend", "skip",
    ])

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(project),
        "--filesystem-path", str(tmp_path.parent / "other"), "--dvc-backend", "skip",
    ])

    assert result.exit_code == 0
    assert "already has a tracking source configured" in result.output
    config = yaml.safe_load((project / "mldna.yaml").read_text())
    assert config["filesystem"] == {"path": str(store)}


def test_init_azure_dvc_does_not_write_connection_string_to_local_remote(tmp_path, monkeypatch):
    # An existing local default remote is left alone by configure_dvc_remote,
    # so the connection string must not land on it either.
    from mldna.core import dvc
    from mldna.core.scaffold import scaffold_producer

    scaffold_producer(tmp_path, "fraud-detector")
    dvc.run(["remote", "add", "-d", "local-remote", str(tmp_path.parent / "store")], cwd=tmp_path)
    monkeypatch.setattr(
        "mldna.azure.blob.list_containers_via_connection_string",
        lambda connection_string: ["alpha"],
    )

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "azure",
        "--blob-auth-mode", "connection-string",
        "--connection-string", "DefaultEndpointsProtocol=https;AccountName=myacct;AccountKey=xxx",
        "--dvc-container", "alpha", "--dvc-prefix", "fraud-detector",
    ])

    assert result.exit_code == 0
    assert "is not an Azure Blob remote" in result.output
    config_local = tmp_path / ".dvc" / "config.local"
    assert "AccountKey" not in (config_local.read_text() if config_local.exists() else "")


def test_init_refuses_in_subfolder_of_existing_project(tmp_path):
    parent = tmp_path / "parent"
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(parent),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])
    child = parent / "nested" / "sub"

    result = runner.invoke(app, [
        "init", "other-model", "--dir", str(child),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    assert result.exit_code == 1
    assert str(parent / "mldna.yaml") in result.output
    assert "mldna init" not in result.output
    assert not (child / "mldna.yaml").exists()


def test_init_consumer_refuses_in_subfolder_of_existing_project(tmp_path):
    parent = tmp_path / "parent"
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(parent),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])
    child = parent / "nested"

    result = runner.invoke(app, ["init", "--dir", str(child), "--consumer"])

    assert result.exit_code == 1
    assert str(parent / "mldna.yaml") in result.output


def test_init_without_model_name_requires_consumer_flag(tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "--consumer" in result.output


def test_init_consumer_scaffolds_lockfile_and_ci_snippet(tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--consumer"])

    assert result.exit_code == 0
    assert (tmp_path / "models.lock.yaml").exists()
    assert (tmp_path / "ci" / "mldna-fetch.snippet.yml").exists()
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["repo_role"] == "consumer"


def test_init_dvc_connection_string_derives_account_and_lists_containers(tmp_path, monkeypatch):
    # Account name must come from the connection string itself, never asked
    # separately; the container is picked from a real listing instead of typed.
    monkeypatch.setattr(
        "mldna.azure.blob.list_containers_via_connection_string",
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
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
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
        "mldna.azure.blob.list_containers_via_credential",
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
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["dvc"] == {
        "remote_url": "azure://datasets/fraud-detector", "account_name": "myacct",
    }


def test_init_dvc_container_listing_can_create_new_container(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "mldna.azure.blob.list_containers_via_credential",
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
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["dvc"]["remote_url"] == "azure://brand-new/fraud-detector"


def test_init_dvc_container_listing_failure_falls_back_to_manual_prompt(tmp_path, monkeypatch):
    def _boom(account_name, credential):
        raise RuntimeError("403 Forbidden")

    monkeypatch.setattr("mldna.azure.blob.list_containers_via_credential", _boom)
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", lambda: "fake-credential")

    result = runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "azure",
        "--blob-auth-mode", "aad", "--dvc-account", "myacct",
        "--dvc-prefix", "fraud-detector",
    ], input="typed-container\n")

    assert result.exit_code == 0
    assert "Could not list containers" in result.output
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["dvc"]["remote_url"] == "azure://typed-container/fraud-detector"


def test_init_consumer_azure_flag_writes_azure_section_only(tmp_path):
    result = runner.invoke(app, [
        "init", "--dir", str(tmp_path), "--consumer", "--azure",
        "--subscription-id", "sub", "--resource-group", "rg", "--workspace", "ws",
    ])

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["azure"] == {"subscription_id": "sub", "resource_group": "rg", "workspace": "ws"}
    assert "dvc" not in config


def test_id_verb_is_idempotent(tmp_path):
    (tmp_path / "mldna.yaml").write_text("project:\n  model_name: m\n  namespace: ns\n")

    first = runner.invoke(app, ["id", "--dir", str(tmp_path)])
    written = yaml.safe_load((tmp_path / "mldna.yaml").read_text())["project"]["model_id"]
    second = runner.invoke(app, ["id", "--dir", str(tmp_path)])

    assert first.exit_code == 0 and "Commit mldna.yaml" in first.stdout
    assert second.exit_code == 0 and "already" in second.stdout
    assert yaml.safe_load((tmp_path / "mldna.yaml").read_text())["project"]["model_id"] == written


def test_id_new_replaces_the_id_and_warns(tmp_path):
    # D#77: a project copied from another must be able to shed the copied id.
    (tmp_path / "mldna.yaml").write_text(
        "project:\n  model_name: m\n  namespace: ns\n  model_id: copied-id\n")

    result = runner.invoke(app, ["id", "--new", "--dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "will now refuse this project" in result.stdout
    assert yaml.safe_load((tmp_path / "mldna.yaml").read_text())["project"]["model_id"] != "copied-id"


def test_init_generates_model_id(tmp_path):
    runner.invoke(app, [
        "init", "m", "--dir", str(tmp_path),
        "--tracking-source", "local", "--dvc-backend", "skip",
    ])

    assert yaml.safe_load((tmp_path / "mldna.yaml").read_text())["project"]["model_id"]
