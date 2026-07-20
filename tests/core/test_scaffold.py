import yaml

from mlops.core.scaffold import (
    configure_dvc_remote,
    configure_local_dvc_remote,
    has_azure_config,
    has_dvc_config,
    scaffold_consumer,
    scaffold_producer,
    write_azure_config,
    write_azure_tracking_uri,
)


def test_creates_expected_layout(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    for name in ("dataset", "models", "outputs", "main.py", "params.yaml",
                 "requirements.txt", ".gitignore", "mlops.yaml", ".git", ".dvc"):
        assert (tmp_path / name).exists()


def test_writes_mlops_yaml_with_model_name(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["project"]["repo_role"] == "producer"


def test_gitignore_covers_mandatory_entries(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    gitignore = (tmp_path / ".gitignore").read_text()
    for entry in ("models/", "outputs/", ".mlops/", "mlruns/"):
        assert entry in gitignore


def test_dvc_add_creates_dataset_pointer(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    assert (tmp_path / "dataset.dvc").exists()


def test_idempotent_on_second_run(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    (tmp_path / "mlops.yaml").write_text("project:\n  model_name: unchanged\n  repo_role: producer\n")

    created = scaffold_producer(tmp_path, "different-name")

    assert created == []
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["project"]["model_name"] == "unchanged"


def test_existing_git_repo_is_reused(tmp_path):
    import subprocess
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    created = scaffold_producer(tmp_path, "fraud-detector")

    assert str(tmp_path / ".git") not in created


def test_has_azure_config_false_before_and_true_after(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    assert has_azure_config(tmp_path) is False

    write_azure_config(tmp_path, "sub", "rg", "ws")

    assert has_azure_config(tmp_path) is True


def test_write_azure_config_preserves_project_section(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    write_azure_config(tmp_path, "sub-id", "my-rg", "my-ws")

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["azure"] == {
        "subscription_id": "sub-id", "resource_group": "my-rg", "workspace": "my-ws",
    }


def test_configure_dvc_remote_writes_config_and_adds_remote(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    configure_dvc_remote(tmp_path, "myaccount", "mycontainer", "fraud-detector")

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["dvc"] == {
        "remote_url": "azure://mycontainer/fraud-detector", "account_name": "myaccount",
    }
    from mlops.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "azure"


def test_configure_dvc_remote_idempotent(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    configure_dvc_remote(tmp_path, "myaccount", "mycontainer", "fraud-detector")

    configure_dvc_remote(tmp_path, "other-account", "other-container", "other-prefix")

    from mlops.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "azure"


def test_scaffold_consumer_creates_expected_layout(tmp_path):
    scaffold_consumer(tmp_path)

    for name in ("mlops.yaml", "models.lock.yaml", "ci/mlops-fetch.snippet.yml",
                 ".gitignore", ".git"):
        assert (tmp_path / name).exists()

    for unwanted in ("dataset", "models", "outputs", "main.py", ".dvc"):
        assert not (tmp_path / unwanted).exists()


def test_scaffold_consumer_writes_consumer_repo_role(tmp_path):
    scaffold_consumer(tmp_path)

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["project"]["repo_role"] == "consumer"
    assert "model_name" not in config["project"]


def test_scaffold_consumer_gitignore_protects_mlops_dir(tmp_path):
    scaffold_consumer(tmp_path)

    assert ".mlops/" in (tmp_path / ".gitignore").read_text()


def test_scaffold_consumer_appends_to_existing_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n")

    scaffold_consumer(tmp_path)

    gitignore = (tmp_path / ".gitignore").read_text()
    assert "node_modules/" in gitignore
    assert ".mlops/" in gitignore


def test_scaffold_consumer_idempotent(tmp_path):
    scaffold_consumer(tmp_path)
    (tmp_path / "models.lock.yaml").write_text("fraud-detector:\n  version: 1\n  format: pkl\n  dest: x\n")

    created = scaffold_consumer(tmp_path)

    assert created == []
    assert "fraud-detector" in (tmp_path / "models.lock.yaml").read_text()


def test_write_azure_tracking_uri_preserves_project_section(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    write_azure_tracking_uri(tmp_path, "azureml://pasted-endpoint")

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["azure"] == {"tracking_uri": "azureml://pasted-endpoint"}


def test_has_dvc_config_false_before_and_true_after(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    assert has_dvc_config(tmp_path) is False

    configure_dvc_remote(tmp_path, "acct", "container", "prefix")

    assert has_dvc_config(tmp_path) is True


def test_configure_local_dvc_remote_writes_config_and_adds_remote(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    local_path = tmp_path.parent / "fraud-detector-dvc-storage"

    configure_local_dvc_remote(tmp_path, str(local_path))

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["dvc"] == {"remote_url": str(local_path)}
    assert "account_name" not in config["dvc"]
    from mlops.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "local"
    assert local_path.exists()


def test_configure_local_dvc_remote_idempotent(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    first_path = tmp_path.parent / "first"
    second_path = tmp_path.parent / "second"
    configure_local_dvc_remote(tmp_path, str(first_path))

    configure_local_dvc_remote(tmp_path, str(second_path))

    from mlops.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "local"
    assert not second_path.exists()
