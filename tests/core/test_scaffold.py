import yaml

from mldna.core.scaffold import (
    configure_dvc_remote,
    configure_local_dvc_remote,
    has_azure_config,
    has_dvc_config,
    has_filesystem_config,
    has_mlflow_config,
    scaffold_consumer,
    scaffold_producer,
    write_azure_config,
    write_azure_tracking_uri,
    write_filesystem_path,
    write_mlflow_tracking_uri,
)


def test_creates_expected_layout(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    for name in ("dataset", "models", "outputs", "main.py", "params.yaml",
                 "requirements.txt", ".gitignore", "mldna.yaml", ".git", ".dvc"):
        assert (tmp_path / name).exists()


def test_writes_mldna_yaml_with_model_name(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["project"]["repo_role"] == "producer"


def test_namespace_defaults_to_the_project_directory_name(tmp_path):
    # issue #16: zero friction - init writes a namespace automatically.
    project = tmp_path / "alice-fraud-detector"
    scaffold_producer(project, "fraud-detector")

    config = yaml.safe_load((project / "mldna.yaml").read_text())
    assert config["project"]["namespace"] == "alice-fraud-detector"


def test_namespace_can_be_given_explicitly(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector", namespace="alice")

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["namespace"] == "alice"


def test_requirements_pins_installable_distribution_name(tmp_path):
    from mldna import __version__

    scaffold_producer(tmp_path, "fraud-detector")

    assert (tmp_path / "requirements.txt").read_text() == f"mldna-lineage=={__version__}\n"


def test_gitignore_covers_mandatory_entries(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    gitignore = (tmp_path / ".gitignore").read_text()
    for entry in ("models/", "outputs/", ".mldna/", "mlruns/"):
        assert entry in gitignore


def test_gitignore_covers_secret_patterns(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    gitignore = (tmp_path / ".gitignore").read_text()
    for entry in (".env", ".env.*", "*.pem", "*.key"):
        assert entry in gitignore


def test_dvc_add_creates_dataset_pointer(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    assert (tmp_path / "dataset.dvc").exists()


def test_idempotent_on_second_run(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    (tmp_path / "mldna.yaml").write_text("project:\n  model_name: unchanged\n  repo_role: producer\n")

    created = scaffold_producer(tmp_path, "different-name")

    assert created == []
    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
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

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["azure"] == {
        "subscription_id": "sub-id", "resource_group": "my-rg", "workspace": "my-ws",
    }


def test_configure_dvc_remote_writes_config_and_adds_remote(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    configure_dvc_remote(tmp_path, "myaccount", "mycontainer", "fraud-detector")

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["dvc"] == {
        "remote_url": "azure://mycontainer/fraud-detector", "account_name": "myaccount",
    }
    from mldna.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "azure"


def test_configure_dvc_remote_idempotent(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    configure_dvc_remote(tmp_path, "myaccount", "mycontainer", "fraud-detector")

    configure_dvc_remote(tmp_path, "other-account", "other-container", "other-prefix")

    from mldna.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "azure"


def test_scaffold_consumer_creates_expected_layout(tmp_path):
    scaffold_consumer(tmp_path)

    for name in ("mldna.yaml", "models.lock.yaml", "ci/mldna-fetch.snippet.yml",
                 ".gitignore", ".git"):
        assert (tmp_path / name).exists()

    for unwanted in ("dataset", "models", "outputs", "main.py", ".dvc"):
        assert not (tmp_path / unwanted).exists()


def test_ci_snippet_contains_pinned_base_install_line(tmp_path):
    from mldna import __version__

    scaffold_consumer(tmp_path)

    snippet = (tmp_path / "ci" / "mldna-fetch.snippet.yml").read_text()
    # Base install line (active, not commented) without [azure] extra
    assert f"pip install 'mldna-lineage=={__version__}'" in snippet
    # Verify it's the base install (comes before Azure's extra version)
    base_install = f"pip install 'mldna-lineage=={__version__}'\n"
    azure_install = f"pip install 'mldna-lineage[azure]=={__version__}'"
    assert base_install in snippet
    assert azure_install in snippet  # Azure version is in commented section


def test_ci_snippet_contains_mldna_fetch_command(tmp_path):
    scaffold_consumer(tmp_path)

    snippet = (tmp_path / "ci" / "mldna-fetch.snippet.yml").read_text()
    # mldna fetch must appear (in multiple backend options, all commented)
    assert snippet.count("mldna fetch") >= 2
    assert "Local store" not in snippet


def test_ci_snippet_shows_the_three_shared_tracking_backends(tmp_path):
    scaffold_consumer(tmp_path)

    snippet = (tmp_path / "ci" / "mldna-fetch.snippet.yml").read_text()
    # Each backend clearly labeled as an option
    assert "BACKEND OPTION 1: Filesystem store" in snippet
    assert "BACKEND OPTION 2: MLflow server" in snippet
    assert "BACKEND OPTION 3: Azure ML" in snippet
    assert "OPTION 4" not in snippet


def test_ci_snippet_explains_filesystem_path_requirement(tmp_path):
    scaffold_consumer(tmp_path)

    snippet = (tmp_path / "ci" / "mldna-fetch.snippet.yml").read_text()
    # Filesystem option must note path must be reachable
    assert "reachable from this CI runner" in snippet


def test_ci_snippet_points_to_mlflow_auth_docs(tmp_path):
    scaffold_consumer(tmp_path)

    snippet = (tmp_path / "ci" / "mldna-fetch.snippet.yml").read_text()
    # MLflow option should reference MLflow's docs for auth
    assert "see MLflow's docs" in snippet
    assert "MLFLOW_TRACKING_USERNAME" in snippet
    assert "MLFLOW_TRACKING_PASSWORD" in snippet


def test_ci_snippet_azure_section_documents_secrets_needed(tmp_path):
    scaffold_consumer(tmp_path)

    snippet = (tmp_path / "ci" / "mldna-fetch.snippet.yml").read_text()
    # Azure section must document all three required secrets
    assert "AZURE_TENANT_ID" in snippet
    assert "AZURE_CLIENT_ID" in snippet
    assert "AZURE_CLIENT_SECRET" in snippet
    # Verify they're shown as commented options (in the Azure backend block)
    assert "#    AZURE_TENANT_ID:" in snippet
    assert "service-principal secrets" in snippet
    # Azure install must include [azure] extra
    assert "#  run: pip install 'mldna-lineage[azure]" in snippet


def test_scaffold_consumer_writes_consumer_repo_role(tmp_path):
    scaffold_consumer(tmp_path)

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["repo_role"] == "consumer"
    assert "model_name" not in config["project"]
    assert "namespace" not in config["project"]


def test_scaffold_consumer_gitignore_protects_mldna_dir(tmp_path):
    scaffold_consumer(tmp_path)

    assert ".mldna/" in (tmp_path / ".gitignore").read_text()


def test_scaffold_consumer_appends_to_existing_gitignore(tmp_path):
    (tmp_path / ".gitignore").write_text("node_modules/\n")

    scaffold_consumer(tmp_path)

    gitignore = (tmp_path / ".gitignore").read_text()
    assert "node_modules/" in gitignore
    assert ".mldna/" in gitignore


def test_scaffold_consumer_idempotent(tmp_path):
    scaffold_consumer(tmp_path)
    (tmp_path / "models.lock.yaml").write_text("fraud-detector:\n  version: 1\n  format: pkl\n  dest: x\n")

    created = scaffold_consumer(tmp_path)

    assert created == []
    assert "fraud-detector" in (tmp_path / "models.lock.yaml").read_text()


def test_write_azure_tracking_uri_preserves_project_section(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    write_azure_tracking_uri(tmp_path, "azureml://pasted-endpoint")

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["azure"] == {"tracking_uri": "azureml://pasted-endpoint"}


def test_has_mlflow_config_false_before_and_true_after(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    assert has_mlflow_config(tmp_path) is False

    write_mlflow_tracking_uri(tmp_path, "http://my-mlflow-host:5000")

    assert has_mlflow_config(tmp_path) is True


def test_write_mlflow_tracking_uri_preserves_project_section(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    write_mlflow_tracking_uri(tmp_path, "http://my-mlflow-host:5000")

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["mlflow"] == {"tracking_uri": "http://my-mlflow-host:5000"}


def test_has_dvc_config_false_before_and_true_after(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    assert has_dvc_config(tmp_path) is False

    configure_dvc_remote(tmp_path, "acct", "container", "prefix")

    assert has_dvc_config(tmp_path) is True


def test_configure_local_dvc_remote_writes_config_and_adds_remote(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    local_path = tmp_path.parent / "fraud-detector-dvc-storage"

    configure_local_dvc_remote(tmp_path, str(local_path))

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["dvc"] == {"remote_url": str(local_path)}
    assert "account_name" not in config["dvc"]
    from mldna.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "local"
    assert local_path.exists()


def test_scaffolded_config_parses_to_unchanged_project_config(tmp_path):
    # The regression guard for the commented optional-sections block: a
    # commented block must leave the parsed config byte-for-byte equivalent.
    from mldna.core.config import ProjectConfig
    from mldna.core.discovery import DEFAULT_EXCLUDES, DEFAULT_MODEL_GLOBS

    scaffold_producer(tmp_path, "fraud-detector")

    config = ProjectConfig.load(tmp_path)
    assert config.model_name == "fraud-detector"
    assert config.repo_role == "producer"
    assert config.model_globs == list(DEFAULT_MODEL_GLOBS)
    assert config.scan_exclude == list(DEFAULT_EXCLUDES)
    assert config.key_metric is None
    assert config.require_formats == []
    assert config.namespace == config.raw["project"]["namespace"]
    assert config.raw == {
        "project": {
            "model_name": "fraud-detector", "repo_role": "producer",
            "namespace": config.namespace, "model_id": config.model_id,
        }
    }


def test_scaffolded_config_documents_optional_sections(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    text = (tmp_path / "mldna.yaml").read_text()
    for key in ("key_metric", "model_globs", "scan_exclude", "require_formats"):
        mentions = [line for line in text.splitlines() if key in line]
        assert mentions, f"{key} is undiscoverable in the scaffolded mldna.yaml"
        assert all(line.lstrip().startswith("#") for line in mentions)


def test_optional_sections_comment_survives_a_later_section_write(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    write_mlflow_tracking_uri(tmp_path, "http://my-mlflow-host:5000")

    text = (tmp_path / "mldna.yaml").read_text()
    assert "key_metric" in text
    assert yaml.safe_load(text)["mlflow"] == {"tracking_uri": "http://my-mlflow-host:5000"}


def test_dvc_section_is_labeled_an_informational_mirror(tmp_path):
    # Gap 4 resolution (a): the section is write-only by design, so it says so
    # and names the real source of truth instead of being parsed.
    scaffold_producer(tmp_path, "fraud-detector")

    configure_dvc_remote(tmp_path, "myaccount", "mycontainer", "fraud-detector")

    text = (tmp_path / "mldna.yaml").read_text()
    assert "Informational mirror" in text
    assert ".dvc/config" in text
    assert "dvc remote modify" in text
    assert text.index("Informational mirror") < text.index("dvc:\n")
    assert yaml.safe_load(text)["dvc"]["remote_url"] == "azure://mycontainer/fraud-detector"


def test_local_dvc_section_is_labeled_too(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    configure_local_dvc_remote(tmp_path, str(tmp_path.parent / "store"))

    assert "Informational mirror" in (tmp_path / "mldna.yaml").read_text()


def test_has_filesystem_config_false_before_and_true_after(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    assert has_filesystem_config(tmp_path) is False

    write_filesystem_path(tmp_path, "/shared/mlruns")

    assert has_filesystem_config(tmp_path) is True


def test_write_filesystem_path_preserves_project_section(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")

    write_filesystem_path(tmp_path, "/shared/mlruns")

    config = yaml.safe_load((tmp_path / "mldna.yaml").read_text())
    assert config["project"]["model_name"] == "fraud-detector"
    assert config["filesystem"] == {"path": "/shared/mlruns"}


def test_configure_local_dvc_remote_idempotent(tmp_path):
    scaffold_producer(tmp_path, "fraud-detector")
    first_path = tmp_path.parent / "first"
    second_path = tmp_path.parent / "second"
    configure_local_dvc_remote(tmp_path, str(first_path))

    configure_local_dvc_remote(tmp_path, str(second_path))

    from mldna.core import blobauth
    assert blobauth.get_default_remote(tmp_path) == "local"
    assert not second_path.exists()


def test_scaffold_producer_appends_to_existing_gitignore(tmp_path):
    """Existing .gitignore with unrelated content gets mandatory entries appended."""
    (tmp_path / ".gitignore").write_text("node_modules/\nbuild/\n")

    scaffold_producer(tmp_path, "fraud-detector")

    gitignore = (tmp_path / ".gitignore").read_text()
    assert "node_modules/" in gitignore
    assert "build/" in gitignore
    for entry in ("models/", "outputs/", ".mldna/", "mlruns/", ".env", ".env.*", "*.pem", "*.key"):
        assert entry in gitignore


def test_scaffold_producer_gitignore_idempotent(tmp_path):
    """Running scaffold_producer twice doesn't duplicate .gitignore entries."""
    scaffold_producer(tmp_path, "fraud-detector")
    first_content = (tmp_path / ".gitignore").read_text()

    created = scaffold_producer(tmp_path, "different-name")

    second_content = (tmp_path / ".gitignore").read_text()
    assert first_content == second_content
    assert created == []


def test_scaffold_producer_creates_full_gitignore_when_missing(tmp_path):
    """Empty directory gets the full .gitignore as before."""
    assert not (tmp_path / ".gitignore").exists()

    scaffold_producer(tmp_path, "fraud-detector")

    gitignore = (tmp_path / ".gitignore").read_text()
    for entry in ("models/", "outputs/", ".mldna/", "mlruns/", ".dvc/cache/", ".env", ".env.*", "*.pem", "*.key"):
        assert entry in gitignore


def test_scaffold_producer_gitignore_partial_overlap(tmp_path):
    """Existing .gitignore with some entries gains only the missing ones."""
    (tmp_path / ".gitignore").write_text("models/\n.mldna/\n")

    scaffold_producer(tmp_path, "fraud-detector")

    gitignore = (tmp_path / ".gitignore").read_text()
    lines = gitignore.splitlines()
    assert lines.count("models/") == 1
    assert lines.count(".mldna/") == 1
    for entry in ("outputs/", "mlruns/", ".env", ".env.*", "*.pem", "*.key"):
        assert entry in gitignore


def test_scaffold_consumer_lockfile_stub_shows_alias_form(tmp_path):
    scaffold_consumer(tmp_path)

    stub = (tmp_path / "models.lock.yaml").read_text()
    assert "fraud-detector:" in stub
    assert "model: fraud-detector" in stub
