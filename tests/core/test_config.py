import pytest

from mldna.core.config import (
    ProjectConfig,
    find_parent_project,
    missing_namespace_hint,
    missing_project_hint,
)

MINIMAL_YAML = """
project:
  model_name: fraud-detector
  repo_role: producer
"""

NAMESPACED_YAML = """
project:
  model_name: fraud-detector
  namespace: alice
  repo_role: producer
"""

WITH_TRACKING_YAML = """
project:
  model_name: fraud-detector
  repo_role: producer
tracking:
  model_globs: ["*.custom"]
  scan_exclude: ["checkpoints_tmp/"]
"""


WITH_FILESYSTEM_YAML = """
project:
  model_name: fraud-detector
  repo_role: producer
filesystem:
  path: /shared/mlruns
"""


def test_loads_minimal_config(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)
    assert config.model_name == "fraud-detector"
    assert config.repo_role == "producer"
    assert "*.pkl" in config.model_globs


def test_tracking_overrides_extend_defaults(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(WITH_TRACKING_YAML)

    config = ProjectConfig.from_yaml(path)
    assert "*.custom" in config.model_globs
    assert "*.pkl" in config.model_globs
    assert "checkpoints_tmp/" in config.scan_exclude


def test_consumer_config_without_model_name_loads(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text("project:\n  repo_role: consumer\n")

    config = ProjectConfig.from_yaml(path)

    assert config.model_name is None
    assert config.repo_role == "consumer"


def test_key_metric_parsed_from_tracking_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "tracking:\n  key_metric: accuracy\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.key_metric == "accuracy"


def test_scan_limit_parsed_from_tracking_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "tracking:\n  scan_limit: 500\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.scan_limit == 500


def test_scan_limit_defaults_to_200(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.scan_limit == 200


def test_env_allowlist_parsed_from_tracking_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "tracking:\n  env_allowlist: [\"SEED_OVERRIDE\", \"LR_OVERRIDE\"]\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.env_allowlist == ["SEED_OVERRIDE", "LR_OVERRIDE"]


def test_env_allowlist_defaults_to_empty(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.env_allowlist == []


def test_record_argv_defaults_to_false(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    assert ProjectConfig.from_yaml(path).record_argv is False


def test_record_argv_parsed_from_tracking_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "tracking:\n  record_argv: true\n"
    )

    assert ProjectConfig.from_yaml(path).record_argv is True


def test_record_argv_non_bool_refuses(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "tracking:\n  record_argv: \"yes\"\n"
    )

    with pytest.raises(ValueError, match="tracking.record_argv must be true or false"):
        ProjectConfig.from_yaml(path)


def test_key_metric_defaults_to_none(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.key_metric is None


def test_require_formats_parsed_from_publish_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "publish:\n  require_formats: [openvino]\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.require_formats == ["openvino"]


def test_require_formats_defaults_to_empty(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.require_formats == []


def test_load_returns_defaults_when_mldna_yaml_missing(tmp_path):
    config = ProjectConfig.load(tmp_path)

    assert config.model_name is None
    assert config.require_formats == []


def test_load_reads_existing_mldna_yaml(tmp_path):
    (tmp_path / "mldna.yaml").write_text(MINIMAL_YAML)

    config = ProjectConfig.load(tmp_path)

    assert config.model_name == "fraud-detector"


def test_tracking_uri_parsed_from_azure_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "azure:\n  tracking_uri: azureml://region.api.azureml.ms/mlflow/v1.0/x\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.tracking_uri == "azureml://region.api.azureml.ms/mlflow/v1.0/x"
    assert config.subscription_id is None


def test_tracking_uri_defaults_to_none(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.tracking_uri is None


def test_mlflow_tracking_uri_parsed_from_mlflow_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "mlflow:\n  tracking_uri: http://my-mlflow-host:5000\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.mlflow_tracking_uri == "http://my-mlflow-host:5000"
    assert config.tracking_uri is None


def test_mlflow_tracking_uri_defaults_to_none(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.mlflow_tracking_uri is None


def test_filesystem_path_parsed_from_filesystem_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(WITH_FILESYSTEM_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.filesystem_path == "/shared/mlruns"
    assert config.mlflow_tracking_uri is None
    assert config.tracking_uri is None


def test_filesystem_path_defaults_to_none(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.filesystem_path is None


def test_find_parent_project_returns_none_with_no_ancestor_project(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    assert find_parent_project(nested) is None


def test_find_parent_project_finds_ancestor_mldna_yaml(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "mldna.yaml").write_text(MINIMAL_YAML)
    nested = root / "a" / "b"
    nested.mkdir(parents=True)

    assert find_parent_project(nested) == root / "mldna.yaml"


def test_find_parent_project_ignores_start_dir_itself(tmp_path):
    # Callers already know whether start_dir itself has an mldna.yaml - the
    # walk-up must never report start_dir's own file as a "parent" one.
    (tmp_path / "mldna.yaml").write_text(MINIMAL_YAML)

    assert find_parent_project(tmp_path) is None


def test_missing_project_hint_names_parent_and_omits_init(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "mldna.yaml").write_text(MINIMAL_YAML)
    nested = root / "a"
    nested.mkdir()

    hint = missing_project_hint(nested)

    assert str(root / "mldna.yaml") in hint
    assert "mldna init" not in hint


def test_missing_project_hint_suggests_init_with_no_ancestor_project(tmp_path):
    nested = tmp_path / "a"
    nested.mkdir()

    hint = missing_project_hint(nested)

    assert "mldna init" in hint


def test_namespace_parsed_from_project_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(NAMESPACED_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.namespace == "alice"


def test_namespace_defaults_to_none(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.namespace is None


def test_full_model_name_combines_namespace_and_model_name(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(NAMESPACED_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.full_model_name() == "alice-fraud-detector"


def test_full_model_name_is_none_without_a_namespace(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.full_model_name() is None


def test_full_model_name_is_none_without_a_model_name(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text("project:\n  namespace: alice\n  repo_role: consumer\n")

    config = ProjectConfig.from_yaml(path)

    assert config.full_model_name() is None


def test_missing_namespace_hint_names_the_fix(tmp_path):
    hint = missing_namespace_hint(tmp_path)

    assert "project.namespace" in hint
    assert str(tmp_path) in hint


def test_max_file_mb_parsed_from_push_section(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "push:\n  max_file_mb: 20\n"
    )

    assert ProjectConfig.from_yaml(path).max_file_mb == 20


def test_max_file_mb_defaults_to_5(tmp_path):
    path = tmp_path / "mldna.yaml"
    path.write_text(MINIMAL_YAML)

    assert ProjectConfig.from_yaml(path).max_file_mb == 5
