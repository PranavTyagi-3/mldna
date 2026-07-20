from mlops.core.config import ProjectConfig

MINIMAL_YAML = """
project:
  model_name: fraud-detector
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


def test_loads_minimal_config(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)
    assert config.model_name == "fraud-detector"
    assert config.repo_role == "producer"
    assert "*.pkl" in config.model_globs


def test_tracking_overrides_extend_defaults(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(WITH_TRACKING_YAML)

    config = ProjectConfig.from_yaml(path)
    assert "*.custom" in config.model_globs
    assert "*.pkl" in config.model_globs
    assert "checkpoints_tmp/" in config.scan_exclude


def test_consumer_config_without_model_name_loads(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text("project:\n  repo_role: consumer\n")

    config = ProjectConfig.from_yaml(path)

    assert config.model_name is None
    assert config.repo_role == "consumer"


def test_key_metric_parsed_from_tracking_section(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "tracking:\n  key_metric: accuracy\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.key_metric == "accuracy"


def test_key_metric_defaults_to_none(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.key_metric is None


def test_require_formats_parsed_from_publish_section(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "publish:\n  require_formats: [openvino]\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.require_formats == ["openvino"]


def test_require_formats_defaults_to_empty(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.require_formats == []


def test_load_returns_defaults_when_mlops_yaml_missing(tmp_path):
    config = ProjectConfig.load(tmp_path)

    assert config.model_name is None
    assert config.require_formats == []


def test_load_reads_existing_mlops_yaml(tmp_path):
    (tmp_path / "mlops.yaml").write_text(MINIMAL_YAML)

    config = ProjectConfig.load(tmp_path)

    assert config.model_name == "fraud-detector"


def test_tracking_uri_parsed_from_azure_section(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(
        "project:\n  model_name: fraud-detector\n  repo_role: producer\n"
        "azure:\n  tracking_uri: azureml://region.api.azureml.ms/mlflow/v1.0/x\n"
    )

    config = ProjectConfig.from_yaml(path)

    assert config.tracking_uri == "azureml://region.api.azureml.ms/mlflow/v1.0/x"
    assert config.subscription_id is None


def test_tracking_uri_defaults_to_none(tmp_path):
    path = tmp_path / "mlops.yaml"
    path.write_text(MINIMAL_YAML)

    config = ProjectConfig.from_yaml(path)

    assert config.tracking_uri is None
