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
