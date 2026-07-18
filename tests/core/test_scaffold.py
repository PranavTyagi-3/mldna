import yaml

from mlops.core.scaffold import scaffold_producer


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
