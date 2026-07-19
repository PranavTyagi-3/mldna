import yaml
from typer.testing import CliRunner

from mlops.cli.main import app

runner = CliRunner()


def test_init_scaffolds_project(tmp_path):
    result = runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path)])

    assert result.exit_code == 0
    assert (tmp_path / "mlops.yaml").exists()
    assert "Initialized producer project 'fraud-detector'" in result.stdout


def test_init_without_azure_flag_has_no_azure_section(tmp_path):
    runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path)])

    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert "azure" not in config


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


def test_init_azure_flag_prompts_for_missing_values(tmp_path):
    result = runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path), "--azure"], input=(
        "sub\nrg\nws\nacct\ncont\nprefix\n"
    ))

    assert result.exit_code == 0
    config = yaml.safe_load((tmp_path / "mlops.yaml").read_text())
    assert config["azure"]["subscription_id"] == "sub"
    assert config["dvc"]["account_name"] == "acct"


def test_init_azure_flag_does_not_reprompt_when_already_configured(tmp_path):
    runner.invoke(app, [
        "init", "fraud-detector", "--dir", str(tmp_path), "--azure",
        "--subscription-id", "sub", "--resource-group", "rg", "--workspace", "ws",
        "--dvc-account", "acct", "--dvc-container", "cont", "--dvc-prefix", "prefix",
    ])

    result = runner.invoke(app, ["init", "fraud-detector", "--dir", str(tmp_path), "--azure"])

    assert result.exit_code == 0
    assert "already has an azure: section" in result.output
