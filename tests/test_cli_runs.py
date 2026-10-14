import subprocess

import mlflow
import yaml
from typer.testing import CliRunner

from mldna.cli.main import KEY_METRIC_WIDTH, _format_key_metric, app
from mldna.core.push import push as core_push
from mldna.tracking import track_run

runner = CliRunner()


def _commit(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)


def _set_key_metric(repo, name):
    config_path = repo / "mldna.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["tracking"] = {"key_metric": name}
    config_path.write_text(yaml.dump(config, sort_keys=False))


def test_runs_cli_no_runs_yet(producer_repo):
    result = runner.invoke(app, ["runs", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "No runs found." in result.output


def test_runs_cli_lists_a_run(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo) as run:
        pass

    result = runner.invoke(app, ["runs", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert run.run_id[:8] in result.output


def test_runs_cli_missing_mldna_yaml_fails_cleanly(tmp_path):
    result = runner.invoke(app, ["runs", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "runs failed" in result.output


def _metric_cell(output):
    header, row = output.splitlines()[0], output.splitlines()[1]
    start = header.index("KEY METRIC")
    return row[start:start + KEY_METRIC_WIDTH]


def test_long_float_metric_fits_the_column():
    assert len(_format_key_metric(0.8734918273645)) < KEY_METRIC_WIDTH


def test_close_but_different_metrics_stay_distinguishable():
    assert _format_key_metric(0.8734918273645) != _format_key_metric(0.8734911111111)
    assert _format_key_metric(1234.56789) != _format_key_metric(1234.56123)
    assert _format_key_metric(1.2345678e-07) != _format_key_metric(1.2345611e-07)


def test_non_float_metric_values_render_without_crashing():
    assert _format_key_metric("-") == "-"
    assert _format_key_metric(3) == "3"
    assert _format_key_metric(None) == "None"


def test_overlong_metric_text_is_visibly_truncated():
    text = _format_key_metric("x" * 40)

    assert len(text) < KEY_METRIC_WIDTH
    assert text.endswith("~")


def test_runs_cli_unset_key_metric_renders_placeholder(producer_repo):
    _commit(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        pass

    result = runner.invoke(app, ["runs", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert _metric_cell(result.output).strip() == "-"


def test_runs_cli_metric_never_logged_renders_placeholder(producer_repo):
    _set_key_metric(producer_repo, "accuracy")
    _commit(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        pass

    result = runner.invoke(app, ["runs", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert _metric_cell(result.output).strip() == "-"


def test_runs_cli_long_float_metric_keeps_the_table_aligned(producer_repo):
    _set_key_metric(producer_repo, "accuracy")
    _commit(producer_repo)
    core_push(producer_repo)
    with track_run(workspace_dir=producer_repo):
        mlflow.log_metric("accuracy", 0.8734918273645)

    result = runner.invoke(app, ["runs", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    header, row = result.output.splitlines()[0], result.output.splitlines()[1]
    assert _metric_cell(result.output).strip() == "0.873492"
    assert row.index("false") == header.index("DIRTY")
