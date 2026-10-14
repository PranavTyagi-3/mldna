import subprocess

from typer.testing import CliRunner

from mldna.cli.main import app
from mldna.core.push import push as core_push
from mldna.tracking import track_run

runner = CliRunner()


def _tracked_pkl(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)
    with track_run(workspace_dir=repo) as run:
        pkl = repo / "models" / "model.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"trained model bytes")
    return pkl, run.run_id


def test_publish_yes_succeeds_noninteractively(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    result = runner.invoke(app, [
        "publish", str(pkl), "--dir", str(producer_repo), "--yes",
    ])

    assert result.exit_code == 0
    assert "Published project-test-model v" in result.stdout


def test_publish_without_yes_declines(producer_repo):
    pkl, run_id = _tracked_pkl(producer_repo)

    result = runner.invoke(app, [
        "publish", str(pkl), "--dir", str(producer_repo),
    ], input="n\n")

    assert result.exit_code == 1
    assert "Aborted" in result.output


def test_publish_missing_pkl_prints_refusal_without_traceback(producer_repo):
    result = runner.invoke(app, [
        "publish", str(producer_repo / "models" / "gone.pkl"),
        "--dir", str(producer_repo), "--yes",
    ])

    assert result.exit_code == 1
    assert "publish failed:" in result.output
    assert "does not exist" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_publish_prints_refusal_without_traceback_when_dataset_moved_on(producer_repo):
    import sys as _sys
    pkl, run_id = _tracked_pkl(producer_repo)

    (producer_repo / "dataset" / "extra.bin").write_bytes(b"new data, never seen by the run")
    subprocess.run([_sys.executable, "-m", "dvc", "add", "dataset", "-q"],
                    cwd=producer_repo, check=True)

    result = runner.invoke(app, [
        "publish", str(pkl), "--dir", str(producer_repo), "--yes",
    ])

    assert result.exit_code == 1
    assert "publish failed:" in result.output
    assert "dataset was changed and re-added" in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def _two_clean_runs_identical_bytes(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    with track_run(workspace_dir=repo):
        pkl = repo / "models" / "model_a.pkl"
        pkl.parent.mkdir(exist_ok=True)
        pkl.write_bytes(b"deterministic model bytes")
    (repo / "README.md").write_text("docs only\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "docs only"], cwd=repo, check=True)
    core_push(repo)
    with track_run(workspace_dir=repo) as run_b:
        pkl = repo / "models" / "model_b.pkl"
        pkl.write_bytes(b"deterministic model bytes")
    commit_b = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    return pkl, run_b.run_id, commit_b


def test_publish_differing_lineage_noninteractive_refuses_naming_commit(producer_repo):
    pkl, _, commit_b = _two_clean_runs_identical_bytes(producer_repo)

    # CliRunner never provides a real tty, so this exercises the refusal path.
    result = runner.invoke(app, ["publish", str(pkl), "--dir", str(producer_repo), "--yes"])

    assert result.exit_code == 1
    assert "--commit <sha>" in result.output
    assert commit_b[:8] in result.output
    assert "Published" not in result.output


def test_publish_commit_option_selects_run(producer_repo):
    pkl, run_b, commit_b = _two_clean_runs_identical_bytes(producer_repo)

    result = runner.invoke(app, [
        "publish", str(pkl), "--dir", str(producer_repo), "--yes", "--commit", commit_b[:8],
    ])

    assert result.exit_code == 0, result.output
    assert "HUMAN-SELECTED" in result.output
    assert f"from run {run_b}" in result.output


def test_publish_interactive_choice_selects_run(producer_repo, monkeypatch):
    import mldna.cli.main as cli_main
    pkl, run_b, _ = _two_clean_runs_identical_bytes(producer_repo)
    monkeypatch.setattr(cli_main, "_stdin_is_interactive", lambda: True)

    # Candidates are listed newest-first, so run B is option 1.
    result = runner.invoke(app, [
        "publish", str(pkl), "--dir", str(producer_repo), "--yes",
    ], input="1\n")

    assert result.exit_code == 0, result.output
    assert "interactive choice" in result.output
    assert f"from run {run_b}" in result.output
