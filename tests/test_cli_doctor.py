import subprocess

from typer.testing import CliRunner

from mldna.cli.main import app

runner = CliRunner()


def test_doctor_reports_all_checks(producer_repo):
    subprocess.run(["git", "add", "-A"], cwd=producer_repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=producer_repo, check=True)

    result = runner.invoke(app, ["doctor", "--dir", str(producer_repo)])

    assert "tools:" in result.output
    assert "gitignore:" in result.output
    assert "run_lock:" in result.output
    assert "git_remote:" in result.output
    # producer_repo configures no tracking section, so the tracking check is
    # named for the local store - "aml_auth:" here asserted a label for a plane
    # the project never uses (D#42).
    assert "tracking_store:" in result.output
    assert "blob_auth:" in result.output


def test_doctor_exit_code_reflects_failures(producer_repo):
    (producer_repo / ".gitignore").write_text("models/\n")

    result = runner.invoke(app, ["doctor", "--dir", str(producer_repo)])

    assert result.exit_code == 1
    assert "[FAIL] gitignore" in result.output


def test_doctor_refuses_a_malformed_timeout(producer_repo):
    result = runner.invoke(app, ["doctor", "--dir", str(producer_repo), "--timeout", "abc"])

    assert result.exit_code == 2
    assert "--timeout must be a whole number of seconds" in result.output
    assert "Fix: mldna doctor --timeout 60s" in result.output
    assert "tools:" not in result.output


def test_doctor_passes_the_parsed_timeout_through(producer_repo, monkeypatch):
    seen = []
    monkeypatch.setattr("mldna.cli.main.run_doctor", lambda d, t: seen.append(t) or [])

    runner.invoke(app, ["doctor", "--dir", str(producer_repo), "--timeout", "100s"])
    runner.invoke(app, ["doctor", "--dir", str(producer_repo)])

    assert seen == ["100", None]
