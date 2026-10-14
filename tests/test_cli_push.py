import subprocess
import sys

from typer.testing import CliRunner

from mldna.cli.main import app

runner = CliRunner()


def test_push_commits_and_pushes(producer_repo):
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])
    result = runner.invoke(app, ["push", "--message", "cli push", "--yes", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "Committed: cli push" in result.stdout
    assert "Pushed." in result.stdout


def test_push_cli_with_nothing_staged_refuses_naming_stage(producer_repo):
    # the scaffold's `dvc add` already staged files, so push once first
    runner.invoke(app, ["push", "-m", "first", "--yes", "--dir", str(producer_repo)])
    result = runner.invoke(app, ["push", "-m", "x", "--dir", str(producer_repo)])

    assert result.exit_code == 1
    assert "Nothing is staged" in result.output
    assert "mldna stage" in result.output
    assert "Committed" not in result.output


def test_stage_cli_lists_files_and_points_to_push(producer_repo):
    (producer_repo / "notes.txt").write_text("hello\n")

    result = runner.invoke(app, ["stage", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "A\tnotes.txt" in result.stdout
    assert ".gitignore" in result.stdout
    assert 'mldna push -m' in result.stdout
    assert "Committed" not in result.stdout


def test_stage_cli_is_idempotent(producer_repo):
    first = runner.invoke(app, ["stage", "--dir", str(producer_repo)])
    second = runner.invoke(app, ["stage", "--dir", str(producer_repo)])

    assert second.exit_code == 0
    assert first.stdout == second.stdout


def test_push_cli_refuses_without_message_when_noninteractive(producer_repo):
    # CliRunner never provides a real tty, so omitting --message exercises
    # the non-interactive refusal rather than typer.prompt.
    before = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True, text=True
    ).stdout

    result = runner.invoke(app, ["push", "--dir", str(producer_repo)])

    assert result.exit_code == 1
    assert "-m" in result.output
    assert "Committed" not in result.output

    after = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True, text=True
    ).stdout
    assert before == after


def test_push_cli_skips_dvc_when_no_remote(tmp_path):
    import subprocess
    import sys
    from mldna.core.scaffold import scaffold_producer

    scaffold_producer(tmp_path, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    origin = tmp_path.parent / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True)

    runner.invoke(app, ["stage", "--dir", str(tmp_path)])
    result = runner.invoke(app, ["push", "--message", "test push", "--yes", "--dir", str(tmp_path)])

    # Even if git push fails (no commits to push initially), the skip message should appear
    assert "Skipped dvc push: no DVC remote configured" in result.stdout


def _write_big(repo):
    (repo / "big.csv").write_bytes(b"x" * (6 * 1024 * 1024))


def test_push_cli_refuses_noninteractive_without_yes_naming_yes(producer_repo):
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True)

    result = runner.invoke(app, ["push", "-m", "x", "--dir", str(producer_repo)])

    assert result.exit_code == 1
    assert "Fix: pass --yes" in result.output
    assert "Committed" not in result.output
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True).stdout == head.stdout


def test_push_cli_yes_prints_the_file_list_before_committing(producer_repo):
    (producer_repo / "notes.txt").write_text("hello\n")
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])

    result = runner.invoke(app, ["push", "-m", "x", "--yes", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "Files to be committed" in result.stdout
    assert "A\tnotes.txt" in result.stdout
    assert result.stdout.index("Files to be committed") < result.stdout.index("Committed: x")


def test_push_cli_large_file_is_listed_with_its_reason(producer_repo):
    _write_big(producer_repo)
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])

    result = runner.invoke(app, ["push", "-m", "big", "--yes", "--dir", str(producer_repo)])

    assert "big.csv (larger than 5 MB" in result.stdout


def test_push_cli_yes_commits_a_large_file(producer_repo):
    _write_big(producer_repo)
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])

    result = runner.invoke(app, ["push", "-m", "big", "--yes", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "Committed: big" in result.stdout


def test_push_cli_interactive_decline_refuses(producer_repo, monkeypatch):
    _write_big(producer_repo)
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])
    monkeypatch.setattr("mldna.cli.main._stdin_is_interactive", lambda: True)

    result = runner.invoke(app, ["push", "-m", "big", "--dir", str(producer_repo)], input="n\n")

    assert result.exit_code == 1
    assert "Commit these" in result.output and "[y/N]" in result.output
    assert "Committed" not in result.output
    assert "big.csv" in subprocess.run(
        ["git", "diff", "--cached", "--name-only"], cwd=producer_repo, capture_output=True, text=True
    ).stdout


def test_push_cli_interactive_accept_commits(producer_repo, monkeypatch):
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])
    monkeypatch.setattr("mldna.cli.main._stdin_is_interactive", lambda: True)

    result = runner.invoke(app, ["push", "-m", "ok", "--dir", str(producer_repo)], input="y\n")

    assert result.exit_code == 0
    assert "Committed: ok" in result.output


def test_stage_cli_prints_unstaged_and_history_notices_before_the_staged_list(producer_repo):
    (producer_repo / "config.py").write_text("KEY = 'x'\n")
    (producer_repo / "old.py").write_text("KEY = 'y'\n")
    runner.invoke(app, ["stage", "--dir", str(producer_repo)])
    subprocess.run(["git", "commit", "-q", "-m", "old", "--", "old.py"], cwd=producer_repo, check=True)
    with (producer_repo / ".gitignore").open("a") as f:
        f.write("config.py\nold.py\n")

    result = runner.invoke(app, ["stage", "--dir", str(producer_repo)])

    assert "Unstaged (now ignored): config.py" in result.stdout
    assert "old.py is ignored but already in git history" in result.stdout
    assert result.stdout.index("Unstaged (now ignored)") < result.stdout.index("Staged files")


def test_stdin_is_not_interactive_when_stdin_is_devnull():
    # On Windows isatty() is True for NUL; the console check must override it.
    code = "from mldna.cli.main import _stdin_is_interactive; print(_stdin_is_interactive())"
    result = subprocess.run(
        [sys.executable, "-c", code], stdin=subprocess.DEVNULL, capture_output=True, text=True
    )

    assert result.stdout.strip() == "False"


def test_push_cli_with_nothing_staged_refuses_before_asking_for_a_message(producer_repo):
    runner.invoke(app, ["push", "-m", "first", "--yes", "--dir", str(producer_repo)])

    result = runner.invoke(app, ["push", "--dir", str(producer_repo)], input="should not be read\n")

    assert result.exit_code == 1
    assert "Nothing is staged" in result.output
    assert "Commit message" not in result.output
