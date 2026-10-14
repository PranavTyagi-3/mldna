import subprocess
from types import SimpleNamespace

from typer.testing import CliRunner

from mldna.cli.main import app
from mldna.core.push import push as core_push

runner = CliRunner()


def _commit_and_push(repo):
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "scaffold"], cwd=repo, check=True)
    core_push(repo)


def test_pull_cli_reports_no_published_model(producer_repo):
    _commit_and_push(producer_repo)

    result = runner.invoke(app, ["pull", "--dir", str(producer_repo)])

    assert result.exit_code == 0
    assert "Pulled code and data." in result.output
    assert "No published model found" in result.output


def test_pull_cli_reports_published_model(published):
    repo, _, published_result = published

    result = runner.invoke(app, ["pull", "--dir", str(repo)])

    assert result.exit_code == 0
    assert f"Pulled published model bundle from run {published_result['run_id']}" in result.output
    assert "into models" in result.output and "test-model-v" in result.output


def test_pull_cli_refuses_ambiguous_commit_when_noninteractive(monkeypatch, tmp_path):
    # The two-published-runs scenario itself is covered against the real core
    # (tests/test_pull.py); here the core reports two candidates and only the
    # CLI's refusal is under test.
    _fake_pull(monkeypatch, IDS[:2])
    main_py = tmp_path / "main.py"
    main_py.write_text("print('mine')")

    # CliRunner never provides a real tty, so omitting --run-id exercises the
    # non-interactive refusal rather than typer.prompt.
    result = runner.invoke(app, ["pull", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "--run-id" in result.output
    assert "mldna runs" not in result.output
    assert all(rid[:8] in result.output for rid in IDS[:2])
    assert "  abcd1234  v1  published ?" in result.output
    assert main_py.read_text() == "print('mine')"


def test_pull_cli_fails_in_consumer_repo(tmp_path):
    result = runner.invoke(app, ["init", "--dir", str(tmp_path), "--consumer"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["pull", "--dir", str(tmp_path)])

    assert result.exit_code == 1
    assert "mldna fetch" in result.output


def _fake_pull(monkeypatch, run_ids):
    candidates = [
        SimpleNamespace(info=SimpleNamespace(run_id=rid),
                        data=SimpleNamespace(tags={"registry_version": str(i + 1)}))
        for i, rid in enumerate(run_ids)
    ]

    def fake_core_pull(dir, select):
        chosen = select(candidates)
        return {"model_pulled": True, "run_id": chosen.info.run_id,
                "registry_version": 1, "dest": "models"}

    monkeypatch.setattr("mldna.cli.main.core_pull", fake_core_pull)


IDS = ["abcd1234" + "0" * 24, "abcd9999" + "1" * 24, "ffff0000" + "2" * 24]


def test_pull_run_id_shorter_than_four_characters_refuses(monkeypatch, tmp_path):
    _fake_pull(monkeypatch, IDS)

    result = runner.invoke(app, ["pull", "--dir", str(tmp_path), "--run-id", "abc"])

    assert result.exit_code == 1
    assert "too short" in result.output and "mldna runs" in result.output


def test_pull_run_id_matching_nothing_lists_candidates(monkeypatch, tmp_path):
    _fake_pull(monkeypatch, IDS)

    result = runner.invoke(app, ["pull", "--dir", str(tmp_path), "--run-id", "zzzz"])

    assert result.exit_code == 1
    assert "is not among" in result.output
    assert all(rid[:8] in result.output for rid in IDS)


def test_pull_run_id_ambiguous_prefix_lists_matches(monkeypatch, tmp_path):
    _fake_pull(monkeypatch, IDS)

    result = runner.invoke(app, ["pull", "--dir", str(tmp_path), "--run-id", "abcd"])

    assert result.exit_code == 1
    assert "matches several" in result.output
    assert IDS[0][:8] in result.output and IDS[1][:8] in result.output
    assert IDS[2][:8] not in result.output


def test_pull_run_id_accepts_eight_character_prefix(monkeypatch, tmp_path):
    _fake_pull(monkeypatch, IDS)

    result = runner.invoke(app, ["pull", "--dir", str(tmp_path), "--run-id", IDS[1][:8]])

    assert result.exit_code == 0
    assert f"from run {IDS[1]}" in result.output


def test_pull_run_id_accepts_full_id(monkeypatch, tmp_path):
    _fake_pull(monkeypatch, IDS)

    result = runner.invoke(app, ["pull", "--dir", str(tmp_path), "--run-id", IDS[0]])

    assert result.exit_code == 0
    assert f"from run {IDS[0]}" in result.output
