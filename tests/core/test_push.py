import subprocess
import sys

import pytest

from mldna.core import dvc
from mldna.core.push import PushError, push, stage


def test_push_commits_and_pushes(producer_repo):
    stage(producer_repo)
    result = push(producer_repo, message="first push")

    assert result["committed"] is True

    local_head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True, text=True
    ).stdout.strip()
    remote_head = subprocess.run(
        ["git", "rev-parse", "origin/master"], cwd=producer_repo, capture_output=True, text=True
    )
    if remote_head.returncode != 0:
        remote_head = subprocess.run(
            ["git", "rev-parse", "origin/main"], cwd=producer_repo, capture_output=True, text=True
        )
    assert remote_head.stdout.strip() == local_head


def test_push_with_nothing_staged_and_nothing_unpushed_refuses_naming_stage(producer_repo):
    stage(producer_repo)
    push(producer_repo, message="first push")

    stage(producer_repo)
    with pytest.raises(PushError) as e:
        push(producer_repo, message="second push")

    text = str(e.value)
    assert "Nothing is staged" in text
    assert "Fix: run `mldna stage`" in text


def test_push_again_with_nothing_new_staged_refuses(producer_repo):
    # the scaffold's `dvc add` already staged files, so push once first
    push(producer_repo, message="first")

    with pytest.raises(PushError, match="mldna stage"):
        push(producer_repo, message="oops")


def test_push_does_not_stage_anything_itself(producer_repo):
    stage(producer_repo)
    (producer_repo / "late.txt").write_text("written after stage\n")

    push(producer_repo, message="staged only")

    assert "late.txt" not in _git(producer_repo, "ls-files")


def test_stage_lists_the_staged_files_without_committing(producer_repo):
    (producer_repo / "notes.txt").write_text("hello\n")

    lines = stage(producer_repo).lines

    assert "A\tnotes.txt" in lines
    assert "notes.txt" in _git(producer_repo, "diff", "--cached", "--name-only")
    assert subprocess.run(["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True).returncode != 0


def test_stage_is_idempotent(producer_repo):
    (producer_repo / "notes.txt").write_text("hello\n")

    first = stage(producer_repo)
    second = stage(producer_repo)

    assert first == second


def test_stage_honors_gitignore(producer_repo):
    (producer_repo / "secret.env.txt").write_text("x\n")
    with (producer_repo / ".gitignore").open("a") as f:
        f.write("secret.env.txt\n")

    lines = stage(producer_repo).lines

    assert not any("secret.env.txt" in line for line in lines)


def test_stage_unstages_a_newly_ignored_staged_file_and_reports_it(producer_repo):
    (producer_repo / "config.py").write_text("KEY = 'x'\n")
    stage(producer_repo)
    assert "config.py" in _git(producer_repo, "diff", "--cached", "--name-only")
    with (producer_repo / ".gitignore").open("a") as f:
        f.write("config.py\n")

    result = stage(producer_repo)

    assert result.unstaged == ["config.py"]
    assert "config.py" not in _git(producer_repo, "diff", "--cached", "--name-only")
    assert (producer_repo / "config.py").exists()
    assert stage(producer_repo).unstaged == []


def test_stage_leaves_a_committed_then_ignored_file_alone_and_reports_history(producer_repo):
    (producer_repo / "config.py").write_text("KEY = 'x'\n")
    stage(producer_repo)
    push(producer_repo, message="oops")
    with (producer_repo / ".gitignore").open("a") as f:
        f.write("config.py\n")

    result = stage(producer_repo)

    assert result.unstaged == []
    assert result.in_history == ["config.py"]
    assert "config.py" in _git(producer_repo, "ls-files")
    assert "D\tconfig.py" not in result.lines


def test_push_with_only_unpushed_commits_still_pushes(producer_repo):
    stage(producer_repo)
    subprocess.run(["git", "commit", "-q", "-m", "local only"], cwd=producer_repo, check=True)

    result = push(producer_repo)

    assert result["committed"] is False
    assert _git(producer_repo, "rev-parse", "HEAD") == _git(producer_repo, "rev-parse", f"origin/{_git(producer_repo, 'branch', '--show-current')}")


def test_push_fails_clearly_without_git_remote(tmp_path):
    from mldna.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    dvc_remote = tmp_path.with_name(tmp_path.name + "-dvc-remote")
    subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "add", "-d", "local-remote", str(dvc_remote)],
        cwd=tmp_path, check=True,
    )

    stage(tmp_path)
    with pytest.raises(PushError, match="git push failed"):
        push(tmp_path)


def test_push_skips_dvc_when_no_remote_is_configured(tmp_path, capsys):
    from mldna.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)
    origin = tmp_path.with_name(tmp_path.name + "-origin.git")
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True)

    stage(tmp_path)
    result = push(tmp_path)

    assert result["committed"] is True
    captured = capsys.readouterr()
    assert "Skipped dvc push: no DVC remote configured" in captured.out


def test_push_git_rejection_advises_pull(tmp_path):
    from mldna.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)

    # Create origin bare repo
    origin = tmp_path.with_name(tmp_path.name + "-origin.git")
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)

    # Set up producer and push initial commit
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "initial"], cwd=tmp_path, check=True)
    subprocess.run(["git", "push", "-u", "origin", "master"], cwd=tmp_path, check=True)

    # Create an upstream clone and push a new commit to simulate remote being ahead
    upstream = tmp_path.with_name(tmp_path.name + "-upstream")
    subprocess.run(["git", "clone", str(origin), str(upstream)], check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "other@example.com"], cwd=upstream, check=True)
    subprocess.run(["git", "config", "user.name", "other"], cwd=upstream, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-m", "remote commit"], cwd=upstream, check=True)
    subprocess.run(["git", "push"], cwd=upstream, check=True)

    dvc_remote = tmp_path.with_name(tmp_path.name + "-dvc-remote")
    subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "add", "-d", "local-remote", str(dvc_remote)],
        cwd=tmp_path, check=True,
    )

    stage(tmp_path)
    with pytest.raises(PushError) as exc_info:
        push(tmp_path, message="attempt push")

    error_msg = str(exc_info.value)
    assert "mldna pull" in error_msg
    assert "git remote add" not in error_msg


def _config_local_content(workspace_dir):
    config_local = workspace_dir / ".dvc" / "config.local"
    return config_local.read_text() if config_local.exists() else ""


def test_push_skips_blob_override_for_local_remote(producer_repo, monkeypatch, capsys):
    monkeypatch.setenv("MLDNA_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;test=1")

    stage(producer_repo)
    push(producer_repo)

    config_content = _config_local_content(producer_repo)
    assert "test=1" not in config_content
    assert "connection_string" not in config_content


def test_push_applies_blob_override_for_azure_remote(tmp_path, monkeypatch):
    from mldna.core.scaffold import scaffold_producer

    workspace_dir = tmp_path / "workspace"
    scaffold_producer(workspace_dir, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=workspace_dir, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=workspace_dir, check=True)

    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=workspace_dir, check=True)

    dvc.run(["remote", "remove", "local-remote"], cwd=workspace_dir, check=False)
    dvc.run(["remote", "add", "-d", "azure-remote", "azure://container/prefix"], cwd=workspace_dir)

    monkeypatch.setenv("MLDNA_BLOB_CONNECTION_STRING", "DefaultEndpointsProtocol=https;test=1")

    # dvc push/pull to a real azure:// remote cannot succeed here - there is no
    # Azure. The connection string is written BEFORE that attempt, which is what
    # this test is about, so expect the transfer failure and assert the write.
    stage(workspace_dir)
    with pytest.raises(PushError):
        push(workspace_dir)

    config_content = _config_local_content(workspace_dir)
    assert "test=1" in config_content


def _write_big(repo, name="big.csv", mb=6):
    (repo / name).write_bytes(b"x" * (mb * 1024 * 1024))


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()


def test_push_refuses_a_large_file_without_confirmation_and_does_not_commit(producer_repo):
    _write_big(producer_repo)
    (producer_repo / "notes.txt").write_text("keep staged\n")
    subprocess.run(["git", "add", "notes.txt"], cwd=producer_repo, check=True)
    before = subprocess.run(["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True)

    stage(producer_repo)
    index_before = _git(producer_repo, "diff", "--cached", "--name-status")
    with pytest.raises(PushError, match="mldna push --yes"):
        push(producer_repo, message="oops", confirm=lambda msg: False)

    after = subprocess.run(["git", "rev-parse", "HEAD"], cwd=producer_repo, capture_output=True)
    assert before.stdout == after.stdout
    assert _git(producer_repo, "diff", "--cached", "--name-status") == index_before


def test_push_confirm_message_lists_files_and_states_the_permanence_and_the_fixes(producer_repo):
    _write_big(producer_repo)
    seen = []

    stage(producer_repo)
    push(producer_repo, confirm=lambda msg: seen.append(msg) or True)

    [text] = seen
    assert "Files to be committed" in text and "A\tbig.csv" in text
    assert "big.csv (larger than 5 MB, 6.0 MB)" in text
    assert "rewriting history" in text
    assert "dvc add" in text and ".gitignore" in text
    assert text.rsplit("\n", 1)[1].startswith("Commit these ")


def test_push_asks_every_time_even_with_nothing_risky(producer_repo):
    (producer_repo / "notes.txt").write_text("hello\n")
    seen = []

    stage(producer_repo)
    push(producer_repo, confirm=lambda msg: seen.append(msg) or True)

    [text] = seen
    assert "A\tnotes.txt" in text
    assert "large or model-shaped" not in text


def test_push_proceeds_when_confirmed(producer_repo):
    _write_big(producer_repo)
    seen = []

    def confirm(msg):
        seen.append(msg)
        return True

    stage(producer_repo)
    result = push(producer_repo, message="big on purpose", confirm=confirm)

    assert result["committed"] is True
    assert len(seen) == 1
    assert "big.csv" in _git(producer_repo, "ls-files")


def test_push_flags_a_model_shaped_file_but_not_a_small_xml(producer_repo):
    (producer_repo / "pom.xml").write_text("<project/>\n")
    (producer_repo / "net.pkl").write_bytes(b"x" * 200 * 1024)

    seen = []
    stage(producer_repo)
    push(producer_repo, confirm=lambda msg: seen.append(msg) or True)

    risky_block = seen[0].split("large or model-shaped")[1]
    assert "net.pkl" in risky_block
    assert "pom.xml" not in risky_block


def test_push_with_only_unpushed_commits_never_calls_confirm(producer_repo):
    stage(producer_repo)
    subprocess.run(["git", "commit", "-q", "-m", "local only"], cwd=producer_repo, check=True)

    def confirm(msg):
        raise AssertionError("confirm must not be called")

    assert push(producer_repo, confirm=confirm)["committed"] is False


def _commit_dataset_with(repo, files):
    for name in files:
        (repo / "dataset" / name).write_text("row\n" * 100)
    stage(repo)
    push(repo, message="data")


def test_push_shrunk_dataset_pointer_shows_in_the_confirmation(producer_repo):
    _commit_dataset_with(producer_repo, ["a.csv", "b.csv"])
    for name in ("a.csv", "b.csv"):
        (producer_repo / "dataset" / name).unlink()
    seen = []

    stage(producer_repo)
    push(producer_repo, message="empty", confirm=lambda msg: seen.append(msg) or True)

    assert "Dataset shrink" in seen[0]
    assert "dataset.dvc" in seen[0] and "-> 0" in seen[0]


def test_push_shrink_declined_commits_nothing(producer_repo):
    _commit_dataset_with(producer_repo, ["a.csv", "b.csv"])
    head = _git(producer_repo, "rev-parse", "HEAD")
    for name in ("a.csv", "b.csv"):
        (producer_repo / "dataset" / name).unlink()

    stage(producer_repo)
    with pytest.raises(PushError, match="Not committed"):
        push(producer_repo, message="empty", confirm=lambda msg: False)

    assert _git(producer_repo, "rev-parse", "HEAD") == head


def _git_ready_project(tmp_path):
    from mldna.core.scaffold import scaffold_producer
    scaffold_producer(tmp_path, "test-model")
    subprocess.run(["git", "config", "user.email", "dev@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "dev"], cwd=tmp_path, check=True)


def test_push_hook_rejection_shows_git_message_and_does_not_say_behind(tmp_path):
    _git_ready_project(tmp_path)
    origin = tmp_path.with_name(tmp_path.name + "-origin.git")
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True)
    hook = origin / "hooks" / "pre-receive"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    subprocess.run(["git", "remote", "add", "origin", str(origin)], cwd=tmp_path, check=True)

    stage(tmp_path)
    with pytest.raises(PushError) as exc_info:
        push(tmp_path)

    error_msg = str(exc_info.value)
    assert error_msg.startswith("git push failed:")
    assert "pre-receive hook declined" in error_msg
    assert "behind" not in error_msg
    assert "mldna pull" not in error_msg
    assert "git remote add" not in error_msg


def test_push_with_no_remote_hints_git_remote_add(tmp_path):
    _git_ready_project(tmp_path)

    stage(tmp_path)
    with pytest.raises(PushError) as exc_info:
        push(tmp_path)

    error_msg = str(exc_info.value)
    assert "does not appear to be a git repository" in error_msg
    assert "git remote add origin <url>" in error_msg
    assert "mldna pull" not in error_msg


def test_push_unrelated_failure_gets_no_guessed_hint(tmp_path):
    _git_ready_project(tmp_path)
    missing = tmp_path.with_name(tmp_path.name + "-missing-dir")
    subprocess.run(["git", "remote", "add", "origin", str(missing)], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "remote.origin.pushurl", "ssh://127.0.0.1:1/x.git"],
                   cwd=tmp_path, check=True)

    stage(tmp_path)
    with pytest.raises(PushError) as exc_info:
        push(tmp_path)

    error_msg = str(exc_info.value)
    assert error_msg.startswith("git push failed:")
    assert "mldna pull" not in error_msg
    assert "git remote add" not in error_msg


def test_run_dvc_with_empty_stderr_never_ends_in_a_bare_colon(tmp_path, monkeypatch):
    from mldna.core import push as push_module

    class Result:
        returncode = 3
        stderr = ""
        stdout = ""

    monkeypatch.setattr(push_module.dvc, "run", lambda *a, **k: Result())
    with pytest.raises(PushError) as exc_info:
        push_module._run_dvc(["add", "dataset"], tmp_path, "dvc add failed")
    assert str(exc_info.value) == "dvc add failed: exit code 3"

    Result.stdout = "from stdout\n"
    with pytest.raises(PushError) as exc_info:
        push_module._run_dvc(["add", "dataset"], tmp_path, "dvc add failed")
    assert str(exc_info.value) == "dvc add failed: from stdout"
