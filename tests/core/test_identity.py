from mlops.core.identity import resolve_identity


def test_resolves_from_git_config(git_repo):
    assert resolve_identity(git_repo) == "dev@example.com"


def test_user_file_takes_priority(git_repo):
    (git_repo / ".mlops").mkdir()
    (git_repo / ".mlops" / "user").write_text("override@example.com\n")

    assert resolve_identity(git_repo) == "override@example.com"


def _isolate_global_git_config(tmp_path, monkeypatch):
    empty_config = tmp_path / "empty_gitconfig"
    empty_config.write_text("")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(empty_config))
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(empty_config))


def test_no_identity_returns_none(tmp_path, monkeypatch):
    import subprocess
    _isolate_global_git_config(tmp_path, monkeypatch)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    assert resolve_identity(tmp_path) is None


def test_extra_resolver_used_as_last_resort(tmp_path, monkeypatch):
    import subprocess
    _isolate_global_git_config(tmp_path, monkeypatch)
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)

    assert resolve_identity(tmp_path, extra_resolver=lambda: "aad@example.com") == "aad@example.com"
