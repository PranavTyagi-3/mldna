from mlops.core import git


def test_is_clean_on_fresh_commit(git_repo):
    assert git.is_clean(git_repo) is True


def test_is_clean_false_after_edit(git_repo):
    (git_repo / "README.md").write_text("changed\n")
    assert git.is_clean(git_repo) is False


def test_get_commit_sha_is_full_hash(git_repo):
    sha = git.get_commit_sha(git_repo)
    assert len(sha) == 40


def test_get_branch(git_repo):
    branch = git.get_branch(git_repo)
    assert branch in ("main", "master")
