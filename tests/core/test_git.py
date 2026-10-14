import subprocess

import pytest

from mldna.core import git


def test_is_clean_raises_outside_a_repo(tmp_path):
    with pytest.raises(git.GitStatusError):
        git.is_clean(tmp_path)


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


def test_worktree_list_raises_outside_a_repo(tmp_path):
    with pytest.raises(git.GitWorktreeError):
        git.worktree_list(tmp_path)


def test_worktree_add_list_remove(git_repo):
    path = git_repo / "wt"

    git.worktree_add(git_repo, path)
    assert [p.resolve() for p in git.worktree_list(git_repo)] == [git_repo.resolve(), path.resolve()]

    git.worktree_remove(git_repo, path)
    assert [p.resolve() for p in git.worktree_list(git_repo)] == [git_repo.resolve()]


def test_ignored_files_raises_outside_a_repo(tmp_path):
    with pytest.raises(git.GitStatusError):
        git.ignored_files(tmp_path)


def test_ignored_files_lists_only_untracked_ignored(git_repo):
    (git_repo / "models").mkdir()
    (git_repo / "models" / "best.pkl").write_bytes(b"x")
    (git_repo / "notes.txt").write_text("untracked but not ignored")

    assert git.ignored_files(git_repo) == ["models/best.pkl"]


def test_staged_files_raises_outside_a_repo(tmp_path):
    with pytest.raises(git.GitStatusError):
        git.staged_files(tmp_path)


def test_staged_files_lists_added_and_modified_only(git_repo):
    (git_repo / "new.txt").write_text("new\n")
    (git_repo / "README.md").write_text("changed\n")
    subprocess.run(["git", "add", "-A"], cwd=git_repo, check=True)

    assert sorted(git.staged_files(git_repo)) == ["README.md", "new.txt"]


def test_staged_files_is_empty_when_nothing_is_staged(git_repo):
    assert git.staged_files(git_repo) == []


def test_tracked_files_raises_outside_a_repo(tmp_path):
    with pytest.raises(git.GitStatusError):
        git.tracked_files(tmp_path)


def test_tracked_files_lists_committed_paths(git_repo):
    assert sorted(git.tracked_files(git_repo)) == [".gitignore", "README.md", "mldna.yaml"]


# D#77: the origin stamped on a registry model must never carry credentials.
@pytest.mark.parametrize("url, expected", [
    ("https://user:s3cret@github.com/org/repo.git", "https://github.com/org/repo"),
    ("https://x-access-token:ghp_abc@github.com/org/repo/", "https://github.com/org/repo"),
    ("git@github.com:org/repo.git", "github.com:org/repo"),
    ("ssh://git@host:22/org/repo.git/", "ssh://host:22/org/repo"),
    ("https://github.com/org/repo", "https://github.com/org/repo"),
])
def test_normalise_remote_url_strips_credentials_and_suffix(url, expected):
    assert git.normalise_remote_url(url) == expected


def test_origin_url_is_none_without_origin(git_repo):
    assert git.origin_url(git_repo) is None


def test_origin_url_is_normalised(git_repo):
    subprocess.run(["git", "remote", "add", "origin", "https://u:p@example.com/org/repo.git"],
                   cwd=git_repo, check=True)
    assert git.origin_url(git_repo) == "https://example.com/org/repo"


# D#77: one repository, many spellings - compare canonically, never literally.
@pytest.mark.parametrize("a, b", [
    ("git@github.com:acme/churn.git", "https://github.com/acme/churn"),
    ("ssh://git@github.com:22/acme/churn", "https://GitHub.com/acme/churn/"),
    ("https://u:secret@github.com/acme/churn.git", "https://github.com/acme/churn"),
    ("../origin.git", "{base}/origin.git"),
    ("file:///{base}/origin.git", "{base}/origin"),
])
def test_same_remote_ignores_spelling(tmp_path, a, b):
    base = tmp_path.as_posix()
    repo = tmp_path / "project"
    assert git.same_remote(a.format(base=base), b.format(base=base), repo)


@pytest.mark.parametrize("a, b", [
    ("https://github.com/acme/churn", "https://github.com/acme/other"),
    ("https://github.com/acme/churn", "https://gitlab.com/acme/churn"),
    ("{base}/one/origin.git", "{base}/two/origin.git"),
])
def test_same_remote_still_tells_different_repositories_apart(tmp_path, a, b):
    base = tmp_path.as_posix()
    assert not git.same_remote(a.format(base=base), b.format(base=base), tmp_path)


def test_origin_url_makes_a_relative_local_path_absolute(git_repo):
    subprocess.run(["git", "remote", "add", "origin", "../origin.git"], cwd=git_repo, check=True)
    assert git.origin_url(git_repo) == str((git_repo.parent / "origin").resolve())
