import shutil
import subprocess

import mlflow
import mlflow.tracking.fluent as mlflow_fluent
import pytest


@pytest.fixture(autouse=True)
def _isolated_mlflow_tracking_uri():
    """Reset mlflow's process-global tracking state after every test.

    mlflow.set_tracking_uri() alone is not enough: mlflow also caches the
    active experiment id in a plain module global (mlflow.tracking.fluent.
    _active_experiment_id, not thread-local, not tied to the tracking URI)
    and keeps any unclosed run on a stack. Either can survive a URI change
    and get looked up against the next test's store, producing a
    "Run/Experiment not found" error that depends on test order.

    Resetting to a fixed, known-clean state (rather than restoring whatever
    URI was active before the test) means every test starts from the same
    blank slate regardless of what ran before it or in what order -- a
    leak becomes impossible, not just unlikely.
    """
    yield
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)
    mlflow_fluent._active_experiment_id = None


def run(repo_dir, *args):
    subprocess.run(["git", *args], cwd=repo_dir, check=True, capture_output=True)


@pytest.fixture
def git_repo(tmp_path):
    run(tmp_path, "init", "-q")
    run(tmp_path, "config", "user.email", "dev@example.com")
    run(tmp_path, "config", "user.name", "dev")
    (tmp_path / ".gitignore").write_text("models/\noutputs/\n.mldna/\nmlruns/\n")
    (tmp_path / "README.md").write_text("hello\n")
    (tmp_path / "mldna.yaml").write_text(
        "project:\n  model_name: test-model\n  namespace: test-ns\n  repo_role: producer\n"
        "  model_id: 11111111-2222-4333-8444-555555555555\n"
    )
    run(tmp_path, "add", ".gitignore", "README.md", "mldna.yaml")
    run(tmp_path, "commit", "-q", "-m", "initial commit")
    return tmp_path


@pytest.fixture(scope="session", autouse=True)
def _no_pip_freeze():
    """track_run snapshots the environment with a `pip freeze` subprocess;
    nothing under test depends on its content."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("mldna.tracking._pip_freeze", lambda: "pkg==1.0\n")
        yield


@pytest.fixture(scope="session")
def producer_template(tmp_path_factory):
    """Scaffold a producer once per session (dvc init/add are slow). The dir is
    named `project` because the namespace defaults to the directory name."""
    from mldna.core.scaffold import scaffold_producer

    template = tmp_path_factory.mktemp("producer_template") / "project"
    scaffold_producer(template, "test-model")
    run(template, "config", "user.email", "dev@example.com")
    run(template, "config", "user.name", "dev")
    return template


def new_producer(base, template):
    """Copy the template into base/project with a per-test git origin and DVC
    remote. The DVC remote is written straight into .dvc/config, which is what
    `dvc remote add -d` does, without paying for a dvc subprocess."""
    origin = base / "origin.git"
    subprocess.run(["git", "init", "--bare", "-q", str(origin)], check=True, capture_output=True)

    project_dir = base / "project"
    shutil.copytree(template, project_dir, symlinks=True)
    run(project_dir, "remote", "add", "origin", str(origin))

    dvc_remote = (base / "dvc-remote").as_posix()
    (project_dir / ".dvc" / "config").write_text(
        f"[core]\n    remote = local-remote\n['remote \"local-remote\"']\n    url = {dvc_remote}\n"
    )
    return project_dir


@pytest.fixture
def producer_repo(tmp_path, producer_template):
    return new_producer(tmp_path, producer_template)


@pytest.fixture(scope="session")
def producer_factory(producer_template):
    """new_producer for wider-scoped fixtures: producer_factory(base_dir)."""
    return lambda base: new_producer(base, producer_template)


def _publish_versions(template, tmp_path_factory, pkl_contents):
    """Publish one pkl-only run per entry of pkl_contents from a fresh producer.
    Returns (repo, [publish results])."""
    from mldna.core.push import push as core_push
    from mldna.publish import publish
    from mldna.tracking import track_run

    repo = new_producer(tmp_path_factory.mktemp("published"), template)
    run(repo, "add", "-A")
    run(repo, "commit", "-q", "-m", "scaffold")
    core_push(repo)
    results = []
    for content in pkl_contents:
        with track_run(workspace_dir=repo):
            pkl = repo / "models" / "model.pkl"
            pkl.parent.mkdir(exist_ok=True)
            pkl.write_bytes(content)
        results.append(publish(pkl, workspace_dir=repo, confirm=lambda s: True))

    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(None)
    mlflow_fluent._active_experiment_id = None
    return repo, results


def _private_copy(source, tmp_path):
    """Copy a published producer, registry included (mlruns lives in the
    project), so a test may mutate it freely; mlflow is pointed at the copy."""
    from mldna.trackinguri import resolve_and_set_tracking_uri

    repo = tmp_path / "project"
    shutil.copytree(source, repo, symlinks=True)
    resolve_and_set_tracking_uri(repo)
    return repo


@pytest.fixture(scope="session")
def published_producer(producer_template, tmp_path_factory):
    """One pkl-only publish per session (per xdist worker), for tests that only
    read the registry."""
    return _publish_versions(producer_template, tmp_path_factory, [b"trained model bytes"])


@pytest.fixture(scope="session")
def published_twice_producer(producer_template, tmp_path_factory):
    """Two published runs of the same commit (the ambiguous-pull scenario)."""
    return _publish_versions(
        producer_template, tmp_path_factory, [b"model bytes v1", b"model bytes v2"]
    )


@pytest.fixture
def published(published_producer, tmp_path):
    """(repo, pkl_path, publish_result) of a private copy of the one publish."""
    source, (result,) = published_producer
    repo = _private_copy(source, tmp_path)
    return repo, repo / "models" / "model.pkl", result


@pytest.fixture
def published_twice(published_twice_producer, tmp_path):
    """(repo, [first_result, second_result]) of a private copy."""
    source, results = published_twice_producer
    return _private_copy(source, tmp_path), results
