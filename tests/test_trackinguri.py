import os

import mlflow
import pytest
import yaml

from mldna.trackinguri import TrackingUriError, resolve_and_set_tracking_uri
import mldna.trackinguri as trackinguri_module

MINIMAL = {"project": {"model_name": "test-model", "repo_role": "producer"}}


def _write_config(tmp_path, extra=None):
    data = dict(MINIMAL)
    if extra:
        data.update(extra)
    (tmp_path / "mldna.yaml").write_text(yaml.dump(data))


def test_no_mldna_yaml_uses_local(tmp_path):
    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert mode == "local"
    assert uri.startswith("file:")
    assert mlflow.get_tracking_uri() == uri


def test_no_azure_section_uses_local(tmp_path):
    _write_config(tmp_path)

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert mode == "local"


def test_incomplete_azure_section_raises(tmp_path):
    _write_config(tmp_path, {"azure": {"subscription_id": "s", "resource_group": "rg"}})

    with pytest.raises(TrackingUriError, match="workspace"):
        resolve_and_set_tracking_uri(tmp_path)


def test_complete_azure_section_bootstraps(tmp_path, monkeypatch):
    _write_config(tmp_path, {
        "azure": {"subscription_id": "s", "resource_group": "rg", "workspace": "ws"},
    })

    calls = {}

    def fake_resolve_credential(workspace_dir):
        calls["auth_workspace_dir"] = workspace_dir
        return "fake-credential", "fake mode"

    def fake_bootstrap(subscription_id, resource_group, workspace, credential):
        calls["bootstrap_args"] = (subscription_id, resource_group, workspace, credential)
        return "azureml://fake-tracking-uri"

    monkeypatch.setattr("mldna.azure.auth.resolve_aml_credential", fake_resolve_credential)
    monkeypatch.setattr("mldna.azure.tracking.bootstrap_tracking_uri", fake_bootstrap)

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert uri == "azureml://fake-tracking-uri"
    assert mode == "fake mode"
    assert calls["bootstrap_args"] == ("s", "rg", "ws", "fake-credential")
    assert mlflow.get_tracking_uri() == uri


def test_tracking_uri_shortcut_skips_bootstrap(tmp_path, monkeypatch):
    _write_config(tmp_path, {
        "azure": {"tracking_uri": "azureml://pasted-endpoint"},
    })

    def fake_resolve_credential(workspace_dir):
        return "fake-credential", "fake mode"

    def boom(*args, **kwargs):
        raise AssertionError("bootstrap_tracking_uri should not be called for the tracking_uri shortcut")

    monkeypatch.setattr("mldna.azure.auth.resolve_aml_credential", fake_resolve_credential)
    monkeypatch.setattr("mldna.azure.tracking.bootstrap_tracking_uri", boom)

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert uri == "azureml://pasted-endpoint"
    assert mode == "fake mode"
    assert mlflow.get_tracking_uri() == uri


def test_azure_mode_makes_stdout_unicode_safe(tmp_path, monkeypatch):
    # Regression: mlflow crashes mid-call printing a run URL with an emoji on
    # Windows' default console codepage when talking to a real REST tracking
    # server - confirmed live against mlops-spike-ws, and it happened before
    # the run's terminal status was even written. Local mode never talks to a
    # real server, so it must NOT be touched.
    _write_config(tmp_path, {
        "azure": {"subscription_id": "s", "resource_group": "rg", "workspace": "ws"},
    })
    monkeypatch.setattr("mldna.azure.auth.resolve_aml_credential", lambda d: ("cred", "mode"))
    monkeypatch.setattr(
        "mldna.azure.tracking.bootstrap_tracking_uri", lambda *a: "azureml://fake"
    )

    calls = []

    class FakeStdout:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(trackinguri_module.sys, "stdout", FakeStdout())

    resolve_and_set_tracking_uri(tmp_path)

    assert calls == [{"errors": "replace"}]


def test_local_mode_does_not_touch_stdout(tmp_path, monkeypatch):
    class FakeStdout:
        def reconfigure(self, **kwargs):
            raise AssertionError("local mode never talks to a real server - must not reconfigure stdout")

    monkeypatch.setattr(trackinguri_module.sys, "stdout", FakeStdout())

    resolve_and_set_tracking_uri(tmp_path)  # no raise


def test_mlflow_section_sets_tracking_uri(tmp_path):
    _write_config(tmp_path, {"mlflow": {"tracking_uri": "http://my-mlflow-host:5000"}})

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert uri == "http://my-mlflow-host:5000"
    assert mode == "remote mlflow server"
    assert mlflow.get_tracking_uri() == uri


def test_mlflow_section_missing_tracking_uri_raises(tmp_path):
    _write_config(tmp_path, {"mlflow": {}})

    with pytest.raises(TrackingUriError, match="tracking_uri"):
        resolve_and_set_tracking_uri(tmp_path)


def test_both_mlflow_and_azure_sections_raises(tmp_path):
    _write_config(tmp_path, {
        "mlflow": {"tracking_uri": "http://my-mlflow-host:5000"},
        "azure": {"subscription_id": "s", "resource_group": "rg", "workspace": "ws"},
    })

    with pytest.raises(TrackingUriError, match="only one tracking"):
        resolve_and_set_tracking_uri(tmp_path)


def test_filesystem_and_azure_sections_raises(tmp_path):
    _write_config(tmp_path, {
        "filesystem": {"path": str(tmp_path)},
        "azure": {"subscription_id": "s", "resource_group": "rg", "workspace": "ws"},
    })

    with pytest.raises(TrackingUriError, match="only one tracking"):
        resolve_and_set_tracking_uri(tmp_path)


def test_filesystem_and_mlflow_sections_raises(tmp_path):
    _write_config(tmp_path, {
        "filesystem": {"path": str(tmp_path)},
        "mlflow": {"tracking_uri": "http://my-mlflow-host:5000"},
    })

    with pytest.raises(TrackingUriError, match="only one tracking"):
        resolve_and_set_tracking_uri(tmp_path)


def test_filesystem_nonexistent_path_raises_naming_resolved_path(tmp_path):
    missing = tmp_path / "producer" / "mlruns"
    _write_config(tmp_path, {"filesystem": {"path": str(missing)}})

    with pytest.raises(TrackingUriError, match="does not exist") as excinfo:
        resolve_and_set_tracking_uri(tmp_path)

    assert str(missing.resolve()) in str(excinfo.value)


def test_filesystem_unset_env_var_refuses_naming_it(tmp_path, monkeypatch):
    monkeypatch.delenv("MLDNA_TRACKING_ROOT", raising=False)
    _write_config(tmp_path, {"filesystem": {"path": "$MLDNA_TRACKING_ROOT/mlruns"}})

    with pytest.raises(TrackingUriError, match="is not set") as excinfo:
        resolve_and_set_tracking_uri(tmp_path)

    assert "MLDNA_TRACKING_ROOT is not set (filesystem.path in mldna.yaml uses it)" in str(excinfo.value)
    assert "Fix: set MLDNA_TRACKING_ROOT, or use a literal path." in str(excinfo.value)
    assert "$MLDNA" not in str(excinfo.value)


def test_filesystem_section_missing_path_raises(tmp_path):
    _write_config(tmp_path, {"filesystem": {}})

    with pytest.raises(TrackingUriError, match="no path key"):
        resolve_and_set_tracking_uri(tmp_path)


def test_mlflow_missing_tracking_uri_error_offers_filesystem_section(tmp_path):
    _write_config(tmp_path, {"mlflow": {}})

    with pytest.raises(TrackingUriError, match="filesystem:"):
        resolve_and_set_tracking_uri(tmp_path)


def test_filesystem_absolute_path_sets_tracking_uri(tmp_path):
    store = tmp_path / "producer-mlruns"
    store.mkdir()
    _write_config(tmp_path, {"filesystem": {"path": str(store)}})

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert uri == f"file:{store.resolve()}"
    assert mode == "filesystem store"
    assert mode != "remote mlflow server"
    assert mlflow.get_tracking_uri() == uri


def test_filesystem_path_expands_env_var(tmp_path, monkeypatch):
    store = tmp_path / "shared" / "mlruns"
    store.mkdir(parents=True)
    monkeypatch.setenv("MLDNA_TRACKING_ROOT", str(tmp_path / "shared"))
    _write_config(tmp_path, {"filesystem": {"path": "$MLDNA_TRACKING_ROOT/mlruns"}})

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert uri == f"file:{store.resolve()}"
    assert mode == "filesystem store"


def test_filesystem_path_relative_to_repo_root(tmp_path):
    store = tmp_path / "sibling-mlruns"
    store.mkdir()
    _write_config(tmp_path, {"filesystem": {"path": "sibling-mlruns"}})

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert uri == f"file:{store.resolve()}"
    assert mode == "filesystem store"


def test_filesystem_mode_does_not_touch_stdout(tmp_path, monkeypatch):
    # A file-based store is not a REST server, so the emoji print that forced
    # the stdout fix (D#30) never happens - same as local mode.
    store = tmp_path / "producer-mlruns"
    store.mkdir()
    _write_config(tmp_path, {"filesystem": {"path": str(store)}})

    class FakeStdout:
        def reconfigure(self, **kwargs):
            raise AssertionError("a filesystem store never talks to a real server")

    monkeypatch.setattr(trackinguri_module.sys, "stdout", FakeStdout())

    resolve_and_set_tracking_uri(tmp_path)  # no raise


def test_mlflow_section_with_file_uri_is_labeled_filesystem(tmp_path):
    _write_config(tmp_path, {"mlflow": {"tracking_uri": f"file:{tmp_path}"}})

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert uri == f"file:{tmp_path}"
    assert mode == "filesystem store"


def test_default_local_uri_and_mode_unchanged(tmp_path):
    _write_config(tmp_path)

    uri, mode = resolve_and_set_tracking_uri(tmp_path)

    assert mode == "local"
    assert uri == f"file:{tmp_path / 'mlruns'}"


def test_mlflow_mode_makes_stdout_unicode_safe(tmp_path, monkeypatch):
    # Same emoji-crash risk as the Azure branch (D#30's stdout fix) - a plain
    # remote MLflow server is also a real REST tracking server, unlike local.
    _write_config(tmp_path, {"mlflow": {"tracking_uri": "http://my-mlflow-host:5000"}})

    calls = []

    class FakeStdout:
        def reconfigure(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(trackinguri_module.sys, "stdout", FakeStdout())

    resolve_and_set_tracking_uri(tmp_path)

    assert calls == [{"errors": "replace"}]


RETRY_VARS = ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR")
TIMEOUT_VARS = (
    "MLFLOW_HTTP_REQUEST_TIMEOUT",
    "MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT",
    "MLFLOW_DOWNLOAD_CHUNK_TIMEOUT",
)


@pytest.fixture
def clean_retry_env():
    # _bound_http_retries writes process-wide env, so snapshot/restore rather
    # than monkeypatch.delenv (which records nothing for an absent var and
    # would leak the default into later tests).
    saved = {name: os.environ.pop(name, None) for name in RETRY_VARS + TIMEOUT_VARS}
    yield
    for name, old in saved.items():
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def test_local_mode_does_not_bound_http_retries(tmp_path, clean_retry_env):
    _write_config(tmp_path)

    resolve_and_set_tracking_uri(tmp_path)

    assert [name for name in RETRY_VARS if name in os.environ] == []


def test_filesystem_mode_does_not_bound_http_retries(tmp_path, clean_retry_env):
    store = tmp_path / "producer-mlruns"
    store.mkdir()
    _write_config(tmp_path, {"filesystem": {"path": str(store)}})

    resolve_and_set_tracking_uri(tmp_path)

    assert [name for name in RETRY_VARS if name in os.environ] == []


def test_mlflow_file_uri_mode_does_not_bound_http_retries(tmp_path, clean_retry_env):
    _write_config(tmp_path, {"mlflow": {"tracking_uri": f"file:{tmp_path}"}})

    resolve_and_set_tracking_uri(tmp_path)

    assert [name for name in RETRY_VARS if name in os.environ] == []


def test_user_set_retry_values_are_not_overridden(tmp_path, monkeypatch, clean_retry_env):
    # The bound is a default, not a policy: an operator who needs mlflow's
    # original patience must keep it.
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "7")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "2")
    _write_config(tmp_path, {"mlflow": {"tracking_uri": "http://my-mlflow-host:5000"}})

    resolve_and_set_tracking_uri(tmp_path)

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "7"
    assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "2"


def test_rest_mode_leaves_timeouts_untouched(tmp_path, clean_retry_env):
    # A multi-GB bundle transfer legitimately needs mlflow's generous
    # per-request timeouts - only retry count/spacing is bounded.
    _write_config(tmp_path, {"mlflow": {"tracking_uri": "http://my-mlflow-host:5000"}})

    resolve_and_set_tracking_uri(tmp_path)

    assert [name for name in TIMEOUT_VARS if name in os.environ] == []


def test_mlflow_mode_bounds_http_retries(tmp_path, clean_retry_env):
    _write_config(tmp_path, {"mlflow": {"tracking_uri": "http://my-mlflow-host:5000"}})

    resolve_and_set_tracking_uri(tmp_path)

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "3"
    assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "1"


def test_azure_mode_bounds_http_retries(tmp_path, monkeypatch, clean_retry_env):
    _write_config(tmp_path, {
        "azure": {"subscription_id": "s", "resource_group": "rg", "workspace": "ws"},
    })
    monkeypatch.setattr("mldna.azure.auth.resolve_aml_credential", lambda d: ("cred", "mode"))
    monkeypatch.setattr(
        "mldna.azure.tracking.bootstrap_tracking_uri", lambda *a: "azureml://fake"
    )

    resolve_and_set_tracking_uri(tmp_path)

    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "3"
    assert os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] == "1"


def test_filesystem_missing_path_error_names_no_mldna_env_var(tmp_path):
    # MLDNA_TRACKING_ROOT is not a variable mldna defines or reads - naming it
    # in the example made it read like one.
    _write_config(tmp_path, {"filesystem": {}})

    with pytest.raises(TrackingUriError) as excinfo:
        resolve_and_set_tracking_uri(tmp_path)

    assert "MLDNA_TRACKING_ROOT" not in str(excinfo.value)
    assert "MLDNA_" not in str(excinfo.value)
    assert "environment variable" in str(excinfo.value)
