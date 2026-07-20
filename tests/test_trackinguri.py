import mlflow
import pytest
import yaml

from mlops.trackinguri import TrackingUriError, resolve_and_set_tracking_uri
import mlops.trackinguri as trackinguri_module

MINIMAL = {"project": {"model_name": "test-model", "repo_role": "producer"}}


def _write_config(tmp_path, extra=None):
    data = dict(MINIMAL)
    if extra:
        data.update(extra)
    (tmp_path / "mlops.yaml").write_text(yaml.dump(data))


def test_no_mlops_yaml_uses_local(tmp_path):
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

    monkeypatch.setattr("mlops.azure.auth.resolve_aml_credential", fake_resolve_credential)
    monkeypatch.setattr("mlops.azure.tracking.bootstrap_tracking_uri", fake_bootstrap)

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

    monkeypatch.setattr("mlops.azure.auth.resolve_aml_credential", fake_resolve_credential)
    monkeypatch.setattr("mlops.azure.tracking.bootstrap_tracking_uri", boom)

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
    monkeypatch.setattr("mlops.azure.auth.resolve_aml_credential", lambda d: ("cred", "mode"))
    monkeypatch.setattr(
        "mlops.azure.tracking.bootstrap_tracking_uri", lambda *a: "azureml://fake"
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
