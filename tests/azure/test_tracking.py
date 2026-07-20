from types import SimpleNamespace

from mlops.azure.tracking import list_workspaces


def test_list_workspaces_lists_by_subscription_scope(monkeypatch):
    calls = {}

    class FakeWorkspaceOperations:
        def list(self, *, scope):
            calls["scope"] = scope
            return [
                SimpleNamespace(name="ml-workspace-dev", resource_group="rg-dev"),
                SimpleNamespace(name="ml-workspace-prod", resource_group="rg-prod"),
            ]

    class FakeMLClient:
        def __init__(self, credential, subscription_id):
            calls["credential"] = credential
            calls["subscription_id"] = subscription_id
            self.workspaces = FakeWorkspaceOperations()

    monkeypatch.setattr("mlops.azure.tracking.MLClient", FakeMLClient)

    result = list_workspaces("sub-123", "fake-credential")

    assert calls["scope"] == "subscription"
    assert calls["subscription_id"] == "sub-123"
    assert calls["credential"] == "fake-credential"
    assert result == [("ml-workspace-dev", "rg-dev"), ("ml-workspace-prod", "rg-prod")]


def test_list_workspaces_empty(monkeypatch):
    class FakeWorkspaceOperations:
        def list(self, *, scope):
            return []

    class FakeMLClient:
        def __init__(self, credential, subscription_id):
            self.workspaces = FakeWorkspaceOperations()

    monkeypatch.setattr("mlops.azure.tracking.MLClient", FakeMLClient)

    assert list_workspaces("sub-123", "fake-credential") == []
