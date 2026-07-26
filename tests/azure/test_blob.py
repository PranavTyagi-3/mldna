from types import SimpleNamespace

from mlops.azure.blob import (
    list_containers_via_connection_string,
    list_containers_via_credential,
    parse_account_name,
)


def test_parse_account_name_finds_it_among_other_fields():
    cs = "DefaultEndpointsProtocol=https;AccountName=myacct;AccountKey=xxx;EndpointSuffix=core.windows.net"
    assert parse_account_name(cs) == "myacct"


def test_parse_account_name_missing_returns_none():
    assert parse_account_name("SharedAccessSignature=sv=2020-01-01&sig=xxx") is None


class _FakeContainers:
    def __init__(self, names):
        self._names = names

    def list_containers(self):
        return [SimpleNamespace(name=n) for n in self._names]


def test_list_containers_via_connection_string_sorts_names(monkeypatch):
    calls = {}

    class FakeBlobServiceClient:
        @classmethod
        def from_connection_string(cls, connection_string):
            calls["connection_string"] = connection_string
            return _FakeContainers(["bravo", "alpha"])

    monkeypatch.setattr("mlops.azure.blob._import_blob_client", lambda: FakeBlobServiceClient)

    result = list_containers_via_connection_string("fake-connection-string")

    assert calls["connection_string"] == "fake-connection-string"
    assert result == ["alpha", "bravo"]


def test_list_containers_via_credential_uses_account_url(monkeypatch):
    calls = {}

    class FakeBlobServiceClient:
        def __init__(self, account_url, credential):
            calls["account_url"] = account_url
            calls["credential"] = credential
            self._inner = _FakeContainers(["datasets"])

        def list_containers(self):
            return self._inner.list_containers()

    monkeypatch.setattr("mlops.azure.blob._import_blob_client", lambda: FakeBlobServiceClient)

    result = list_containers_via_credential("myacct", "fake-credential")

    assert calls["account_url"] == "https://myacct.blob.core.windows.net"
    assert calls["credential"] == "fake-credential"
    assert result == ["datasets"]
