def _import_blob_client():
    # Lazy for the same reason as auth._import_azure_identity: cli/main.py
    # imports this module at module level for parse_account_name (pure
    # string parsing), which must work on a base install without the
    # [azure] extra.
    try:
        from azure.storage.blob import BlobServiceClient
    except ImportError as e:
        raise ImportError(
            f'Azure Blob discovery needs the `azure` extra. Fix: pip install "mlops[azure]" ({e})'
        )
    return BlobServiceClient


def parse_account_name(connection_string):
    """Extracts AccountName=... from a storage connection string, or None."""
    for part in connection_string.split(";"):
        key, sep, value = part.partition("=")
        if sep and key.strip() == "AccountName":
            return value.strip()
    return None


def list_containers_via_connection_string(connection_string):
    client = _import_blob_client().from_connection_string(connection_string)
    return sorted(c.name for c in client.list_containers())


def list_containers_via_credential(account_name, credential):
    client = _import_blob_client()(
        f"https://{account_name}.blob.core.windows.net", credential=credential,
    )
    return sorted(c.name for c in client.list_containers())
