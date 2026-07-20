from azure.storage.blob import BlobServiceClient


def parse_account_name(connection_string):
    """Extracts AccountName=... from a storage connection string, or None."""
    for part in connection_string.split(";"):
        key, sep, value = part.partition("=")
        if sep and key.strip() == "AccountName":
            return value.strip()
    return None


def list_containers_via_connection_string(connection_string):
    client = BlobServiceClient.from_connection_string(connection_string)
    return sorted(c.name for c in client.list_containers())


def list_containers_via_credential(account_name, credential):
    client = BlobServiceClient(
        f"https://{account_name}.blob.core.windows.net", credential=credential,
    )
    return sorted(c.name for c in client.list_containers())
