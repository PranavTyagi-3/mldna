import hashlib

CHUNK_SIZE = 8 * 1024 * 1024


def sha256_file(path, chunk_size=CHUNK_SIZE) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()
