import hashlib

from mlops.core.hashing import sha256_file


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "data.bin"
    payload = b"some model bytes" * 1000
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_sha256_file_small_chunk_size(tmp_path):
    path = tmp_path / "data.bin"
    payload = b"x" * 100
    path.write_bytes(payload)

    assert sha256_file(path, chunk_size=7) == hashlib.sha256(payload).hexdigest()
