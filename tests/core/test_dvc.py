from mldna.core.dvc import read_data_hash


def test_read_data_hash_from_real_pointer_file(producer_repo):
    data_hash = read_data_hash(producer_repo)

    assert data_hash is not None
    assert data_hash.endswith(".dir")


def test_read_data_hash_returns_none_when_absent(tmp_path):
    assert read_data_hash(tmp_path) is None
