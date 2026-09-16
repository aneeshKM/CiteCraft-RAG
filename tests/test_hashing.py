from citecraft.hashing import sha256_file, stable_parent_id


def test_sha256_file_is_content_based(tmp_path):
    first = tmp_path / "first.pdf"
    second = tmp_path / "renamed.pdf"
    first.write_bytes(b"same PDF bytes")
    second.write_bytes(b"same PDF bytes")

    assert sha256_file(first) == sha256_file(second)


def test_parent_id_is_stable_and_position_sensitive():
    first = stable_parent_id("abc", 1, "content")

    assert first == stable_parent_id("abc", 1, "content")
    assert first != stable_parent_id("abc", 2, "content")

