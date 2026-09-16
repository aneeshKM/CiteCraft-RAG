"""Stable content hashing utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_handle:
        while block := file_handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def stable_parent_id(file_hash: str, position: int, text: str) -> str:
    payload = f"{file_hash}:{position}:{text}".encode()
    return hashlib.sha256(payload).hexdigest()
