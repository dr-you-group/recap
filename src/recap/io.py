"""Deterministic JSON, immutable artifacts, and hashes."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_bytes(path: Path, data: bytes, *, immutable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and immutable:
        if path.read_bytes() != data:
            raise ValueError(f"immutable artifact differs: {path}")
        return
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_json(path: Path, value, *, immutable: bool = False) -> None:
    write_bytes(path, canonical(value), immutable=immutable)


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows, *, immutable: bool = False) -> None:
    write_bytes(path, b"".join(canonical(x) for x in rows), immutable=immutable)
