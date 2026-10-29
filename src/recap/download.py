"""Bounded public-data retrieval, verified cache, and validator-based resume."""
from __future__ import annotations

import hashlib
import json
import tarfile
import time
from pathlib import Path, PurePosixPath

import httpx
from lxml import etree
from PIL import Image
from pypdf import PdfReader

from .io import now, sha256, write_json


def xml_root(data: bytes):
    parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False,
                             recover=False, huge_tree=False)
    return etree.fromstring(data, parser=parser)


def validate_file(path: Path, kind: str | None) -> None:
    with path.open("rb") as stream:
        prefix = stream.read(1024)
    if kind == "pdf":
        if not prefix.startswith(b"%PDF-"):
            raise ValueError("not a PDF signature")
        reader = PdfReader(path)
        if not reader.pages:
            raise ValueError("PDF has no pages")
    elif kind == "image":
        with Image.open(path) as img:
            if img.format not in {"PNG", "JPEG", "WEBP", "TIFF", "GIF"}:
                raise ValueError("unsupported image signature")
            img.verify()
    elif kind == "json":
        json.loads(path.read_bytes())
    elif kind == "xml":
        root = xml_root(path.read_bytes())
        if etree.QName(root).localname.lower() == "html":
            raise ValueError("HTML is not the requested XML")


def validate_tar(path: Path, *, max_unpacked_bytes: int = 512 * 1024 * 1024) -> list[str]:
    """Validate before extraction. M0 collectors use individual objects, not archives."""
    names, total = [], 0
    with tarfile.open(path) as archive:
        for member in archive:
            p = PurePosixPath(member.name)
            if p.is_absolute() or ".." in p.parts or "\\" in member.name:
                raise ValueError("archive path escape")
            if not (member.isfile() or member.isdir()):
                raise ValueError("archive links/devices prohibited")
            total += member.size
            if total > max_unpacked_bytes:
                raise ValueError("archive unpacked size limit")
            names.append(member.name)
    return names


class DownloadCache:
    def __init__(self, root: Path, *, offline: bool = False, client=None,
                 max_bytes: int = 5 * 1024**3, interval: float = 0.5,
                 sleep=time.sleep, clock=time.monotonic):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.offline = offline
        self.client = client or httpx.Client(timeout=30, follow_redirects=True,
                                             headers={"User-Agent": "RECAP-M0/0.1 public-research"})
        self.max_bytes = max_bytes
        self.initial_bytes = sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())
        self.received = 0
        self.interval, self.sleep, self.clock = interval, sleep, clock
        self.last_request = -float("inf")
        self.hits = 0
        self.downloads = 0
        self.events: list[dict] = []

    def _pace(self):
        pause = self.interval - (self.clock() - self.last_request)
        if pause > 0:
            self.sleep(pause)
        self.last_request = self.clock()

    def _event(self, event):
        event = {"at": now(), **event}
        self.events.append(event)
        with (self.root / "requests.jsonl").open("a") as stream:
            stream.write(json.dumps(event, sort_keys=True) + "\n")

    def fetch(self, url: str, relative: str, *, kind: str | None = None,
              expected_md5: str | None = None, expected_sha256: str | None = None) -> tuple[Path, dict]:
        path = self.root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not path.resolve().is_relative_to(self.root):
            raise ValueError("download destination escapes cache")
        path.parent.mkdir(parents=True, exist_ok=True)
        info_path = path.with_name(path.name + ".source.json")
        if path.exists():
            if not info_path.exists():
                raise ValueError("existing raw file has no provenance; will not overwrite")
            info = json.loads(info_path.read_text())
            if info["source_url"] != url or sha256(path) != info["sha256"]:
                raise ValueError("cache URL/hash mismatch")
            self._verify_hash(path, expected_md5, expected_sha256)
            validate_file(path, kind)
            self.hits += 1
            return path, info
        if self.offline:
            raise FileNotFoundError(f"offline cache miss: {relative}")
        partial = path.with_name(path.name + ".part")
        partial_info = path.with_name(path.name + ".part.json")
        last_error = None
        for attempt in range(1, 4):
            headers = {"Accept-Encoding": "identity"}
            offset = 0
            if partial.exists() and partial_info.exists():
                old = json.loads(partial_info.read_text())
                if old.get("url") == url and old.get("validator"):
                    offset = partial.stat().st_size
                    headers.update({"Range": f"bytes={offset}-", "If-Range": old["validator"]})
            self._pace()
            try:
                with self.client.stream("GET", url, headers=headers) as response:
                    if response.status_code == 416 and offset:
                        partial.unlink(missing_ok=True)
                        partial_info.unlink(missing_ok=True)
                        raise ValueError("resume range rejected; restart required")
                    response.raise_for_status()
                    if response.status_code == 206:
                        value = response.headers.get("content-range", "")
                        if not offset or not value.startswith(f"bytes {offset}-"):
                            raise ValueError("invalid resume Content-Range")
                    else:
                        offset = 0  # server ignored Range or validator changed
                    remaining = response.headers.get("content-length")
                    if remaining and self.initial_bytes + self.received + int(remaining) > self.max_bytes:
                        raise OverflowError("download budget exceeded before transfer")
                    validator = response.headers.get("etag") or response.headers.get("last-modified")
                    write_json(partial_info, {"url": url, "validator": validator})
                    received_this_attempt = 0
                    with partial.open("ab" if offset else "wb") as stream:
                        for chunk in response.iter_bytes():
                            self.received += len(chunk)
                            received_this_attempt += len(chunk)
                            if self.initial_bytes + self.received > self.max_bytes:
                                raise OverflowError("download budget exceeded")
                            stream.write(chunk)
                    if remaining and received_this_attempt != int(remaining):
                        raise ValueError("partial HTTP body")
                    if response.status_code == 206:
                        total = response.headers["content-range"].rsplit("/", 1)[-1]
                        if total != "*" and partial.stat().st_size != int(total):
                            raise ValueError("resume did not complete the object")
                    self._verify_hash(partial, expected_md5, expected_sha256)
                    validate_file(partial, kind)
                    info = {"source_url": url, "retrieved_at": now(), "sha256": sha256(partial),
                            "size": partial.stat().st_size, "source_md5": expected_md5,
                            "etag": response.headers.get("etag"),
                            "last_modified": response.headers.get("last-modified"),
                            "content_type": response.headers.get("content-type"),
                            "attempts": attempt, "resumed_from": offset}
                    partial.replace(path)
                    write_json(info_path, info, immutable=True)
                    partial_info.unlink(missing_ok=True)
                    self.downloads += 1
                    self._event({"url": url, "status": "downloaded", "attempt": attempt,
                                 "size": info["size"], "sha256": info["sha256"]})
                    return path, info
            except (httpx.HTTPError, ValueError, OSError) as error:
                last_error = error
                code = error.response.status_code if isinstance(error, httpx.HTTPStatusError) else None
                self._event({"url": url, "status": "failed", "attempt": attempt,
                             "error_type": type(error).__name__, "http_status": code})
                if code is not None and code not in {408, 429, 500, 502, 503, 504}:
                    break
                # A completed but invalid file must not be resumed as if it were valid.
                if isinstance(error, ValueError) and "partial" not in str(error):
                    partial.unlink(missing_ok=True)
                    partial_info.unlink(missing_ok=True)
                if attempt < 3:
                    self.sleep(min(2 ** (attempt - 1), 4))
        raise RuntimeError(f"public download failed after bounded attempts: {relative}; {type(last_error).__name__}") from last_error

    @staticmethod
    def _verify_hash(path, md5, digest):
        if digest and sha256(path) != digest:
            raise ValueError("SHA-256 mismatch")
        if md5:
            hasher = hashlib.md5(usedforsecurity=False)
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            if hasher.hexdigest() != md5:
                raise ValueError("source MD5 mismatch")
