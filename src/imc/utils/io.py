"""Filesystem and download helpers shared across scripts."""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Iterable

import requests
from tqdm import tqdm


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure.

    Returns
    -------
    Path
        The directory path, guaranteed to exist.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def md5sum(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Compute the MD5 hex digest of a file in streaming fashion.

    Parameters
    ----------
    path : str or Path
        File to hash.
    chunk_size : int
        Read chunk size in bytes.

    Returns
    -------
    str
        Lowercase hex MD5 digest.
    """
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: str | Path,
    expected_md5: str | None = None,
    chunk_size: int = 1 << 20,
    timeout: int = 60,
    overwrite: bool = False,
) -> Path:
    """Download ``url`` to ``dest`` with resume + optional MD5 verification.

    Parameters
    ----------
    url : str
        Source URL (HTTPS).
    dest : str or Path
        Destination path on disk.
    expected_md5 : str or None
        If given, validate the downloaded file against this MD5 hex digest
        and raise ``ValueError`` on mismatch.
    chunk_size : int
        Streaming read chunk size in bytes.
    timeout : int
        Per-request timeout in seconds.
    overwrite : bool
        If True, re-download even if the destination file already exists.

    Returns
    -------
    Path
        The destination path on disk.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and not overwrite:
        if expected_md5 is None or md5sum(dest) == expected_md5:
            return dest

    resume_byte = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={resume_byte}-"} if resume_byte else {}

    with requests.get(url, stream=True, timeout=timeout, headers=headers) as resp:
        resp.raise_for_status()
        total_extra = int(resp.headers.get("Content-Length", "0") or "0")
        total = total_extra + resume_byte
        mode = "ab" if resume_byte else "wb"
        with open(part, mode) as f, tqdm(
            total=total or None,
            initial=resume_byte,
            unit="B",
            unit_scale=True,
            desc=dest.name,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                bar.update(len(chunk))

    shutil.move(part, dest)
    if expected_md5 is not None:
        actual = md5sum(dest)
        if actual != expected_md5:
            raise ValueError(f"MD5 mismatch for {dest}: expected {expected_md5}, got {actual}")
    return dest


def iter_lines(path: str | Path, encoding: str = "utf-8") -> Iterable[str]:
    """Yield decoded lines from ``path``, transparently handling ``.gz``.

    Parameters
    ----------
    path : str or Path
        File path. ``.gz`` files are opened with gzip decompression.
    encoding : str
        Text encoding.

    Yields
    ------
    str
        Each decoded line of the file (newline stripped).
    """
    import gzip

    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    with opener(p, "rt", encoding=encoding) as f:
        for line in f:
            yield line.rstrip("\n")
