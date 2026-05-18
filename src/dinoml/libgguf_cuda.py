from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from dinoml.ir import canonical_json


def libgguf_source_provenance(source_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    return {
        "schema_version": 1,
        "source_kind": "vendored_source",
        "source_root": str(source_root),
        "git_revision": _git_output(source_root, "rev-parse", "HEAD"),
        "git_tree": _git_output(source_root, "rev-parse", "HEAD^{tree}"),
        "tracked_source_hash": _tracked_source_hash(source_root),
    }


def libgguf_provenance_key(provenance: dict[str, object]) -> str:
    return hashlib.sha256(canonical_json(provenance).encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _git_output(source_root: Path, *args: str) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(source_root), *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _tracked_source_hash(source_root: Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", str(source_root), "ls-files", "-z"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except OSError:
        proc = None
    if proc is not None and proc.returncode == 0:
        files = [item.decode("utf-8") for item in proc.stdout.split(b"\0") if item]
    else:
        files = [
            str(path.relative_to(source_root))
            for path in source_root.rglob("*")
            if path.is_file() and ".git" not in path.parts
        ]
    digest = hashlib.sha256()
    for rel_path in sorted(files):
        path = source_root / rel_path
        if not path.is_file():
            continue
        digest.update(rel_path.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest()
