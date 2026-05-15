from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path

from dinoml.ir import canonical_json

LIBGGUF_CUDA_NATIVE_LIBRARY_ENV = "LIBGGUF_CUDA_NATIVE_LIBRARY"
LIBGGUF_CUDA_EXTENSION_ENV = "LIBGGUF_CUDA_EXTENSION"


def resolve_libgguf_cuda_direct_link_library() -> Path | None:
    explicit_native = os.environ.get(LIBGGUF_CUDA_NATIVE_LIBRARY_ENV)
    if not explicit_native:
        return None
    resolved = Path(explicit_native).resolve()
    if resolved.is_file():
        return resolved
    return None


def resolve_libgguf_cuda_symbol_library() -> Path | None:
    explicit_native = resolve_libgguf_cuda_direct_link_library()
    if explicit_native is not None and _is_dynamic_library(explicit_native):
        return explicit_native
    extension_path = _extension_path_from_env_or_import()
    if extension_path is None:
        return None
    for candidate in (*_sibling_native_library_candidates(extension_path, dynamic_only=True), extension_path):
        resolved = candidate.resolve()
        if resolved.is_file() and _is_dynamic_library(resolved):
            return resolved
    return None


def _extension_path_from_env_or_import() -> Path | None:
    explicit_extension = os.environ.get(LIBGGUF_CUDA_EXTENSION_ENV)
    if explicit_extension:
        return Path(explicit_extension)
    try:
        import libgguf.libgguf_cuda.ops as cuda_ops  # type: ignore[import-not-found]
    except ImportError:
        return None
    extension = getattr(cuda_ops, "_C_gguf", None)
    extension_file = getattr(extension, "__file__", None)
    if not extension_file:
        return None
    return Path(str(extension_file))


def _sibling_native_library_candidates(extension_path: Path, *, dynamic_only: bool = False) -> tuple[Path, ...]:
    names = ("gguf_cuda_native.so", "libgguf_cuda_native.so")
    if not dynamic_only:
        names = (*names, "libgguf_cuda_native.a")
    candidates: list[Path] = []
    for parent in (extension_path.parent, *extension_path.parents[:4]):
        for name in names:
            candidates.append(parent / name)
    return tuple(candidates)


def libgguf_submodule_source_root(repo_root: Path) -> Path | None:
    candidate = repo_root / "third_party" / "libgguf"
    if (candidate / "CMakeLists.txt").exists():
        return candidate
    return None


def libgguf_source_provenance(source_root: Path) -> dict[str, object]:
    source_root = source_root.resolve()
    return {
        "schema_version": 1,
        "source_kind": "vendored_submodule",
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


def _is_dynamic_library(path: Path) -> bool:
    return path.suffix in {".so", ".dylib", ".dll"}


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
