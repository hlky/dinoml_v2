from __future__ import annotations

import os
from pathlib import Path


def resolve_libgguf_cuda_direct_link_library() -> Path | None:
    for candidate in _direct_link_candidates():
        resolved = candidate.resolve()
        if resolved.is_file():
            return resolved
    return None


def resolve_libgguf_cuda_symbol_library() -> Path | None:
    direct = resolve_libgguf_cuda_direct_link_library()
    if direct is not None:
        return direct
    extension_path = _extension_path_from_env_or_import()
    if extension_path is None:
        return None
    resolved = extension_path.resolve()
    return resolved if resolved.is_file() else None


def _direct_link_candidates() -> tuple[Path, ...]:
    explicit_native = os.environ.get("LIBGGUF_CUDA_NATIVE_LIBRARY")
    if explicit_native:
        return (Path(explicit_native),)
    extension_path = _extension_path_from_env_or_import()
    if extension_path is None:
        return ()
    return _sibling_native_library_candidates(extension_path)


def _extension_path_from_env_or_import() -> Path | None:
    explicit_extension = os.environ.get("LIBGGUF_CUDA_EXTENSION")
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


def _sibling_native_library_candidates(extension_path: Path) -> tuple[Path, ...]:
    names = (
        "gguf_cuda_native.so",
        "libgguf_cuda_native.so",
        "libgguf_cuda_native.a",
    )
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
