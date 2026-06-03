from __future__ import annotations

import importlib.machinery
import os

from dinoml.backends import rocm as rocm_backend


def test_rocm_profiler_artifact_helpers_prefer_loadable_module_and_executable(tmp_path):
    stem = "dinoml_ck_gemm_profiler_gemm_rcr_float16"
    bind_suffix = importlib.machinery.EXTENSION_SUFFIXES[0]
    bind_import = tmp_path / f"{stem}_bind{bind_suffix}"
    bind_import.write_bytes(b"bind-import")
    (tmp_path / f"{stem}_bind.exp").write_bytes(b"bind-exp")
    (tmp_path / f"{stem}_bind.lib").write_bytes(b"bind-lib")
    executable = tmp_path / (f"{stem}.exe" if os.name == "nt" else stem)
    executable.write_bytes(b"exe")
    (tmp_path / f"{stem}.exp").write_bytes(b"exe-exp")

    assert rocm_backend._profiler_bind_artifact(tmp_path, stem) == bind_import
    assert rocm_backend._profiler_bind_artifact_sha256(tmp_path, stem) == rocm_backend.file_sha256(bind_import)
    assert rocm_backend._profiler_executable_artifact(tmp_path, stem) == executable
    assert rocm_backend._profiler_executable_artifact_sha256(tmp_path, stem) == rocm_backend.file_sha256(executable)
