from __future__ import annotations

from dinoml.backends import build_parallelism
from dinoml.backends import cpu as cpu_backend
from dinoml.backends import cuda as cuda_backend
from dinoml.backends import rocm as rocm_backend


def test_cmake_parallel_args_use_effective_cpu_count(monkeypatch):
    monkeypatch.setattr(build_parallelism, "effective_cpu_count", lambda: 7)

    assert build_parallelism.cmake_parallel_args() == ["--parallel", "7"]


def test_cpu_support_build_passes_explicit_cmake_parallelism(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    calls = []

    def fake_run_cmake(cmd, *, cwd):
        del cwd
        calls.append(cmd)
        if "--build" in cmd:
            lib_dir = cache_root / "support" / "cpu" / "full" / "lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            (lib_dir / cpu_backend._shared_library_name("dinoml_runtime")).write_bytes(b"runtime")
            (lib_dir / cpu_backend._shared_library_name("dinoml_cpu_kernels")).write_bytes(b"kernels")
            if cpu_backend.os.name == "nt":
                (lib_dir / "dinoml_runtime.lib").write_bytes(b"implib")

    monkeypatch.setenv("DINOML_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(cpu_backend, "cmake_parallel_args", lambda parallel=None: ["--parallel", "7"])
    monkeypatch.setattr(cpu_backend, "_run_cmake", fake_run_cmake)
    monkeypatch.setenv("CMAKE_GENERATOR", "Ninja")

    cpu_backend.ensure_cpu_support_libs()

    assert calls[1][-2:] == ["--parallel", "7"]


def test_cuda_support_build_passes_explicit_cmake_parallelism(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    calls = []

    def fake_run_cmake(cmd, *, cwd):
        del cwd
        calls.append(cmd)
        if "--build" in cmd:
            lib_dir = cache_root / "support" / "cuda-120" / "full" / "lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            (lib_dir / "libdinoml_runtime.so").write_bytes(b"runtime")
            (lib_dir / "libdinoml_cuda_runtime.so").write_bytes(b"cuda-runtime")
            (lib_dir / "libdinoml_cuda_kernels.so").write_bytes(b"kernels")

    monkeypatch.setenv("DINOML_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(cuda_backend, "cmake_parallel_args", lambda parallel=None: ["--parallel", "7"])
    monkeypatch.setattr(cuda_backend, "_run_cmake", fake_run_cmake)

    cuda_backend.ensure_cuda_support_libs("sm_120")

    assert calls[1][-2:] == ["--parallel", "7"]


def test_rocm_support_build_passes_explicit_cmake_parallelism(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    calls = []

    def fake_run_cmake(cmd, *, cwd):
        del cwd
        calls.append(cmd)
        if "--build" in cmd:
            lib_dir = cache_root / "support" / "rocm-gfx1201" / "full" / "lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            for name in ("dinoml_runtime", "dinoml_rocm_runtime", "dinoml_rocm_kernels"):
                (lib_dir / rocm_backend._shared_library_name(name)).write_bytes(name.encode("utf-8"))

    monkeypatch.setenv("DINOML_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(rocm_backend, "cmake_parallel_args", lambda parallel=None: ["--parallel", "7"])
    monkeypatch.setattr(rocm_backend, "_prepare_cmake_build_dir", lambda _build_dir: None)
    monkeypatch.setattr(rocm_backend, "_run_cmake", fake_run_cmake)

    rocm_backend.ensure_rocm_support_libs("gfx1201")

    assert calls[1][-2:] == ["--parallel", "7"]


def test_cuda_flash_attention_build_preserves_parallel_override(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    calls = []

    def fake_run_cmake(cmd, *, cwd):
        del cwd
        calls.append(cmd)
        if "--build" in cmd:
            lib_dir = cache_root / "support" / "cuda-120" / "flash-attn-cuda" / "cmake-full" / "lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            (lib_dir / cuda_backend.flash_attn_cuda_static_library_name("float16")).write_bytes(b"wrapper")
            (lib_dir / cuda_backend.flash_attn_cuda_upstream_static_library_name()).write_bytes(b"upstream")

    monkeypatch.setenv("DINOML_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("DINOML_CUDA_FLASH_ATTN_BUILD_PARALLEL", "3")
    monkeypatch.setattr(cuda_backend, "require_cuda_library", lambda _name: None)
    monkeypatch.setattr(cuda_backend, "_prepare_cmake_build_dir", lambda _build_dir: None)
    monkeypatch.setattr(cuda_backend, "cmake_parallel_args", lambda parallel=None: ["--parallel", str(parallel or "7")])
    monkeypatch.setattr(cuda_backend, "_run_cmake", fake_run_cmake)

    cuda_backend._ensure_cmake_flash_attn_cuda_archives("sm_120", {"required_kernels": []})

    assert calls[1][-2:] == ["--parallel", "3"]


def test_cutlass_gemm_support_build_can_request_profiler_bindings(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    calls = []

    def fake_run_cmake(cmd, *, cwd):
        del cwd
        calls.append(cmd)
        if "--build" in cmd:
            lib_dir = cache_root / "support" / "cuda-89" / "cutlass-gemm" / "cmake-full" / "lib"
            lib_dir.mkdir(parents=True, exist_ok=True)
            (lib_dir / "libdinoml_cutlass_dual_gemm_rcr_fast_gelu_float16.a").write_bytes(b"archive")

    monkeypatch.setenv("DINOML_CACHE_DIR", str(cache_root))
    monkeypatch.setenv("DINOML_BUILD_CUTLASS_GEMM_PROFILERS", "1")
    monkeypatch.setattr(cuda_backend, "require_cuda_library", lambda _name: None)
    monkeypatch.setattr(cuda_backend, "_repo_root", lambda: tmp_path)
    monkeypatch.setattr(cuda_backend, "_prepare_cmake_build_dir", lambda _build_dir: None)
    monkeypatch.setattr(cuda_backend, "_cutlass_gemm_source_sha256", lambda _repo_root: "fake-sha")
    monkeypatch.setattr(cuda_backend, "cmake_parallel_args", lambda parallel=None: ["--parallel", str(parallel or "7")])
    monkeypatch.setattr(cuda_backend, "_run_cmake", fake_run_cmake)

    cuda_backend._ensure_cmake_cutlass_gemm_archives(
        "sm_89",
        {
            "required_kernels": [
                {
                    "kernel_library": "cutlass_gemm",
                    "op": "dual_gemm_rcr_fast_gelu",
                    "dtype": "float16",
                }
            ]
        },
    )

    assert any(
        arg == "-DDINOML_CUTLASS_GEMM_PROFILER_TARGETS=dinoml_cutlass_gemm_dual_gemm_rcr_fast_gelu_float16"
        for arg in calls[0]
    )
    assert "dinoml_cutlass_gemm_profiler_dual_gemm_rcr_fast_gelu_float16_bind" in calls[1]
