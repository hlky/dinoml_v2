import pytest
import shutil
import ctypes
import hashlib

import dinoml as dml
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.backends.cutlass import ensure_cutlass_gemm_support_lib
from dinoml.backends.registry import BackendSpec, get_backend_spec, registered_backend_names, registered_backend_specs
from dinoml.ir import read_json


class Identity(dml.Module):
    def forward(self, x):
        return dml.ops.output(x, "y")


def test_target_defaults_are_loaded_from_backend_registry():
    assert dml.Target("cuda").arch == "sm_86"
    assert dml.Target("cuda", arch="sm_90").to_json() == {"name": "cuda", "arch": "sm_90"}

    assert dml.Target("cpu").arch == "native"
    assert dml.Target("cpu", arch="sm_86").arch == "native"
    assert dml.Target("cpu", arch="x86_64").to_json() == {"name": "cpu", "arch": "x86_64"}


def test_target_rejects_names_missing_from_backend_registry():
    with pytest.raises(ValueError, match="Unsupported DinoML target 'metal'.*cpu, cuda"):
        dml.Target("metal")


def test_backend_registry_describes_cpu_and_cuda_support():
    assert registered_backend_names() == ("cpu", "cuda")
    assert [spec.name for spec in registered_backend_specs()] == ["cpu", "cuda"]

    cpu = get_backend_spec("cpu")
    assert cpu.default_arch == "native"
    assert cpu.supported_dtypes == frozenset({"float16", "float32", "bfloat16"})
    assert cpu.build_function == "dinoml.backends.cpu.build_cpu_module"
    assert cpu.cmake.supports_openmp is True
    assert cpu.cmake.requires_cuda is False
    assert cpu.support_libraries == {
        "runtime_library": "lib/libdinoml_runtime.so",
        "kernel_library": "lib/libdinoml_cpu_kernels.so",
    }

    cuda = get_backend_spec("cuda")
    assert cuda.default_arch == "sm_86"
    assert cuda.supported_dtypes == frozenset({"float16", "float32", "bfloat16"})
    assert cuda.build_function == "dinoml.backends.cuda.build_cuda_module"
    assert cuda.cmake.requires_cuda is True
    assert cuda.cmake.supports_cuda_fast_math is True
    assert cuda.support_libraries == {
        "runtime_library": "lib/libdinoml_runtime.so",
        "cuda_runtime_library": "lib/libdinoml_cuda_runtime.so",
        "kernel_library": "lib/libdinoml_cuda_kernels.so",
    }


def test_backend_registry_build_functions_resolve_to_callables():
    assert get_backend_spec("cpu").resolve_build_function().__name__ == "build_cpu_module"
    assert get_backend_spec("cuda").resolve_build_function().__name__ == "build_cuda_module"


def test_cuda_library_discovery_reports_expected_keys():
    libraries = discover_cuda_libraries()
    assert sorted(libraries) == ["cub", "cublaslt", "cuda", "cudnn", "cutlass"]
    for library in libraries.values():
        payload = library.to_json()
        assert payload["name"] == library.name
        assert isinstance(payload["available"], bool)
        assert isinstance(payload["include_roots"], list)
    assert libraries["cutlass"].headers == ("cutlass/gemm/device/gemm_universal.h",)


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cutlass_gemm_support_library_builds_once(tmp_path, monkeypatch):
    libraries = discover_cuda_libraries()
    if not libraries["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    support = ensure_cutlass_gemm_support_lib("sm_86", cache_key="test-cutlass")

    assert support.library.exists()
    assert support.source.exists()
    assert support.manifest.exists()
    assert support.source_manifest.exists()
    manifest = read_json(support.manifest)
    source_manifest = read_json(support.source_manifest)
    assert manifest["schema_version"] == 2
    assert manifest["provider"] == "cutlass"
    assert manifest["library_sha256"] == hashlib.sha256(support.library.read_bytes()).hexdigest()
    assert len(manifest["source_sha256"]) == 64
    assert manifest["source_manifest"] == "../src/source_manifest.json"
    assert source_manifest["schema_version"] == 2
    assert source_manifest["kind"] == "dinoml.support_source_manifest"
    assert source_manifest["provider"] == "cutlass"
    assert source_manifest["family_cache_key"] == manifest["family_cache_key"]
    assert source_manifest["used_candidate_plan_key"] == manifest["used_candidate_plan_key"]
    assert source_manifest["used_candidate_plan"]["used_candidate_plan_key"] == manifest["used_candidate_plan_key"]
    assert len(manifest["used_candidate_plan_key"]) == 64
    assert len(source_manifest["used_candidate_plan"]["entries"]) == 18
    assert len(source_manifest["source_manifest_key"]) == 64
    static_source = next(item for item in source_manifest["sources"] if item["source_id"] == "cutlass_gemm_static_default")
    assert static_source["emitted_source_path"] == support.source.name
    assert static_source["source_key"] == manifest["source_sha256"]
    assert static_source["source_sha256"] == manifest["source_sha256"]
    assert len(static_source["repo_source_sha256"]) == 64
    assert sorted({item["candidate_set_key"] for item in source_manifest["candidate_sets"]}) == static_source["candidate_set_keys"]
    assert sorted({item["candidate_config_key"] for item in source_manifest["candidates"]}) == static_source["candidate_config_keys"]
    assert {"kernel", "profiler"} == {item["kind"] for item in static_source["symbols"]}
    assert source_manifest["build_units"][0]["source_ids"] == ["cutlass_gemm_static_default"]
    assert len(manifest["family_cache_key"]) == 64
    assert len(manifest["used_candidate_plan"]["candidate_set_keys"]) == 18
    assert len(manifest["used_candidate_plan"]["candidate_config_keys"]) == 18
    assert manifest["used_candidate_plan"]["kernel_symbols"] == sorted(
        symbol
        for family in manifest["families"]
        for symbol in family["kernel_symbols_by_dtype"].values()
    )
    assert len(manifest["build_fingerprint"]) == 64
    assert manifest["provenance_key"]
    assert manifest["build_fingerprint"] == manifest["provenance_key"]
    assert manifest["provenance"]["provenance_key"] == manifest["provenance_key"]
    assert manifest["provenance"]["family_cache_key"] == manifest["family_cache_key"]
    assert manifest["provenance"]["nvcc"]["available"] is True
    assert "-arch=sm_86" in manifest["compile"]["flags"]
    assert manifest["compile"]["flags"] == manifest["provenance"]["compile_flags"]
    assert manifest["provenance"]["dependencies"]["cutlass"]["headers"][0]["sha256"]
    assert manifest["provenance"]["dependencies"]["cublaslt"]["headers"][0]["sha256"]
    assert {family["op_name"] for family in manifest["families"]} == {
        "gemm_rcr",
        "gemm_rcr_bias",
        "gemm_rcr_bias_relu",
        "gemm_rrr",
        "gemm_rrr_bias",
        "gemm_rrr_bias_relu",
    }
    for family in manifest["families"]:
        assert sorted(family["kernel_symbols_by_dtype"]) == ["bfloat16", "float16", "float32"]
        assert sorted(family["profiler_symbols_by_dtype"]) == ["bfloat16", "float16", "float32"]
        assert sorted(family["candidates_by_dtype"]) == ["bfloat16", "float16", "float32"]
        assert sorted(family["candidate_sets_by_dtype"]) == ["bfloat16", "float16", "float32"]
        for dtype, candidates in family["candidates_by_dtype"].items():
            assert len(candidates) == 1
            candidate = candidates[0]
            candidate_set = family["candidate_sets_by_dtype"][dtype]
            assert candidate_set["candidate_count"] == 1
            assert candidate_set["candidate_config_keys"] == [candidate["candidate_config_key"]]
            assert len(candidate_set["candidate_set_key"]) == 64
            assert candidate["candidate_id"] == "cutlass_default"
            assert candidate["dtype"] == dtype
            assert candidate["kernel_symbol"] == family["kernel_symbols_by_dtype"][dtype]
            assert candidate["profiler_symbol"] == family["profiler_symbols_by_dtype"][dtype]
            assert len(candidate["candidate_config_key"]) == 64
    bias_candidates = [
        candidate
        for family in manifest["families"]
        if family["op_name"] == "gemm_rcr_bias"
        for candidate in family["candidates_by_dtype"]["float32"]
    ]
    assert bias_candidates[0]["epilogue"] == "bias"
    assert bias_candidates[0]["epilogue_config"]["inputs"] == ["bias"]
    assert bias_candidates[0]["launch_abi"] == "dinoml_cutlass_gemm_bias_v1"
    bias_relu_candidates = [
        candidate
        for family in manifest["families"]
        if family["op_name"] == "gemm_rcr_bias_relu"
        for candidate in family["candidates_by_dtype"]["float32"]
    ]
    assert bias_relu_candidates[0]["epilogue"] == "bias_relu"
    assert bias_relu_candidates[0]["epilogue_config"]["inputs"] == ["bias"]
    assert bias_relu_candidates[0]["epilogue_config"]["activation"] == "relu"
    assert bias_relu_candidates[0]["launch_abi"] == "dinoml_cutlass_gemm_bias_v1"


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cutlass_gemm_support_library_runs_rrr_and_rcr(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    support = ensure_cutlass_gemm_support_lib("sm_86", cache_key="test-cutlass-run")
    dll = ctypes.CDLL(str(support.library))
    for name in (
        "dinoml_cutlass_gemm_rrr_f32",
        "dinoml_cutlass_gemm_rcr_f32",
        "dinoml_cutlass_gemm_rrr_f16",
        "dinoml_cutlass_gemm_rcr_f16",
        "dinoml_cutlass_gemm_rrr_bf16",
        "dinoml_cutlass_gemm_rcr_bf16",
    ):
        fn = getattr(dll, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int
    for name in (
        "dinoml_cutlass_gemm_rrr_bias_f32",
        "dinoml_cutlass_gemm_rcr_bias_f32",
        "dinoml_cutlass_gemm_rrr_bias_f16",
        "dinoml_cutlass_gemm_rcr_bias_f16",
        "dinoml_cutlass_gemm_rrr_bias_bf16",
        "dinoml_cutlass_gemm_rcr_bias_bf16",
        "dinoml_cutlass_gemm_rrr_bias_relu_f32",
        "dinoml_cutlass_gemm_rcr_bias_relu_f32",
        "dinoml_cutlass_gemm_rrr_bias_relu_f16",
        "dinoml_cutlass_gemm_rcr_bias_relu_f16",
        "dinoml_cutlass_gemm_rrr_bias_relu_bf16",
        "dinoml_cutlass_gemm_rcr_bias_relu_bf16",
    ):
        fn = getattr(dll, name)
        fn.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        fn.restype = ctypes.c_int

    torch.manual_seed(7)
    for torch_dtype, suffix, atol, rtol in (
        (torch.float32, "f32", 1e-4, 1e-4),
        (torch.float16, "f16", 2e-2, 2e-2),
        (torch.bfloat16, "bf16", 3e-2, 3e-2),
    ):
        a = torch.randn((16, 32), device="cuda", dtype=torch_dtype)
        b_rrr = torch.randn((32, 24), device="cuda", dtype=torch_dtype)
        c_rrr = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rrr_{suffix}")(
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b_rrr.data_ptr()),
            ctypes.c_void_p(c_rrr.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_rrr, a @ b_rrr, atol=atol, rtol=rtol)

        b_rcr = torch.randn((24, 32), device="cuda", dtype=torch_dtype)
        c_rcr = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_{suffix}")(
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b_rcr.data_ptr()),
            ctypes.c_void_p(c_rcr.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_rcr, a @ b_rcr.t(), atol=atol, rtol=rtol)

        bias = torch.randn((24,), device="cuda", dtype=torch_dtype)
        c_bias = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_{suffix}")(
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b_rcr.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()),
            ctypes.c_void_p(c_bias.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_bias, a @ b_rcr.t() + bias, atol=atol, rtol=rtol)

        c_bias_relu = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_relu_{suffix}")(
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(b_rcr.data_ptr()),
            ctypes.c_void_p(bias.data_ptr()),
            ctypes.c_void_p(c_bias_relu.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_bias_relu, torch.relu(a @ b_rcr.t() + bias), atol=atol, rtol=rtol)


def test_compile_uses_backend_registry_for_manifest_and_build_dispatch(tmp_path, monkeypatch):
    calls = []

    def fake_build(ir, *, target, artifact_dir, generated_src_dir, kernel_manifest):
        calls.append(
            {
                "target": target.to_json(),
                "artifact_dir": artifact_dir,
                "generated_src_dir": generated_src_dir,
                "kernel_manifest": kernel_manifest,
            }
        )

    monkeypatch.setattr(BackendSpec, "resolve_build_function", lambda self: fake_build)

    spec = dml.trace(Identity(), inputs={"x": dml.TensorSpec([1, 4], "float32")}, name="registry_dispatch")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "registry_dispatch.dinoml")

    manifest = read_json(artifact.path / "manifest.json")
    assert manifest["files"]["runtime_library"] == get_backend_spec("cpu").support_libraries["runtime_library"]
    assert manifest["files"]["kernel_library"] == get_backend_spec("cpu").support_libraries["kernel_library"]
    assert calls == [
        {
            "target": {"name": "cpu", "arch": "native"},
            "artifact_dir": artifact.path,
            "generated_src_dir": artifact.path / "debug" / "generated_src",
            "kernel_manifest": read_json(artifact.path / "kernel_manifest.json"),
        }
    ]
