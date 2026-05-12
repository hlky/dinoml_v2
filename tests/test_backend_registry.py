import ctypes
import hashlib
import shutil
from collections import Counter

import pytest

import dinoml as dml
import dinoml.backends.build_parallelism as build_parallelism
import dinoml.backends.cutlass as cutlass_backend
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.backends.cutlass import ensure_cutlass_gemm_support_lib
from dinoml.backends.registry import BackendSpec, get_backend_spec, registered_backend_names, registered_backend_specs
from dinoml.ir import canonical_json, read_json
from dinoml.kernels.bmm import BMM_OPS
from dinoml.kernels.gemm import GEMM_OPS
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_candidates
from dinoml.kernels.providers.cutlass.gemm import (
    CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE,
    cutlass_gemm_candidate_set,
    cutlass_gemm_candidates,
    cutlass_gemm_used_candidate_plan,
)
from dinoml.passes import PassManager


FLOAT32_CANDIDATE_MATH_COUNTS = {
    "tf32": 57,
    "fast_f16": 57,
    "fast_bf16": 57,
    "tf32_fast_f32": 39,
    "f32": 11,
}
FLOAT32_OPTIONAL_FAST_OPERATOR_BY_MATH = {
    "fast_f16": "multiply_add_fast_f16",
    "fast_bf16": "multiply_add_fast_bf16",
    "tf32_fast_f32": "multiply_add_fast_f32",
}
FLOAT32_OPTIONAL_MATH_COUNTS = {
    math: count for math, count in FLOAT32_CANDIDATE_MATH_COUNTS.items() if math != "f32"
}
SPLIT_K_LAUNCH_ABIS = {"dinoml_cutlass_gemm_v1", "dinoml_cutlass_gemm_bias_v1"}
SPLIT_K_RESIDUAL_EPILOGUES = {"bias_add", "bias_add_add", "bias_add_relu", "bias_add_add_relu"}
SPLIT_K_RESIDUAL_LAUNCH_ABIS = {
    "dinoml_cutlass_gemm_bias_residual_v1",
    "dinoml_cutlass_gemm_bias_residual2_v1",
}
SM80_TENSOROP_ACCESS_SIZE_BITS = 128
SM80_TENSOROP_WARP_SIZE = 32
SM80_TENSOROP_MAX_WARP_THREAD_CONTIGUOUS = 8


def _assert_float32_candidate_math_families(candidates, *, expect_simt: bool = True):
    expected_counts = FLOAT32_CANDIDATE_MATH_COUNTS if expect_simt else FLOAT32_OPTIONAL_MATH_COUNTS
    assert len(candidates) == sum(expected_counts.values())
    assert Counter(item["cutlass"]["math"] for item in candidates) == Counter(expected_counts)
    assert Counter(item["cutlass"]["math"] for item in candidates if item["optional"]) == Counter(
        FLOAT32_OPTIONAL_MATH_COUNTS
    )
    if expect_simt:
        assert Counter(item["cutlass"]["math"] for item in candidates if not item["optional"]) == Counter({"f32": 11})
        assert {item["cutlass"]["opclass"] for item in candidates if not item["optional"]} == {"simt"}
    else:
        assert all(item["optional"] for item in candidates)
    assert {item["cutlass"]["opclass"] for item in candidates if item["optional"]} == {"tensorop"}
    for math, math_operator in FLOAT32_OPTIONAL_FAST_OPERATOR_BY_MATH.items():
        assert {
            item["cutlass"].get("math_operator", "multiply_add")
            for item in candidates
            if item["cutlass"]["math"] == math
        } == {math_operator}


def _cutlass_candidate_count(dtype: str, op_name: str = "gemm_rrr") -> int:
    return len(cutlass_gemm_candidates(op_name, dtype))


def _execution_plan_key(plan):
    return hashlib.sha256(
        canonical_json({key: value for key, value in plan.items() if key != "execution_plan_key"}).encode("utf-8")
    ).hexdigest()


def _cutlass_candidate_ids(dtype: str, op_name: str = "gemm_rrr") -> list[str]:
    return [str(candidate["candidate_id"]) for candidate in cutlass_gemm_candidates(op_name, dtype)]


def _cutlass_default_symbol_id(dtype: str) -> str:
    return str(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[dtype][0]["symbol_id"])


def _tiny_cutlass_used_candidate_plan(target=None):
    target = target or {"name": "cuda", "arch": "sm_86"}
    candidates = [dict(cutlass_gemm_candidates("gemm_rrr", "float32", target=target)[0])]
    candidate_set = cutlass_gemm_candidate_set("gemm_rrr", "float32", target=target)
    required = {
        "op": "gemm_rrr",
        "kernel_symbol": candidates[0]["kernel_symbol"],
        "kernel_library": "cutlass_gemm",
        "profiler_symbol": candidates[0]["profiler_symbol"],
        "selected_candidate_id": candidates[0]["candidate_id"],
        "candidates": candidates,
        "candidate_set_id": candidate_set["candidate_set_id"],
        "candidate_set_key": candidate_set["candidate_set_key"],
        "candidate_set": candidate_set,
    }
    return cutlass_gemm_used_candidate_plan(
        {
            "target": dict(target),
            "cache_key": "test-tiny-cutlass-cache-key",
            "support_cache_key": "test-tiny-cutlass-support-key",
            "required_kernels": [required],
        }
    )


def _assert_split_k_metadata(payload, launch_abi: str | None = None) -> None:
    launch_abi = str(payload["launch_abi"] if launch_abi is None else launch_abi)
    epilogue = str(payload.get("epilogue", ""))
    supports_split_k = launch_abi in SPLIT_K_LAUNCH_ABIS or (
        launch_abi in SPLIT_K_RESIDUAL_LAUNCH_ABIS and epilogue in SPLIT_K_RESIDUAL_EPILOGUES
    )
    assert payload["split_k_values"] == [1]
    assert payload["split_k_default"] == 1
    assert payload["supports_split_k"] is supports_split_k
    assert payload["workspace_nbytes"] == 0
    if supports_split_k:
        assert payload["split_k_search"] == {"strategy": "v1_gemm_factor", "max_split_k": 32}
    else:
        assert "split_k_search" not in payload


def _assert_sm80_tensorop_thread_map_divisible(candidate) -> None:
    cutlass = candidate["cutlass"]
    if cutlass["opclass"] != "tensorop":
        return
    m, n, k = [int(dim) for dim in cutlass["threadblock"]]
    elements_per_access = SM80_TENSOROP_ACCESS_SIZE_BITS // _dtype_bits(str(candidate["dtype"]))
    layouts = candidate["layouts"]
    assert _sm80_operand_thread_map_divisible(
        "a",
        str(layouts["a"]),
        m,
        n,
        k,
        elements_per_access,
    ), candidate["candidate_id"]
    assert _sm80_operand_thread_map_divisible(
        "b",
        str(layouts["b"]),
        m,
        n,
        k,
        elements_per_access,
    ), candidate["candidate_id"]


def _sm80_operand_thread_map_divisible(
    operand: str,
    layout: str,
    m: int,
    n: int,
    k: int,
    elements_per_access: int,
) -> bool:
    if operand == "a":
        shape_contiguous, shape_strided = (k, m) if layout == "row" else (m, k)
        warp_thread_contiguous = (
            k // elements_per_access
            if layout == "row"
            else min(m // elements_per_access, SM80_TENSOROP_MAX_WARP_THREAD_CONTIGUOUS)
        )
    else:
        shape_contiguous, shape_strided = (n, k) if layout == "row" else (k, n)
        warp_thread_contiguous = (
            min(n // elements_per_access, SM80_TENSOROP_MAX_WARP_THREAD_CONTIGUOUS)
            if layout == "row"
            else k // elements_per_access
        )
    if warp_thread_contiguous <= 0 or SM80_TENSOROP_WARP_SIZE % warp_thread_contiguous:
        return False
    if shape_contiguous % elements_per_access:
        return False
    shape_in_accesses_contiguous = shape_contiguous // elements_per_access
    warp_thread_strided = SM80_TENSOROP_WARP_SIZE // warp_thread_contiguous
    return shape_in_accesses_contiguous % warp_thread_contiguous == 0 and shape_strided % warp_thread_strided == 0


def _dtype_bits(dtype: str) -> int:
    return 16 if dtype in {"float16", "bfloat16"} else 32


def test_cutlass_float32_candidate_registry_lists_v1_fast_math_families():
    _assert_float32_candidate_math_families(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE["float32"])


def test_cutlass_reduced_precision_gemm_candidates_prune_sm80_thread_map_rejects():
    target = {"name": "cuda", "arch": "sm_86"}
    rrr_float16 = cutlass_gemm_candidates("gemm_rrr", "float16", target=target)
    rcr_float16 = cutlass_gemm_candidates("gemm_rcr", "float16", target=target)
    rrr_bfloat16 = cutlass_gemm_candidates("gemm_rrr", "bfloat16", target=target)
    invalid_rrr_ids = {
        "cutlass_tensorop_sm80_16816_256x96x32_s2_w4x2x1_f32_align8",
        "cutlass_tensorop_sm80_16816_128x224x32_s4_w2x4x1_f32_align8",
        "cutlass_tensorop_sm80_16816_192x160x32_s3_w4x2x1_f32_align2",
        "cutlass_tensorop_sm80_16816_192x96x32_s3_w4x2x1_f32_align8",
    }

    assert len(rrr_float16) == 111
    assert len(rrr_bfloat16) == 111
    assert len(rcr_float16) == 138
    assert invalid_rrr_ids.isdisjoint({candidate["candidate_id"] for candidate in rrr_float16})
    assert invalid_rrr_ids.isdisjoint({candidate["candidate_id"] for candidate in rrr_bfloat16})
    assert invalid_rrr_ids.issubset({candidate["candidate_id"] for candidate in rcr_float16})


def test_cutlass_reduced_precision_candidates_satisfy_sm80_thread_map_divisibility():
    for op_name in GEMM_OPS:
        for dtype in ("float16", "bfloat16"):
            for candidate in cutlass_gemm_candidates(op_name, dtype):
                _assert_sm80_tensorop_thread_map_divisible(candidate)
    for op_name in BMM_OPS:
        for dtype in ("float16", "bfloat16"):
            for candidate in cutlass_bmm_candidates(op_name, dtype):
                _assert_sm80_tensorop_thread_map_divisible(candidate)


def test_cutlass_compile_flags_enable_bounded_split_compile(monkeypatch):
    monkeypatch.setattr(cutlass_backend, "_nvcc_supports_option", lambda option: option == "--split-compile")
    monkeypatch.setattr(cutlass_backend, "effective_cpu_count", lambda: 6)

    assert "--split-compile=6" in cutlass_backend._compile_flags("86")
    monkeypatch.setenv("DINOML_NVCC_SPLIT_COMPILE", "4")
    assert "--split-compile=4" in cutlass_backend._compile_flags("86")
    monkeypatch.setenv("DINOML_NVCC_SPLIT_COMPILE", "1")
    assert not any(flag.startswith("--split-compile") for flag in cutlass_backend._compile_flags("86"))


def test_effective_cpu_count_caps_physical_cores_by_cgroup_quota(monkeypatch):
    monkeypatch.setattr(build_parallelism, "_logical_cpu_count", lambda: 32)
    monkeypatch.setattr(build_parallelism, "_linux_physical_cpu_count", lambda: 16)
    monkeypatch.setattr(build_parallelism, "_linux_cgroup_cpu_quota_count", lambda cpu_count: 8)

    assert build_parallelism.effective_cpu_count() == 8


class Identity(dml.Module):
    def forward(self, x):
        return dml.ops.output(x, "y")


class GemmRRR(dml.Module):
    def forward(self, a, b):
        return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")


def test_target_defaults_are_loaded_from_backend_registry():
    assert dml.Target("cuda").arch == "sm_86"
    assert dml.Target("cuda", arch="sm_90").to_json() == {
        "name": "cuda",
        "arch": "sm_90",
        "no_tf32": False,
        "use_fp16_acc": False,
    }
    assert dml.Target("cuda", no_tf32=True, use_fp16_acc=True).to_json() == {
        "name": "cuda",
        "arch": "sm_86",
        "no_tf32": True,
        "use_fp16_acc": True,
    }

    assert dml.Target("cpu").arch == "native"
    assert dml.Target("cpu", arch="sm_86").arch == "native"
    assert dml.Target("cpu", arch="x86_64").to_json() == {
        "name": "cpu",
        "arch": "x86_64",
        "no_tf32": False,
        "use_fp16_acc": False,
    }


def test_target_rejects_names_missing_from_backend_registry():
    with pytest.raises(ValueError, match="Unsupported DinoML target 'metal'.*cpu, cuda"):
        dml.Target("metal")


def test_backend_registry_describes_cpu_and_cuda_support():
    assert registered_backend_names() == ("cpu", "cuda")
    assert [spec.name for spec in registered_backend_specs()] == ["cpu", "cuda"]

    cpu = get_backend_spec("cpu")
    assert cpu.default_arch == "native"
    assert cpu.supported_dtypes == frozenset({"float16", "float32", "bfloat16", "bool"})
    assert cpu.build_function == "dinoml.backends.cpu.build_cpu_module"
    assert cpu.cmake.supports_openmp is True
    assert cpu.cmake.requires_cuda is False
    assert cpu.support_libraries == {
        "runtime_library": "lib/libdinoml_runtime.so",
        "kernel_library": "lib/libdinoml_cpu_kernels.so",
    }

    cuda = get_backend_spec("cuda")
    assert cuda.default_arch == "sm_86"
    assert cuda.supported_dtypes == frozenset({"float16", "float32", "bfloat16", "bool"})
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
    used_candidate_plan = _tiny_cutlass_used_candidate_plan()

    support = ensure_cutlass_gemm_support_lib(
        "sm_86",
        cache_key="test-cutlass",
        used_candidate_plan=used_candidate_plan,
    )

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
    assert manifest["used_candidate_plan_key"] == used_candidate_plan["used_candidate_plan_key"]
    expected_gemm_entry_count = 1
    assert len(source_manifest["used_candidate_plan"]["entries"]) == expected_gemm_entry_count
    assert len(source_manifest["source_manifest_key"]) == 64
    static_source = next(item for item in source_manifest["sources"] if item["source_id"] == "cutlass_gemm_static_default")
    rendered_source = support.source.read_text(encoding="utf-8")
    assert static_source["emitted_source_path"] == support.source.name
    assert static_source["source_key"] == manifest["source_sha256"]
    assert static_source["source_sha256"] == manifest["source_sha256"]
    assert len(static_source["repo_source_sha256"]) == 64
    assert static_source["source_metrics"]["candidate_count"] == 1
    assert static_source["source_metrics"]["entry_count"] == 1
    assert static_source["source_metrics"]["candidate_set_count"] == 1
    assert static_source["source_metrics"]["kernel_symbol_count"] == 1
    assert static_source["source_metrics"]["profiler_symbol_count"] == 1
    assert static_source["source_metrics"]["symbol_count"] == 2
    assert static_source["source_metrics"]["source_line_count"] == rendered_source.count("\n")
    assert static_source["source_metrics"]["source_nbytes"] == len(rendered_source.encode("utf-8"))
    assert sorted({item["candidate_set_key"] for item in source_manifest["candidate_sets"]}) == static_source["candidate_set_keys"]
    assert sorted({item["candidate_config_key"] for item in source_manifest["candidates"]}) == static_source["candidate_config_keys"]
    assert {"kernel", "profiler"} == {item["kind"] for item in static_source["symbols"]}
    assert source_manifest["build_units"][0]["source_ids"] == ["cutlass_gemm_static_default"]
    assert len(manifest["family_cache_key"]) == 64
    assert len(manifest["used_candidate_plan"]["candidate_set_keys"]) == expected_gemm_entry_count
    assert len(manifest["used_candidate_plan"]["candidate_config_keys"]) == expected_gemm_entry_count
    assert manifest["used_candidate_plan"]["kernel_symbols"] == used_candidate_plan["kernel_symbols"]
    assert "DINOML_FORWARD_GEMM_EXPORT(gemm_rrr, float32" in rendered_source
    assert "DINOML_FORWARD_GEMM_EXPORT(gemm_rcr, float32" not in rendered_source
    assert len(manifest["build_fingerprint"]) == 64
    assert manifest["provenance_key"]
    assert manifest["build_fingerprint"] == manifest["provenance_key"]
    assert manifest["provenance"]["provenance_key"] == manifest["provenance_key"]
    assert manifest["provenance"]["family_cache_key"] == manifest["family_cache_key"]
    assert manifest["provenance"]["nvcc"]["available"] is True
    assert "-arch=sm_86" in manifest["compile"]["flags"]
    assert manifest["compile"]["flags"] == manifest["provenance"]["compile_flags"]
    assert manifest["compile"]["duration_ms"] >= 0
    assert manifest["compile"]["source_metrics"] == static_source["source_metrics"]
    assert manifest["provenance"]["dependencies"]["cutlass"]["headers"][0]["sha256"]
    assert manifest["provenance"]["dependencies"]["cublaslt"]["headers"][0]["sha256"]
    assert {family["op_name"] for family in manifest["families"]} == set(GEMM_OPS)
    for family in manifest["families"]:
        op_name = str(family["op_name"])
        assert sorted(family["kernel_symbols_by_dtype"]) == ["bfloat16", "float16", "float32"]
        assert sorted(family["profiler_symbols_by_dtype"]) == ["bfloat16", "float16", "float32"]
        assert sorted(family["candidates_by_dtype"]) == ["bfloat16", "float16", "float32"]
        assert sorted(family["candidate_sets_by_dtype"]) == ["bfloat16", "float16", "float32"]
        for dtype, candidates in family["candidates_by_dtype"].items():
            assert len(candidates) == _cutlass_candidate_count(dtype, op_name)
            candidate = candidates[0]
            candidate_set = family["candidate_sets_by_dtype"][dtype]
            assert candidate_set["candidate_count"] == _cutlass_candidate_count(dtype, op_name)
            assert candidate_set["candidate_config_keys"] == [
                item["candidate_config_key"] for item in candidates
            ]
            assert len(candidate_set["candidate_set_key"]) == 64
            _assert_split_k_metadata(candidate_set, str(candidate_set["launch_abi"]))
            assert [item["candidate_id"] for item in candidates] == _cutlass_candidate_ids(dtype, op_name)
            assert candidate["dtype"] == dtype
            assert candidate["kernel_symbol"] == family["kernel_symbols_by_dtype"][dtype]
            assert candidate["profiler_symbol"] == family["profiler_symbols_by_dtype"][dtype]
            _assert_split_k_metadata(candidate, str(candidate["launch_abi"]))
            assert candidate["cutlass"]["opclass"] == "tensorop"
            assert candidate["cutlass"]["arch"] == "sm80"
            assert candidate["optional"] is (dtype == "float32")
            assert candidate["cutlass"]["threadblock"] == ([256, 128, 16] if dtype == "float32" else [256, 128, 32])
            assert candidates[1]["cutlass"]["opclass"] == "tensorop"
            assert candidates[1]["cutlass"]["align"] in ({1, 2, 4} if dtype == "float32" else {2, 4, 8})
            assert candidates[-1]["cutlass"]["threadblock"] == ([32, 128, 8] if dtype == "float32" else [96, 192, 32])
            if dtype == "float32":
                _assert_float32_candidate_math_families(candidates)
                assert {item["cutlass"]["instruction"][0] for item in candidates if item["cutlass"]["opclass"] == "simt"} == {1}
            if dtype == "float16":
                assert {item["accumulator_dtype"] for item in candidates} == {"float16", "float32"}
            else:
                assert {item["accumulator_dtype"] for item in candidates} == {"float32"}
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
    bias_gelu_candidates = [
        candidate
        for family in manifest["families"]
        if family["op_name"] == "gemm_rcr_bias_gelu"
        for candidate in family["candidates_by_dtype"]["float32"]
    ]
    assert bias_gelu_candidates[0]["epilogue"] == "bias_gelu"
    assert bias_gelu_candidates[0]["epilogue_config"]["inputs"] == ["bias"]
    assert bias_gelu_candidates[0]["epilogue_config"]["activation"] == "gelu"
    assert bias_gelu_candidates[0]["launch_abi"] == "dinoml_cutlass_gemm_bias_v1"
    bias_elup1_candidates = [
        candidate
        for family in manifest["families"]
        if family["op_name"] == "gemm_rcr_bias_elup1"
        for candidate in family["candidates_by_dtype"]["float32"]
    ]
    assert bias_elup1_candidates[0]["epilogue"] == "bias_elup1"
    assert bias_elup1_candidates[0]["epilogue_config"]["inputs"] == ["bias"]
    assert bias_elup1_candidates[0]["epilogue_config"]["activation"] == "elup1"
    assert bias_elup1_candidates[0]["epilogue_config"]["cutlass_functor"] == "cutlass::epilogue::thread::LinearCombinationELUp1"
    assert bias_elup1_candidates[0]["launch_abi"] == "dinoml_cutlass_gemm_bias_v1"
    residual_ops = {
        f"gemm_{layout}_bias_{suffix}": (inputs, launch_abi)
        for layout in ("rcr", "rrr")
        for suffix, inputs, launch_abi in (
            ("add", ["bias", "d0"], "dinoml_cutlass_gemm_bias_residual_v1"),
            ("add_add", ["bias", "d0", "d1"], "dinoml_cutlass_gemm_bias_residual2_v1"),
            ("mul", ["bias", "d0"], "dinoml_cutlass_gemm_bias_residual_v1"),
            ("mul_add", ["bias", "d0", "d1"], "dinoml_cutlass_gemm_bias_residual2_v1"),
        )
    }
    residual_ops.update(
        {
            f"gemm_{layout}_bias_{suffix}": (inputs, launch_abi)
            for layout in ("rcr", "rrr")
            for suffix, inputs, launch_abi in (
                ("add_relu", ["bias", "d0"], "dinoml_cutlass_gemm_bias_residual_v1"),
                ("add_add_relu", ["bias", "d0", "d1"], "dinoml_cutlass_gemm_bias_residual2_v1"),
                ("mul_tanh", ["bias", "d0"], "dinoml_cutlass_gemm_bias_residual_v1"),
                ("sigmoid_mul", ["bias", "d0"], "dinoml_cutlass_gemm_bias_residual_v1"),
                ("sigmoid_mul_tanh", ["bias", "d0"], "dinoml_cutlass_gemm_bias_residual_v1"),
            )
        }
    )
    residual_candidates = {
        family["op_name"]: family["candidates_by_dtype"]["float32"][0]
        for family in manifest["families"]
        if family["op_name"] in residual_ops
    }
    assert sorted(residual_candidates) == sorted(residual_ops)
    for op_name, (inputs, launch_abi) in residual_ops.items():
        assert residual_candidates[op_name]["epilogue_config"]["inputs"] == inputs
        assert residual_candidates[op_name]["launch_abi"] == launch_abi


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
    symbol_suffixes = {
        "float32": f"float32_{_cutlass_default_symbol_id("float32")}",
        "float16": f"float16_{_cutlass_default_symbol_id("float16")}",
        "bfloat16": f"bfloat16_{_cutlass_default_symbol_id("bfloat16")}",
    }
    for name in (
        f"dinoml_cutlass_gemm_rrr_float32_{_cutlass_default_symbol_id("float32")}",
        f"dinoml_cutlass_gemm_rcr_float32_{_cutlass_default_symbol_id("float32")}",
        f"dinoml_cutlass_gemm_rrr_float16_{_cutlass_default_symbol_id("float16")}",
        f"dinoml_cutlass_gemm_rcr_float16_{_cutlass_default_symbol_id("float16")}",
        f"dinoml_cutlass_gemm_rrr_bfloat16_{_cutlass_default_symbol_id("bfloat16")}",
        f"dinoml_cutlass_gemm_rcr_bfloat16_{_cutlass_default_symbol_id("bfloat16")}",
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
        f"dinoml_cutlass_gemm_rrr_bias_float32_{_cutlass_default_symbol_id("float32")}",
        f"dinoml_cutlass_gemm_rcr_bias_float32_{_cutlass_default_symbol_id("float32")}",
        f"dinoml_cutlass_gemm_rrr_bias_float16_{_cutlass_default_symbol_id("float16")}",
        f"dinoml_cutlass_gemm_rcr_bias_float16_{_cutlass_default_symbol_id("float16")}",
        f"dinoml_cutlass_gemm_rrr_bias_bfloat16_{_cutlass_default_symbol_id("bfloat16")}",
        f"dinoml_cutlass_gemm_rcr_bias_bfloat16_{_cutlass_default_symbol_id("bfloat16")}",
        f"dinoml_cutlass_gemm_rrr_bias_relu_float32_{_cutlass_default_symbol_id("float32")}",
        f"dinoml_cutlass_gemm_rcr_bias_relu_float32_{_cutlass_default_symbol_id("float32")}",
        f"dinoml_cutlass_gemm_rrr_bias_relu_float16_{_cutlass_default_symbol_id("float16")}",
        f"dinoml_cutlass_gemm_rcr_bias_relu_float16_{_cutlass_default_symbol_id("float16")}",
        f"dinoml_cutlass_gemm_rrr_bias_relu_bfloat16_{_cutlass_default_symbol_id("bfloat16")}",
        f"dinoml_cutlass_gemm_rcr_bias_relu_bfloat16_{_cutlass_default_symbol_id("bfloat16")}",
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
    for layout in ("rcr", "rrr"):
        single_source_suffixes = (
            ("add", "add_relu", "mul", "mul_tanh", "sigmoid_mul", "sigmoid_mul_tanh")
            if layout == "rcr"
            else ("add", "mul")
        )
        for op_suffix in single_source_suffixes:
            for dtype_name in ("float32", "float16", "bfloat16"):
                name = f"dinoml_cutlass_gemm_{layout}_bias_{op_suffix}_{dtype_name}_{_cutlass_default_symbol_id(dtype_name)}"
                fn = getattr(dll, name)
                fn.argtypes = [
                    ctypes.c_void_p,
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
        double_source_suffixes = ("add_add", "add_add_relu", "mul_add") if layout == "rcr" else ("add_add", "mul_add")
        for op_suffix in double_source_suffixes:
            for dtype_name in ("float32", "float16", "bfloat16"):
                name = f"dinoml_cutlass_gemm_{layout}_bias_{op_suffix}_{dtype_name}_{_cutlass_default_symbol_id(dtype_name)}"
                fn = getattr(dll, name)
                fn.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
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
        (torch.float32, symbol_suffixes["float32"], 1e-2, 1e-2),
        (torch.float16, symbol_suffixes["float16"], 2e-2, 2e-2),
        (torch.bfloat16, symbol_suffixes["bfloat16"], 3e-2, 3e-2),
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

        for layout, b_residual, b_reference in (("rcr", b_rcr, b_rcr.t()), ("rrr", b_rrr, b_rrr)):
            d0 = torch.randn((16, 24), device="cuda", dtype=torch_dtype)
            d1 = torch.randn((16, 24), device="cuda", dtype=torch_dtype)
            base = a @ b_reference + bias

            c_bias_add = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_add_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(c_bias_add.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_add, base + d0, atol=atol, rtol=rtol)

            c_bias_add_add = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_add_add_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(d1.data_ptr()),
                ctypes.c_void_p(c_bias_add_add.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_add_add, base + d0 + d1, atol=atol, rtol=rtol)

            c_bias_add_relu = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_add_relu_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(c_bias_add_relu.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_add_relu, torch.relu(base + d0), atol=atol, rtol=rtol)

            c_bias_add_add_relu = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_add_add_relu_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(d1.data_ptr()),
                ctypes.c_void_p(c_bias_add_add_relu.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_add_add_relu, torch.relu(base + d0 + d1), atol=atol, rtol=rtol)

            c_bias_mul_tanh = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_mul_tanh_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(c_bias_mul_tanh.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_mul_tanh, torch.tanh(base * d0), atol=atol, rtol=rtol)

            c_bias_sigmoid_mul = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_sigmoid_mul_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(c_bias_sigmoid_mul.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_sigmoid_mul, torch.sigmoid(base) * d0, atol=atol, rtol=rtol)

            c_bias_sigmoid_mul_tanh = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_sigmoid_mul_tanh_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(c_bias_sigmoid_mul_tanh.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_sigmoid_mul_tanh, torch.tanh(torch.sigmoid(base) * d0), atol=atol, rtol=rtol)

            c_bias_mul = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_mul_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(c_bias_mul.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_mul, base * d0, atol=atol, rtol=rtol)

            c_bias_mul_add = torch.empty((16, 24), device="cuda", dtype=torch_dtype)
            err = getattr(dll, f"dinoml_cutlass_gemm_{layout}_bias_mul_add_{suffix}")(
                ctypes.c_void_p(a.data_ptr()),
                ctypes.c_void_p(b_residual.data_ptr()),
                ctypes.c_void_p(bias.data_ptr()),
                ctypes.c_void_p(d0.data_ptr()),
                ctypes.c_void_p(d1.data_ptr()),
                ctypes.c_void_p(c_bias_mul_add.data_ptr()),
                ctypes.c_int(16),
                ctypes.c_int(24),
                ctypes.c_int(32),
                ctypes.c_void_p(0),
            )
            assert err == 0
            torch.cuda.synchronize()
            torch.testing.assert_close(c_bias_mul_add, base * d0 + d1, atol=atol, rtol=rtol)

        a_folded = torch.randn((2, 8, 32), device="cuda", dtype=torch_dtype)
        b_folded = torch.randn((24, 32), device="cuda", dtype=torch_dtype)
        bias_folded = torch.randn((24,), device="cuda", dtype=torch_dtype)
        d0_folded = torch.randn((2, 8, 24), device="cuda", dtype=torch_dtype)
        d1_folded = torch.randn((2, 8, 24), device="cuda", dtype=torch_dtype)
        folded_base = a_folded @ b_folded.t() + bias_folded

        c_folded_add = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_add_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(c_folded_add.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_folded_add, folded_base + d0_folded, atol=atol, rtol=rtol)

        c_folded_mul = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_mul_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(c_folded_mul.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_folded_mul, folded_base * d0_folded, atol=atol, rtol=rtol)

        c_folded_mul_tanh = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_mul_tanh_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(c_folded_mul_tanh.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_folded_mul_tanh, torch.tanh(folded_base * d0_folded), atol=atol, rtol=rtol)

        c_folded_sigmoid_mul = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_sigmoid_mul_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(c_folded_sigmoid_mul.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_folded_sigmoid_mul, torch.sigmoid(folded_base) * d0_folded, atol=atol, rtol=rtol)

        c_folded_sigmoid_mul_tanh = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_sigmoid_mul_tanh_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(c_folded_sigmoid_mul_tanh.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_folded_sigmoid_mul_tanh,
            torch.tanh(torch.sigmoid(folded_base) * d0_folded),
            atol=atol,
            rtol=rtol,
        )

        c_folded_add_add = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_add_add_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(d1_folded.data_ptr()),
            ctypes.c_void_p(c_folded_add_add.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_folded_add_add, folded_base + d0_folded + d1_folded, atol=atol, rtol=rtol)

        c_folded_mul_add = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_mul_add_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(d1_folded.data_ptr()),
            ctypes.c_void_p(c_folded_mul_add.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(c_folded_mul_add, folded_base * d0_folded + d1_folded, atol=atol, rtol=rtol)

        c_folded_add_add_relu = torch.empty((2, 8, 24), device="cuda", dtype=torch_dtype)
        err = getattr(dll, f"dinoml_cutlass_gemm_rcr_bias_add_add_relu_{suffix}")(
            ctypes.c_void_p(a_folded.data_ptr()),
            ctypes.c_void_p(b_folded.data_ptr()),
            ctypes.c_void_p(bias_folded.data_ptr()),
            ctypes.c_void_p(d0_folded.data_ptr()),
            ctypes.c_void_p(d1_folded.data_ptr()),
            ctypes.c_void_p(c_folded_add_add_relu.data_ptr()),
            ctypes.c_int(16),
            ctypes.c_int(24),
            ctypes.c_int(32),
            ctypes.c_void_p(0),
        )
        assert err == 0
        torch.cuda.synchronize()
        torch.testing.assert_close(
            c_folded_add_add_relu,
            torch.relu(folded_base + d0_folded + d1_folded),
            atol=atol,
            rtol=rtol,
        )


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
            "target": {"name": "cpu", "arch": "native", "no_tf32": False, "use_fp16_acc": False},
            "artifact_dir": artifact.path,
            "generated_src_dir": artifact.path / "debug" / "generated_src",
            "kernel_manifest": read_json(artifact.path / "kernel_manifest.json"),
        }
    ]


def test_compile_applies_execution_plan_before_build_and_codegen(tmp_path, monkeypatch):
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

    spec = dml.trace(
        GemmRRR(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="compile_profile_selected_gemm",
    )
    target = dml.Target("cuda", arch="sm_86")
    lowered, _ = PassManager().run(spec.ir)
    base_manifest = build_kernel_manifest(lowered, target.to_json())
    required = base_manifest["required_kernels"][0]
    selected_candidate = required["candidates"][1]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "target": target.to_json(),
        "kernel_manifest_cache_key": base_manifest["cache_key"],
        "selection_policy": "lowest_median_elapsed_ms_per_node_shape",
        "selection_confidence_policy": {"name": "test-confidence"},
        "static_selection_policy": "unique_selected_candidate_per_op_dtype_candidate_set",
        "summary": {"selection_count": 1, "low_confidence_count": 0, "static_selection_count": 1, "conflict_count": 0},
        "static_selections": [
            {
                "selection_key": "profile-selection",
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": selected_candidate["candidate_config_key"],
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "shape": {"m": 4, "n": 6, "k": 8},
                "avg_ms": 0.01,
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
    }
    execution_plan["execution_plan_key"] = _execution_plan_key(execution_plan)
    artifact = dml.compile(
        spec,
        target,
        tmp_path / "compile_profile_selected_gemm.dinoml",
        execution_plan=execution_plan,
    )

    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    codegen_plan = read_json(artifact.path / "kernel_codegen_plan.json")
    compile_config = read_json(artifact.path / "compile_config.json")
    manifest = read_json(artifact.path / "manifest.json")
    selected_required = kernel_manifest["required_kernels"][0]

    assert calls[0]["kernel_manifest"] == kernel_manifest
    assert selected_required["selected_candidate_id"] == selected_candidate["candidate_id"]
    assert selected_required["kernel_symbol"] == selected_candidate["kernel_symbol"]
    assert selected_required["profiler_symbol"] == selected_candidate["profiler_symbol"]
    assert codegen_plan["kernel_symbols"] == [selected_candidate["kernel_symbol"]]
    assert codegen_plan["profiler_symbols"] == [selected_candidate["profiler_symbol"]]
    assert compile_config["execution_plan"]["execution_plan_key"] == execution_plan["execution_plan_key"]
    assert manifest["execution_plan"] == compile_config["execution_plan"]
    assert compile_config["execution_plan"]["selection_confidence_policy"] == {"name": "test-confidence"}
    assert read_json(artifact.path / "debug" / "execution_plan.json") == execution_plan


def test_compile_rejects_stale_execution_plan(tmp_path, monkeypatch):
    monkeypatch.setattr(BackendSpec, "resolve_build_function", lambda self: lambda **kwargs: None)
    spec = dml.trace(
        GemmRRR(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="compile_stale_profile_plan",
    )
    target = dml.Target("cuda", arch="sm_86")
    stale_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "target": target.to_json(),
        "kernel_manifest_cache_key": "stale-kernel-manifest-key",
        "static_selections": [
            {
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": "unused",
                "selected_candidate_id": "unused",
            }
        ],
    }

    with pytest.raises(ValueError, match="different kernel manifest"):
        dml.compile(
            spec,
            target,
            tmp_path / "compile_stale_profile_plan.dinoml",
            execution_plan=stale_plan,
        )


def test_compile_rejects_tampered_execution_plan_key(tmp_path, monkeypatch):
    monkeypatch.setattr(BackendSpec, "resolve_build_function", lambda self: lambda **kwargs: None)
    spec = dml.trace(
        GemmRRR(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="compile_tampered_profile_plan",
    )
    target = dml.Target("cuda", arch="sm_86")
    lowered, _ = PassManager().run(spec.ir)
    base_manifest = build_kernel_manifest(lowered, target.to_json())
    required = base_manifest["required_kernels"][0]
    selected_candidate = required["candidates"][1]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "target": target.to_json(),
        "kernel_manifest_cache_key": base_manifest["cache_key"],
        "execution_plan_key": "not-the-plan-key",
        "static_selections": [
            {
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": selected_candidate["candidate_config_key"],
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "shape": {"m": 4, "n": 6, "k": 8},
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
    }

    with pytest.raises(ValueError, match="Execution plan key does not match payload"):
        dml.compile(
            spec,
            target,
            tmp_path / "compile_tampered_profile_plan.dinoml",
            execution_plan=execution_plan,
        )


def test_compile_rejects_duplicate_execution_plan_selection_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(BackendSpec, "resolve_build_function", lambda self: lambda **kwargs: None)
    spec = dml.trace(
        GemmRRR(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="compile_duplicate_profile_plan",
    )
    target = dml.Target("cuda", arch="sm_86")
    lowered, _ = PassManager().run(spec.ir)
    base_manifest = build_kernel_manifest(lowered, target.to_json())
    required = base_manifest["required_kernels"][0]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "target": target.to_json(),
        "kernel_manifest_cache_key": base_manifest["cache_key"],
        "static_selections": [
            {
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": required["candidates"][0]["candidate_id"],
                "candidate_config_key": required["candidates"][0]["candidate_config_key"],
                "kernel_symbol": required["candidates"][0]["kernel_symbol"],
                "profiler_symbol": required["candidates"][0]["profiler_symbol"],
                "shape": {"m": 4, "n": 6, "k": 8},
                "split_k": 1,
                "workspace_nbytes": 0,
            },
            {
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": required["candidates"][1]["candidate_id"],
                "candidate_config_key": required["candidates"][1]["candidate_config_key"],
                "kernel_symbol": required["candidates"][1]["kernel_symbol"],
                "profiler_symbol": required["candidates"][1]["profiler_symbol"],
                "shape": {"m": 4, "n": 6, "k": 8},
                "split_k": 1,
                "workspace_nbytes": 0,
            },
        ],
    }

    with pytest.raises(ValueError, match="duplicate static selections"):
        dml.compile(
            spec,
            target,
            tmp_path / "compile_duplicate_profile_plan.dinoml",
            execution_plan=execution_plan,
        )
