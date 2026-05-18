import hashlib
from collections import Counter
from pathlib import Path

import pytest

import dinoml as dml
import dinoml.backends.build_parallelism as build_parallelism
import dinoml.backends.cutlass as cutlass_backend
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.backends.registry import BackendSpec, get_backend_spec, registered_backend_names, registered_backend_specs
from dinoml.ir import canonical_json, read_json
from dinoml.kernels.bmm import BMM_OPS
from dinoml.kernels.gemm import GEMM_OPS
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_candidates
from dinoml.kernels.providers.cutlass.gemm import (
    CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE,
    cutlass_gemm_candidates,
)
from dinoml.passes import PassManager


FLOAT32_CANDIDATE_MATH_COUNTS = {
    "tf32": 57,
    "f32": 11,
}
FLOAT32_OPTIONAL_MATH_COUNTS = {"tf32": 57}
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
    assert all(item["cutlass"].get("math_operator", "multiply_add") == "multiply_add" for item in candidates)

def _execution_plan_key(plan):
    return hashlib.sha256(
        canonical_json({key: value for key, value in plan.items() if key != "execution_plan_key"}).encode("utf-8")
    ).hexdigest()

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


def test_cutlass_float32_candidate_registry_lists_v1_math_families():
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
