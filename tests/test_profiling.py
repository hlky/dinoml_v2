import ctypes
from dataclasses import replace
import hashlib
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

import dinoml as dml
import dinoml.cli as cli
import dinoml.compiler as compiler_mod
import dinoml.kernels.profiling as profiling_mod
from dinoml.backends.registry import BackendSpec
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.ir import read_json, write_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION, build_kernel_manifest
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_candidates
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidates
from dinoml.kernels.profiling import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    PROFILE_REPORT_SCHEMA_VERSION,
    _cache_entry,
    _profile_key,
    _profile_key_payload,
    _profile_result,
    _profile_timing,
    build_execution_plan,
    build_profile_workloads,
    parse_shape_overrides,
    profile_artifact,
    profile_cache_path,
)
from dinoml.passes import PassManager


DEFAULT_CUDA_TARGET = {"name": "cuda", "arch": "sm_86"}


def _cutlass_candidates(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> tuple[dict[str, object], ...]:
    return cutlass_gemm_candidates(op_name, dtype, target=target or DEFAULT_CUDA_TARGET)


def _cutlass_bmm_candidates(dtype: str, *, op_name: str = "bmm_rrr", target=None) -> tuple[dict[str, object], ...]:
    return cutlass_bmm_candidates(op_name, dtype, target=target or DEFAULT_CUDA_TARGET)


def _cutlass_candidate_count(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> int:
    return len(_cutlass_candidates(dtype, op_name=op_name, target=target))


def _assert_folded_residual_workload_alignment(workloads, *, layout: str, op_name: str) -> None:
    full_candidate_count = _cutlass_candidate_count("float32", op_name=op_name)
    if layout == "rcr":
        assert len(workloads) == full_candidate_count
    else:
        assert len(workloads) < full_candidate_count
        assert {workload.candidate["cutlass"]["align"] for workload in workloads} == {1}
        assert all(
            workload.alignment_context["candidate_filter"]["max_operand_alignment"] == 1
            for workload in workloads
        )


def _cutlass_default_candidate_id(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> str:
    return str(_cutlass_candidates(dtype, op_name=op_name, target=target)[0]["candidate_id"])


def _cutlass_default_symbol_id(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> str:
    return str(_cutlass_candidates(dtype, op_name=op_name, target=target)[0]["symbol_id"])


def _set_tensor_layout_alignment(graph, names, alignment: int) -> None:
    for tensor in graph["tensors"]:
        if tensor["name"] in names:
            tensor.setdefault("layout", {})["alignment"] = int(alignment)


def _clear_tensor_layout_alignment(graph, names) -> None:
    for tensor in graph["tensors"]:
        if tensor["name"] in names:
            tensor.setdefault("layout", {}).pop("alignment", None)


def _set_tensor_layout_storage_offset(graph, names, offset: int) -> None:
    for tensor in graph["tensors"]:
        if tensor["name"] in names:
            tensor.setdefault("layout", {})["storage_offset"] = int(offset)


class GemmModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b), "y")


class BmmModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b), "y")


class GemmBiasModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias):
        op = getattr(dml.ops, self.op_name)
        return dml.ops.output(op(a, b, bias), "y")


class GemmResidualModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b, bias, d0, d1=None):
        op = getattr(dml.ops, self.op_name)
        if d1 is None:
            return dml.ops.output(op(a, b, bias, d0), "y")
        return dml.ops.output(op(a, b, bias, d0, d1), "y")


GEMM_BIAS_RESIDUAL_CASES = tuple(
    (f"gemm_{layout}_bias_{suffix}", layout, epilogue, inputs)
    for layout in ("rcr", "rrr")
    for suffix, epilogue, inputs in (
        ("add", "bias_add", ("bias", "d0")),
        ("add_add", "bias_add_add", ("bias", "d0", "d1")),
        ("mul", "bias_mul", ("bias", "d0")),
        ("mul_add", "bias_mul_add", ("bias", "d0", "d1")),
    )
)
GEMM_BIAS_RESIDUAL_CASES = (
    *GEMM_BIAS_RESIDUAL_CASES,
    *(
        (f"gemm_{layout}_bias_{suffix}", layout, epilogue, inputs)
        for layout in ("rcr", "rrr")
        for suffix, epilogue, inputs in (
            ("add_relu", "bias_add_relu", ("bias", "d0")),
            ("add_add_relu", "bias_add_add_relu", ("bias", "d0", "d1")),
            ("mul_tanh", "bias_mul_tanh", ("bias", "d0")),
            ("sigmoid_mul", "bias_sigmoid_mul", ("bias", "d0")),
            ("sigmoid_mul_tanh", "bias_sigmoid_mul_tanh", ("bias", "d0")),
        )
    ),
)


def test_parse_shape_overrides():
    assert parse_shape_overrides(["x=1,128,768", "tokens=77"]) == {
        "x": (1, 128, 768),
        "tokens": (77,),
    }
    with pytest.raises(ValueError, match="Expected shape override"):
        parse_shape_overrides(["x:1,2"])


def test_build_profile_workloads_uses_runtime_shape_overrides():
    batch = dml.Dim("batch", min=1, max=16)
    tokens = dml.Dim("tokens", min=8, max=24, divisible_by=8)
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_dynamic_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest, input_shapes={"a": (7, 32), "b": (32, 16)})

    assert len(workloads) == _cutlass_candidate_count("float16")
    workload = workloads[0]
    assert workload.profiler_symbol == f"dinoml_profile_cutlass_gemm_rrr_float16_{_cutlass_default_symbol_id('float16')}"
    assert workload.dtype == "float16"
    assert workload.candidate_set_id == "cutlass_gemm_rrr_float16_linear_combination_v1"
    assert workload.candidate_set_key
    assert workload.candidate_id == _cutlass_default_candidate_id("float16")
    assert workload.candidate["provider"] == "cutlass"
    assert workload.candidate["layouts"] == {"a": "row", "b": "row", "c": "row"}
    assert workload.candidate["cutlass"]["opclass"] == "tensorop"
    assert workload.candidate["cutlass"]["threadblock"] == [256, 128, 32]
    assert workload.candidate_config_key
    assert (workload.m, workload.n, workload.k) == (7, 16, 32)
    assert workload.output_shape == (7, 16)
    assert workload.to_json()["shape_case"] == {
        "source": "runtime_override",
        "case_id": "runtime_override",
        "dims": {},
        "dim_sources": {},
    }


def test_build_profile_workloads_supports_cutlass_bmm_batch_broadcast_and_column_output(tmp_path):
    spec = dml.trace(
        BmmModule("bmm_ccc"),
        inputs={"a": dml.TensorSpec([1, 8, 4], "float32"), "b": dml.TensorSpec([3, 5, 8], "float32")},
        name="profile_bmm_ccc",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(manifest, tmp_path / "cache").to_json()
    expected_candidates = [
        candidate for candidate in _cutlass_bmm_candidates("float32", op_name="bmm_ccc") if int(candidate["cutlass"]["align"]) <= 4
    ]

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == len(expected_candidates)
    workload = workloads[0]
    assert workload.kernel_library == "cutlass_bmm"
    assert workload.profiler_symbol == f"dinoml_profile_cutlass_bmm_ccc_float32_{_cutlass_default_symbol_id('float32')}"
    assert workload.candidate_set_id == "cutlass_bmm_ccc_float32_linear_combination_v1"
    assert workload.batch_count == 3
    assert (workload.m, workload.n, workload.k) == (4, 5, 8)
    assert workload.a_shape == (1, 8, 4)
    assert workload.b_shape == (3, 5, 8)
    assert workload.output_shape == (3, 5, 4)
    assert workload.batch_stride_a == 0
    assert workload.batch_stride_b == 40
    assert workload.batch_stride_c == 20
    assert (workload.lda, workload.ldb, workload.ldc) == (4, 8, 4)
    assert workload.alignment_context["kind"] == "cutlass_bmm_alignment_context"
    assert workload.alignment_context["candidate_filter"]["max_operand_alignment"] == 4
    payload = workload.to_json()
    assert payload["kernel_library"] == "cutlass_bmm"
    assert payload["batch_strides"] == {"a": 0, "b": 40, "c": 20}
    assert payload["leading_dimensions"] == {"a": 4, "b": 8, "c": 4}
    key_payload = _profile_key_payload(
        workload,
        {"target": DEFAULT_CUDA_TARGET},
        manifest,
        codegen_plan,
        context={
            "fingerprint": {
                "hardware_key": "hardware-key",
                "support_libraries_key": "support-key",
            }
        },
    )
    assert key_payload["kernel_library"] == "cutlass_bmm"
    assert key_payload["shape"] == {"m": 4, "n": 5, "k": 8, "batch_count": 3}


def test_build_profile_workloads_supports_cutlass_bmm_add_full_output_epilogue(tmp_path):
    class BmmAddModule(dml.Module):
        def forward(self, a, b, d0):
            return dml.ops.output(dml.ops.bmm_rrr_add(a, b, d0), "y")

    spec = dml.trace(
        BmmAddModule(),
        inputs={
            "a": dml.TensorSpec([2, 4, 8], "float32"),
            "b": dml.TensorSpec([2, 8, 6], "float32"),
            "d0": dml.TensorSpec([2, 4, 6], "float32"),
        },
        name="profile_bmm_rrr_add",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(manifest, tmp_path / "cache").to_json()

    workloads = build_profile_workloads(lowered, manifest)

    assert workloads
    workload = workloads[0]
    assert workload.kernel_library == "cutlass_bmm"
    assert workload.op == "bmm_rrr_add"
    assert workload.candidate_set_id == "cutlass_bmm_rrr_add_float32_add_v1"
    assert workload.candidate["epilogue"] == "add"
    assert workload.candidate["epilogue_config"]["launch_abi"] == "dinoml_cutlass_bmm_add_v1"
    assert workload.residual_tensors == ("d0",)
    assert workload.residual_shapes == ((2, 4, 6),)
    assert workload.output_shape == (2, 4, 6)
    assert workload.batch_stride_c == 24
    assert workload.alignment_context["epilogue"]["inputs"][0]["tensor"] == "d0"
    payload = workload.to_json()
    assert payload["inputs"]["d0"] == [2, 4, 6]
    key_payload = _profile_key_payload(
        workload,
        {"target": DEFAULT_CUDA_TARGET},
        manifest,
        codegen_plan,
        context={
            "fingerprint": {
                "hardware_key": "hardware-key",
                "support_libraries_key": "support-key",
            }
        },
    )
    assert key_payload["epilogue"] == "add"
    assert key_payload["epilogue_config"]["launch_abi"] == "dinoml_cutlass_bmm_add_v1"


def test_build_profile_workloads_supports_cutlass_bmm_add_trailing_bias_epilogue(tmp_path):
    class BmmAddModule(dml.Module):
        def forward(self, a, b, d0):
            return dml.ops.output(dml.ops.bmm_rrr_add(a, b, d0), "y")

    spec = dml.trace(
        BmmAddModule(),
        inputs={
            "a": dml.TensorSpec([2, 4, 8], "float32"),
            "b": dml.TensorSpec([2, 8, 6], "float32"),
            "d0": dml.TensorSpec([6], "float32"),
        },
        name="profile_bmm_rrr_add_bias",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(manifest, tmp_path / "cache").to_json()

    workloads = build_profile_workloads(lowered, manifest)

    assert workloads
    workload = workloads[0]
    assert workload.op == "bmm_rrr_add"
    assert workload.residual_tensors == ("d0",)
    assert workload.residual_shapes == ((6,),)
    assert workload.batch_stride_d0 == 0
    assert workload.ldd0 == 0
    payload = workload.to_json()
    assert payload["batch_strides"]["d0"] == 0
    assert payload["leading_dimensions"]["d0"] == 0
    key_payload = _profile_key_payload(
        workload,
        {"target": DEFAULT_CUDA_TARGET},
        manifest,
        codegen_plan,
        context={
            "fingerprint": {
                "hardware_key": "hardware-key",
                "support_libraries_key": "support-key",
            }
        },
    )
    assert key_payload["epilogue"] == "add"
    assert key_payload["candidate_set_id"] == "cutlass_bmm_rrr_add_float32_add_v1"


def test_profile_result_records_bmm_batch_shape_and_cost_model():
    spec = dml.trace(
        BmmModule("bmm_ccc"),
        inputs={"a": dml.TensorSpec([1, 8, 4], "float32"), "b": dml.TensorSpec([3, 5, 8], "float32")},
        name="profile_bmm_result",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workload = build_profile_workloads(lowered, manifest)[0]

    result = _profile_result(workload, 0.25, 5, profile_key="bmm-profile", status="ok")

    assert result["kernel_library"] == "cutlass_bmm"
    assert result["shape"]["batch_count"] == 3
    assert result["batch_count"] == 3
    assert result["batch_strides"] == {"a": 0, "b": 40, "c": 20}
    assert result["leading_dimensions"] == {"a": 4, "b": 8, "c": 4}
    assert result["flops"] == 2 * 3 * 4 * 5 * 8
    assert result["bytes"] == 4 * ((1 * 8 * 4) + (3 * 5 * 8) + (3 * 5 * 4))


def test_build_execution_plan_keeps_bmm_batch_count_in_shape_key():
    spec = dml.trace(
        BmmModule("bmm_rrr"),
        inputs={"a": dml.TensorSpec([3, 4, 8], "float32"), "b": dml.TensorSpec([3, 8, 5], "float32")},
        name="profile_bmm_execution_plan",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workload = build_profile_workloads(lowered, manifest)[0]
    batch3 = _profile_result(workload, 0.20, 5, profile_key="bmm-batch3", status="ok")
    batch7_workload = replace(
        workload,
        batch_count=7,
        a_shape=(7, 4, 8),
        b_shape=(7, 8, 5),
        output_shape=(7, 4, 5),
        batch_stride_a=32,
        batch_stride_b=40,
        batch_stride_c=20,
    )
    batch7 = _profile_result(batch7_workload, 0.30, 5, profile_key="bmm-batch7", status="ok")

    plan = build_execution_plan(
        {
            "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
            "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": {"name": "cuda", "arch": "sm_86"},
            "kernel_manifest_cache_key": manifest["cache_key"],
            "codegen_plan_cache_key": "codegen-key",
            "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
            "hardware_cache_key": "hardware-key",
            "support_libraries_cache_key": "support-key",
            "problems": [batch3, batch7],
            "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
        }
    )

    assert len(plan["selections"]) == 2
    assert {selection["shape"]["batch_count"] for selection in plan["selections"]} == {3, 7}
    assert plan["static_selections"][0]["shape"]["profiled_shapes"][0]["batch_count"] in {3, 7}


def test_build_execution_plan_marks_bmm_static_conflict_for_batch_specific_winners():
    spec = dml.trace(
        BmmModule("bmm_rrr"),
        inputs={"a": dml.TensorSpec([3, 4, 8], "float32"), "b": dml.TensorSpec([3, 8, 5], "float32")},
        name="profile_bmm_execution_plan_conflict",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workloads = build_profile_workloads(lowered, manifest)
    batch3 = _profile_result(workloads[0], 0.20, 5, profile_key="bmm-batch3", status="ok")
    batch7_workload = replace(
        workloads[1],
        batch_count=7,
        a_shape=(7, 4, 8),
        b_shape=(7, 8, 5),
        output_shape=(7, 4, 5),
        batch_stride_a=32,
        batch_stride_b=40,
        batch_stride_c=20,
    )
    batch7 = _profile_result(batch7_workload, 0.10, 5, profile_key="bmm-batch7", status="ok")

    plan = build_execution_plan(
        {
            "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
            "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": {"name": "cuda", "arch": "sm_86"},
            "kernel_manifest_cache_key": manifest["cache_key"],
            "codegen_plan_cache_key": "codegen-key",
            "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
            "hardware_cache_key": "hardware-key",
            "support_libraries_cache_key": "support-key",
            "problems": [batch3, batch7],
            "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
        }
    )

    assert len(plan["selections"]) == 2
    assert plan["static_selections"] == []
    assert plan["summary"]["conflict_count"] == 1
    conflict = plan["conflicts"][0]
    assert conflict["op"] == "bmm_rrr"
    assert conflict["selected_candidate_ids"] == sorted({workloads[0].candidate_id, workloads[1].candidate_id})
    assert {shape["batch_count"] for shape in conflict["profiled_shapes"]} == {3, 7}


def test_build_profile_workloads_expands_dim_buckets():
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    tokens = dml.Dim("tokens", min=8, max=16, divisible_by=8, buckets=(8, 16))
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_bucket_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    candidate_count = _cutlass_candidate_count("float16")
    assert len(workloads) == candidate_count * 6
    cases = {
        (workload.m, workload.n, workload.k, workload.shape_case_id, tuple(sorted(workload.dim_values.items())))
        for workload in workloads
    }
    assert cases == {
        (2, 8, 32, "bucket_batch=2_tokens=8", (("batch", 2), ("tokens", 8))),
        (2, 16, 32, "bucket_batch=2_tokens=16", (("batch", 2), ("tokens", 16))),
        (4, 8, 32, "bucket_batch=4_tokens=8", (("batch", 4), ("tokens", 8))),
        (4, 16, 32, "bucket_batch=4_tokens=16", (("batch", 4), ("tokens", 16))),
    }
    assert {workload.shape_source for workload in workloads} == {"dim_buckets"}
    assert {tuple(sorted(workload.dim_sources.items())) for workload in workloads} == {
        (("batch", "bucket"), ("tokens", "bucket"))
    }
    splits_by_case = {}
    for workload in workloads:
        splits_by_case.setdefault(workload.shape_case_id, set()).add(workload.split_k)
    assert splits_by_case == {
        "bucket_batch=2_tokens=8": {1, 2},
        "bucket_batch=2_tokens=16": {1},
        "bucket_batch=4_tokens=8": {1, 2},
        "bucket_batch=4_tokens=16": {1},
    }


def test_build_profile_workloads_evaluates_sourceable_symbolic_expr_buckets():
    tokens = dml.Dim("tokens", min=4, max=8, buckets=(4, 8))
    token_dim = dml.TensorSpec([tokens]).shape_spec[0]
    tokens_plus_one = dml.ops.int_add(token_dim, 1)
    spec = dml.trace(
        GemmModule("gemm_rcr"),
        inputs={
            "a": dml.TensorSpec([token_dim, 32], "float16"),
            "b": dml.TensorSpec([tokens_plus_one, 32], "float16"),
        },
        name="profile_symbolic_expr_bucket_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float16", op_name="gemm_rcr") * 5
    assert {
        (workload.m, workload.n, workload.output_shape, workload.shape_case_id, tuple(sorted(workload.dim_values.items())))
        for workload in workloads
    } == {
        (4, 5, (4, 5), "bucket_tokens=4", (("tokens", 4),)),
        (8, 9, (8, 9), "bucket_tokens=8", (("tokens", 8),)),
    }
    assert {tuple(sorted(workload.dim_sources.items())) for workload in workloads} == {(("tokens", "bucket"),)}


def test_build_profile_workloads_overrides_disable_bucket_expansion():
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    tokens = dml.Dim("tokens", min=8, max=16, divisible_by=8, buckets=(8, 16))
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_override_bucket_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest, input_shapes={"a": (3, 32), "b": (32, 8)})

    assert len(workloads) == _cutlass_candidate_count("float16") * 2
    assert {(workload.m, workload.n, workload.k) for workload in workloads} == {(3, 8, 32)}
    assert {workload.split_k for workload in workloads} == {1, 2}
    assert {workload.shape_source for workload in workloads} == {"runtime_override"}


def test_build_profile_workloads_records_max_sourced_dynamic_dims_with_buckets():
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    tokens = dml.Dim("tokens", min=8, max=16, divisible_by=8)
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_bucket_and_max_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float16") * 2
    assert {
        (workload.m, workload.n, workload.shape_case_id, tuple(sorted(workload.dim_sources.items())))
        for workload in workloads
    } == {
        (2, 16, "bucket_batch=2_tokens=16", (("batch", "bucket"), ("tokens", "max"))),
        (4, 16, "bucket_batch=4_tokens=16", (("batch", "bucket"), ("tokens", "max"))),
    }


def test_build_profile_workloads_bucket_expansion_preserves_shared_dim_values():
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    spec = dml.trace(
        GemmResidualModule("gemm_rcr_bias_add"),
        inputs={
            "a": dml.TensorSpec([batch, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([batch, 11], "float32"),
        },
        name="profile_bucket_residual_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32") * 2
    case_inputs = {workload.shape_case_id: workload.to_json()["inputs"] for workload in workloads[: _cutlass_candidate_count("float32") * 2]}
    assert case_inputs["bucket_batch=2"]["a"] == [2, 32]
    assert case_inputs["bucket_batch=2"]["d0"] == [2, 11]
    assert case_inputs["bucket_batch=4"]["a"] == [4, 32]
    assert case_inputs["bucket_batch=4"]["d0"] == [4, 11]


def test_build_profile_workloads_rejects_inconsistent_same_name_buckets():
    batch_a = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    batch_d0 = dml.Dim("batch", min=1, max=4, buckets=(3, 4))
    spec = dml.trace(
        GemmResidualModule("gemm_rcr_bias_add"),
        inputs={
            "a": dml.TensorSpec([batch_a, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([batch_d0, 11], "float32"),
        },
        name="profile_inconsistent_bucket_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    with pytest.raises(ValueError, match="Inconsistent profiling bucket metadata"):
        build_profile_workloads(lowered, manifest)


def test_build_profile_workloads_uses_manifest_selected_fp16_accumulation_policy():
    target = {**DEFAULT_CUDA_TARGET, "use_fp16_acc": True}
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float16"), "b": dml.TensorSpec([32, 16], "float16")},
        name="profile_fp16_acc_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float16", target=target)
    assert workloads
    assert {workload.candidate["accumulator_dtype"] for workload in workloads} == {"float16"}
    assert all("_f16_" in workload.kernel_symbol for workload in workloads)
    assert all("_f16_" in workload.profiler_symbol for workload in workloads)
    assert all("_f32_" not in workload.kernel_symbol for workload in workloads)
    assert workloads[0].candidate_id == _cutlass_default_candidate_id("float16", target=target)


def test_build_profile_workloads_uses_manifest_selected_no_tf32_policy():
    target = {**DEFAULT_CUDA_TARGET, "no_tf32": True}
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 11], "float32")},
        name="profile_no_tf32_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == 11
    assert {workload.candidate["cutlass"]["opclass"] for workload in workloads} == {"simt"}
    assert {workload.candidate["cutlass"]["math"] for workload in workloads} == {"f32"}
    assert all("simt_sm80_f32" in workload.kernel_symbol for workload in workloads)
    assert all("tf32" not in workload.kernel_symbol for workload in workloads)
    assert workloads[0].candidate_id == _cutlass_default_candidate_id("float32", target=target)


def test_build_profile_workloads_expands_v1_split_k_for_supported_gemm():
    target = {**DEFAULT_CUDA_TARGET, "no_tf32": True}
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([64, 1024], "float32"), "b": dml.TensorSpec([1024, 64], "float32")},
        name="profile_split_k_search_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    expected_split_k = [1, 4, 6, 8, 10, 12, 14]
    first_candidate = workloads[0].candidate_id
    assert len(workloads) == 11 * len(expected_split_k)
    assert [workload.split_k for workload in workloads if workload.candidate_id == first_candidate] == expected_split_k
    assert all(workload.workspace_nbytes == 0 for workload in workloads if workload.split_k == 1)
    assert all(workload.workspace_nbytes > 0 for workload in workloads if workload.split_k > 1)


@pytest.mark.parametrize(
    ("op_name", "has_d1"),
    [
        ("gemm_rcr_bias_add", False),
        ("gemm_rcr_bias_add_add_relu", True),
    ],
)
def test_build_profile_workloads_expands_split_k_for_additive_residual_epilogues(op_name, has_d1):
    target = {**DEFAULT_CUDA_TARGET, "no_tf32": True}
    inputs = {
        "a": dml.TensorSpec([64, 1024], "float32"),
        "b": dml.TensorSpec([64, 1024], "float32"),
        "bias": dml.TensorSpec([64], "float32"),
        "d0": dml.TensorSpec([64, 64], "float32"),
    }
    if has_d1:
        inputs["d1"] = dml.TensorSpec([64, 64], "float32")
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs=inputs,
        name=f"profile_{op_name}_split_k_search_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    expected_split_k = [1, 4, 6, 8, 10, 12, 14]
    first_candidate = workloads[0].candidate_id
    assert len(workloads) == 11 * len(expected_split_k)
    assert [workload.split_k for workload in workloads if workload.candidate_id == first_candidate] == expected_split_k
    assert all(workload.candidate["supports_split_k"] is True for workload in workloads)
    assert all(workload.candidate["split_k_search"] == {"strategy": "v1_gemm_factor", "max_split_k": 32} for workload in workloads)
    assert all(workload.workspace_nbytes == 0 for workload in workloads if workload.split_k == 1)
    assert all(workload.workspace_nbytes > 0 for workload in workloads if workload.split_k > 1)


@pytest.mark.parametrize("op_name", ["gemm_rcr_bias_mul", "gemm_rcr_bias_sigmoid_mul"])
def test_build_profile_workloads_keeps_non_additive_residual_epilogues_split_k_one(op_name):
    target = {**DEFAULT_CUDA_TARGET, "no_tf32": True}
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([64, 1024], "float32"),
            "b": dml.TensorSpec([64, 1024], "float32"),
            "bias": dml.TensorSpec([64], "float32"),
            "d0": dml.TensorSpec([64, 64], "float32"),
        },
        name=f"profile_{op_name}_no_split_k_search_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == 11
    assert {workload.split_k for workload in workloads} == {1}
    assert all(workload.workspace_nbytes == 0 for workload in workloads)
    assert all(workload.candidate["supports_split_k"] is False for workload in workloads)
    assert all("split_k_search" not in workload.candidate for workload in workloads)


def test_build_profile_workloads_filters_candidates_by_v1_rrr_shape_alignment():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 6], "float32")},
        name="profile_shape_alignment_filtered_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    expected_candidates = [
        candidate for candidate in _cutlass_candidates("float32") if int(candidate["cutlass"]["align"]) <= 2
    ]

    workloads = build_profile_workloads(lowered, manifest)

    assert {workload.candidate_id for workload in workloads} == {
        str(candidate["candidate_id"]) for candidate in expected_candidates
    }
    assert {workload.candidate["cutlass"]["align"] for workload in workloads} <= {1, 2}
    assert {workload.split_k for workload in workloads} == {1, 2}


def test_build_profile_workloads_rejects_fp16_rrr_shape_alignment_without_candidate():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float16"), "b": dml.TensorSpec([32, 11], "float16")},
        name="profile_fp16_shape_alignment_rejected_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    with pytest.raises(ValueError, match="manifest alignment filter removed all candidates"):
        build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})


def test_build_profile_workloads_filters_candidates_by_layout_alignment():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 12], "float32")},
        name="profile_alignment_filtered_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"a", "b"}, 2)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    expected_candidates = [
        candidate for candidate in _cutlass_candidates("float32") if int(candidate["cutlass"]["align"]) <= 2
    ]

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == len(expected_candidates)
    assert [workload.candidate_id for workload in workloads] == [
        str(candidate["candidate_id"]) for candidate in expected_candidates
    ]
    assert {workload.candidate["cutlass"]["align"] for workload in workloads} <= {1, 2}


@pytest.mark.parametrize("annotated_tensor", ["a", "b"])
def test_build_profile_workloads_uses_partial_a_b_layout_alignment(annotated_tensor):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 12], "float32")},
        name=f"profile_partial_alignment_{annotated_tensor}_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _clear_tensor_layout_alignment(lowered, {"a", "b"})
    _set_tensor_layout_alignment(lowered, {annotated_tensor}, 2)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert workloads
    assert len(workloads) < _cutlass_candidate_count("float32")
    assert {workload.candidate["cutlass"]["align"] for workload in workloads} <= {1, 2}
    assert all(
        workload.alignment_context["candidate_filter"]["max_operand_alignment"] == 2
        for workload in workloads
    )


def test_build_profile_workloads_uses_minimum_a_b_layout_alignment():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 12], "float32")},
        name="profile_mixed_alignment_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"a"}, 8)
    _set_tensor_layout_alignment(lowered, {"b"}, 2)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert {workload.candidate["cutlass"]["align"] for workload in workloads} <= {1, 2}


def test_build_profile_workloads_filters_candidates_by_layout_storage_offset():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 12], "float32")},
        name="profile_storage_offset_alignment_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"a", "b"}, 4)
    _set_tensor_layout_storage_offset(lowered, {"a"}, 1)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    required = manifest["required_kernels"][0]
    assert required["cutlass_alignment_cap"] == 1
    assert required["cutlass_alignment"]["candidate_filter"]["max_operand_alignment"] == 1
    assert {workload.candidate["cutlass"]["align"] for workload in workloads} == {1}
    assert workloads[0].alignment_context["operands"]["a"]["sources"][-1] == {
        "source": "layout.storage_offset",
        "offset_elements": 1,
        "alignment": 1,
    }


def test_build_profile_workloads_filters_rcr_candidates_by_a_b_layout_alignment():
    spec = dml.trace(
        GemmModule("gemm_rcr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([11, 32], "float32")},
        name="profile_alignment_filtered_rcr_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"a", "b"}, 2)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert workloads
    assert all(workload.op == "gemm_rcr" for workload in workloads)
    assert {workload.candidate["layouts"]["b"] for workload in workloads} == {"column"}
    assert {workload.candidate["cutlass"]["align"] for workload in workloads} <= {1, 2}


def test_build_profile_workloads_ignores_epilogue_input_layout_alignment_for_candidate_filtering():
    spec = dml.trace(
        GemmResidualModule("gemm_rcr_bias_add"),
        inputs={
            "a": dml.TensorSpec([7, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([7, 11], "float32"),
        },
        name="profile_epilogue_alignment_ignored_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"bias", "d0"}, 1)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")


def test_build_profile_workloads_rejects_layout_alignment_without_candidate():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float16"), "b": dml.TensorSpec([32, 16], "float16")},
        name="profile_alignment_rejected_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"a", "b"}, 1)
    with pytest.raises(ValueError, match="manifest alignment filter removed all candidates"):
        build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})


@pytest.mark.parametrize(
    ("op_name", "epilogue"),
    [
        ("gemm_rcr_bias", "bias"),
        ("gemm_rcr_bias_relu", "bias_relu"),
        ("gemm_rcr_bias_elup1", "bias_elup1"),
    ],
)
def test_build_profile_workloads_supports_gemm_bias_epilogue(op_name, epilogue):
    spec = dml.trace(
        GemmBiasModule(op_name),
        inputs={
            "a": dml.TensorSpec([7, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
        },
        name=f"profile_{op_name}",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")
    workload = workloads[0]
    assert workload.profiler_symbol == f"dinoml_profile_cutlass_{op_name}_float32_{_cutlass_default_symbol_id('float32')}"
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert workload.bias_tensor == "bias"
    assert workload.bias_shape == (11,)
    assert workload.candidate["epilogue"] == epilogue
    assert workload.candidate["epilogue_config"]["inputs"] == ["bias"]
    expected_activation = {"bias_relu": "relu", "bias_elup1": "elup1"}.get(epilogue)
    assert workload.candidate["epilogue_config"]["activation"] == expected_activation
    assert workload.to_json()["inputs"]["bias"] == [11]


@pytest.mark.parametrize(("op_name", "layout", "epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_CASES)
def test_build_profile_workloads_supports_gemm_residual_epilogue_inputs(op_name, layout, epilogue, epilogue_inputs):
    n = 12
    input_specs = {
        "a": dml.TensorSpec([7, 32], "float32"),
        "b": dml.TensorSpec([n, 32] if layout == "rcr" else [32, n], "float32"),
        "bias": dml.TensorSpec([n], "float32"),
        "d0": dml.TensorSpec([7, n], "float32"),
    }
    if "d1" in epilogue_inputs:
        input_specs["d1"] = dml.TensorSpec([7, n], "float32")
    spec = dml.trace(GemmResidualModule(op_name), inputs=input_specs, name=f"profile_{op_name}")
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    required = manifest["required_kernels"][0]
    assert len(workloads) == len(required["candidates"])
    assert len(workloads) == _cutlass_candidate_count("float32", op_name=op_name)
    workload = workloads[0]
    assert workload.profiler_symbol == f"dinoml_profile_cutlass_{op_name}_float32_{_cutlass_default_symbol_id('float32')}"
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert workload.bias_tensor == "bias"
    assert workload.bias_shape == (n,)
    assert workload.candidate["epilogue"] == epilogue
    assert workload.candidate["epilogue_config"]["inputs"] == list(epilogue_inputs)
    expected_activation = {
        "bias_add_relu": "relu",
        "bias_add_add_relu": "relu",
        "bias_mul_tanh": "tanh",
        "bias_sigmoid_mul_tanh": "tanh",
    }.get(epilogue)
    expected_pre_activation = "sigmoid" if epilogue in {"bias_sigmoid_mul", "bias_sigmoid_mul_tanh"} else None
    assert workload.candidate["epilogue_config"]["activation"] == expected_activation
    assert workload.candidate["epilogue_config"].get("pre_residual_activation") == expected_pre_activation
    assert workload.candidate["launch_abi"].startswith("dinoml_cutlass_gemm_")
    expected_split_k_support = epilogue in {"bias_add", "bias_add_add", "bias_add_relu", "bias_add_add_relu"}
    assert workload.candidate["supports_split_k"] is expected_split_k_support
    if expected_split_k_support:
        assert workload.candidate["split_k_search"] == {"strategy": "v1_gemm_factor", "max_split_k": 32}
    else:
        assert "split_k_search" not in workload.candidate
    inputs = workload.to_json()["inputs"]
    assert inputs["bias"] == [n]
    assert inputs["d0"] == [7, n]
    if "d1" in epilogue_inputs:
        assert inputs["d1"] == [7, n]
    else:
        assert "d1" not in inputs


@pytest.mark.parametrize(
    ("op_name", "layout", "epilogue"),
    [
        (f"gemm_{layout}_bias_{suffix}", layout, epilogue)
        for layout in ("rcr", "rrr")
        for suffix, epilogue in (
            ("add", "bias_add"),
            ("add_relu", "bias_add_relu"),
            ("mul", "bias_mul"),
            ("mul_tanh", "bias_mul_tanh"),
            ("sigmoid_mul", "bias_sigmoid_mul"),
            ("sigmoid_mul_tanh", "bias_sigmoid_mul_tanh"),
        )
    ],
)
def test_build_profile_workloads_flattens_gemm_single_residual_folded_m(op_name, layout, epilogue):
    b_shape = [11, 32] if layout == "rcr" else [32, 11]
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 32], "float32"),
            "b": dml.TensorSpec(b_shape, "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([2, 3, 11], "float32"),
        },
        name=f"profile_{op_name}_folded_m",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    _assert_folded_residual_workload_alignment(workloads, layout=layout, op_name=op_name)
    workload = workloads[0]
    assert (workload.m, workload.n, workload.k) == (6, 11, 32)
    assert workload.a_shape == (2, 3, 32)
    assert workload.b_shape == tuple(b_shape)
    assert workload.residual_shapes == ((2, 3, 11),)
    assert workload.output_shape == (2, 3, 11)
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert workload.to_json()["inputs"]["b"] == b_shape
    assert workload.to_json()["inputs"]["a"] == [2, 3, 32]
    assert workload.to_json()["inputs"]["d0"] == [2, 3, 11]
    assert list(workload.to_json()["output"].values()) == [[2, 3, 11]]


@pytest.mark.parametrize(
    ("op_name", "layout", "epilogue"),
    [
        (f"gemm_{layout}_bias_{suffix}", layout, epilogue)
        for layout in ("rcr", "rrr")
        for suffix, epilogue in (
            ("add_add", "bias_add_add"),
            ("mul_add", "bias_mul_add"),
            ("add_add_relu", "bias_add_add_relu"),
        )
    ],
)
def test_build_profile_workloads_flattens_gemm_dual_residual_folded_m(op_name, layout, epilogue):
    b_shape = [11, 32] if layout == "rcr" else [32, 11]
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 32], "float32"),
            "b": dml.TensorSpec(b_shape, "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([2, 3, 11], "float32"),
            "d1": dml.TensorSpec([2, 3, 11], "float32"),
        },
        name=f"profile_{op_name}_folded_m",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    _assert_folded_residual_workload_alignment(workloads, layout=layout, op_name=op_name)
    workload = workloads[0]
    assert (workload.m, workload.n, workload.k) == (6, 11, 32)
    assert workload.b_shape == tuple(b_shape)
    assert workload.residual_tensors == ("d0", "d1")
    assert workload.residual_shapes == ((2, 3, 11), (2, 3, 11))
    assert workload.output_shape == (2, 3, 11)
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    inputs = workload.to_json()["inputs"]
    assert inputs["b"] == b_shape
    assert inputs["d0"] == [2, 3, 11]
    assert inputs["d1"] == [2, 3, 11]
    assert list(workload.to_json()["output"].values()) == [[2, 3, 11]]


def test_profile_key_changes_with_fingerprint_keys(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_key_fingerprint",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    manifest = {"target": {"name": "cuda", "arch": "sm_86"}}
    workload = build_profile_workloads(lowered, kernel_manifest)[0]
    other_candidate = {**workload.candidate, "candidate_id": "cutlass_other", "candidate_config_key": "candidate-b"}
    other_workload = replace(
        workload,
        candidate_id="cutlass_other",
        candidate_config_key="candidate-b",
        candidate_set_key="candidate-set-b",
        candidate=other_candidate,
    )
    context_a = {"fingerprint": {"hardware_key": "hardware-a", "support_libraries_key": "support-a"}}
    context_b = {"fingerprint": {"hardware_key": "hardware-b", "support_libraries_key": "support-a"}}

    payload_a = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context_a)
    payload_b = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context_b)
    payload_c = _profile_key_payload(other_workload, manifest, kernel_manifest, codegen_plan, context=context_a)
    payload_d = _profile_key_payload(replace(workload, split_k=4), manifest, kernel_manifest, codegen_plan, context=context_a)

    assert payload_a["hardware_fingerprint_key"] == "hardware-a"
    assert payload_a["support_libraries_fingerprint_key"] == "support-a"
    assert payload_a["profile_variant"] == {"split_k": 1}
    assert payload_d["profile_variant"] == {"split_k": 4}
    expected_candidate = next(
        candidate for candidate in _cutlass_candidates("float32") if int(candidate["cutlass"]["align"]) <= 2
    )
    assert payload_a["candidate_id"] == expected_candidate["candidate_id"]
    assert payload_a["candidate_set_id"] == "cutlass_gemm_rrr_float32_linear_combination_v1"
    assert payload_a["candidate_set_key"] == workload.candidate_set_key
    assert payload_a["candidate_config_key"] == workload.candidate_config_key
    assert payload_a["layouts"] == workload.candidate["layouts"]
    assert payload_a["epilogue"] == workload.candidate["epilogue"]
    assert payload_a["epilogue_config"] == workload.candidate["epilogue_config"]
    assert payload_a["alignment_context"] == workload.alignment_context
    assert _profile_key(payload_a) != _profile_key(payload_b)
    assert _profile_key(payload_a) != _profile_key(payload_c)
    assert _profile_key(payload_a) != _profile_key(payload_d)


def test_profile_result_records_timing_statistics_and_residual_bytes(tmp_path):
    spec = dml.trace(
        GemmResidualModule("gemm_rcr_bias_add_add"),
        inputs={
            "a": dml.TensorSpec([7, 32], "float32"),
            "b": dml.TensorSpec([12, 32], "float32"),
            "bias": dml.TensorSpec([12], "float32"),
            "d0": dml.TensorSpec([7, 12], "float32"),
            "d1": dml.TensorSpec([7, 12], "float32"),
        },
        name="profile_timing_stats",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    workload = build_profile_workloads(lowered, kernel_manifest)[0]
    timing = _profile_timing([0.20, 0.10, 0.30], iterations=5)

    result = _profile_result(workload, timing["median_ms"], 5, profile_key="timed-profile", status="ok", timing=timing)
    cache_entry = _cache_entry(
        workload,
        result,
        _profile_key_payload(
            workload,
            {"target": {"name": "cuda", "arch": "sm_86"}},
            kernel_manifest,
            codegen_plan,
            context={"fingerprint": {"hardware_key": "hardware", "support_libraries_key": "support"}},
        ),
    )

    assert result["elapsed_ms"] == pytest.approx(0.20)
    assert result["bytes"] == 4 * (7 * 32 + 12 * 32 + 7 * 12 + 12 + 7 * 12 + 7 * 12)
    assert result["repeats"] == 3
    assert result["timing"]["samples_ms"] == [0.20, 0.10, 0.30]
    assert result["timing"]["median_ms"] == pytest.approx(0.20)
    assert result["timing"]["mean_ms"] == pytest.approx(0.20)
    assert result["timing"]["stddev_ms"] > 0.0
    assert result["timing"]["standard_error_ms"] > 0.0
    assert result["timing"]["mean_ci95_ms"]["half_width"] > 0.0
    assert result["timing"]["statistics_schema_version"] == profiling_mod.PROFILE_STATISTICS_SCHEMA_VERSION
    assert result["candidates"][0]["repeats"] == 3
    assert cache_entry["repeats"] == 3
    assert cache_entry["timing"]["sample_count"] == 3
    assert cache_entry["statistics_schema_version"] == profiling_mod.PROFILE_STATISTICS_SCHEMA_VERSION
    cached = profiling_mod._profile_result_from_cache(workload, cache_entry)
    assert cached["status"] == "cached"
    assert cached["timing"]["samples_ms"] == [0.20, 0.10, 0.30]


def test_profile_cache_rejects_malformed_entries_payload(tmp_path):
    cache_path = tmp_path / "profile_cache.json"
    target = {"name": "cuda", "arch": "sm_86"}
    write_json(cache_path, {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": target, "entries": []})

    cache = profiling_mod._read_profile_cache(cache_path, target)

    assert cache == {"schema_version": PROFILE_CACHE_SCHEMA_VERSION, "target": target, "entries": {}}

    write_json(
        cache_path,
        {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": target,
            "entries": {"bad": [], "good": {"profile_key": "good"}},
        },
    )

    cache = profiling_mod._read_profile_cache(cache_path, target)

    assert cache == {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": target,
        "entries": {"good": {"profile_key": "good"}},
    }


def test_profile_cache_rejects_mismatched_profile_key_entries(tmp_path):
    cache_path = tmp_path / "profile_cache.json"
    target = {"name": "cuda", "arch": "sm_86"}
    write_json(
        cache_path,
        {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": target,
            "entries": {
                "good": {"profile_key": "good", "elapsed_ms": 1.0},
                "missing": {"elapsed_ms": 2.0},
                "mismatch": {"profile_key": "other", "elapsed_ms": 3.0},
                "nonstr": {"profile_key": 4, "elapsed_ms": 4.0},
            },
        },
    )

    cache = profiling_mod._read_profile_cache(cache_path, target)

    assert cache["entries"] == {"good": {"profile_key": "good", "elapsed_ms": 1.0}}


def test_profile_cache_write_preserves_existing_same_target_entries(tmp_path):
    cache_path = tmp_path / "profile_cache.json"
    target = {"name": "cuda", "arch": "sm_86"}
    write_json(
        cache_path,
        {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": target,
            "entries": {
                "existing": {"profile_key": "existing", "elapsed_ms": 2.0},
                "shared": {"profile_key": "shared", "elapsed_ms": 3.0},
            },
        },
    )

    profiling_mod._write_profile_cache(
        cache_path,
        {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": target,
            "entries": {
                "fresh": {"profile_key": "fresh", "elapsed_ms": 1.0},
                "shared": {"profile_key": "shared", "elapsed_ms": 0.5},
            },
        },
    )

    cache = profiling_mod._read_profile_cache(cache_path, target)

    assert cache["entries"] == {
        "existing": {"profile_key": "existing", "elapsed_ms": 2.0},
        "fresh": {"profile_key": "fresh", "elapsed_ms": 1.0},
        "shared": {"profile_key": "shared", "elapsed_ms": 0.5},
    }


def test_profile_cache_hit_requires_enough_timing_samples(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_cache_timing_samples",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    workload = build_profile_workloads(lowered, kernel_manifest)[0]
    timing = _profile_timing([0.20, 0.20, 0.20], iterations=5)
    result = _profile_result(workload, timing["median_ms"], 5, profile_key="timed-profile", status="ok", timing=timing)
    cache_entry = _cache_entry(
        workload,
        result,
        _profile_key_payload(
            workload,
            {"target": {"name": "cuda", "arch": "sm_86"}},
            kernel_manifest,
            codegen_plan,
            context={"fingerprint": {"hardware_key": "hardware", "support_libraries_key": "support"}},
        ),
    )
    stale_entry = {
        **cache_entry,
        "repeats": 3,
        "timing": {**cache_entry["timing"], "samples_ms": [0.20], "sample_count": 1, "repeats": 1},
    }

    assert profiling_mod._cache_entry_satisfies(cache_entry, iterations=5, repeats=3) is True
    assert profiling_mod._cache_entry_satisfies(stale_entry, iterations=5, repeats=3) is False
    assert profiling_mod._cache_entry_satisfies({**cache_entry, "timing": {}}, iterations=5, repeats=3) is False


def test_profile_cache_hit_rejects_malformed_timing_fields(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_cache_malformed_timing_fields",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    workload = build_profile_workloads(lowered, kernel_manifest)[0]
    timing = _profile_timing([0.20, 0.20, 0.20], iterations=5)
    result = _profile_result(workload, timing["median_ms"], 5, profile_key="timed-profile", status="ok", timing=timing)
    cache_entry = _cache_entry(
        workload,
        result,
        _profile_key_payload(
            workload,
            {"target": {"name": "cuda", "arch": "sm_86"}},
            kernel_manifest,
            codegen_plan,
            context={"fingerprint": {"hardware_key": "hardware", "support_libraries_key": "support"}},
        ),
    )

    malformed_cases = (
        {**cache_entry, "iterations": "five"},
        {**cache_entry, "repeats": object()},
        {**cache_entry, "timing": {**cache_entry["timing"], "repeats": "three"}},
        {**cache_entry, "timing": {**cache_entry["timing"], "sample_count": []}},
        {**cache_entry, "iterations": -1},
        {**cache_entry, "timing": {**cache_entry["timing"], "sample_count": -1}},
    )

    for malformed_entry in malformed_cases:
        assert profiling_mod._cache_entry_satisfies(malformed_entry, iterations=5, repeats=3) is False


def test_build_execution_plan_selects_fastest_candidate_for_profiled_shape(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_execution_plan_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    workloads = build_profile_workloads(lowered, kernel_manifest)
    slow_timing = _profile_timing([0.25, 0.25, 0.25], iterations=5)
    fast_timing = _profile_timing([0.10, 0.10, 0.10], iterations=5)
    slow = _profile_result(workloads[0], slow_timing["median_ms"], 5, profile_key="slow-profile", status="ok", timing=slow_timing)
    fast = _profile_result(workloads[1], fast_timing["median_ms"], 5, profile_key="fast-profile", status="ok", timing=fast_timing)
    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "artifact": str(tmp_path / "profile_execution_plan_gemm.dinoml"),
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [slow, fast],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    assert plan["schema_version"] == EXECUTION_PLAN_SCHEMA_VERSION
    assert plan["selection_policy"] == "lowest_median_elapsed_ms_per_node_shape"
    assert plan["selection_confidence_policy"] == {
        "name": "confidence_interval_margin_v1",
        "statistics_schema_version": profiling_mod.PROFILE_STATISTICS_SCHEMA_VERSION,
        "confidence_level": profiling_mod.PROFILE_CONFIDENCE_LEVEL,
        "z_score": profiling_mod.PROFILE_CONFIDENCE_Z_SCORE,
        "min_repeats": profiling_mod.PROFILE_CONFIDENCE_MIN_REPEATS,
        "min_absolute_margin_ms": profiling_mod.PROFILE_CONFIDENCE_MIN_ABSOLUTE_MARGIN_MS,
        "min_relative_speedup": profiling_mod.PROFILE_CONFIDENCE_MIN_RELATIVE_SPEEDUP,
    }
    assert plan["summary"] == {"selection_count": 1, "low_confidence_count": 0, "static_selection_count": 1, "conflict_count": 0}
    assert plan["execution_plan_key"]
    selection = plan["selections"][0]
    assert selection["selected_candidate_id"] == workloads[1].candidate_id
    assert selection["candidate_config_key"] == workloads[1].candidate_config_key
    assert selection["kernel_symbol"] == workloads[1].kernel_symbol
    assert selection["profiler_symbol"] == workloads[1].profiler_symbol
    assert selection["shape"] == {
        "m": 4,
        "n": 6,
        "k": 8,
        "source": "graph_max_shape",
        "case_id": "max",
        "dims": {},
        "dim_sources": {},
    }
    assert selection["workspace_nbytes"] == 0
    assert selection["split_k"] == 1
    assert selection["confidence"]["confident"] is True
    assert selection["confidence"]["level"] == "high"
    static_selection = plan["static_selections"][0]
    assert static_selection["selected_candidate_id"] == workloads[1].candidate_id
    assert static_selection["node_id"] is None
    assert static_selection["shape"]["source"] == "static_overlay_from_consistent_profiled_shapes"
    assert static_selection["confidence"]["confident"] is True
    assert static_selection["confidence"]["profiled_shape_count"] == 1


def test_build_execution_plan_records_low_confidence_for_close_or_noisy_candidates(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_execution_plan_confidence",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workloads = build_profile_workloads(lowered, kernel_manifest)
    best_timing = _profile_timing([0.05, 0.10, 0.15], iterations=5)
    best = _profile_result(workloads[0], best_timing["median_ms"], 5, profile_key="best-profile", status="ok", timing=best_timing)
    runner_timing = _profile_timing([0.101, 0.101, 0.101], iterations=5)
    close_runner = _profile_result(workloads[1], runner_timing["median_ms"], 5, profile_key="runner-profile", status="ok", timing=runner_timing)
    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": "codegen-key",
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [best, close_runner],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    assert plan["selections"] == []
    assert plan["static_selections"] == []
    assert plan["summary"] == {"selection_count": 0, "low_confidence_count": 1, "static_selection_count": 0, "conflict_count": 0}
    confidence = plan["low_confidence_selections"][0]["confidence"]
    assert confidence["confident"] is False
    assert confidence["level"] == "low"
    assert confidence["relative_speedup_over_runner_up"] < profiling_mod.PROFILE_CONFIDENCE_MIN_RELATIVE_SPEEDUP
    assert confidence["margin_ms"] < confidence["required_margin_ms"]
    assert confidence["combined_standard_error_ms"] > 0.0
    assert confidence["sample_counts"] == {"best": 3, "runner_up": 3}
    assert confidence["reasons"] == ["margin_below_required_threshold"]


def test_build_execution_plan_requires_repeat_samples_for_confidence(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_execution_plan_insufficient_repeats",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workloads = build_profile_workloads(lowered, kernel_manifest)
    best = _profile_result(workloads[0], 0.10, 5, profile_key="best-profile", status="ok")
    runner = _profile_result(workloads[1], 0.25, 5, profile_key="runner-profile", status="ok")
    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": "codegen-key",
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [best, runner],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    assert plan["selections"] == []
    confidence = plan["low_confidence_selections"][0]["confidence"]
    assert confidence["sample_counts"] == {"best": 1, "runner_up": 1}
    assert confidence["reasons"] == ["best_insufficient_repeats", "runner_up_insufficient_repeats"]


def test_build_execution_plan_preserves_profiled_split_k_and_workspace(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_execution_plan_split_k_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workloads = build_profile_workloads(lowered, kernel_manifest)
    split1_timing = _profile_timing([0.25, 0.25, 0.25], iterations=5)
    split4_timing = _profile_timing([0.10, 0.10, 0.10], iterations=5)
    split1 = _profile_result(workloads[0], split1_timing["median_ms"], 5, profile_key="split1-profile", status="ok", timing=split1_timing)
    split4 = _profile_result(
        replace(workloads[0], split_k=4, workspace_nbytes=8192),
        split4_timing["median_ms"],
        5,
        profile_key="split4-profile",
        status="ok",
        timing=split4_timing,
    )
    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "artifact": str(tmp_path / "profile_execution_plan_split_k_gemm.dinoml"),
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": "codegen-key",
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [split1, split4],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    selection = plan["selections"][0]
    assert selection["selected_candidate_id"] == workloads[0].candidate_id
    assert selection["split_k"] == 4
    assert selection["profile_variant"] == {"split_k": 4}
    assert selection["workspace_nbytes"] == 8192
    assert plan["static_selections"][0]["split_k"] == 4


def test_build_execution_plan_marks_static_conflict_for_shape_specific_winners(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_execution_plan_conflict",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workloads = build_profile_workloads(lowered, kernel_manifest)
    shape_a = _profile_result(workloads[0], 0.10, 5, profile_key="shape-a", status="ok")
    shape_b = _profile_result(replace(workloads[1], node_id="n1", m=8), 0.09, 5, profile_key="shape-b", status="ok")
    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": "codegen-key",
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [shape_a, shape_b],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    assert plan["summary"] == {"selection_count": 2, "low_confidence_count": 0, "static_selection_count": 0, "conflict_count": 1}
    assert plan["static_selections"] == []
    assert plan["conflicts"][0]["reason"] == "profiled_shapes_selected_different_candidate_or_split_k"
    assert plan["conflicts"][0]["selected_candidate_ids"] == sorted(
        [workloads[0].candidate_id, workloads[1].candidate_id]
    )
    assert plan["conflicts"][0]["selected_split_k"] == [1]


def test_build_execution_plan_marks_static_conflict_for_shape_specific_split_k(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_execution_plan_split_k_conflict",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workloads = build_profile_workloads(lowered, kernel_manifest)
    shape_a = _profile_result(workloads[0], 0.10, 5, profile_key="shape-a", status="ok")
    shape_b = _profile_result(
        replace(workloads[0], node_id="n1", m=8, split_k=4, workspace_nbytes=8192),
        0.09,
        5,
        profile_key="shape-b",
        status="ok",
    )
    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": "codegen-key",
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [shape_a, shape_b],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    assert plan["summary"] == {"selection_count": 2, "low_confidence_count": 0, "static_selection_count": 0, "conflict_count": 1}
    assert plan["conflicts"][0]["reason"] == "profiled_shapes_selected_different_candidate_or_split_k"
    assert plan["conflicts"][0]["selected_candidate_ids"] == [workloads[0].candidate_id]
    assert plan["conflicts"][0]["selected_split_k"] == [1, 4]


def test_build_execution_plan_preserves_bucket_shape_metadata(tmp_path):
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, 16], "float16")},
        name="profile_execution_plan_bucket_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    workloads = build_profile_workloads(lowered, kernel_manifest)
    candidate_count = _cutlass_candidate_count("float16")
    case_a = _profile_result(workloads[0], 0.25, 5, profile_key="bucket-a", status="ok")
    case_b = _profile_result(workloads[candidate_count], 0.30, 5, profile_key="bucket-b", status="ok")
    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [case_a, case_b],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    assert [selection["shape"]["source"] for selection in plan["selections"]] == ["dim_buckets", "dim_buckets"]
    assert [selection["shape"]["case_id"] for selection in plan["selections"]] == [
        "bucket_batch=2",
        "bucket_batch=4",
    ]
    assert plan["static_selections"][0]["shape"]["profiled_shapes"][0]["dims"] == {"batch": 2}
    assert plan["static_selections"][0]["shape"]["profiled_shapes"][1]["dims"] == {"batch": 4}


def test_symbolic_expr_profile_shapes_feed_cache_keys_and_execution_plan(tmp_path):
    tokens = dml.Dim("tokens", min=4, max=8, buckets=(4, 8))
    token_dim = dml.TensorSpec([tokens]).shape_spec[0]
    tokens_plus_one = dml.ops.int_add(token_dim, 1)
    spec = dml.trace(
        GemmModule("gemm_rcr"),
        inputs={
            "a": dml.TensorSpec([token_dim, 32], "float16"),
            "b": dml.TensorSpec([tokens_plus_one, 32], "float16"),
        },
        name="profile_symbolic_expr_plan_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    workloads = build_profile_workloads(lowered, kernel_manifest)
    case_4 = next(workload for workload in workloads if workload.shape_case_id == "bucket_tokens=4" and workload.split_k == 1)
    case_8 = next(workload for workload in workloads if workload.shape_case_id == "bucket_tokens=8" and workload.split_k == 1)
    manifest = {"target": {"name": "cuda", "arch": "sm_86"}}
    context = {"fingerprint": {"hardware_key": "hardware-key", "support_libraries_key": "support-key"}}

    payload_4 = _profile_key_payload(case_4, manifest, kernel_manifest, codegen_plan, context=context)
    payload_8 = _profile_key_payload(case_8, manifest, kernel_manifest, codegen_plan, context=context)

    assert payload_4["shape"] == {"m": 4, "n": 5, "k": 32}
    assert payload_8["shape"] == {"m": 8, "n": 9, "k": 32}
    assert _profile_key(payload_4) != _profile_key(payload_8)

    report = {
        "schema_version": PROFILE_REPORT_SCHEMA_VERSION,
        "profile_cache_schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "kernel_manifest_cache_key": kernel_manifest["cache_key"],
        "codegen_plan_cache_key": codegen_plan["cache_key"],
        "fingerprint": {"schema_version": 1, "key": "fingerprint-key"},
        "hardware_cache_key": "hardware-key",
        "support_libraries_cache_key": "support-key",
        "problems": [
            _profile_result(case_4, 0.25, 5, profile_key="symbolic-bucket-4", status="ok"),
            _profile_result(case_8, 0.30, 5, profile_key="symbolic-bucket-8", status="ok"),
        ],
        "summary": {"cached": 0, "failed": 0, "profiled": 2, "skipped": 0},
    }

    plan = build_execution_plan(report)

    assert [selection["shape"]["case_id"] for selection in plan["selections"]] == [
        "bucket_tokens=4",
        "bucket_tokens=8",
    ]
    assert plan["static_selections"][0]["shape"]["profiled_shapes"][0]["dims"] == {"tokens": 4}
    assert plan["static_selections"][0]["shape"]["profiled_shapes"][1]["dims"] == {"tokens": 8}


def test_profile_artifact_uses_cache_before_running(tmp_path, monkeypatch):
    monkeypatch.setattr(
        profiling_mod,
        "_cuda_hardware_fingerprint",
        lambda target: {
            "backend": "cuda",
            "target_arch": target["arch"],
            "cuda_visible_devices": "",
            "nvidia_smi": "available",
            "devices": [
                {
                    "index": 0,
                    "name": "NVIDIA GeForce RTX 3090",
                    "compute_capability": "8.6",
                    "driver_version": "555.42",
                    "memory_total_mib": 24576,
                }
            ],
            "nvcc": {"available": "true", "release": "12.8", "build": "12.8.42"},
        },
    )
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="cached_profile_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    manifest = {
        "artifact_schema_version": 1,
        "runtime_abi_version": 7,
        "name": "cached_profile_gemm",
        "target": {"name": "cuda", "arch": "sm_86"},
        "files": {
            "graph": "graph.dinoir.json",
            "kernel_manifest": "kernel_manifest.json",
            "kernel_codegen_plan": "kernel_codegen_plan.json",
            "cutlass_gemm_library": "lib/libdinoml_cutlass_gemm.so",
        },
    }
    artifact = tmp_path / "cached_profile_gemm.dinoml"
    artifact.mkdir()
    artifact_lib = artifact / "lib" / "libdinoml_cutlass_gemm.so"
    artifact_lib.parent.mkdir()
    artifact_lib.write_bytes(b"artifact cutlass gemm")
    cache_dir = Path(codegen_plan["external_support_libraries"][0]["cache_dir"])
    cache_lib = cache_dir / "lib" / "libdinoml_cutlass_gemm.so"
    cache_lib.parent.mkdir(parents=True)
    cache_lib.write_bytes(b"cached cutlass gemm")
    write_json(
        cache_dir / "lib" / "cutlass_gemm_manifest.json",
        {
            "schema_version": 1,
            "target": {"name": "cuda", "arch": "sm_86"},
            "provider": "cutlass",
            "source_sha256": "source-hash",
            "library_sha256": "library-hash",
            "source_manifest": "../src/source_manifest.json",
            "provenance_key": "provenance-hash",
            "build_fingerprint": "provenance-hash",
            "family_cache_key": "family-hash",
            "used_candidate_plan_key": "used-plan-hash",
            "external_kernel_plan_cache_key": "external-plan-hash",
            "compile": {"flags": ["-arch=sm_86"]},
            "provenance": {
                "provenance_key": "provenance-hash",
                "compile_flags": ["-arch=sm_86"],
            },
            "cache_key": "cutlass-cache",
        },
    )
    write_json(artifact / "manifest.json", manifest)
    write_json(artifact / "graph.dinoir.json", lowered)
    write_json(artifact / "kernel_manifest.json", kernel_manifest)
    write_json(artifact / "kernel_codegen_plan.json", codegen_plan)

    context = profiling_mod._profile_context(artifact, manifest, codegen_plan)
    workloads = build_profile_workloads(lowered, kernel_manifest)
    cached_entries = {}
    key_payload = None
    for idx, workload in enumerate(workloads):
        key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context)
        profile_key = _profile_key(key_payload)
        elapsed = 0.125 if idx == 0 else 0.25
        timing = _profile_timing([elapsed, elapsed, elapsed], iterations=9)
        cached_result = _profile_result(workload, timing["median_ms"], 9, profile_key=profile_key, status="ok", timing=timing)
        cached_entries[profile_key] = _cache_entry(workload, cached_result, key_payload)
    cache = {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "entries": cached_entries,
    }
    cache_path = profile_cache_path(codegen_plan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, cache)

    report = profile_artifact(artifact, iterations=3, repeats=3)

    assert report["summary"] == {"cached": len(workloads), "failed": 0, "profiled": 0, "skipped": 0}
    execution_plan_path = artifact / "debug" / "execution_plan.json"
    assert report["execution_plan"]["path"] == str(execution_plan_path.resolve())
    assert report["execution_plan"]["schema_version"] == EXECUTION_PLAN_SCHEMA_VERSION
    assert report["execution_plan"]["selection_count"] == 1
    assert read_json(execution_plan_path)["summary"]["selection_count"] == 1
    assert report["profile_cache_schema_version"] == PROFILE_CACHE_SCHEMA_VERSION
    assert report["kernel_manifest_cache_key"] == kernel_manifest["cache_key"]
    assert report["codegen_plan_cache_key"] == codegen_plan["cache_key"]
    assert report["fingerprint"]["schema_version"] == 1
    assert report["fingerprint"]["key"]
    assert report["hardware"]["devices"][0]["name"] == "NVIDIA GeForce RTX 3090"
    assert key_payload is not None
    assert report["fingerprint"]["hardware_key"] == key_payload["hardware_fingerprint_key"]
    assert report["hardware_cache_key"] == key_payload["hardware_fingerprint_key"]
    assert report["libraries"][0]["artifact_sha256"] == hashlib.sha256(b"artifact cutlass gemm").hexdigest()
    assert report["fingerprint"]["support_libraries"][0]["source_sha256"] == "source-hash"
    assert report["fingerprint"]["support_libraries"][0]["library_sha256"] == "library-hash"
    assert report["fingerprint"]["support_libraries"][0]["source_manifest"] == "../src/source_manifest.json"
    assert report["fingerprint"]["support_libraries"][0]["provenance_key"] == "provenance-hash"
    assert report["fingerprint"]["support_libraries"][0]["build_fingerprint"] == "provenance-hash"
    assert report["fingerprint"]["support_libraries"][0]["family_cache_key"] == "family-hash"
    assert report["fingerprint"]["support_libraries"][0]["used_candidate_plan_key"] == "used-plan-hash"
    assert report["fingerprint"]["support_libraries"][0]["compile"]["flags"] == ["-arch=sm_86"]
    assert report["fingerprint"]["support_libraries"][0]["provenance"]["provenance_key"] == "provenance-hash"
    assert report["support_libraries_cache_key"] == key_payload["support_libraries_fingerprint_key"]
    assert report["problems"][0]["status"] == "cached"
    expected_candidate = next(
        candidate for candidate in _cutlass_candidates("float32") if int(candidate["cutlass"]["align"]) <= 2
    )
    assert report["problems"][0]["selected"]["candidate_id"] == expected_candidate["candidate_id"]
    assert report["problems"][0]["selected"]["reason"] == "cache_hit"
    assert report["problems"][0]["candidates"][0]["candidate_config_key"]
    assert report["problems"][0]["profile_key"] in cached_entries


def test_profile_artifact_rejects_cache_hit_with_mismatched_embedded_key(tmp_path, monkeypatch):
    monkeypatch.setattr(
        profiling_mod,
        "_cuda_hardware_fingerprint",
        lambda target: {
            "backend": "cuda",
            "target_arch": target["arch"],
            "cuda_visible_devices": "",
            "nvidia_smi": "unavailable",
            "devices": [],
            "nvcc": {"available": "false"},
        },
    )
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="stale_key_profile_gemm",
    )
    target = {"name": "cuda", "arch": "sm_86", "no_tf32": True}
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, target)
    codegen_plan = create_codegen_plan(kernel_manifest, tmp_path / "cache").to_json()
    manifest = {
        "artifact_schema_version": 1,
        "runtime_abi_version": 7,
        "name": "stale_key_profile_gemm",
        "target": target,
        "files": {
            "graph": "graph.dinoir.json",
            "kernel_manifest": "kernel_manifest.json",
            "kernel_codegen_plan": "kernel_codegen_plan.json",
            "cutlass_gemm_library": "lib/libdinoml_cutlass_gemm.so",
        },
    }
    artifact = tmp_path / "stale_key_profile_gemm.dinoml"
    artifact.mkdir()
    artifact_lib = artifact / "lib" / "libdinoml_cutlass_gemm.so"
    artifact_lib.parent.mkdir()
    artifact_lib.write_bytes(b"artifact cutlass gemm")
    cache_dir = Path(codegen_plan["external_support_libraries"][0]["cache_dir"])
    cache_lib = cache_dir / "lib" / "libdinoml_cutlass_gemm.so"
    cache_lib.parent.mkdir(parents=True)
    cache_lib.write_bytes(b"cached cutlass gemm")
    write_json(
        cache_dir / "lib" / "cutlass_gemm_manifest.json",
        {
            "schema_version": 1,
            "target": target,
            "provider": "cutlass",
            "source_sha256": "source-hash",
            "library_sha256": "library-hash",
            "source_manifest": "../src/source_manifest.json",
            "provenance_key": "provenance-hash",
            "build_fingerprint": "provenance-hash",
            "family_cache_key": "family-hash",
            "used_candidate_plan_key": "used-plan-hash",
            "external_kernel_plan_cache_key": "external-plan-hash",
            "cache_key": "cutlass-cache",
        },
    )
    write_json(artifact / "manifest.json", manifest)
    write_json(artifact / "graph.dinoir.json", lowered)
    write_json(artifact / "kernel_manifest.json", kernel_manifest)
    write_json(artifact / "kernel_codegen_plan.json", codegen_plan)

    context = profiling_mod._profile_context(artifact, manifest, codegen_plan)
    workloads = build_profile_workloads(lowered, kernel_manifest)
    workload = workloads[0]
    key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context)
    profile_key = _profile_key(key_payload)
    timing = _profile_timing([0.10, 0.10, 0.10], iterations=9)
    cached_result = _profile_result(workload, timing["median_ms"], 9, profile_key=profile_key, status="ok", timing=timing)
    stale_entry = _cache_entry(workload, cached_result, key_payload)
    stale_entry["key"] = {**key_payload, "support_libraries_fingerprint_key": "stale-support-key"}
    cache_path = profile_cache_path(codegen_plan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(
        cache_path,
        {
            "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
            "target": target,
            "entries": {profile_key: stale_entry},
        },
    )

    profile_calls = []

    class FakeProfiler:
        def __init__(self, artifact_dir, manifest):
            self.artifact_dir = artifact_dir
            self.manifest = manifest

        def profile(self, workload, *, iterations, rng):
            profile_calls.append((workload.candidate_id, iterations))
            return (0.125, workload.workspace_nbytes)

        def close(self):
            profile_calls.append(("close", 0))

    monkeypatch.setattr(profiling_mod, "_CudaProfiler", FakeProfiler)

    report = profile_artifact(artifact, iterations=3, repeats=3)

    assert report["summary"] == {"cached": 0, "failed": 0, "profiled": len(workloads), "skipped": 0}
    assert report["problems"][0]["status"] == "ok"
    assert report["problems"][0]["selected"]["reason"] == "only_candidate"
    profiled_calls = profile_calls[:-1]
    assert len(profiled_calls) == len(workloads) * 3
    assert {iterations for _, iterations in profiled_calls} == {3}
    assert [candidate_id for candidate_id, _ in profiled_calls[::3]] == [workload.candidate_id for workload in workloads]
    assert profile_calls[-1] == ("close", 0)
    refreshed_cache = profiling_mod._read_profile_cache(cache_path, target)
    assert refreshed_cache["entries"][profile_key]["key"] == key_payload


def test_cuda_profiler_check_reads_cuda_runtime_last_error():
    class FakeGetter:
        restype = None

        def __init__(self, message):
            self.message = message

        def __call__(self):
            return self.message

    profiler = object.__new__(profiling_mod._CudaProfiler)
    profiler._runtime = SimpleNamespace(dino_get_last_error=FakeGetter(b"stale runtime failure"))
    profiler._cuda_runtime = SimpleNamespace(dino_get_last_error=FakeGetter(b"fresh cuda helper failure"))

    with pytest.raises(RuntimeError, match="fresh cuda helper failure"):
        profiler._check(7)


def test_cuda_profiler_close_retains_failed_free_for_retry():
    frees = []
    fail_ptr = 0x1000

    class FakeCudaRuntime:
        def dino_device_free(self, ptr):
            frees.append(ptr.value)
            return 9 if ptr.value == fail_ptr else 0

    def check(code):
        if code:
            raise RuntimeError("profiler free failed")

    profiler = object.__new__(profiling_mod._CudaProfiler)
    profiler._buffers = [ctypes.c_void_p(0x1000), ctypes.c_void_p(0x2000)]
    profiler._cuda_runtime = FakeCudaRuntime()
    profiler._check = check

    with pytest.raises(RuntimeError, match="profiler free failed"):
        profiler.close()

    assert frees == [0x2000, 0x1000]
    assert len(profiler._buffers) == 1
    assert profiler._buffers[0].value == 0x1000

    fail_ptr = 0
    profiler.close()

    assert frees == [0x2000, 0x1000, 0x1000]
    assert profiler._buffers == []


@pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc is required")
def test_cuda_profile_artifact_writes_cutlass_gemm_report(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA device is required")
    if not discover_cuda_libraries()["cutlass"].available:
        pytest.skip("CUTLASS headers are not available")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    spec = dml.trace(
        GemmModule("gemm_rcr"),
        inputs={"a": dml.TensorSpec([8, 16], "float32"), "b": dml.TensorSpec([12, 16], "float32")},
        name="profile_gemm_rcr",
    )
    artifact = dml.compile(spec, dml.Target("cuda", arch="sm_86"), tmp_path / "profile_gemm_rcr.dinoml")
    report = profile_artifact(artifact.path, iterations=3)

    report_path = artifact.path / "debug" / "profile_report.json"
    assert read_json(report_path) == report
    assert report["schema_version"] == PROFILE_REPORT_SCHEMA_VERSION
    assert report["profile_cache_schema_version"] == PROFILE_CACHE_SCHEMA_VERSION
    assert report["fingerprint"]["schema_version"] == 1
    assert report["hardware"]["backend"] == "cuda"
    assert report["fingerprint"]["hardware_key"] == report["hardware_cache_key"]
    assert report["fingerprint"]["support_libraries_key"] == report["support_libraries_cache_key"]
    assert report["fingerprint"]["support_libraries"][0]["name"] == "cutlass_gemm"
    assert report["libraries"][0]["artifact_sha256"]
    assert report["summary"] == {"cached": 0, "failed": 0, "profiled": _cutlass_candidate_count("float32"), "skipped": 0}
    assert len(report["problems"]) == _cutlass_candidate_count("float32")
    workload = report["problems"][0]
    assert workload["status"] == "ok"
    assert workload["profiler_symbol"] == f"dinoml_profile_cutlass_gemm_rcr_float32_{_cutlass_default_symbol_id('float32')}"
    assert workload["m"] == 8
    assert workload["n"] == 12
    assert workload["k"] == 16
    assert workload["elapsed_ms"] >= 0.0
    assert workload["flops"] == 2 * 8 * 12 * 16
    assert workload["selected"]["reason"] == "only_candidate"
    assert workload["selected"]["candidate_id"] == _cutlass_default_candidate_id("float32")
    candidate = workload["candidates"][0]
    assert candidate["candidate_id"] == _cutlass_default_candidate_id("float32")
    assert candidate["provider"] == "cutlass"
    assert candidate["family"] == "gemm_universal"
    assert candidate["layouts"] == {"a": "row", "b": "column", "c": "row"}
    assert candidate["kernel_symbol"] == f"dinoml_cutlass_gemm_rcr_float32_{_cutlass_default_symbol_id('float32')}"
    assert candidate["profiler_symbol"] == f"dinoml_profile_cutlass_gemm_rcr_float32_{_cutlass_default_symbol_id('float32')}"
    assert candidate["candidate_config_key"]


def test_cli_profile_smoke(monkeypatch, capsys):
    calls = []

    def fake_profile_artifact(artifact, *, input_shapes, iterations, repeats, output, execution_plan_output, refresh):
        calls.append((artifact, input_shapes, iterations, repeats, output, execution_plan_output, refresh))
        return {
            "artifact": str(artifact),
            "target": {"name": "cuda", "arch": "sm_86"},
            "iterations": iterations,
            "repeats": repeats,
            "execution_plan": {"path": "plan.json", "selection_count": 1},
            "summary": {"cached": 0, "failed": 0, "profiled": 1, "skipped": 0},
            "problems": [
                {
                    "node_id": "n0",
                    "op": "gemm_rrr",
                    "dtype": "float32",
                    "profiler_symbol": f"dinoml_profile_cutlass_gemm_rrr_float32_{_cutlass_default_symbol_id("float32")}",
                    "m": 4,
                    "n": 6,
                    "k": 8,
                    "split_k": 2,
                    "workspace_nbytes": 4096,
                    "elapsed_ms": 0.01,
                    "timing": {"samples_ms": [0.011, 0.01], "median_ms": 0.0105, "repeats": 2},
                    "tflops": 0.00384,
                }
            ],
        }

    monkeypatch.setattr(cli, "profile_artifact", fake_profile_artifact)

    assert (
        cli.main(
            [
                "profile",
                "artifact.dinoml",
                "--iterations",
                "2",
                "--repeats",
                "2",
                "--shape",
                "a=4,8",
                "--out",
                "report.json",
                "--execution-plan-out",
                "plan.json",
                "--refresh",
            ]
        )
        == 0
    )
    stdout = capsys.readouterr().out

    assert calls == [("artifact.dinoml", {"a": (4, 8)}, 2, 2, "report.json", "plan.json", True)]
    assert f"dinoml_profile_cutlass_gemm_rrr_float32_{_cutlass_default_symbol_id("float32")}" in stdout
    assert '"repeats": 2' in stdout
    assert '"split_k": 2' in stdout
    assert '"workspace_nbytes": 4096' in stdout
    assert "plan.json" in stdout


def test_cli_compile_forwards_execution_plan(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "model.py"
    model_path.write_text("def build_spec():\n    return 'spec'\n", encoding="utf-8")
    calls = []

    def fake_compile(spec, *, target, output, execution_plan, constant_load_policy):
        calls.append((spec, target.to_json(), output, execution_plan, constant_load_policy))

        class FakeArtifact:
            path = Path(output)

        return FakeArtifact()

    monkeypatch.setattr(cli.dml, "compile", fake_compile)

    assert (
        cli.main(
            [
                "compile",
                str(model_path),
                "--target",
                "cuda",
                "--arch",
                "sm_86",
                "--execution-plan",
                "plan.json",
                "--out",
                str(tmp_path / "artifact.dinoml"),
            ]
        )
        == 0
    )

    assert calls == [
        (
            "spec",
            {"name": "cuda", "arch": "sm_86", "no_tf32": False, "use_fp16_acc": False},
            str(tmp_path / "artifact.dinoml"),
            "plan.json",
            "eager",
        )
    ]
    assert "artifact.dinoml" in capsys.readouterr().out


def test_compile_profile_runs_two_phase_rebuild(tmp_path, monkeypatch):
    build_calls = []
    lower_calls = []
    report_calls = []
    lowered_ir = {"name": "lowered-once"}
    reports = [{"name": "pass-report"}]

    def fake_lower_for_compile(spec, target, *, artifact_dir, pass_manager):
        lower_calls.append((spec, target.to_json(), str(artifact_dir), pass_manager))
        return lowered_ir, reports

    def fake_build_artifact(
        spec,
        target,
        *,
        artifact_dir,
        generated_src_dir,
        lowered_ir,
        reports,
        backend,
        execution_plan_payload,
        constant_load_policy,
    ):
        del backend
        assert constant_load_policy == "eager"
        (artifact_dir / "debug").mkdir(parents=True, exist_ok=True)
        stale_source = generated_src_dir / "stale_candidate_source.cu"
        if execution_plan_payload is None:
            generated_src_dir.mkdir(parents=True, exist_ok=True)
            stale_source.write_text("// candidate-only source\n", encoding="utf-8")
        else:
            assert not stale_source.exists()
        build_calls.append(
            {
                "spec": spec,
                "target": target.to_json(),
                "output": str(artifact_dir),
                "lowered_ir": lowered_ir,
                "reports": reports,
                "execution_plan": execution_plan_payload,
            }
        )
        return compiler_mod.Artifact(artifact_dir)

    def fake_profile_artifact(artifact, *, input_shapes, iterations, repeats, refresh):
        artifact = Path(artifact)
        plan_path = artifact / "debug" / "execution_plan.json"
        plan = {
            "schema_version": 1,
            "kind": "dinoml.execution_plan",
            "static_selections": [{"selection_key": "profile-selection"}],
        }
        write_json(plan_path, plan)
        report_calls.append((str(artifact), input_shapes, iterations, repeats, refresh))
        return {
            "artifact": str(artifact),
            "execution_plan": {"path": str(plan_path), "selection_count": 1},
            "summary": {"cached": 0, "failed": 0, "profiled": 1, "skipped": 0},
            "problems": [],
        }

    monkeypatch.setattr(compiler_mod, "_lower_for_compile", fake_lower_for_compile)
    monkeypatch.setattr(compiler_mod, "_build_artifact_from_lowered_ir", fake_build_artifact)
    monkeypatch.setattr(compiler_mod, "profile_artifact", fake_profile_artifact)

    artifact = compiler_mod.compile(
        "spec",
        dml.Target("cuda", arch="sm_86"),
        tmp_path / "profiled.dinoml",
        profile=True,
        profile_iterations=7,
        profile_repeats=3,
        profile_input_shapes={"a": (4, 8)},
        profile_refresh=True,
    )

    assert artifact.path == (tmp_path / "profiled.dinoml").resolve()
    assert len(lower_calls) == 1
    assert [call["execution_plan"] for call in build_calls] == [
        None,
        {"schema_version": 1, "kind": "dinoml.execution_plan", "static_selections": [{"selection_key": "profile-selection"}]},
    ]
    assert build_calls[0]["lowered_ir"] is lowered_ir
    assert build_calls[1]["lowered_ir"] is lowered_ir
    assert build_calls[0]["reports"] is reports
    assert build_calls[1]["reports"] is reports
    assert report_calls == [(str(artifact.path), {"a": (4, 8)}, 7, 3, True)]
    assert read_json(artifact.path / "debug" / "bootstrap_profile_report.json")["execution_plan"]["selection_count"] == 1


def test_cuda_linear_profile_compile_consumes_profiled_execution_plan(tmp_path, monkeypatch):
    from examples.cuda_linear import IN_FEATURES, OUT_FEATURES, VALIDATION_BATCH, build_spec

    build_calls = []

    def fake_build(ir, *, target, artifact_dir, generated_src_dir, kernel_manifest):
        del ir, generated_src_dir
        build_calls.append(
            {
                "target": target.to_json(),
                "artifact_dir": artifact_dir,
                "kernel_manifest": kernel_manifest,
            }
        )

    def fake_profile_artifact(artifact, *, input_shapes, iterations, repeats, refresh):
        artifact = Path(artifact)
        kernel_manifest = read_json(artifact / "kernel_manifest.json")
        manifest = read_json(artifact / "manifest.json")
        required = kernel_manifest["required_kernels"][0]
        selected_candidate = required["candidates"][1]
        plan_path = artifact / "debug" / "execution_plan.json"
        plan = {
            "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
            "kind": "dinoml.execution_plan",
            "target": manifest["target"],
            "kernel_manifest_cache_key": kernel_manifest["cache_key"],
            "execution_plan_key": "cuda-linear-profile-plan",
            "selection_policy": "test-profile-assisted-linear",
            "selection_confidence_policy": {"name": "single-sample-smoke"},
            "static_selection_policy": "unique_selected_candidate_per_op_dtype_candidate_set",
            "summary": {
                "selection_count": 1,
                "static_selection_count": 1,
                "conflict_count": 0,
                "low_confidence_count": 0,
            },
            "static_selections": [
                {
                    "selection_key": "cuda-linear-profile-selection",
                    "op": required["op"],
                    "dtype": "float32",
                    "candidate_set_key": required["candidate_set_key"],
                    "selected_candidate_id": selected_candidate["candidate_id"],
                    "candidate_config_key": selected_candidate["candidate_config_key"],
                    "kernel_symbol": selected_candidate["kernel_symbol"],
                    "profiler_symbol": selected_candidate["profiler_symbol"],
                    "shape": {"m": VALIDATION_BATCH, "n": OUT_FEATURES, "k": IN_FEATURES},
                    "avg_ms": 0.01,
                    "confidence": {"confident": True, "level": "high"},
                    "split_k": 1,
                    "workspace_nbytes": 0,
                }
            ],
        }
        write_json(plan_path, plan)
        return {
            "artifact": str(artifact),
            "target": manifest["target"],
            "iterations": iterations,
            "repeats": repeats,
            "execution_plan": {
                "path": str(plan_path),
                "schema_version": EXECUTION_PLAN_SCHEMA_VERSION,
                "execution_plan_key": "cuda-linear-profile-plan",
                "selection_count": 1,
            },
            "summary": {"cached": 0, "failed": 0, "profiled": 1, "skipped": 0},
            "problems": [],
            "input_shapes": {name: list(shape) for name, shape in (input_shapes or {}).items()},
            "refresh": refresh,
        }

    monkeypatch.setattr(BackendSpec, "resolve_build_function", lambda self: fake_build)
    monkeypatch.setattr(compiler_mod, "profile_artifact", fake_profile_artifact)

    target = dml.Target("cuda", arch="sm_86", no_tf32=True)
    artifact = dml.compile(
        build_spec(),
        target,
        tmp_path / "cuda_linear_profiled.dinoml",
        profile=True,
        profile_iterations=1,
        profile_repeats=1,
        profile_input_shapes={"x": (VALIDATION_BATCH, IN_FEATURES)},
    )

    kernel_manifest = read_json(artifact.path / "kernel_manifest.json")
    codegen_plan = read_json(artifact.path / "kernel_codegen_plan.json")
    compile_config = read_json(artifact.path / "compile_config.json")
    bootstrap_report = read_json(artifact.path / "debug" / "bootstrap_profile_report.json")
    metadata = read_json(artifact.path / "metadata.json")
    required = kernel_manifest["required_kernels"][0]
    final_selection = required["execution_plan_selection"]
    default_selection_id = build_calls[0]["kernel_manifest"]["required_kernels"][0]["selected_candidate_id"]

    assert len(build_calls) == 2
    assert default_selection_id != final_selection["selected_candidate_id"]
    assert build_calls[1]["kernel_manifest"] == kernel_manifest
    assert required["op"] == "gemm_rrr_bias"
    assert required["candidate_set_id"] == "cutlass_gemm_rrr_bias_float32_bias_v1"
    assert required["candidate_set"]["target_policy"]["no_tf32"] is True
    expected_candidate_count = _cutlass_candidate_count(
        "float32",
        op_name="gemm_rrr_bias",
        target=target.to_json(),
    )
    assert required["candidate_set"]["candidate_count"] == expected_candidate_count
    assert len(required["candidates"]) == expected_candidate_count
    assert required["selected_candidate_id"] == final_selection["selected_candidate_id"]
    assert required["kernel_symbol"] == final_selection["kernel_symbol"]
    assert required["profiler_symbol"] == final_selection["profiler_symbol"]
    assert final_selection["shape"] == {"m": VALIDATION_BATCH, "n": OUT_FEATURES, "k": IN_FEATURES}
    assert codegen_plan["kernel_symbols"] == [final_selection["kernel_symbol"]]
    assert codegen_plan["profiler_symbols"] == [final_selection["profiler_symbol"]]
    assert compile_config["execution_plan"]["execution_plan_key"] == "cuda-linear-profile-plan"
    assert bootstrap_report["execution_plan"]["selection_count"] == 1
    assert bootstrap_report["iterations"] == 1
    assert bootstrap_report["repeats"] == 1
    assert bootstrap_report["input_shapes"] == {"x": [VALIDATION_BATCH, IN_FEATURES]}
    assert [constant["name"] for constant in metadata["constants"]] == ["weight", "bias"]
    assert metadata["inputs"][0]["shape_spec"][0]["buckets"] == [1, VALIDATION_BATCH, 4]


def test_compile_profile_keeps_initial_artifact_when_no_candidates(tmp_path, monkeypatch):
    build_calls = []
    lower_calls = []

    def fake_lower_for_compile(spec, target, *, artifact_dir, pass_manager):
        lower_calls.append((spec, target.to_json(), str(artifact_dir), pass_manager))
        return {"name": "lowered-once"}, []

    def fake_build_artifact(
        spec,
        target,
        *,
        artifact_dir,
        generated_src_dir,
        lowered_ir,
        reports,
        backend,
        execution_plan_payload,
        constant_load_policy,
    ):
        del spec, target, generated_src_dir, lowered_ir, reports, backend
        assert constant_load_policy == "eager"
        (artifact_dir / "debug").mkdir(parents=True, exist_ok=True)
        build_calls.append(execution_plan_payload)
        return compiler_mod.Artifact(artifact_dir)

    def fake_profile_artifact(artifact, *, input_shapes, iterations, repeats, refresh):
        del input_shapes, iterations, repeats, refresh
        artifact = Path(artifact)
        plan_path = artifact / "debug" / "execution_plan.json"
        write_json(
            plan_path,
            {
                "schema_version": 1,
                "kind": "dinoml.execution_plan",
                "selections": [],
                "static_selections": [],
                "summary": {"selection_count": 0},
            },
        )
        return {
            "artifact": str(artifact),
            "execution_plan": {"path": str(plan_path), "selection_count": 0},
            "summary": {"cached": 0, "failed": 0, "profiled": 0, "skipped": 0},
            "problems": [],
        }

    monkeypatch.setattr(compiler_mod, "_lower_for_compile", fake_lower_for_compile)
    monkeypatch.setattr(compiler_mod, "_build_artifact_from_lowered_ir", fake_build_artifact)
    monkeypatch.setattr(compiler_mod, "profile_artifact", fake_profile_artifact)

    artifact = compiler_mod.compile("spec", dml.Target("cuda", arch="sm_86"), tmp_path / "profiled.dinoml", profile=True)

    assert artifact.path == (tmp_path / "profiled.dinoml").resolve()
    assert len(lower_calls) == 1
    assert build_calls == [None]
    assert read_json(artifact.path / "debug" / "bootstrap_profile_report.json")["execution_plan"]["selection_count"] == 0


def test_compile_profile_accepts_sourceable_symbolic_shape_expressions(tmp_path, monkeypatch):
    build_calls = []
    profile_calls = []
    tokens = {"kind": "dim", "name": "tokens", "min": 4, "max": 16, "buckets": [4, 8, 16]}
    pooled_tokens = {"kind": "int_expr", "op": "div", "lhs": tokens, "rhs": 2}
    lowered_ir = {
        "name": "profile_sourceable_expr",
        "inputs": [
            {"name": "x", "tensor": "x", "shape": [16, 8], "shape_spec": [tokens, 8], "dtype": "float32"}
        ],
        "outputs": [
            {"name": "y", "tensor": "y", "shape": [8, 8], "shape_spec": [pooled_tokens, 8], "dtype": "float32"}
        ],
        "constants": [],
        "tensors": [
            {"name": "x", "shape": [16, 8], "shape_spec": [tokens, 8], "dtype": "float32"},
            {"name": "y", "shape": [8, 8], "shape_spec": [pooled_tokens, 8], "dtype": "float32"},
        ],
        "nodes": [],
        "metadata": {},
    }

    def fake_lower_for_compile(spec, target, *, artifact_dir, pass_manager):
        del spec, target, artifact_dir, pass_manager
        return lowered_ir, []

    def fake_build_artifact(
        spec,
        target,
        *,
        artifact_dir,
        generated_src_dir,
        lowered_ir,
        reports,
        backend,
        execution_plan_payload,
        constant_load_policy,
    ):
        del spec, target, generated_src_dir, lowered_ir, reports, backend
        assert constant_load_policy == "eager"
        (artifact_dir / "debug").mkdir(parents=True, exist_ok=True)
        build_calls.append(execution_plan_payload)
        return compiler_mod.Artifact(artifact_dir)

    def fake_profile_artifact(artifact, *, input_shapes, iterations, repeats, refresh):
        del input_shapes, iterations, repeats, refresh
        artifact = Path(artifact)
        profile_calls.append(str(artifact))
        plan_path = artifact / "debug" / "execution_plan.json"
        write_json(
            plan_path,
            {
                "schema_version": 1,
                "kind": "dinoml.execution_plan",
                "selections": [],
                "static_selections": [],
                "summary": {"selection_count": 0},
            },
        )
        return {
            "artifact": str(artifact),
            "execution_plan": {"path": str(plan_path), "selection_count": 0},
            "summary": {"cached": 0, "failed": 0, "profiled": 0, "skipped": 0},
            "problems": [],
        }

    monkeypatch.setattr(compiler_mod, "_lower_for_compile", fake_lower_for_compile)
    monkeypatch.setattr(compiler_mod, "_build_artifact_from_lowered_ir", fake_build_artifact)
    monkeypatch.setattr(compiler_mod, "profile_artifact", fake_profile_artifact)

    artifact = compiler_mod.compile("spec", dml.Target("cuda", arch="sm_86"), tmp_path / "profiled.dinoml", profile=True)

    assert artifact.path == (tmp_path / "profiled.dinoml").resolve()
    assert build_calls == [None]
    assert profile_calls == [str(artifact.path)]
    assert read_json(artifact.path / "debug" / "bootstrap_profile_report.json")["execution_plan"]["selection_count"] == 0


def test_compile_profile_rejects_ambiguous_or_non_cuda_requests(tmp_path):
    with pytest.raises(ValueError, match="cannot also consume"):
        compiler_mod.compile(
            "spec",
            dml.Target("cuda", arch="sm_86"),
            tmp_path / "profiled.dinoml",
            profile=True,
            execution_plan={"schema_version": 1, "static_selections": []},
        )

    with pytest.raises(ValueError, match="CUDA targets only"):
        compiler_mod.compile("spec", dml.Target("cpu"), tmp_path / "profiled_cpu.dinoml", profile=True)


def test_compile_profile_rejects_unsourced_symbolic_shape_expressions(tmp_path, monkeypatch):
    expr = {
        "kind": "int_expr",
        "op": "div",
        "lhs": {"kind": "dim", "name": "tokens", "min": 4, "max": 16},
        "rhs": 2,
    }

    def fake_lower_for_compile(spec, target, *, artifact_dir, pass_manager):
        del spec, target, artifact_dir, pass_manager
        return (
            {
                "name": "profile_expr",
                "inputs": [{"name": "x", "tensor": "x", "shape": [8, 8], "shape_spec": [expr, 8], "dtype": "float32"}],
                "outputs": [{"name": "y", "tensor": "y", "shape": [8, 8], "shape_spec": [8, 8], "dtype": "float32"}],
                "constants": [],
                "tensors": [
                    {"name": "x", "shape": [8, 8], "shape_spec": [expr, 8], "dtype": "float32"},
                    {"name": "y", "shape": [8, 8], "shape_spec": [8, 8], "dtype": "float32"},
                ],
                "nodes": [],
                "metadata": {},
            },
            [],
        )

    monkeypatch.setattr(compiler_mod, "_lower_for_compile", fake_lower_for_compile)

    with pytest.raises(NotImplementedError, match="direct runtime sources.*tokens"):
        compiler_mod.compile("spec", dml.Target("cuda", arch="sm_86"), tmp_path / "profiled.dinoml", profile=True)


def test_compile_profile_rejects_output_only_symbolic_shape_sources(tmp_path, monkeypatch):
    expr = {
        "kind": "int_expr",
        "op": "div",
        "lhs": {"kind": "dim", "name": "tokens", "min": 4, "max": 16},
        "rhs": 2,
    }
    output_tokens = {"kind": "dim", "name": "tokens", "min": 4, "max": 16}
    build_calls = []

    def fake_lower_for_compile(spec, target, *, artifact_dir, pass_manager):
        del spec, target, artifact_dir, pass_manager
        return (
            {
                "name": "profile_expr_output_only_source",
                "inputs": [{"name": "x", "tensor": "x", "shape": [8, 8], "shape_spec": [expr, 8], "dtype": "float32"}],
                "outputs": [
                    {"name": "y", "tensor": "y", "shape": [16, 8], "shape_spec": [output_tokens, 8], "dtype": "float32"}
                ],
                "constants": [],
                "tensors": [
                    {"name": "x", "shape": [8, 8], "shape_spec": [expr, 8], "dtype": "float32"},
                    {"name": "y", "shape": [16, 8], "shape_spec": [output_tokens, 8], "dtype": "float32"},
                ],
                "nodes": [],
                "metadata": {},
            },
            [],
        )

    def fake_build_artifact(*args, **kwargs):
        build_calls.append((args, kwargs))
        return compiler_mod.Artifact(tmp_path / "profiled.dinoml")

    monkeypatch.setattr(compiler_mod, "_lower_for_compile", fake_lower_for_compile)
    monkeypatch.setattr(compiler_mod, "_build_artifact_from_lowered_ir", fake_build_artifact)

    with pytest.raises(NotImplementedError, match="direct runtime sources.*tokens"):
        compiler_mod.compile("spec", dml.Target("cuda", arch="sm_86"), tmp_path / "profiled.dinoml", profile=True)
    assert build_calls == []


def test_profile_artifact_rejects_unsourced_symbolic_shape_expressions(tmp_path):
    expr = {
        "kind": "int_expr",
        "op": "div",
        "lhs": {"kind": "dim", "name": "tokens", "min": 4, "max": 16},
        "rhs": 2,
    }
    artifact = tmp_path / "expr_profile.dinoml"
    artifact.mkdir()
    write_json(
        artifact / "manifest.json",
        {
            "target": {"name": "cuda", "arch": "sm_86"},
            "files": {
                "graph": "graph.dinoir.json",
                "kernel_manifest": "kernel_manifest.json",
                "kernel_codegen_plan": "kernel_codegen_plan.json",
            },
        },
    )
    write_json(
        artifact / "graph.dinoir.json",
        {
            "inputs": [],
            "outputs": [{"name": "y", "tensor": "y", "shape": [8], "shape_spec": [expr], "dtype": "float32"}],
            "constants": [],
            "tensors": [{"name": "y", "shape": [8], "shape_spec": [expr], "dtype": "float32"}],
            "nodes": [],
            "metadata": {},
        },
    )

    with pytest.raises(NotImplementedError, match="direct runtime sources.*tokens"):
        profile_artifact(artifact)


def test_profile_artifact_rejects_symbolic_shape_expressions_in_views(tmp_path):
    expr = {
        "kind": "int_expr",
        "op": "div",
        "lhs": {"kind": "dim", "name": "tokens", "min": 4, "max": 16},
        "rhs": 2,
    }
    artifact = tmp_path / "expr_view_profile.dinoml"
    artifact.mkdir()
    write_json(
        artifact / "manifest.json",
        {
            "target": {"name": "cuda", "arch": "sm_86"},
            "files": {
                "graph": "graph.dinoir.json",
                "kernel_manifest": "kernel_manifest.json",
                "kernel_codegen_plan": "kernel_codegen_plan.json",
            },
        },
    )
    write_json(
        artifact / "graph.dinoir.json",
        {
            "inputs": [],
            "outputs": [{"name": "y", "tensor": "y", "shape": [8], "shape_spec": [8], "dtype": "float32"}],
            "constants": [],
            "tensors": [{"name": "y", "shape": [8], "shape_spec": [8], "dtype": "float32"}],
            "nodes": [],
            "metadata": {
                "memory_plan": {
                    "views": {
                        "views": [
                            {
                                "tensor": "y_view",
                                "source": "y",
                                "shape": [8],
                                "shape_spec": [expr],
                            }
                        ]
                    }
                }
            },
        },
    )

    with pytest.raises(NotImplementedError, match="direct runtime sources.*view tensor 'y_view'.*tokens"):
        profile_artifact(artifact)


def test_cli_compile_forwards_profile_options(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "model.py"
    model_path.write_text("def build_spec():\n    return 'spec'\n", encoding="utf-8")
    calls = []

    def fake_compile(
        spec,
        *,
        target,
        output,
        execution_plan,
        profile,
        profile_iterations,
        profile_repeats,
        profile_input_shapes,
        profile_refresh,
        constant_load_policy,
    ):
        calls.append(
            (
                spec,
                target.to_json(),
                output,
                execution_plan,
                profile,
                profile_iterations,
                profile_repeats,
                profile_input_shapes,
                profile_refresh,
                constant_load_policy,
            )
        )

        class FakeArtifact:
            path = Path(output)

        return FakeArtifact()

    monkeypatch.setattr(cli.dml, "compile", fake_compile)

    assert (
        cli.main(
            [
                "compile",
                str(model_path),
                "--target",
                "cuda",
                "--arch",
                "sm_86",
                "--profile",
                "--profile-iterations",
                "7",
                "--profile-repeats",
                "3",
                "--shape",
                "a=4,8",
                "--profile-refresh",
                "--constant-load-policy",
                "deferred",
                "--out",
                str(tmp_path / "artifact.dinoml"),
            ]
        )
        == 0
    )

    assert calls == [
        (
            "spec",
            {"name": "cuda", "arch": "sm_86", "no_tf32": False, "use_fp16_acc": False},
            str(tmp_path / "artifact.dinoml"),
            None,
            True,
            7,
            3,
            {"a": (4, 8)},
            True,
            "deferred",
        )
    ]
    assert "artifact.dinoml" in capsys.readouterr().out
