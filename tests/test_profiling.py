from dataclasses import replace
import hashlib
import shutil
from pathlib import Path

import pytest

import dinoml as dml
import dinoml.cli as cli
import dinoml.kernels.profiling as profiling_mod
from dinoml.backends.cuda_libraries import discover_cuda_libraries
from dinoml.ir import read_json, write_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION, build_kernel_manifest
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidates
from dinoml.kernels.profiling import (
    EXECUTION_PLAN_SCHEMA_VERSION,
    PROFILE_REPORT_SCHEMA_VERSION,
    _cache_entry,
    _profile_key,
    _profile_key_payload,
    _profile_result,
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


def _cutlass_candidate_count(dtype: str, *, op_name: str = "gemm_rrr", target=None) -> int:
    return len(_cutlass_candidates(dtype, op_name=op_name, target=target))


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


class GemmModule(dml.Module):
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
    ("gemm_rcr_bias_add_relu", "rcr", "bias_add_relu", ("bias", "d0")),
    ("gemm_rcr_bias_add_add_relu", "rcr", "bias_add_add_relu", ("bias", "d0", "d1")),
    ("gemm_rcr_bias_mul_tanh", "rcr", "bias_mul_tanh", ("bias", "d0")),
    ("gemm_rcr_bias_sigmoid_mul", "rcr", "bias_sigmoid_mul", ("bias", "d0")),
    ("gemm_rcr_bias_sigmoid_mul_tanh", "rcr", "bias_sigmoid_mul_tanh", ("bias", "d0")),
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
    tokens = dml.Dim("tokens", min=1, max=24)
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_dynamic_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest, input_shapes={"a": (7, 32), "b": (32, 11)})

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
    assert (workload.m, workload.n, workload.k) == (7, 11, 32)
    assert workload.output_shape == (7, 11)
    assert workload.to_json()["shape_case"] == {
        "source": "runtime_override",
        "case_id": "runtime_override",
        "dims": {},
        "dim_sources": {},
    }


def test_build_profile_workloads_expands_dim_buckets():
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    tokens = dml.Dim("tokens", min=1, max=11, buckets=(5, 11))
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_bucket_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    candidate_count = _cutlass_candidate_count("float16")
    assert len(workloads) == candidate_count * 8
    cases = {
        (workload.m, workload.n, workload.k, workload.shape_case_id, tuple(sorted(workload.dim_values.items())))
        for workload in workloads
    }
    assert cases == {
        (2, 5, 32, "bucket_batch=2_tokens=5", (("batch", 2), ("tokens", 5))),
        (2, 11, 32, "bucket_batch=2_tokens=11", (("batch", 2), ("tokens", 11))),
        (4, 5, 32, "bucket_batch=4_tokens=5", (("batch", 4), ("tokens", 5))),
        (4, 11, 32, "bucket_batch=4_tokens=11", (("batch", 4), ("tokens", 11))),
    }
    assert {workload.shape_source for workload in workloads} == {"dim_buckets"}
    assert {tuple(sorted(workload.dim_sources.items())) for workload in workloads} == {
        (("batch", "bucket"), ("tokens", "bucket"))
    }
    splits_by_case = {}
    for workload in workloads:
        splits_by_case.setdefault(workload.shape_case_id, set()).add(workload.split_k)
    assert splits_by_case == {
        "bucket_batch=2_tokens=5": {1, 2, 4},
        "bucket_batch=2_tokens=11": {1},
        "bucket_batch=4_tokens=5": {1, 2, 4},
        "bucket_batch=4_tokens=11": {1},
    }


def test_build_profile_workloads_overrides_disable_bucket_expansion():
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    tokens = dml.Dim("tokens", min=1, max=11, buckets=(5, 11))
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, tokens], "float16")},
        name="profile_override_bucket_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest, input_shapes={"a": (3, 32), "b": (32, 7)})

    assert len(workloads) == _cutlass_candidate_count("float16") * 2
    assert {(workload.m, workload.n, workload.k) for workload in workloads} == {(3, 7, 32)}
    assert {workload.split_k for workload in workloads} == {1, 2}
    assert {workload.shape_source for workload in workloads} == {"runtime_override"}


def test_build_profile_workloads_records_max_sourced_dynamic_dims_with_buckets():
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    tokens = dml.Dim("tokens", min=1, max=11)
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
        (2, 11, "bucket_batch=2_tokens=11", (("batch", "bucket"), ("tokens", "max"))),
        (4, 11, "bucket_batch=4_tokens=11", (("batch", "bucket"), ("tokens", "max"))),
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
        inputs={"a": dml.TensorSpec([7, 32], "float16"), "b": dml.TensorSpec([32, 11], "float16")},
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


def test_build_profile_workloads_keeps_residual_epilogues_split_k_one():
    target = {**DEFAULT_CUDA_TARGET, "no_tf32": True}
    spec = dml.trace(
        GemmResidualModule("gemm_rcr_bias_add"),
        inputs={
            "a": dml.TensorSpec([64, 1024], "float32"),
            "b": dml.TensorSpec([64, 1024], "float32"),
            "bias": dml.TensorSpec([64], "float32"),
            "d0": dml.TensorSpec([64, 64], "float32"),
        },
        name="profile_residual_no_split_k_search_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, target)

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == 11
    assert {workload.split_k for workload in workloads} == {1}
    assert all(workload.workspace_nbytes == 0 for workload in workloads)


def test_build_profile_workloads_filters_candidates_by_layout_alignment():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 11], "float32")},
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
def test_build_profile_workloads_does_not_filter_when_only_one_gemm_input_has_layout_alignment(annotated_tensor):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 11], "float32")},
        name=f"profile_partial_alignment_{annotated_tensor}_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _clear_tensor_layout_alignment(lowered, {"a", "b"})
    _set_tensor_layout_alignment(lowered, {annotated_tensor}, 2)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")


def test_build_profile_workloads_uses_minimum_a_b_layout_alignment():
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([7, 32], "float32"), "b": dml.TensorSpec([32, 11], "float32")},
        name="profile_mixed_alignment_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"a"}, 8)
    _set_tensor_layout_alignment(lowered, {"b"}, 2)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert {workload.candidate["cutlass"]["align"] for workload in workloads} <= {1, 2}


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
        inputs={"a": dml.TensorSpec([7, 32], "float16"), "b": dml.TensorSpec([32, 11], "float16")},
        name="profile_alignment_rejected_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    _set_tensor_layout_alignment(lowered, {"a", "b"}, 1)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    with pytest.raises(ValueError, match="alignment filter removed all candidates"):
        build_profile_workloads(lowered, manifest)


@pytest.mark.parametrize(
    ("op_name", "epilogue"),
    [
        ("gemm_rcr_bias", "bias"),
        ("gemm_rcr_bias_relu", "bias_relu"),
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
    assert workload.candidate["epilogue_config"]["activation"] == ("relu" if epilogue == "bias_relu" else None)
    assert workload.to_json()["inputs"]["bias"] == [11]


@pytest.mark.parametrize(("op_name", "layout", "epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_CASES)
def test_build_profile_workloads_supports_gemm_residual_epilogue_inputs(op_name, layout, epilogue, epilogue_inputs):
    input_specs = {
        "a": dml.TensorSpec([7, 32], "float32"),
        "b": dml.TensorSpec([11, 32] if layout == "rcr" else [32, 11], "float32"),
        "bias": dml.TensorSpec([11], "float32"),
        "d0": dml.TensorSpec([7, 11], "float32"),
    }
    if "d1" in epilogue_inputs:
        input_specs["d1"] = dml.TensorSpec([7, 11], "float32")
    spec = dml.trace(GemmResidualModule(op_name), inputs=input_specs, name=f"profile_{op_name}")
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
    inputs = workload.to_json()["inputs"]
    assert inputs["bias"] == [11]
    assert inputs["d0"] == [7, 11]
    if "d1" in epilogue_inputs:
        assert inputs["d1"] == [7, 11]
    else:
        assert "d1" not in inputs


@pytest.mark.parametrize(
    ("op_name", "epilogue"),
    [
        ("gemm_rcr_bias_add", "bias_add"),
        ("gemm_rcr_bias_mul", "bias_mul"),
        ("gemm_rcr_bias_mul_tanh", "bias_mul_tanh"),
        ("gemm_rcr_bias_sigmoid_mul", "bias_sigmoid_mul"),
        ("gemm_rcr_bias_sigmoid_mul_tanh", "bias_sigmoid_mul_tanh"),
    ],
)
def test_build_profile_workloads_flattens_gemm_rcr_single_residual_folded_m(op_name, epilogue):
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([2, 3, 11], "float32"),
        },
        name=f"profile_{op_name}_folded_m",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")
    workload = workloads[0]
    assert (workload.m, workload.n, workload.k) == (6, 11, 32)
    assert workload.a_shape == (2, 3, 32)
    assert workload.b_shape == (11, 32)
    assert workload.residual_shapes == ((2, 3, 11),)
    assert workload.output_shape == (2, 3, 11)
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert workload.to_json()["inputs"]["a"] == [2, 3, 32]
    assert workload.to_json()["inputs"]["d0"] == [2, 3, 11]
    assert list(workload.to_json()["output"].values()) == [[2, 3, 11]]


@pytest.mark.parametrize(
    ("op_name", "epilogue"),
    [
        ("gemm_rcr_bias_add_add", "bias_add_add"),
        ("gemm_rcr_bias_mul_add", "bias_mul_add"),
        ("gemm_rcr_bias_add_add_relu", "bias_add_add_relu"),
    ],
)
def test_build_profile_workloads_flattens_gemm_rcr_dual_residual_folded_m(op_name, epilogue):
    spec = dml.trace(
        GemmResidualModule(op_name),
        inputs={
            "a": dml.TensorSpec([2, 3, 32], "float32"),
            "b": dml.TensorSpec([11, 32], "float32"),
            "bias": dml.TensorSpec([11], "float32"),
            "d0": dml.TensorSpec([2, 3, 11], "float32"),
            "d1": dml.TensorSpec([2, 3, 11], "float32"),
        },
        name=f"profile_{op_name}_folded_m",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})

    workloads = build_profile_workloads(lowered, manifest)

    assert len(workloads) == _cutlass_candidate_count("float32")
    workload = workloads[0]
    assert (workload.m, workload.n, workload.k) == (6, 11, 32)
    assert workload.residual_tensors == ("d0", "d1")
    assert workload.residual_shapes == ((2, 3, 11), (2, 3, 11))
    assert workload.output_shape == (2, 3, 11)
    assert workload.candidate_set_id == f"cutlass_{op_name}_float32_{epilogue}_v1"
    inputs = workload.to_json()["inputs"]
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
    assert payload_a["candidate_id"] == _cutlass_default_candidate_id("float32")
    assert payload_a["candidate_set_id"] == "cutlass_gemm_rrr_float32_linear_combination_v1"
    assert payload_a["candidate_set_key"] == workload.candidate_set_key
    assert payload_a["candidate_config_key"] == workload.candidate_config_key
    assert payload_a["layouts"] == workload.candidate["layouts"]
    assert payload_a["epilogue"] == workload.candidate["epilogue"]
    assert payload_a["epilogue_config"] == workload.candidate["epilogue_config"]
    assert _profile_key(payload_a) != _profile_key(payload_b)
    assert _profile_key(payload_a) != _profile_key(payload_c)
    assert _profile_key(payload_a) != _profile_key(payload_d)


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
    slow = _profile_result(workloads[0], 0.25, 5, profile_key="slow-profile", status="ok")
    fast = _profile_result(workloads[1], 0.10, 5, profile_key="fast-profile", status="ok")
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
    assert plan["selection_policy"] == "lowest_elapsed_ms_per_node_shape"
    assert plan["summary"] == {"selection_count": 1, "static_selection_count": 1, "conflict_count": 0}
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
    static_selection = plan["static_selections"][0]
    assert static_selection["selected_candidate_id"] == workloads[1].candidate_id
    assert static_selection["node_id"] is None
    assert static_selection["shape"]["source"] == "static_overlay_from_consistent_profiled_shapes"


def test_build_execution_plan_preserves_profiled_split_k_and_workspace(tmp_path):
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="profile_execution_plan_split_k_gemm",
    )
    lowered, _ = PassManager().run(spec.ir)
    kernel_manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    workloads = build_profile_workloads(lowered, kernel_manifest)
    split1 = _profile_result(workloads[0], 0.25, 5, profile_key="split1-profile", status="ok")
    split4 = _profile_result(
        replace(workloads[0], split_k=4, workspace_nbytes=8192),
        0.10,
        5,
        profile_key="split4-profile",
        status="ok",
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

    assert plan["summary"] == {"selection_count": 2, "static_selection_count": 0, "conflict_count": 1}
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

    assert plan["summary"] == {"selection_count": 2, "static_selection_count": 0, "conflict_count": 1}
    assert plan["conflicts"][0]["reason"] == "profiled_shapes_selected_different_candidate_or_split_k"
    assert plan["conflicts"][0]["selected_candidate_ids"] == [workloads[0].candidate_id]
    assert plan["conflicts"][0]["selected_split_k"] == [1, 4]


def test_build_execution_plan_preserves_bucket_shape_metadata(tmp_path):
    batch = dml.Dim("batch", min=1, max=4, buckets=(2, 4))
    spec = dml.trace(
        GemmModule("gemm_rrr"),
        inputs={"a": dml.TensorSpec([batch, 32], "float16"), "b": dml.TensorSpec([32, 11], "float16")},
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
        "runtime_abi_version": 5,
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
    for workload in workloads:
        key_payload = _profile_key_payload(workload, manifest, kernel_manifest, codegen_plan, context=context)
        profile_key = _profile_key(key_payload)
        cached_result = _profile_result(workload, 0.125, 9, profile_key=profile_key, status="ok")
        cached_entries[profile_key] = _cache_entry(workload, cached_result, key_payload)
    cache = {
        "schema_version": PROFILE_CACHE_SCHEMA_VERSION,
        "target": {"name": "cuda", "arch": "sm_86"},
        "entries": cached_entries,
    }
    cache_path = profile_cache_path(codegen_plan)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(cache_path, cache)

    report = profile_artifact(artifact, iterations=3)

    assert report["summary"] == {"cached": _cutlass_candidate_count("float32"), "failed": 0, "profiled": 0, "skipped": 0}
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
    assert report["problems"][0]["selected"]["candidate_id"] == _cutlass_default_candidate_id("float32")
    assert report["problems"][0]["selected"]["reason"] == "cache_hit"
    assert report["problems"][0]["candidates"][0]["candidate_config_key"]
    assert report["problems"][0]["profile_key"] in cached_entries


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

    def fake_profile_artifact(artifact, *, input_shapes, iterations, output, execution_plan_output, refresh):
        calls.append((artifact, input_shapes, iterations, output, execution_plan_output, refresh))
        return {
            "artifact": str(artifact),
            "target": {"name": "cuda", "arch": "sm_86"},
            "iterations": iterations,
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
                    "elapsed_ms": 0.01,
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

    assert calls == [("artifact.dinoml", {"a": (4, 8)}, 2, "report.json", "plan.json", True)]
    assert f"dinoml_profile_cutlass_gemm_rrr_float32_{_cutlass_default_symbol_id("float32")}" in stdout
    assert "plan.json" in stdout


def test_cli_compile_forwards_execution_plan(tmp_path, monkeypatch, capsys):
    model_path = tmp_path / "model.py"
    model_path.write_text("def build_spec():\n    return 'spec'\n", encoding="utf-8")
    calls = []

    def fake_compile(spec, *, target, output, execution_plan):
        calls.append((spec, target.to_json(), output, execution_plan))

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
        )
    ]
    assert "artifact.dinoml" in capsys.readouterr().out
