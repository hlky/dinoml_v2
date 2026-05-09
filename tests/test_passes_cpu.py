import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from dinoml import Target, compile
from dinoml.backends.cpu import execute_cpu
from dinoml.ir import IR_SCHEMA_VERSION, ModelSpec
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.bmm import BMM_BASE_OPS, BMM_OPS
from dinoml.kernels.gemm import GEMM_OPS, render_cutlass_gemm_source
from dinoml.kernels.providers.cutlass.bmm import cutlass_bmm_used_candidate_plan, render_cutlass_bmm_source
from dinoml.kernels.providers.cutlass.gemm import (
    CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE,
    cutlass_gemm_candidates,
    cutlass_gemm_used_candidate_plan,
)
from dinoml.kernels.manifest import (
    PROFILE_CACHE_SCHEMA_VERSION,
    apply_execution_plan,
    build_external_kernel_plan,
    build_kernel_manifest,
)
from dinoml.lowering.ops import collect_generated_sources, render_generated_kernels, render_launch
from dinoml.lowering.cuda import render_cuda_module
from dinoml.lowering.ops.fused_elementwise import _broadcast_function_name, _function_name
from dinoml.lowering.shape_buffers import dynamic_dim_sources, numel_expr, shape_buffer_context, shape_dim_expr
from dinoml.ops.definitions import OP_REGISTRY, get_op_def
from dinoml.passes import PassManager, validate_ir
from dinoml.passes.validation import ValidationError


DEFAULT_CUDA_TARGET = {"name": "cuda", "arch": "sm_86"}
GEMM_BIAS_RESIDUAL_EPILOGUES = tuple(
    (f"gemm_{layout}_bias_{suffix}", layout, epilogue, inputs)
    for layout in ("rcr", "rrr")
    for suffix, epilogue, inputs in (
        ("add", "bias_add", ("bias", "d0")),
        ("add_add", "bias_add_add", ("bias", "d0", "d1")),
        ("mul", "bias_mul", ("bias", "d0")),
        ("mul_add", "bias_mul_add", ("bias", "d0", "d1")),
    )
)
GEMM_BIAS_RESIDUAL_EPILOGUES = (
    *GEMM_BIAS_RESIDUAL_EPILOGUES,
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
GEMM_BIAS_RESIDUAL_EXPORT_MACROS = {
    "bias_add": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL_EXPORT",
    "bias_add_add": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL2_EXPORT",
    "bias_add_relu": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL_EXPORT",
    "bias_add_add_relu": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL2_EXPORT",
    "bias_mul": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL_EXPORT",
    "bias_mul_add": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL2_EXPORT",
    "bias_mul_tanh": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL_EXPORT",
    "bias_sigmoid_mul": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL_EXPORT",
    "bias_sigmoid_mul_tanh": "DINOML_FORWARD_GEMM_BIAS_RESIDUAL_EXPORT",
}
GEMM_BIAS_RESIDUAL_EPILOGUE_ALIASES = {
    "bias_add": "BiasAddEpilogue",
    "bias_add_add": "BiasAddAddEpilogue",
    "bias_add_relu": "BiasAddReluEpilogue",
    "bias_add_add_relu": "BiasAddAddReluEpilogue",
    "bias_mul": "BiasMulEpilogue",
    "bias_mul_add": "BiasMulAddEpilogue",
    "bias_mul_tanh": "BiasMulTanhEpilogue",
    "bias_sigmoid_mul": "BiasSigmoidMulEpilogue",
    "bias_sigmoid_mul_tanh": "BiasSigmoidMulTanhEpilogue",
}
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
FLOAT32_OPTIONAL_FAST_CUTLASS_OPERATOR_BY_MATH = {
    "fast_f16": "cutlass::arch::OpMultiplyAddFastF16",
    "fast_bf16": "cutlass::arch::OpMultiplyAddFastBF16",
    "tf32_fast_f32": "cutlass::arch::OpMultiplyAddFastF32",
}
SPLIT_K_LAUNCH_ABIS = {"dinoml_cutlass_gemm_v1", "dinoml_cutlass_gemm_bias_v1"}
SPLIT_K_RESIDUAL_EPILOGUES = {"bias_add", "bias_add_add", "bias_add_relu", "bias_add_add_relu"}
SPLIT_K_RESIDUAL_LAUNCH_ABIS = {
    "dinoml_cutlass_gemm_bias_residual_v1",
    "dinoml_cutlass_gemm_bias_residual2_v1",
}


def _trace_gemm_bias_residual(op_name: str, layout: str, *, dtype: str = "float32"):
    import dinoml as dml

    class GemmResidualModule(dml.Module):
        def forward(self, a, b, bias, d0, d1=None):
            op = getattr(dml.ops, op_name)
            if d1 is None:
                return dml.ops.output(op(a, b, bias, d0), "y")
            return dml.ops.output(op(a, b, bias, d0, d1), "y")

    inputs = {
        "a": dml.TensorSpec([7, 32], dtype),
        "b": dml.TensorSpec([11, 32] if layout == "rcr" else [32, 11], dtype),
        "bias": dml.TensorSpec([11], dtype),
        "d0": dml.TensorSpec([7, 11], dtype),
    }
    if "add_add" in op_name or op_name.endswith("_mul_add"):
        inputs["d1"] = dml.TensorSpec([7, 11], dtype)
    return dml.trace(GemmResidualModule(), inputs=inputs, name=f"{op_name}_{dtype}_residual")


def _trace_gemm_bias_activation(op_name: str, layout: str, *, dtype: str = "float32"):
    import dinoml as dml

    class GemmBiasActivationModule(dml.Module):
        def forward(self, a, b, bias):
            return dml.ops.output(getattr(dml.ops, op_name)(a, b, bias), "y")

    return dml.trace(
        GemmBiasActivationModule(),
        inputs={
            "a": dml.TensorSpec([7, 32], dtype),
            "b": dml.TensorSpec([11, 32] if layout == "rcr" else [32, 11], dtype),
            "bias": dml.TensorSpec([11], dtype),
        },
        name=f"{op_name}_{dtype}_bias_activation",
    )


def _cutlass_symbol_ids(dtype: str) -> list[str]:
    return [str(config["symbol_id"]) for config in CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[dtype]]


def _cutlass_candidate_ids(dtype: str) -> list[str]:
    return [str(config["candidate_id"]) for config in CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[dtype]]


def _cutlass_candidate_count(dtype: str) -> int:
    return len(CUTLASS_GEMM_CANDIDATE_CONFIGS_BY_DTYPE[dtype])


def _cutlass_default_symbol_id(dtype: str) -> str:
    return _cutlass_symbol_ids(dtype)[0]


def _cutlass_default_candidate_id(dtype: str) -> str:
    return _cutlass_candidate_ids(dtype)[0]


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


def _cutlass_manifest_candidates(dtype: str, target=None, op_name: str = "gemm_rrr") -> tuple[dict[str, object], ...]:
    return cutlass_gemm_candidates(op_name, dtype, target=target or DEFAULT_CUDA_TARGET)


def _cutlass_manifest_first_candidate_with_alignment(dtype: str, max_alignment: int, target=None, op_name: str = "gemm_rrr"):
    return next(
        candidate
        for candidate in _cutlass_manifest_candidates(dtype, target, op_name)
        if int(candidate["cutlass"]["align"]) <= max_alignment
    )


def _cutlass_manifest_symbol_ids(dtype: str, target=None, op_name: str = "gemm_rrr") -> list[str]:
    return [str(candidate["symbol_id"]) for candidate in _cutlass_manifest_candidates(dtype, target, op_name)]


def _cutlass_manifest_candidate_ids(dtype: str, target=None, op_name: str = "gemm_rrr") -> list[str]:
    return [str(candidate["candidate_id"]) for candidate in _cutlass_manifest_candidates(dtype, target, op_name)]


def _cutlass_manifest_default_symbol_id(dtype: str, target=None, op_name: str = "gemm_rrr") -> str:
    return _cutlass_manifest_symbol_ids(dtype, target, op_name)[0]


def _cutlass_manifest_default_candidate_id(dtype: str, target=None, op_name: str = "gemm_rrr") -> str:
    return _cutlass_manifest_candidate_ids(dtype, target, op_name)[0]


def _cutlass_rendered_policy_alias(rendered: str, policy: str) -> str:
    start = rendered.index(f"using {policy} = GemmPolicy<")
    end = rendered.index(";\n", start) + 2
    return rendered[start:end]


def _assert_float32_candidate_math_families(candidates):
    assert len(candidates) == sum(FLOAT32_CANDIDATE_MATH_COUNTS.values())
    assert Counter(candidate["cutlass"]["math"] for candidate in candidates) == Counter(FLOAT32_CANDIDATE_MATH_COUNTS)
    assert Counter(candidate["cutlass"]["math"] for candidate in candidates if candidate["optional"]) == Counter(
        FLOAT32_OPTIONAL_MATH_COUNTS
    )
    assert Counter(candidate["cutlass"]["math"] for candidate in candidates if not candidate["optional"]) == Counter(
        {"f32": 11}
    )
    assert {candidate["cutlass"]["opclass"] for candidate in candidates if candidate["optional"]} == {"tensorop"}
    assert {candidate["cutlass"]["opclass"] for candidate in candidates if not candidate["optional"]} == {"simt"}
    for math, math_operator in FLOAT32_OPTIONAL_FAST_OPERATOR_BY_MATH.items():
        assert {
            candidate["cutlass"].get("math_operator", "multiply_add")
            for candidate in candidates
            if candidate["cutlass"]["math"] == math
        } == {math_operator}


def test_pass_manager_runs_expected_pipeline():
    from tests.models.fused_elementwise import build_spec

    spec = build_spec()
    lowered, reports = PassManager().run(spec.ir)
    validate_ir(lowered)
    assert [report.name for report in reports] == list(PassManager.DEFAULT_PIPELINE)
    assert "memory_plan" in lowered["metadata"]
    assert "fusion_groups" in lowered["metadata"]
    assert any(node["op"] == "fused_elementwise" for node in lowered["nodes"])


def test_memory_plan_records_shape_only_view_alias_metadata():
    ir = _shape_view_ir()

    lowered, _ = PassManager(pipeline=("memory_plan",)).run(ir)
    validate_ir(lowered)

    views = lowered["metadata"]["memory_plan"]["views"]["views"]
    assert views == [
        {
            "tensor": "y",
            "source": "x",
            "kind": "shape_view",
            "transform": "reshape",
            "offset_elements": 0,
            "shape": [3, 2],
            "shape_spec": [3, 2],
        }
    ]
    assert lowered["metadata"]["memory_plan"]["temporaries"] == []


def test_validation_rejects_invalid_dense_layout_metadata():
    ir = _shape_view_ir()
    ir["tensors"][0]["layout"] = {
        "schema_version": 1,
        "kind": "dense",
        "order": "row_major",
        "strides": [1, 2],
        "storage_offset": 0,
    }

    with pytest.raises(ValidationError, match="invalid layout"):
        validate_ir(ir)


def test_view_alias_validation_requires_shape_only_element_count():
    ir = _shape_view_ir(output_shape=[4, 2])

    with pytest.raises(ValidationError, match="preserve source element count"):
        PassManager(pipeline=("memory_plan",)).run(ir)


def test_compile_rejects_view_of_view_aliases(tmp_path):
    ir = _shape_view_ir()
    ir["outputs"] = [{"name": "z", "tensor": "z", "shape": [6], "shape_spec": [6], "dtype": "float32"}]
    ir["tensors"].append({"name": "z", "shape": [6], "shape_spec": [6], "dtype": "float32", "kind": "output", "nbytes": 24})
    ir["metadata"]["views"]["views"].append(
        {
            "tensor": "z",
            "source": "y",
            "kind": "shape_view",
            "transform": "flatten",
            "shape": [6],
            "shape_spec": [6],
        }
    )
    spec = ModelSpec("shape_view_of_view_alias", ir, constants={})

    with pytest.raises(NotImplementedError, match="View-of-view"):
        compile(spec, Target("cpu"), tmp_path / "shape_view_of_view_alias.dinoml")


def test_cuda_lowering_binds_and_materializes_shape_view_output_alias():
    lowered, _ = PassManager().run(_shape_view_ir())

    generated = render_cuda_module(lowered, generated_kernels=[])

    assert "const float* ptr_y = ptr_x;" in generated
    assert (
        "cudaMemcpyAsync(dinoml::module::tensor_data(outputs[0]), ptr_y, runtime_numel_y * sizeof(float), "
        "cudaMemcpyDeviceToDevice, session->stream)"
    ) in generated


def test_cpu_reference_matches_numpy_formula():
    from tests.models.fused_elementwise import build_spec, build_validation_inputs, numpy_reference

    spec = build_spec()
    inputs = build_validation_inputs()
    actual = execute_cpu(spec, inputs)["y"]
    expected = numpy_reference(inputs)["y"]

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def _shape_view_ir(output_shape=None):
    output_shape = list(output_shape or [3, 2])
    output_nbytes = int(np.prod(output_shape, dtype=np.int64) * 4)
    return {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "shape_view_alias",
        "inputs": [{"name": "x", "tensor": "x", "shape": [2, 3], "shape_spec": [2, 3], "dtype": "float32"}],
        "constants": [],
        "outputs": [{"name": "y", "tensor": "y", "shape": output_shape, "shape_spec": output_shape, "dtype": "float32"}],
        "nodes": [],
        "tensors": [
            {"name": "x", "shape": [2, 3], "shape_spec": [2, 3], "dtype": "float32", "kind": "input", "nbytes": 24},
            {
                "name": "y",
                "shape": output_shape,
                "shape_spec": output_shape,
                "dtype": "float32",
                "kind": "output",
                "nbytes": output_nbytes,
            },
        ],
        "metadata": {
            "views": {
                "version": 1,
                "views": [
                    {
                        "tensor": "y",
                        "source": "x",
                        "kind": "shape_view",
                        "transform": "reshape",
                        "shape": output_shape,
                        "shape_spec": output_shape,
                    }
                ],
            }
        },
    }


def test_elementwise_op_definitions_are_frontend_only_until_fused():
    add = get_op_def("add")
    fused = get_op_def("fused_elementwise")
    assert OP_REGISTRY.get_frontend("add") is add
    assert add.schema.inputs == ("x0", "x1")
    assert not add.backend_kernels
    assert fused.backend_kernels["cuda"].symbol == "generated_fused_elementwise"
    assert fused.backend_kernels["cpu"].symbol == "generated_fused_elementwise"


def test_kernel_manifest_lists_required_unique_kernels():
    from tests.models.fused_elementwise import build_spec

    spec = build_spec()
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    symbols = {item["kernel_symbol"] for item in manifest["required_kernels"]}
    assert symbols == {"generated_fused_elementwise"}
    model_generated = [item for item in manifest["required_kernels"] if item["kernel_library"] == "model"]
    assert model_generated and model_generated[0]["op"] == "fused_elementwise"
    assert manifest["support_cache_key"] != manifest["cache_key"]
    assert manifest["profile_cache_schema_version"] == PROFILE_CACHE_SCHEMA_VERSION
    plan = create_codegen_plan(manifest, "/tmp/dinoml-test-cache")
    assert plan.profiler_symbols == ()
    assert plan.support_cache_dir.name == manifest["support_cache_key"][:16]


def test_external_cuda_kernel_plan_lists_cutlass_gemm_families():
    plan = build_external_kernel_plan({"name": "cuda", "arch": "sm_86"})
    families = {family["op_name"]: family for family in plan["families"]}
    assert set(GEMM_OPS).issubset(families)
    assert set(BMM_OPS).issubset(families)
    assert set(families) >= {
        "gemm_rcr",
        "gemm_rcr_bias",
        "gemm_rcr_bias_relu",
        "gemm_rrr",
        "gemm_rrr_bias",
        "gemm_rrr_bias_relu",
    }
    assert families["gemm_rcr"]["provider"] == "cutlass"
    assert families["gemm_rcr"]["required_libraries"] == ["cutlass", "cublaslt"]
    default_symbol_id = _cutlass_default_symbol_id("float32")
    default_f16_symbol_id = _cutlass_default_symbol_id("float16")
    default_bf16_symbol_id = _cutlass_default_symbol_id("bfloat16")
    assert families["gemm_rcr"]["kernel_symbol"] == f"dinoml_cutlass_gemm_rcr_float32_{default_symbol_id}"
    assert families["gemm_rcr"]["kernel_symbols_by_dtype"]["float16"] == f"dinoml_cutlass_gemm_rcr_float16_{default_f16_symbol_id}"
    assert families["gemm_rcr"]["kernel_symbols_by_dtype"]["bfloat16"] == f"dinoml_cutlass_gemm_rcr_bfloat16_{default_bf16_symbol_id}"
    assert families["gemm_rrr"]["profiler_symbol"] == f"dinoml_profile_cutlass_gemm_rrr_float32_{default_symbol_id}"
    assert families["gemm_rrr"]["profiler_symbols_by_dtype"]["float16"] == f"dinoml_profile_cutlass_gemm_rrr_float16_{default_f16_symbol_id}"
    assert families["gemm_rrr"]["attrs"]["b_layout"] == "row"
    assert families["gemm_rrr"]["attrs"]["supported_dtypes"] == ["float16", "float32", "bfloat16"]
    assert families["gemm_rcr_bias"]["attrs"]["epilogue"] == "bias"
    assert families["gemm_rcr_bias"]["attrs"]["epilogue_config"]["inputs"] == ["bias"]
    assert families["gemm_rcr_bias"]["kernel_symbols_by_dtype"]["float32"] == f"dinoml_cutlass_gemm_rcr_bias_float32_{default_symbol_id}"
    assert families["gemm_rcr_bias_relu"]["attrs"]["epilogue"] == "bias_relu"
    assert families["gemm_rcr_bias_relu"]["attrs"]["epilogue_config"]["activation"] == "relu"
    assert families["gemm_rcr_bias_relu"]["attrs"]["epilogue_config"]["inputs"] == ["bias"]
    assert families["gemm_rcr_bias_relu"]["kernel_symbols_by_dtype"]["float32"] == f"dinoml_cutlass_gemm_rcr_bias_relu_float32_{default_symbol_id}"
    assert families["gemm_rcr_bias_gelu"]["attrs"]["epilogue"] == "bias_gelu"
    assert families["gemm_rcr_bias_gelu"]["attrs"]["epilogue_config"]["activation"] == "gelu"
    assert families["gemm_rcr_bias_gelu"]["attrs"]["epilogue_config"]["inputs"] == ["bias"]
    assert families["gemm_rcr_bias_gelu"]["kernel_symbols_by_dtype"]["float32"] == f"dinoml_cutlass_gemm_rcr_bias_gelu_float32_{default_symbol_id}"
    assert families["gemm_rcr_bias_hardswish"]["attrs"]["epilogue"] == "bias_hardswish"
    assert families["gemm_rcr_bias_hardswish"]["attrs"]["epilogue_config"]["activation"] == "hardswish"
    assert families["gemm_rcr_bias_elup1"]["attrs"]["epilogue"] == "bias_elup1"
    assert families["gemm_rcr_bias_elup1"]["attrs"]["epilogue_config"]["activation"] == "elup1"
    assert families["gemm_rcr_bias_elup1"]["attrs"]["epilogue_config"]["inputs"] == ["bias"]
    assert families["gemm_rcr_bias_elup1"]["kernel_symbols_by_dtype"]["float32"] == f"dinoml_cutlass_gemm_rcr_bias_elup1_float32_{default_symbol_id}"
    rrr_f32_candidates = families["gemm_rrr"]["candidates_by_dtype"]["float32"]
    _assert_float32_candidate_math_families(rrr_f32_candidates)
    rrr_f16_candidates = families["gemm_rrr"]["candidates_by_dtype"]["float16"]
    assert [candidate["candidate_id"] for candidate in rrr_f16_candidates] == _cutlass_candidate_ids("float16")
    assert [candidate["symbol_id"] for candidate in rrr_f16_candidates] == _cutlass_symbol_ids("float16")
    rrr_f16_candidate = rrr_f16_candidates[0]
    assert rrr_f16_candidate["kernel_symbol"] == f"dinoml_cutlass_gemm_rrr_float16_{default_f16_symbol_id}"
    assert rrr_f16_candidate["profiler_symbol"] == f"dinoml_profile_cutlass_gemm_rrr_float16_{default_f16_symbol_id}"
    _assert_split_k_metadata(rrr_f16_candidate, str(rrr_f16_candidate["launch_abi"]))
    assert rrr_f16_candidate["cutlass"] == {
        "api": "device_gemm",
        "opclass": "tensorop",
        "arch": "sm80",
        "math": "16816",
        "threadblock": [256, 128, 32],
        "warp_count": [4, 2, 1],
        "warp": [64, 64, 32],
        "instruction": [16, 8, 16],
        "stages": 3,
        "align": 8,
    }
    assert {candidate["accumulator_dtype"] for candidate in rrr_f16_candidates} == {"float16", "float32"}
    assert rrr_f16_candidates[1]["accumulator_dtype"] == "float16"
    assert rrr_f16_candidates[2]["cutlass"]["align"] == 4
    assert rrr_f16_candidates[6]["cutlass"]["threadblock"] == [128, 256, 32]
    assert len(rrr_f16_candidate["candidate_config_key"]) == 64
    rrr_f16_candidate_set = families["gemm_rrr"]["candidate_sets_by_dtype"]["float16"]
    assert rrr_f16_candidate_set["candidate_set_id"] == "cutlass_gemm_rrr_float16_linear_combination_v1"
    assert rrr_f16_candidate_set["candidate_count"] == _cutlass_candidate_count("float16")
    _assert_split_k_metadata(rrr_f16_candidate_set, str(rrr_f16_candidate_set["launch_abi"]))
    assert rrr_f16_candidate_set["candidate_config_keys"] == [
        candidate["candidate_config_key"] for candidate in rrr_f16_candidates
    ]
    assert len(rrr_f16_candidate_set["candidate_set_key"]) == 64
    assert plan["profiler_strategy"] == "generate_used_candidates_once_then_cache_results"
    assert len(plan["cache_key"]) == 64


def test_external_cuda_kernel_plan_lists_cutlass_bmm_base_families():
    plan = build_external_kernel_plan({"name": "cuda", "arch": "sm_86"})
    families = {family["op_name"]: family for family in plan["families"]}
    default_symbol_id = _cutlass_default_symbol_id("float32")
    default_f16_symbol_id = _cutlass_default_symbol_id("float16")

    family = families["bmm_ccc"]
    assert family["provider"] == "cutlass"
    assert family["family"] == "bmm_strided"
    assert family["required_libraries"] == ["cutlass", "cublaslt"]
    assert family["attrs"] == {
        "a_layout": "column",
        "b_layout": "column",
        "c_layout": "column",
        "epilogue": "none",
        "epilogue_config": {"name": "none", "inputs": [], "launch_abi": "dinoml_cutlass_bmm_v1"},
        "supported_dtypes": ["float16", "float32", "bfloat16"],
    }
    assert family["kernel_symbols_by_dtype"]["float32"] == f"dinoml_cutlass_bmm_ccc_float32_{default_symbol_id}"
    assert family["kernel_symbols_by_dtype"]["float16"] == f"dinoml_cutlass_bmm_ccc_float16_{default_f16_symbol_id}"
    assert family["profiler_symbols_by_dtype"]["float32"] == f"dinoml_profile_cutlass_bmm_ccc_float32_{default_symbol_id}"
    candidate = family["candidates_by_dtype"]["float32"][0]
    assert candidate["family"] == "bmm_strided"
    assert candidate["launch_abi"] == "dinoml_cutlass_bmm_v1"
    assert candidate["layouts"] == {"a": "column", "b": "column", "c": "column"}
    assert candidate["cutlass"]["api"] == "device_gemm_batched"
    assert candidate["cutlass"]["align"] == 4
    assert candidate["split_k_values"] == [1]
    assert candidate["supports_split_k"] is False
    candidate_set = family["candidate_sets_by_dtype"]["float32"]
    assert candidate_set["candidate_set_id"] == "cutlass_bmm_ccc_float32_linear_combination_v1"
    assert candidate_set["candidate_count"] == _cutlass_candidate_count("float32")
    assert candidate_set["launch_abi"] == "dinoml_cutlass_bmm_v1"
    assert candidate_set["target_policy"] == {"no_tf32": False, "use_fp16_acc": False}

    add_family = families["bmm_rrr_add"]
    assert add_family["provider"] == "cutlass"
    assert add_family["attrs"]["epilogue"] == "add"
    assert add_family["attrs"]["epilogue_config"] == {
        "name": "add",
        "inputs": ["d0"],
        "launch_abi": "dinoml_cutlass_bmm_add_v1",
    }
    assert add_family["kernel_symbols_by_dtype"]["float32"] == f"dinoml_cutlass_bmm_rrr_add_float32_{default_symbol_id}"
    add_candidate = add_family["candidates_by_dtype"]["float32"][0]
    assert add_candidate["epilogue"] == "add"
    assert add_candidate["launch_abi"] == "dinoml_cutlass_bmm_add_v1"
    assert add_candidate["cutlass"]["epilogue_source"] == "d0"
    assert add_family["candidate_sets_by_dtype"]["float32"]["candidate_set_id"] == "cutlass_bmm_rrr_add_float32_add_v1"


@pytest.mark.parametrize(("op_name", "layout", "epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_EPILOGUES)
def test_external_cuda_kernel_plan_lists_cutlass_gemm_residual_epilogues(op_name, layout, epilogue, epilogue_inputs):
    plan = build_external_kernel_plan({"name": "cuda", "arch": "sm_86"})
    families = {family["op_name"]: family for family in plan["families"]}
    family = families[op_name]
    default_symbol_id = _cutlass_default_symbol_id("float32")

    assert family["provider"] == "cutlass"
    assert family["attrs"]["b_layout"] == ("column" if layout == "rcr" else "row")
    assert family["attrs"]["epilogue"] == epilogue
    epilogue_config = family["attrs"]["epilogue_config"]
    assert epilogue_config["name"] == epilogue
    assert epilogue_config["inputs"] == list(epilogue_inputs)
    assert epilogue_config["launch_abi"].startswith("dinoml_cutlass_gemm_")
    assert epilogue_config["launch_abi"].endswith("_v1")
    assert epilogue_config["launch_abi"] != "dinoml_cutlass_gemm_bias_v1"
    assert family["kernel_symbols_by_dtype"]["float32"] == f"dinoml_cutlass_{op_name}_float32_{default_symbol_id}"
    assert family["profiler_symbols_by_dtype"]["float32"] == f"dinoml_profile_cutlass_{op_name}_float32_{default_symbol_id}"

    candidate_set = family["candidate_sets_by_dtype"]["float32"]
    assert candidate_set["candidate_set_id"] == f"cutlass_{op_name}_float32_{epilogue}_v1"
    assert candidate_set["epilogue"] == epilogue
    assert candidate_set["epilogue_config"] == epilogue_config
    assert candidate_set["launch_abi"] == epilogue_config["launch_abi"]
    _assert_split_k_metadata(candidate_set)
    candidate = family["candidates_by_dtype"]["float32"][0]
    assert candidate["epilogue"] == epilogue
    assert candidate["epilogue_config"] == epilogue_config
    assert candidate["launch_abi"] == epilogue_config["launch_abi"]
    _assert_split_k_metadata(candidate)


@pytest.mark.parametrize(
    ("dtype", "suffix"),
    [
        ("float32", "float32"),
        ("float16", "float16"),
        ("bfloat16", "bfloat16"),
    ],
)
def test_gemm_kernel_manifest_uses_cutlass_external_library(dtype, suffix):
    import dinoml as dml
    from dinoml.lowering.ops import collect_generated_sources
    manifest_candidates = _cutlass_manifest_candidates(dtype)
    manifest_symbol_ids = _cutlass_manifest_symbol_ids(dtype)
    manifest_candidate_ids = _cutlass_manifest_candidate_ids(dtype)
    default_symbol_id = manifest_symbol_ids[0]
    selected_candidate = _cutlass_manifest_first_candidate_with_alignment(dtype, 2)
    selected_symbol_id = str(selected_candidate["symbol_id"])

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], dtype), "b": dml.TensorSpec([8, 6], dtype)},
        name=f"gemm_{dtype}_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    plan = create_codegen_plan(manifest, "/tmp/dinoml-test-cache")

    assert sources["kernels"] == []
    required = manifest["required_kernels"][0]
    assert required["op"] == "gemm_rrr"
    default_symbol = f"dinoml_cutlass_gemm_rrr_{suffix}_{default_symbol_id}"
    default_profiler = f"dinoml_profile_cutlass_gemm_rrr_{suffix}_{default_symbol_id}"
    selected_symbol = f"dinoml_cutlass_gemm_rrr_{suffix}_{selected_symbol_id}"
    selected_profiler = f"dinoml_profile_cutlass_gemm_rrr_{suffix}_{selected_symbol_id}"
    candidate_symbols = [f"dinoml_cutlass_gemm_rrr_{suffix}_{symbol_id}" for symbol_id in manifest_symbol_ids]
    candidate_profilers = [f"dinoml_profile_cutlass_gemm_rrr_{suffix}_{symbol_id}" for symbol_id in manifest_symbol_ids]
    assert required["kernel_symbol"] == selected_symbol
    assert required["kernel_library"] == "cutlass_gemm"
    assert required["profiler_symbol"] == selected_profiler
    assert required["has_profiler"] is True
    assert required["candidate_set_id"] == f"cutlass_gemm_rrr_{suffix}_linear_combination_v1"
    assert len(required["candidate_set_key"]) == 64
    assert required["candidate_set"]["candidate_set_key"] == required["candidate_set_key"]
    assert required["candidate_set"]["candidate_count"] == len(manifest_candidates)
    _assert_split_k_metadata(required["candidate_set"], str(required["candidate_set"]["launch_abi"]))
    assert required["candidate_set"]["target_policy"] == {"no_tf32": False, "use_fp16_acc": False}
    assert required["cutlass_alignment_cap"] == 2
    assert required["selected_candidate_id"] == selected_candidate["candidate_id"]
    assert len(required["candidates"]) == len(manifest_candidates)
    candidate = required["candidates"][0]
    assert [item["candidate_id"] for item in required["candidates"]] == manifest_candidate_ids
    assert candidate["candidate_id"] == manifest_candidate_ids[0]
    assert candidate["provider"] == "cutlass"
    assert candidate["family"] == "gemm_universal"
    assert candidate["dtype"] == dtype
    assert candidate["layouts"] == {"a": "row", "b": "row", "c": "row"}
    assert candidate["epilogue"] == "linear_combination"
    assert candidate["epilogue_config"]["name"] == "linear_combination"
    assert candidate["accumulator_dtype"] == "float32"
    assert candidate["kernel_symbol"] == default_symbol
    assert candidate["profiler_symbol"] == default_profiler
    _assert_split_k_metadata(candidate, str(candidate["launch_abi"]))
    assert candidate["cutlass"]["opclass"] == "tensorop"
    assert candidate["cutlass"]["arch"] == "sm80"
    assert candidate["optional"] is (dtype == "float32")
    assert candidate["cutlass"]["threadblock"] == ([256, 128, 16] if dtype == "float32" else [256, 128, 32])
    assert candidate["cutlass"]["warp"] == ([64, 64, 16] if dtype == "float32" else [64, 64, 32])
    assert candidate["cutlass"]["instruction"] == ([16, 8, 8] if dtype == "float32" else [16, 8, 16])
    assert candidate["cutlass"]["stages"] == 3
    assert candidate["cutlass"]["align"] == (4 if dtype == "float32" else 8)
    assert len(candidate["candidate_config_key"]) == 64
    assert plan.kernel_symbols == (selected_symbol,)
    assert plan.profiler_symbols == (selected_profiler,)
    assert plan.candidate_profiler_symbols == tuple(candidate_profilers)
    assert plan.external_support_libraries[0]["name"] == "cutlass_gemm"
    assert plan.external_support_libraries[0]["library"] == "lib/libdinoml_cutlass_gemm.so"
    assert len(plan.external_support_libraries[0]["used_candidate_plan_key"]) == 64
    assert plan.external_support_libraries[0]["candidate_set_keys"] == [required["candidate_set_key"]]
    assert plan.external_support_libraries[0]["candidate_config_keys"] == [
        item["candidate_config_key"]
        for item in sorted(required["candidates"], key=lambda item: item["candidate_config_key"])
    ]
    assert plan.external_support_libraries[0]["kernel_symbols"] == sorted(candidate_symbols)
    assert plan.external_support_libraries[0]["profiler_symbols"] == sorted(candidate_profilers)


def test_bmm_kernel_manifest_uses_cutlass_external_library():
    import dinoml as dml

    class BmmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.bmm_ccc(a, b), "y")

    spec = dml.trace(
        BmmModel(),
        inputs={"a": dml.TensorSpec([2, 8, 2], "float32"), "b": dml.TensorSpec([1, 6, 8], "float32")},
        name="bmm_ccc_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    plan = create_codegen_plan(manifest, "/tmp/dinoml-test-cache")
    required = manifest["required_kernels"][0]
    selected_symbol_id = str(_cutlass_manifest_first_candidate_with_alignment("float32", 2, op_name="gemm_rrr")["symbol_id"])

    assert required["op"] == "bmm_ccc"
    assert required["kernel_library"] == "cutlass_bmm"
    assert required["kernel_symbol"] == f"dinoml_cutlass_bmm_ccc_float32_{selected_symbol_id}"
    assert required["profiler_symbol"] == f"dinoml_profile_cutlass_bmm_ccc_float32_{selected_symbol_id}"
    assert required["has_profiler"] is True
    assert required["candidate_set_id"] == "cutlass_bmm_ccc_float32_linear_combination_v1"
    assert required["candidate_set"]["family"] == "bmm_strided"
    assert required["candidate_set"]["launch_abi"] == "dinoml_cutlass_bmm_v1"
    assert required["cutlass_alignment_cap"] == 2
    assert required["candidates"][0]["cutlass"]["api"] == "device_gemm_batched"
    assert required["candidates"][0]["layouts"] == {"a": "column", "b": "column", "c": "column"}
    assert plan.external_support_libraries[0]["name"] == "cutlass_bmm"
    assert plan.external_support_libraries[0]["library"] == "lib/libdinoml_cutlass_bmm.so"

    launch = render_launch("cuda", lowered["nodes"][0], tensor_map, kernel_manifest=manifest)
    assert "batch dimension mismatch" in launch
    assert "static_cast<int64_t>(((shape_a_0 == 1" in launch
    assert "static_cast<int>(shape_a_2)" in launch
    assert "static_cast<int>(shape_b_1)" in launch
    assert required["kernel_symbol"] in launch

    module_source = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=manifest)
    assert f'extern "C" int {required["kernel_symbol"]}(' in module_source
    assert "int64_t batch_stride_a" in module_source

    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_bmm.cu").read_text(
        encoding="utf-8"
    )
    rendered_support = render_cutlass_bmm_source(source, cutlass_bmm_used_candidate_plan(manifest))
    assert f"DINOML_FORWARD_BMM_EXPORT(bmm_ccc, float32" in rendered_support
    assert selected_symbol_id in rendered_support


def test_bmm_add_kernel_manifest_uses_cutlass_epilogue_abi():
    import dinoml as dml

    class BmmAddModel(dml.Module):
        def forward(self, a, b, d0):
            return dml.ops.output(dml.ops.bmm_rrr_add(a, b, d0), "y")

    spec = dml.trace(
        BmmAddModel(),
        inputs={
            "a": dml.TensorSpec([2, 4, 8], "float32"),
            "b": dml.TensorSpec([2, 8, 6], "float32"),
            "d0": dml.TensorSpec([2, 4, 6], "float32"),
        },
        name="bmm_rrr_add_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    plan = create_codegen_plan(manifest, "/tmp/dinoml-test-cache")
    required = manifest["required_kernels"][0]

    assert required["op"] == "bmm_rrr_add"
    assert required["kernel_library"] == "cutlass_bmm"
    assert required["candidate_set_id"] == "cutlass_bmm_rrr_add_float32_add_v1"
    assert required["candidate_set"]["epilogue"] == "add"
    assert required["candidate_set"]["epilogue_config"] == {
        "name": "add",
        "inputs": ["d0"],
        "launch_abi": "dinoml_cutlass_bmm_add_v1",
    }
    assert required["candidates"][0]["cutlass"]["epilogue_source"] == "d0"
    assert required["cutlass_alignment"]["nodes"][0]["epilogue"]["inputs"][0]["tensor"] == "d0"
    assert plan.external_support_libraries[0]["name"] == "cutlass_bmm"

    launch = render_launch("cuda", lowered["nodes"][0], tensor_map, kernel_manifest=manifest)
    assert "ptr_a, ptr_b, ptr_d0, ptr_t0" in launch
    assert "d0 shape mismatch" in launch
    assert "static_cast<int64_t>((shape_a_1) * (shape_b_2))" in launch

    module_source = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=manifest)
    assert f'extern "C" int {required["kernel_symbol"]}(' in module_source
    assert "const float* d0" in module_source
    assert "int64_t batch_stride_d0" in module_source
    assert "int ldd0" in module_source

    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_bmm.cu").read_text(
        encoding="utf-8"
    )
    rendered_support = render_cutlass_bmm_source(source, cutlass_bmm_used_candidate_plan(manifest))
    assert "DINOML_FORWARD_BMM_ADD_EXPORT(bmm_rrr_add, float32" in rendered_support


def test_cuda_bmm_add_lowering_uses_zero_stride_for_trailing_bias_broadcast():
    import dinoml as dml

    class BmmAddModel(dml.Module):
        def forward(self, a, b, d0):
            return dml.ops.output(dml.ops.bmm_rrr_add(a, b, d0), "y")

    spec = dml.trace(
        BmmAddModel(),
        inputs={
            "a": dml.TensorSpec([2, 4, 8], "float32"),
            "b": dml.TensorSpec([2, 8, 6], "float32"),
            "d0": dml.TensorSpec([6], "float32"),
        },
        name="bmm_rrr_add_bias_cuda_rejected",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)

    launch = render_launch("cuda", lowered["nodes"][0], tensor_map, kernel_manifest=manifest)

    assert "ptr_a, ptr_b, ptr_d0, ptr_t0" in launch
    assert "shape_d0_0 != shape_t0_2" in launch
    assert "static_cast<int64_t>(0)" in launch
    assert "static_cast<int>(0)" in launch


def test_apply_execution_plan_selects_profiled_cutlass_bmm_candidate_for_lowering():
    import dinoml as dml

    class BmmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.bmm_ccc(a, b), "y")

    spec = dml.trace(
        BmmModel(),
        inputs={"a": dml.TensorSpec([2, 8, 2], "float32"), "b": dml.TensorSpec([1, 6, 8], "float32")},
        name="bmm_profile_selected_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    default_candidate = next(candidate for candidate in required["candidates"] if candidate["candidate_id"] == required["selected_candidate_id"])
    selected_candidate = next(
        candidate
        for candidate in required["candidates"]
        if candidate["candidate_id"] != default_candidate["candidate_id"]
        and int(candidate["cutlass"]["align"]) <= int(required["cutlass_alignment_cap"])
    )
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "static_selections": [
            {
                "selection_key": "bmm-profile-selection",
                "op": "bmm_ccc",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": selected_candidate["candidate_config_key"],
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "shape": {"m": 2, "n": 6, "k": 8, "batch_count": 2},
                "avg_ms": 0.01,
                "confidence": {"confident": True, "level": "high"},
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
    }

    selected_manifest = apply_execution_plan(manifest, execution_plan, strict=True)
    selected_required = selected_manifest["required_kernels"][0]
    plan = create_codegen_plan(selected_manifest, "/tmp/dinoml-test-cache")
    launch = render_launch("cuda", lowered["nodes"][0], tensor_map, kernel_manifest=selected_manifest)
    rendered_support = render_cutlass_bmm_source(
        (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_bmm.cu").read_text(
            encoding="utf-8"
        ),
        cutlass_bmm_used_candidate_plan(selected_manifest),
    )

    assert selected_required["selected_candidate_id"] == selected_candidate["candidate_id"]
    assert selected_required["kernel_symbol"] == selected_candidate["kernel_symbol"]
    assert selected_required["profiler_symbol"] == selected_candidate["profiler_symbol"]
    assert selected_required["execution_plan_selection"]["candidate_config_key"] == selected_candidate["candidate_config_key"]
    assert selected_required["execution_plan_selection"]["shape"]["batch_count"] == 2
    assert selected_manifest["cache_key"] != manifest["cache_key"]
    assert selected_manifest["support_cache_key"] != manifest["support_cache_key"]
    assert plan.external_support_libraries[0]["name"] == "cutlass_bmm"
    assert plan.kernel_symbols == (selected_candidate["kernel_symbol"],)
    assert plan.profiler_symbols == (selected_candidate["profiler_symbol"],)
    assert selected_candidate["kernel_symbol"] in launch
    assert default_candidate["kernel_symbol"] not in launch
    assert selected_candidate["symbol_id"] in rendered_support


def test_apply_execution_plan_uses_guarded_cutlass_bmm_dispatch():
    import dinoml as dml

    class BmmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.bmm_rrr(a, b), "y")

    spec = dml.trace(
        BmmModel(),
        inputs={"a": dml.TensorSpec([2, 4, 8], "float32"), "b": dml.TensorSpec([2, 8, 6], "float32")},
        name="bmm_guarded_plan_deferred",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    default_candidate = next(candidate for candidate in required["candidates"] if candidate["candidate_id"] == required["selected_candidate_id"])
    selected_candidate = next(
        candidate
        for candidate in required["candidates"]
        if candidate["candidate_id"] != default_candidate["candidate_id"]
    )
    guarded_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "selections": [
            {
                "selection_key": "bmm-guarded-selection",
                "node_id": lowered["nodes"][0]["id"],
                "op": "bmm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": selected_candidate["candidate_config_key"],
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "shape": {"m": 4, "n": 6, "k": 8, "batch_count": 2},
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
        "conflicts": [
            {
                "op": "bmm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "reason": "profiled_shapes_selected_different_candidate_or_split_k",
            }
        ],
    }

    selected_manifest = apply_execution_plan(manifest, guarded_plan, strict=True)
    relaxed = apply_execution_plan(manifest, guarded_plan, strict=False)
    launch = render_launch("cuda", lowered["nodes"][0], tensor_map, kernel_manifest=selected_manifest)

    for payload in (selected_manifest["required_kernels"][0], relaxed["required_kernels"][0]):
        assert "execution_plan_selection" not in payload
        assert payload["selected_candidate_id"] == required["selected_candidate_id"]
        assert [entry["selected_candidate_id"] for entry in payload["execution_plan_dispatch"]] == [
            selected_candidate["candidate_id"]
        ]
        assert payload["execution_plan_dispatch"][0]["shape"]["batch_count"] == 2
    assert selected_candidate["kernel_symbol"] in launch
    assert default_candidate["kernel_symbol"] in launch
    assert "(((shape_a_0 == 1) ? shape_b_0 : shape_a_0)) == 2" in launch
    assert "(shape_a_1) == 4" in launch
    assert "(shape_b_2) == 6" in launch
    dispatch_byte_alignment = int(selected_candidate["cutlass"]["align"]) * 4
    if dispatch_byte_alignment > 4:
        assert f"dinoml::module::is_tensor_pointer_aligned(abi_a, ptr_a, {dispatch_byte_alignment})" in launch
        assert f"dinoml::module::is_tensor_pointer_aligned(abi_b, ptr_b, {dispatch_byte_alignment})" in launch


def test_apply_execution_plan_rejects_cutlass_bmm_candidate_above_alignment_cap():
    import dinoml as dml

    class BmmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.bmm_ccc(a, b), "y")

    spec = dml.trace(
        BmmModel(),
        inputs={"a": dml.TensorSpec([2, 8, 2], "float32"), "b": dml.TensorSpec([1, 6, 8], "float32")},
        name="bmm_profile_alignment_rejected",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    rejected_candidate = next(
        candidate
        for candidate in required["candidates"]
        if int(candidate["cutlass"]["align"]) > int(required["cutlass_alignment_cap"])
    )
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "static_selections": [
            {
                "op": "bmm_ccc",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": rejected_candidate["candidate_id"],
                "candidate_config_key": rejected_candidate["candidate_config_key"],
                "kernel_symbol": rejected_candidate["kernel_symbol"],
                "profiler_symbol": rejected_candidate["profiler_symbol"],
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
    }

    with pytest.raises(ValueError, match="exceeds alignment cap"):
        apply_execution_plan(manifest, execution_plan, strict=True)
    relaxed = apply_execution_plan(manifest, execution_plan, strict=False)

    assert relaxed["required_kernels"][0]["selected_candidate_id"] == required["selected_candidate_id"]
    assert "execution_plan_selection" not in relaxed["required_kernels"][0]


def test_apply_execution_plan_rejects_cutlass_bmm_static_split_k():
    import dinoml as dml

    class BmmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.bmm_rrr(a, b), "y")

    spec = dml.trace(
        BmmModel(),
        inputs={"a": dml.TensorSpec([2, 4, 8], "float32"), "b": dml.TensorSpec([2, 8, 6], "float32")},
        name="bmm_profile_splitk_rejected",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    selected_candidate = next(candidate for candidate in required["candidates"] if candidate["candidate_id"] == required["selected_candidate_id"])
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "static_selections": [
            {
                "op": "bmm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": selected_candidate["candidate_config_key"],
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "split_k": 2,
                "workspace_nbytes": 4096,
            }
        ],
    }

    with pytest.raises(ValueError, match="BMM execution plan selections only support split_k=1"):
        apply_execution_plan(manifest, execution_plan, strict=True)
    relaxed = apply_execution_plan(manifest, execution_plan, strict=False)

    assert relaxed["required_kernels"][0]["selected_candidate_id"] == required["selected_candidate_id"]
    assert "execution_plan_selection" not in relaxed["required_kernels"][0]


def test_cuda_bmm_lowering_rejects_guarded_split_k_dispatch():
    import dinoml as dml

    class BmmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.bmm_rrr(a, b), "y")

    spec = dml.trace(
        BmmModel(),
        inputs={"a": dml.TensorSpec([2, 4, 8], "float32"), "b": dml.TensorSpec([2, 8, 6], "float32")},
        name="bmm_guarded_lowering_rejected",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    manifest["required_kernels"][0]["execution_plan_dispatch"] = [
        {
            "selected_candidate_id": manifest["required_kernels"][0]["selected_candidate_id"],
            "shape": {"m": 4, "n": 6, "k": 8, "batch_count": 2},
            "split_k": 2,
            "workspace_nbytes": 4096,
        }
    ]

    with pytest.raises(NotImplementedError, match="BMM execution-plan selection requires split_k=1"):
        render_launch("cuda", lowered["nodes"][0], tensor_map, kernel_manifest=manifest)


def test_gemm_kernel_manifest_filters_fp16_accumulation_policy():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], "float16"), "b": dml.TensorSpec([8, 6], "float16")},
        name="gemm_float16_acc_policy",
    )
    lowered, _ = PassManager().run(spec.ir)
    default_manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    fp16_acc_target = {**DEFAULT_CUDA_TARGET, "use_fp16_acc": True}
    fp16_acc_manifest = build_kernel_manifest(lowered, fp16_acc_target)
    fp16_acc_plan = create_codegen_plan(fp16_acc_manifest, "/tmp/dinoml-test-cache")

    default_required = default_manifest["required_kernels"][0]
    fp16_required = fp16_acc_manifest["required_kernels"][0]
    assert {candidate["accumulator_dtype"] for candidate in default_required["candidates"]} == {"float32"}
    assert {candidate["accumulator_dtype"] for candidate in fp16_required["candidates"]} == {"float16"}
    assert default_required["selected_candidate_id"] != fp16_required["selected_candidate_id"]
    assert default_required["kernel_symbol"] != fp16_required["kernel_symbol"]
    assert default_required["profiler_symbol"] != fp16_required["profiler_symbol"]
    assert default_required["candidate_set"]["target_policy"] == {"no_tf32": False, "use_fp16_acc": False}
    assert fp16_required["candidate_set"]["target_policy"] == {"no_tf32": False, "use_fp16_acc": True}
    assert default_manifest["support_cache_key"] != fp16_acc_manifest["support_cache_key"]
    assert default_manifest["cache_key"] != fp16_acc_manifest["cache_key"]
    fp16_acc_support = fp16_acc_plan.external_support_libraries[0]
    assert fp16_acc_support["kernel_symbols"] == sorted(candidate["kernel_symbol"] for candidate in fp16_required["candidates"])
    assert fp16_acc_support["profiler_symbols"] == sorted(candidate["profiler_symbol"] for candidate in fp16_required["candidates"])
    assert fp16_acc_support["candidate_config_keys"] == sorted(
        candidate["candidate_config_key"] for candidate in fp16_required["candidates"]
    )

    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=fp16_acc_manifest)
    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(
        encoding="utf-8"
    )
    rendered_support = render_cutlass_gemm_source(source, fp16_acc_support)

    assert fp16_required["kernel_symbol"] in generated
    assert f'extern "C" int {fp16_required["kernel_symbol"]}(' in generated
    assert default_required["kernel_symbol"] not in generated
    assert fp16_required["candidates"][0]["symbol_id"] in rendered_support
    assert default_required["candidates"][0]["symbol_id"] not in rendered_support


def test_gemm_kernel_manifest_no_tf32_selects_simt_float32_candidates():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="gemm_float32_no_tf32",
    )
    lowered, _ = PassManager().run(spec.ir)
    default_manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    no_tf32_target = {**DEFAULT_CUDA_TARGET, "no_tf32": True}
    no_tf32_manifest = build_kernel_manifest(lowered, no_tf32_target)

    default_required = default_manifest["required_kernels"][0]
    no_tf32_required = no_tf32_manifest["required_kernels"][0]
    assert no_tf32_required["candidate_set"]["target_policy"] == {"no_tf32": True, "use_fp16_acc": False}
    assert no_tf32_required["selected_candidate_id"] != default_required["selected_candidate_id"]
    assert no_tf32_required["kernel_symbol"] != default_required["kernel_symbol"]
    assert len(no_tf32_required["candidates"]) == 11
    assert {candidate["cutlass"]["opclass"] for candidate in no_tf32_required["candidates"]} == {"simt"}
    assert {candidate["cutlass"]["math"] for candidate in no_tf32_required["candidates"]} == {"f32"}
    assert not any(candidate["optional"] for candidate in no_tf32_required["candidates"])
    assert {candidate["cutlass"]["align"] for candidate in no_tf32_required["candidates"]} == {1}
    assert no_tf32_required["candidates"][0]["symbol_id"] == "simt_sm80_f32_256x128x8_s5_w4x2x1_f32_align1"
    assert no_tf32_required["candidates"][-1]["symbol_id"] == "simt_sm80_f32_32x128x8_s5_w1x2x1_f32_align1"

    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=no_tf32_manifest)
    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(
        encoding="utf-8"
    )
    rendered_support = render_cutlass_gemm_source(source, cutlass_gemm_used_candidate_plan(no_tf32_manifest))

    assert no_tf32_required["kernel_symbol"] in generated
    assert default_required["kernel_symbol"] not in generated
    assert "check_tensor_pointer_alignment(abi_a" not in generated
    assert "check_tensor_pointer_alignment(abi_b" not in generated
    assert "cutlass::arch::OpClassSimt" in rendered_support
    assert no_tf32_required["candidates"][0]["cutlass_policy"] in rendered_support
    assert no_tf32_required["candidates"][0]["symbol_id"] in rendered_support
    assert default_required["candidates"][0]["symbol_id"] not in rendered_support


def test_apply_execution_plan_selects_profiled_cutlass_candidate_for_lowering():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="gemm_profile_selected_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    default_candidate = required["candidates"][0]
    selected_candidate = required["candidates"][1]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
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
                "confidence": {"confident": True, "level": "high"},
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
    }

    selected_manifest = apply_execution_plan(manifest, execution_plan, strict=True)
    selected_required = selected_manifest["required_kernels"][0]
    plan = create_codegen_plan(selected_manifest, "/tmp/dinoml-test-cache")
    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=selected_manifest)
    rendered_support = render_cutlass_gemm_source(
        (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(
            encoding="utf-8"
        ),
        cutlass_gemm_used_candidate_plan(selected_manifest),
    )

    assert selected_required["selected_candidate_id"] == selected_candidate["candidate_id"]
    assert selected_required["kernel_symbol"] == selected_candidate["kernel_symbol"]
    assert selected_required["profiler_symbol"] == selected_candidate["profiler_symbol"]
    assert selected_required["execution_plan_selection"]["candidate_config_key"] == selected_candidate["candidate_config_key"]
    assert selected_required["execution_plan_selection"]["confidence"] == {"confident": True, "level": "high"}
    assert selected_manifest["cache_key"] != manifest["cache_key"]
    assert selected_manifest["support_cache_key"] != manifest["support_cache_key"]
    assert plan.kernel_symbols == (selected_candidate["kernel_symbol"],)
    assert plan.profiler_symbols == (selected_candidate["profiler_symbol"],)
    assert selected_candidate["kernel_symbol"] in generated
    assert default_candidate["kernel_symbol"] not in generated
    required_alignment = int(selected_candidate["cutlass"]["align"]) * 4
    fallback = selected_required["alignment_fallbacks"][0]
    assert fallback["cutlass_alignment"] == 1
    assert fallback["kernel_symbol"] in generated
    assert f"dinoml::module::is_tensor_pointer_aligned(abi_a, ptr_a, {required_alignment})" in generated
    assert f"dinoml::module::is_tensor_pointer_aligned(abi_b, ptr_b, {required_alignment})" in generated
    assert f"else {{\n  if (int err = {fallback['kernel_symbol']}(" in generated
    assert "cutlass_workspace" not in generated
    assert "dinoml_cutlass_splitk_" not in generated
    assert selected_candidate["symbol_id"] in rendered_support
    fallback_candidate = next(
        candidate for candidate in selected_required["candidates"] if candidate["candidate_id"] == fallback["candidate_id"]
    )
    assert fallback_candidate["symbol_id"] in rendered_support


def test_apply_execution_plan_rejects_cutlass_candidate_above_alignment_cap():
    import dinoml as dml

    tokens = dml.Dim("tokens", min=1, max=16, buckets=(8, 16))

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 32], "float32"), "b": dml.TensorSpec([32, tokens], "float32")},
        name="gemm_profile_alignment_cap_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    unsafe_candidate = next(candidate for candidate in required["candidates"] if int(candidate["cutlass"]["align"]) == 4)
    original_selected = required["selected_candidate_id"]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "static_selections": [
            {
                "selection_key": "profile-selection",
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": unsafe_candidate["candidate_id"],
                "candidate_config_key": unsafe_candidate["candidate_config_key"],
                "kernel_symbol": unsafe_candidate["kernel_symbol"],
                "profiler_symbol": unsafe_candidate["profiler_symbol"],
                "shape": {"m": 4, "n": 16, "k": 32},
                "avg_ms": 0.01,
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
    }

    with pytest.raises(ValueError, match="exceeds alignment cap 1"):
        apply_execution_plan(manifest, execution_plan, strict=True)

    relaxed = apply_execution_plan(manifest, execution_plan, strict=False)

    assert required["cutlass_alignment_cap"] == 1
    assert relaxed["required_kernels"][0]["selected_candidate_id"] == original_selected
    assert relaxed["required_kernels"][0]["kernel_symbol"] == required["kernel_symbol"]


def test_apply_execution_plan_rejects_stale_cutlass_candidate_config_key():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 8], "float32")},
        name="gemm_profile_stale_candidate_config_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    selected_candidate = required["candidates"][0]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "static_selections": [
            {
                "selection_key": "profile-selection",
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": "stale-candidate-config-key",
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "shape": {"m": 4, "n": 8, "k": 8},
                "avg_ms": 0.01,
                "split_k": 1,
                "workspace_nbytes": 0,
            }
        ],
    }

    with pytest.raises(ValueError, match="candidate_config_key mismatch"):
        apply_execution_plan(manifest, execution_plan, strict=True)

    relaxed = apply_execution_plan(manifest, execution_plan, strict=False)

    assert relaxed["required_kernels"][0]["selected_candidate_id"] == required["selected_candidate_id"]
    assert "execution_plan_selection" not in relaxed["required_kernels"][0]


def test_cuda_lowering_uses_guarded_execution_plan_dispatch_for_shape_conflicts():
    import dinoml as dml

    tokens = dml.Dim("tokens", min=1, max=16, buckets=(6, 8))

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 32], "float32"), "b": dml.TensorSpec([32, tokens], "float32")},
        name="gemm_profile_guarded_dispatch_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    align4_candidate = next(candidate for candidate in required["candidates"] if int(candidate["cutlass"]["align"]) == 4)
    align2_candidate = next(candidate for candidate in required["candidates"] if int(candidate["cutlass"]["align"]) == 2)

    def selection(candidate, n, *, split_k=1, workspace_nbytes=0):
        return {
            "selection_key": f"profile-selection-n{n}",
            "node_id": "n0",
            "op": "gemm_rrr",
            "dtype": "float32",
            "candidate_set_key": required["candidate_set_key"],
            "selected_candidate_id": candidate["candidate_id"],
            "candidate_config_key": candidate["candidate_config_key"],
            "kernel_symbol": candidate["kernel_symbol"],
            "profiler_symbol": candidate["profiler_symbol"],
            "shape": {"m": 4, "n": n, "k": 32},
            "avg_ms": 0.01,
            "split_k": split_k,
            "workspace_nbytes": workspace_nbytes,
        }

    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "selections": [selection(align2_candidate, 6, split_k=2, workspace_nbytes=4096), selection(align4_candidate, 8)],
        "static_selections": [],
        "conflicts": [
            {
                "op": "gemm_rrr",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "reason": "profiled_shapes_selected_different_candidate_or_split_k",
            }
        ],
    }

    missing_guard_plan = {
        **execution_plan,
        "selections": [{**execution_plan["selections"][0], "candidate_set_key": "missing-candidate-set"}],
        "conflicts": [{**execution_plan["conflicts"][0], "candidate_set_key": "missing-candidate-set"}],
    }
    with pytest.raises(ValueError, match="guarded selections did not match"):
        apply_execution_plan(manifest, missing_guard_plan, strict=True)
    relaxed_missing_guard = apply_execution_plan(manifest, missing_guard_plan, strict=False)

    selected_manifest = apply_execution_plan(manifest, execution_plan, strict=True)
    selected_required = selected_manifest["required_kernels"][0]
    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=selected_manifest)
    split_symbol = align2_candidate["kernel_symbol"].replace("dinoml_cutlass_", "dinoml_cutlass_splitk_", 1)

    assert "execution_plan_dispatch" not in relaxed_missing_guard["required_kernels"][0]
    assert selected_required["selected_candidate_id"] == required["selected_candidate_id"]
    assert "execution_plan_selection" not in selected_required
    assert [entry["selected_candidate_id"] for entry in selected_required["execution_plan_dispatch"]] == [
        align2_candidate["candidate_id"],
        align4_candidate["candidate_id"],
    ]
    assert selected_required["execution_plan_dispatch"][0]["workspace_nbytes"] == 4096
    assert f'extern "C" int {split_symbol}(' in generated
    assert f'extern "C" int {align4_candidate["kernel_symbol"]}(' in generated
    assert "size_t cutlass_workspace_nbytes = 4096;" in generated
    assert "if ((shape_a_0) == 4 && (shape_b_1) == 6 && (shape_a_1) == 32 &&" in generated
    assert "else if ((shape_a_0) == 4 && (shape_b_1) == 8 && (shape_a_1) == 32 &&" in generated
    assert "const DinoTensor* abi_a = &inputs[0];" in generated
    assert "const DinoTensor* abi_b = &inputs[1];" in generated
    assert "dinoml::module::is_tensor_pointer_aligned(abi_a, ptr_a, 16)" in generated
    assert "dinoml::module::is_tensor_pointer_aligned(abi_b, ptr_b, 16)" in generated
    assert f"if (int err = {split_symbol}(" in generated
    assert ", 2, session->cutlass_workspace, session->cutlass_workspace_nbytes, session->stream" in generated
    assert f"if (int err = {align4_candidate['kernel_symbol']}(" in generated
    assert "else {" in generated


def test_cuda_lowering_uses_split_k_companion_symbol_and_workspace():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="gemm_split_k_selected_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    selected_candidate = required["candidates"][1]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
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
                "split_k": 2,
                "workspace_nbytes": 4096,
            }
        ],
    }
    selected_manifest = apply_execution_plan(manifest, execution_plan, strict=True)

    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=selected_manifest)

    split_symbol = selected_candidate["kernel_symbol"].replace("dinoml_cutlass_", "dinoml_cutlass_splitk_", 1)
    assert f'extern "C" int {split_symbol}(' in generated
    assert f"if (int err = {split_symbol}(" in generated
    assert "void* cutlass_workspace = nullptr;" in generated
    assert "size_t cutlass_workspace_nbytes = 4096;" in generated
    assert "cudaMalloc(&session->cutlass_workspace, 4096)" in generated
    assert ", 2, session->cutlass_workspace, session->cutlass_workspace_nbytes, session->stream" in generated


def test_cuda_lowering_uses_split_k_companion_symbol_and_workspace_for_additive_residual():
    spec = _trace_gemm_bias_residual("gemm_rcr_bias_add", "rcr")
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    selected_candidate = required["candidates"][0]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "static_selections": [
            {
                "selection_key": "profile-selection",
                "op": "gemm_rcr_bias_add",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": selected_candidate["candidate_config_key"],
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "shape": {"m": 7, "n": 11, "k": 32},
                "avg_ms": 0.01,
                "split_k": 2,
                "workspace_nbytes": 4096,
            }
        ],
    }
    selected_manifest = apply_execution_plan(manifest, execution_plan, strict=True)

    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=selected_manifest)

    split_symbol = selected_candidate["kernel_symbol"].replace("dinoml_cutlass_", "dinoml_cutlass_splitk_", 1)
    assert f'extern "C" int {split_symbol}(' in generated
    assert "const float* bias," in generated
    assert "const float* d0," in generated
    assert f"if (int err = {split_symbol}(" in generated
    assert "size_t cutlass_workspace_nbytes = 4096;" in generated
    assert "cudaMalloc(&session->cutlass_workspace, 4096)" in generated
    assert ", 2, session->cutlass_workspace, session->cutlass_workspace_nbytes, session->stream" in generated


def test_cuda_lowering_rejects_split_k_for_non_additive_residual():
    spec = _trace_gemm_bias_residual("gemm_rcr_bias_mul", "rcr")
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    required = manifest["required_kernels"][0]
    selected_candidate = required["candidates"][0]
    execution_plan = {
        "schema_version": 1,
        "kind": "dinoml.execution_plan",
        "static_selections": [
            {
                "selection_key": "profile-selection",
                "op": "gemm_rcr_bias_mul",
                "dtype": "float32",
                "candidate_set_key": required["candidate_set_key"],
                "selected_candidate_id": selected_candidate["candidate_id"],
                "candidate_config_key": selected_candidate["candidate_config_key"],
                "kernel_symbol": selected_candidate["kernel_symbol"],
                "profiler_symbol": selected_candidate["profiler_symbol"],
                "shape": {"m": 7, "n": 11, "k": 32},
                "avg_ms": 0.01,
                "split_k": 2,
                "workspace_nbytes": 4096,
            }
        ],
    }
    selected_manifest = apply_execution_plan(manifest, execution_plan, strict=True)

    with pytest.raises((NotImplementedError, ValueError), match="split-K"):
        render_cuda_module(lowered, generated_kernels=[], kernel_manifest=selected_manifest)


def test_cutlass_gemm_source_renderer_emits_float32_fast_policy_aliases():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="gemm_rrr_fast_tensorop_used_source",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    support = cutlass_gemm_used_candidate_plan(manifest)
    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(
        encoding="utf-8"
    )

    rendered = render_cutlass_gemm_source(source, support)

    required = manifest["required_kernels"][0]
    _assert_float32_candidate_math_families(required["candidates"])
    for math, math_operator in FLOAT32_OPTIONAL_FAST_OPERATOR_BY_MATH.items():
        fast_candidate = next(
            candidate
            for candidate in required["candidates"]
            if candidate["cutlass"]["math"] == math and candidate["cutlass"]["align"] == 4
        )
        policy_alias = _cutlass_rendered_policy_alias(rendered, fast_candidate["cutlass_policy"])
        cutlass_operator = FLOAT32_OPTIONAL_FAST_CUTLASS_OPERATOR_BY_MATH[math]

        assert fast_candidate["optional"] is True
        assert fast_candidate["cutlass"]["math_operator"] == math_operator
        assert fast_candidate["symbol_id"] in rendered
        assert fast_candidate["cutlass_policy"] in rendered
        assert re.search(r"Align4GemmPolicy$", fast_candidate["cutlass_policy"])
        assert cutlass_operator in rendered
        assert cutlass_operator in policy_alias
        assert re.search(r"\n    4[,>]", policy_alias)


def test_cutlass_gemm_source_renderer_keeps_only_used_symbols():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a, b):
            return dml.ops.output(dml.ops.gemm_rrr(a, b), "y")

    spec = dml.trace(
        GemmModel(),
        inputs={"a": dml.TensorSpec([4, 8], "float32"), "b": dml.TensorSpec([8, 6], "float32")},
        name="gemm_rrr_used_source",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    used_plan = cutlass_gemm_used_candidate_plan(manifest)
    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(encoding="utf-8")

    rendered = render_cutlass_gemm_source(source, used_plan)

    assert "template <typename Storage, typename Element, typename LayoutB>" in rendered
    assert "DINOML_CUTLASS_GENERATED_EXPORTS" in rendered
    assert "DINOML_FORWARD_GEMM_EXPORT(gemm_rrr, float32, float, float, f32" in rendered
    assert _cutlass_default_symbol_id("float32") in rendered
    assert _cutlass_symbol_ids("float32")[1] in rendered
    assert "DINOML_FORWARD_GEMM_EXPORT(gemm_rcr, float32" not in rendered
    assert "tensorop_sm80_tf32_256x128x64" not in rendered


def test_cutlass_gemm_source_renderer_emits_elup1_epilogue_export():
    spec = _trace_gemm_bias_activation("gemm_rcr_bias_elup1", "rcr")
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    support = cutlass_gemm_used_candidate_plan(manifest)
    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(
        encoding="utf-8"
    )

    rendered = render_cutlass_gemm_source(source, support)

    required = manifest["required_kernels"][0]
    candidate = required["candidates"][0]
    assert "DINOML_CUTLASS_GENERATED_EXPORTS" in rendered
    assert "DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(gemm_rcr_bias_elup1, float32" in rendered
    assert "BiasElup1Epilogue" in rendered
    assert "LinearCombinationELUp1" in rendered
    assert candidate["symbol_id"] in rendered
    assert candidate["cutlass_policy"] in rendered
    assert required["candidate_set"]["epilogue_config"]["activation"] == "elup1"


@pytest.mark.parametrize(("op_name", "layout", "epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_EPILOGUES)
def test_cutlass_gemm_source_renderer_emits_residual_epilogue_exports(op_name, layout, epilogue, epilogue_inputs):
    spec = _trace_gemm_bias_residual(op_name, layout)
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    support = cutlass_gemm_used_candidate_plan(manifest)
    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(encoding="utf-8")

    rendered = render_cutlass_gemm_source(source, support)

    required = manifest["required_kernels"][0]
    candidate = required["candidates"][0]
    export_macro = GEMM_BIAS_RESIDUAL_EXPORT_MACROS[epilogue]
    assert "DINOML_CUTLASS_GENERATED_EXPORTS" in rendered
    assert f"{export_macro}({op_name}, float32" in rendered
    assert GEMM_BIAS_RESIDUAL_EPILOGUE_ALIASES[epilogue] in rendered
    assert candidate["symbol_id"] in rendered
    assert candidate["cutlass_policy"] in rendered
    assert "d0" in rendered
    if "d1" in epilogue_inputs:
        assert "d1" in rendered
    assert required["candidate_set"]["epilogue_config"]["inputs"] == list(epilogue_inputs)


def test_cutlass_gemm_source_renderer_emits_no_tf32_simt_residual_epilogue_exports():
    spec = _trace_gemm_bias_residual("gemm_rcr_bias_add_add_relu", "rcr")
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {**DEFAULT_CUDA_TARGET, "no_tf32": True})
    support = cutlass_gemm_used_candidate_plan(manifest)
    source = (Path(__file__).resolve().parents[1] / "kernels" / "cuda" / "src" / "cutlass_gemm.cu").read_text(
        encoding="utf-8"
    )

    rendered = render_cutlass_gemm_source(source, support)

    required = manifest["required_kernels"][0]
    candidate = required["candidates"][0]
    assert len(required["candidates"]) == 11
    assert {item["cutlass"]["opclass"] for item in required["candidates"]} == {"simt"}
    assert required["candidate_set"]["supports_split_k"] is True
    assert "DefaultEpilogueWithBroadcastSimt" in rendered
    assert "DINOML_FORWARD_GEMM_BIAS_RESIDUAL2_EXPORT(gemm_rcr_bias_add_add_relu, float32" in rendered
    assert candidate["symbol_id"] in rendered
    assert candidate["cutlass_policy"] in rendered


@pytest.mark.parametrize(("op_name", "layout", "_epilogue", "epilogue_inputs"), GEMM_BIAS_RESIDUAL_EPILOGUES)
def test_cuda_lowering_passes_gemm_residual_epilogue_pointer_args(op_name, layout, _epilogue, epilogue_inputs):
    spec = _trace_gemm_bias_residual(op_name, layout)
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)

    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=manifest)

    symbol = manifest["required_kernels"][0]["kernel_symbol"]
    output_name = lowered["nodes"][0]["outputs"][0]
    output_ptr = f"ptr_{output_name}"
    residual_ptrs = ["ptr_d0", *(["ptr_d1"] if "d1" in epilogue_inputs else [])]
    n_expr = "shape_b_0" if layout == "rcr" else "shape_b_1"
    expected_call = (
        f"{symbol}(ptr_a, ptr_b, ptr_bias, {', '.join(residual_ptrs)}, {output_ptr}, "
        f"static_cast<int>(shape_a_0), static_cast<int>({n_expr}), "
        "static_cast<int>(shape_a_1), session->stream)"
    )
    assert f'extern "C" int {symbol}(' in generated
    assert "const float* bias," in generated
    assert "const float* d0," in generated
    if "d1" in epilogue_inputs:
        assert "const float* d1," in generated
    else:
        assert "const float* d1," not in generated
    assert expected_call in generated


@pytest.mark.parametrize(
    ("op_name", "layout"),
    [
        (f"gemm_{layout}_bias_{suffix}", layout)
        for layout in ("rcr", "rrr")
        for suffix in ("add", "add_relu", "mul", "mul_tanh", "sigmoid_mul", "sigmoid_mul_tanh")
    ],
)
def test_cuda_lowering_flattens_gemm_single_residual_folded_m(op_name, layout):
    import dinoml as dml

    class GemmResidualModel(dml.Module):
        def forward(self, a, b, bias, d0):
            op = getattr(dml.ops, op_name)
            return dml.ops.output(op(a, b, bias, d0), "y")

    spec = dml.trace(
        GemmResidualModel(),
        inputs={
            "a": dml.TensorSpec([2, 3, 8], "float32"),
            "b": dml.TensorSpec([6, 8] if layout == "rcr" else [8, 6], "float32"),
            "bias": dml.TensorSpec([6], "float32"),
            "d0": dml.TensorSpec([2, 3, 6], "float32"),
        },
        name=f"{op_name}_folded_m_lowering",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)

    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=manifest)

    symbol = manifest["required_kernels"][0]["kernel_symbol"]
    output_name = lowered["nodes"][0]["outputs"][0]
    output_ptr = f"ptr_{output_name}"
    n_expr = "shape_b_0" if layout == "rcr" else "shape_b_1"
    expected_call = (
        f"{symbol}(ptr_a, ptr_b, ptr_bias, ptr_d0, {output_ptr}, "
        f"static_cast<int>(shape_a_0 * shape_a_1), static_cast<int>({n_expr}), "
        "static_cast<int>(shape_a_2), session->stream)"
    )
    assert expected_call in generated
    assert f"shape_{output_name}_0 != shape_a_0 || shape_{output_name}_1 != shape_a_1 || shape_{output_name}_2 != {n_expr}" in generated
    assert f"shape_d0_0 != shape_a_0 || shape_d0_1 != shape_a_1 || shape_d0_2 != {n_expr}" in generated


@pytest.mark.parametrize(
    ("op_name", "layout"),
    [
        (f"gemm_{layout}_bias_{suffix}", layout)
        for layout in ("rcr", "rrr")
        for suffix in ("add_add", "mul_add", "add_add_relu")
    ],
)
def test_cuda_lowering_flattens_gemm_dual_residual_folded_m(op_name, layout):
    import dinoml as dml

    class GemmResidualModel(dml.Module):
        def forward(self, a, b, bias, d0, d1):
            op = getattr(dml.ops, op_name)
            return dml.ops.output(op(a, b, bias, d0, d1), "y")

    spec = dml.trace(
        GemmResidualModel(),
        inputs={
            "a": dml.TensorSpec([2, 3, 8], "float32"),
            "b": dml.TensorSpec([6, 8] if layout == "rcr" else [8, 6], "float32"),
            "bias": dml.TensorSpec([6], "float32"),
            "d0": dml.TensorSpec([2, 3, 6], "float32"),
            "d1": dml.TensorSpec([2, 3, 6], "float32"),
        },
        name=f"{op_name}_folded_m_lowering",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)

    generated = render_cuda_module(lowered, generated_kernels=[], kernel_manifest=manifest)

    symbol = manifest["required_kernels"][0]["kernel_symbol"]
    output_name = lowered["nodes"][0]["outputs"][0]
    output_ptr = f"ptr_{output_name}"
    n_expr = "shape_b_0" if layout == "rcr" else "shape_b_1"
    expected_call = (
        f"{symbol}(ptr_a, ptr_b, ptr_bias, ptr_d0, ptr_d1, {output_ptr}, "
        f"static_cast<int>(shape_a_0 * shape_a_1), static_cast<int>({n_expr}), "
        "static_cast<int>(shape_a_2), session->stream)"
    )
    assert expected_call in generated
    assert f"shape_{output_name}_0 != shape_a_0 || shape_{output_name}_1 != shape_a_1 || shape_{output_name}_2 != {n_expr}" in generated
    assert f"shape_d0_0 != shape_a_0 || shape_d0_1 != shape_a_1 || shape_d0_2 != {n_expr}" in generated
    assert f"shape_d1_0 != shape_a_0 || shape_d1_1 != shape_a_1 || shape_d1_2 != {n_expr}" in generated


def test_gemm_kernel_manifest_keeps_distinct_dtype_variants():
    import dinoml as dml

    class GemmModel(dml.Module):
        def forward(self, a32, b32, a16, b16):
            y32 = dml.ops.gemm_rrr(a32, b32)
            y16 = dml.ops.gemm_rrr(a16, b16)
            return {"y32": y32, "y16": y16}

    spec = dml.trace(
        GemmModel(),
        inputs={
            "a32": dml.TensorSpec([4, 8], "float32"),
            "b32": dml.TensorSpec([8, 6], "float32"),
            "a16": dml.TensorSpec([4, 8], "float16"),
            "b16": dml.TensorSpec([8, 6], "float16"),
        },
        name="gemm_mixed_dtype_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, DEFAULT_CUDA_TARGET)
    float32_candidates = _cutlass_manifest_candidates("float32")
    float16_candidates = _cutlass_manifest_candidates("float16")
    float32_symbol_ids = [str(candidate["symbol_id"]) for candidate in float32_candidates]
    float16_symbol_ids = [str(candidate["symbol_id"]) for candidate in float16_candidates]
    float32_candidate_ids = [str(candidate["candidate_id"]) for candidate in float32_candidates]
    float16_candidate_ids = [str(candidate["candidate_id"]) for candidate in float16_candidates]
    selected_float32 = _cutlass_manifest_first_candidate_with_alignment("float32", 2)
    selected_float16 = _cutlass_manifest_first_candidate_with_alignment("float16", 2)

    assert [item["kernel_symbol"] for item in manifest["required_kernels"]] == [
        f"dinoml_cutlass_gemm_rrr_float32_{selected_float32['symbol_id']}",
        f"dinoml_cutlass_gemm_rrr_float16_{selected_float16['symbol_id']}",
    ]
    candidates = [candidate for item in manifest["required_kernels"] for candidate in item["candidates"]]
    assert [candidate["candidate_id"] for candidate in candidates] == [
        *float32_candidate_ids,
        *float16_candidate_ids,
    ]
    assert [candidate["dtype"] for candidate in candidates[: len(float32_candidates)]] == ["float32"] * len(float32_candidates)
    assert [candidate["dtype"] for candidate in candidates[len(float32_candidates): ]] == ["float16"] * len(float16_candidates)
    assert len({candidate["candidate_config_key"] for candidate in candidates}) == len(float32_candidates) + len(float16_candidates)
    assert [item["candidate_set_id"] for item in manifest["required_kernels"]] == [
        "cutlass_gemm_rrr_float32_linear_combination_v1",
        "cutlass_gemm_rrr_float16_linear_combination_v1",
    ]
    assert manifest["required_kernels"][0]["candidate_set_key"] != manifest["required_kernels"][1]["candidate_set_key"]
    plan = create_codegen_plan(manifest, "/tmp/dinoml-test-cache")
    support = plan.external_support_libraries[0]
    assert support["kernel_symbols"] == sorted(
        f"dinoml_cutlass_gemm_rrr_{dtype}_{symbol_id}"
        for dtype, symbol_ids in (("float16", float16_symbol_ids), ("float32", float32_symbol_ids))
        for symbol_id in symbol_ids
    )
    assert support["profiler_symbols"] == sorted(
        f"dinoml_profile_cutlass_gemm_rrr_{dtype}_{symbol_id}"
        for dtype, symbol_ids in (("float16", float16_symbol_ids), ("float32", float32_symbol_ids))
        for symbol_id in symbol_ids
    )
    assert support["candidate_set_keys"] == sorted(item["candidate_set_key"] for item in manifest["required_kernels"])
    assert support["candidate_config_keys"] == sorted(candidate["candidate_config_key"] for candidate in candidates)


def test_softmax_manifest_and_generated_sources_are_model_owned():
    import dinoml as dml

    class SoftmaxModel(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.softmax(x), "y")

    spec = dml.trace(
        SoftmaxModel(),
        inputs={"x": dml.TensorSpec([256, 1024], "float32")},
        name="softmax_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    assert manifest["required_kernels"] == [
        {
            "op": "softmax",
            "kernel_symbol": "generated_softmax",
            "kernel_library": "model",
            "profiler_symbol": None,
            "has_profiler": False,
        }
    ]

    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    assert "expf" in sources["kernels"][0]
    assert "_packed_kernel" in sources["kernels"][0]
    assert "float4" in sources["kernels"][0]
    assert "<<<grid, block, 0, stream>>>" in sources["kernels"][0]


def test_softmax_cuda_source_policy_selects_warp_and_shared_paths():
    import dinoml as dml

    class SoftmaxModel(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.softmax(x), "y")

    warp_spec = dml.trace(
        SoftmaxModel(),
        inputs={"x": dml.TensorSpec([8192, 77], "float32")},
        name="softmax_warp_policy",
    )
    warp_lowered, _ = PassManager().run(warp_spec.ir)
    warp_tensor_map = {tensor["name"]: tensor for tensor in warp_lowered["tensors"]}
    warp_sources = collect_generated_sources("cuda", warp_lowered["nodes"], warp_tensor_map)
    assert "_warp_kernel" in warp_sources["kernels"][0]
    assert "_packed_kernel" not in warp_sources["kernels"][0]

    shared_spec = dml.trace(
        SoftmaxModel(),
        inputs={"x": dml.TensorSpec([256, 4096], "float32")},
        name="softmax_shared_policy",
    )
    shared_lowered, _ = PassManager().run(shared_spec.ir)
    shared_tensor_map = {tensor["name"]: tensor for tensor in shared_lowered["tensors"]}
    shared_sources = collect_generated_sources("cuda", shared_lowered["nodes"], shared_tensor_map)
    assert "_warp_kernel" not in shared_sources["kernels"][0]
    assert "_packed_kernel" not in shared_sources["kernels"][0]
    assert "block * sizeof(float)" in shared_sources["kernels"][0]


def test_reduction_manifest_and_keepdim_shape_inference():
    import dinoml as dml

    class ReduceModel(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.reduce_mean(x, keepdim=True), "y")

    spec = dml.trace(
        ReduceModel(),
        inputs={"x": dml.TensorSpec([2, 3, 4], "float32")},
        name="reduce_mean_manifest",
    )
    lowered, _ = PassManager().run(spec.ir)
    tensor_map = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    output_tensor = tensor_map[lowered["outputs"][0]["tensor"]]
    assert output_tensor["shape"] == [2, 3, 1]
    assert output_tensor["shape_spec"] == [2, 3, 1]

    manifest = build_kernel_manifest(lowered, {"name": "cpu", "arch": "native"})
    assert manifest["required_kernels"] == [
        {
            "op": "reduce_mean",
            "kernel_symbol": "generated_reduction",
            "kernel_library": "model",
            "profiler_symbol": None,
            "has_profiler": False,
        }
    ]

    sources = collect_generated_sources("cuda", lowered["nodes"], tensor_map)
    assert len(sources["kernels"]) == 1
    assert "reduce_mean_" in sources["kernels"][0]
    assert "acc / 4.00000000f" in sources["kernels"][0]


def test_shape_type_infer_propagates_dynamic_shape_spec_through_elementwise_broadcast():
    batch = {"kind": "dim", "name": "batch", "min": 1, "max": 4}
    ir = {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "dynamic_add_shape_spec",
        "inputs": [
            {
                "name": "x",
                "tensor": "x",
                "shape": [4, 16],
                "shape_spec": [batch, 16],
                "dtype": "float32",
            },
            {
                "name": "bias",
                "tensor": "bias",
                "shape": [1, 16],
                "shape_spec": [1, 16],
                "dtype": "float32",
            },
        ],
        "constants": [],
        "outputs": [{"name": "y", "tensor": "y", "shape": [4, 16], "shape_spec": [4, 16], "dtype": "float32"}],
        "nodes": [{"id": "n0", "op": "add", "inputs": ["x", "bias"], "outputs": ["y"], "attrs": {}}],
        "tensors": [
            {
                "name": "x",
                "shape": [4, 16],
                "shape_spec": [batch, 16],
                "dtype": "float32",
                "kind": "input",
                "nbytes": 256,
            },
            {
                "name": "bias",
                "shape": [1, 16],
                "shape_spec": [1, 16],
                "dtype": "float32",
                "kind": "input",
                "nbytes": 64,
            },
            {"name": "y", "shape": [4, 16], "shape_spec": [4, 16], "dtype": "float32", "kind": "output", "nbytes": 256},
        ],
        "metadata": {},
    }

    lowered, _ = PassManager(pipeline=("shape_type_infer",)).run(ir)

    tensors = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    assert tensors["y"]["shape_spec"] == [batch, 16]
    assert lowered["outputs"][0]["shape_spec"] == [batch, 16]


def test_shape_type_infer_propagates_dynamic_shape_spec_through_reductions():
    batch = {"kind": "dim", "name": "batch", "min": 1, "max": 4}
    rows = {"kind": "dim", "name": "rows", "min": 2, "max": 8}
    ir = {
        "schema_version": IR_SCHEMA_VERSION,
        "name": "dynamic_reduce_shape_spec",
        "inputs": [
            {
                "name": "x",
                "tensor": "x",
                "shape": [4, 8, 16],
                "shape_spec": [batch, rows, 16],
                "dtype": "float32",
            }
        ],
        "constants": [],
        "outputs": [
            {"name": "sum", "tensor": "sum", "shape": [4, 8], "shape_spec": [4, 8], "dtype": "float32"},
            {"name": "mean", "tensor": "mean", "shape": [4, 8, 1], "shape_spec": [4, 8, 1], "dtype": "float32"},
        ],
        "nodes": [
            {
                "id": "n0",
                "op": "reduce_sum",
                "inputs": ["x"],
                "outputs": ["sum"],
                "attrs": {"dim": -1, "keepdim": False},
            },
            {
                "id": "n1",
                "op": "reduce_mean",
                "inputs": ["x"],
                "outputs": ["mean"],
                "attrs": {"dim": -1, "keepdim": True},
            },
        ],
        "tensors": [
            {
                "name": "x",
                "shape": [4, 8, 16],
                "shape_spec": [batch, rows, 16],
                "dtype": "float32",
                "kind": "input",
                "nbytes": 2048,
            },
            {
                "name": "sum",
                "shape": [4, 8],
                "shape_spec": [4, 8],
                "dtype": "float32",
                "kind": "output",
                "nbytes": 128,
            },
            {
                "name": "mean",
                "shape": [4, 8, 1],
                "shape_spec": [4, 8, 1],
                "dtype": "float32",
                "kind": "output",
                "nbytes": 128,
            },
        ],
        "metadata": {},
    }

    lowered, _ = PassManager(pipeline=("shape_type_infer",)).run(ir)

    tensors = {tensor["name"]: tensor for tensor in lowered["tensors"]}
    assert tensors["sum"]["shape_spec"] == [batch, rows]
    assert tensors["mean"]["shape_spec"] == [batch, rows, 1]
    assert {output["name"]: output["shape_spec"] for output in lowered["outputs"]} == {
        "sum": [batch, rows],
        "mean": [batch, rows, 1],
    }


def test_shape_buffer_helpers_materialize_dynamic_runtime_dims():
    tensor_map = {
        "x": {
            "name": "x",
            "shape": [4, 16],
            "shape_spec": [{"kind": "dim", "name": "batch", "min": 1, "max": 4}, 16],
        },
        "tmp": {
            "name": "tmp",
            "shape": [4, 16],
            "shape_spec": [{"kind": "dim", "name": "batch", "min": 1, "max": 4}, 16],
        },
    }
    sources = dynamic_dim_sources(input_map={"x": 0}, output_map={}, tensor_map=tensor_map)
    assert sources == {"batch": "inputs[0].shape[0]"}
    assert shape_dim_expr(tensor_map["tmp"], 0, sources) == "inputs[0].shape[0]"
    assert shape_dim_expr(tensor_map["tmp"], 1, sources) == "16"
    assert numel_expr("tmp", 2) == "shape_tmp_0 * shape_tmp_1"
    assert shape_buffer_context(tensor_map["tmp"]) == {"ident": "tmp", "rank": 2, "shape_literal": "4, 16"}


def test_fused_elementwise_function_names_are_stable_and_clean():
    node = {
        "id": "n0_n1_n2_fused",
        "op": "fused_elementwise",
        "inputs": ["x", "scale"],
        "outputs": ["y"],
        "attrs": {
            "sub_ops": [
                {"op": "mul", "inputs": ["x", "scale"], "outputs": ["t0"], "attrs": {}},
                {"op": "relu", "inputs": ["t0"], "outputs": ["y"], "attrs": {}},
            ]
        },
    }
    renamed_node = {**node, "id": "different_graph_node_id"}
    changed_node = {
        **node,
        "attrs": {
            "sub_ops": [
                {"op": "add", "inputs": ["x", "scale"], "outputs": ["t0"], "attrs": {}},
                {"op": "relu", "inputs": ["t0"], "outputs": ["y"], "attrs": {}},
            ]
        },
    }

    name = _function_name(node)

    assert name.startswith("fused_elementwise_")
    assert "dino_fused" not in name
    assert "n0_n1_n2" not in name
    assert _function_name(renamed_node) == name
    assert _function_name(changed_node) != name
    assert _broadcast_function_name(node, "x") == f"{name}_idx_x"


def test_render_generated_kernels_deduplicates_exact_fused_sources(tmp_path):
    node = {
        "id": "n0_n1_fused",
        "op": "fused_elementwise",
        "inputs": ["x", "scale"],
        "outputs": ["y"],
        "attrs": {
            "sub_ops": [
                {"op": "mul", "inputs": ["x", "scale"], "outputs": ["t0"], "attrs": {}},
                {"op": "relu", "inputs": ["t0"], "outputs": ["y"], "attrs": {}},
            ]
        },
    }
    tensor_map = {
        "x": {"name": "x", "shape": [4, 16], "dtype": "float32"},
        "scale": {"name": "scale", "shape": [16], "dtype": "float32"},
        "y": {"name": "y", "shape": [4, 16], "dtype": "float32", "kind": "output"},
    }
    nodes = [node, {**node, "id": "duplicate_fused"}]

    kernels = render_generated_kernels("cpu", nodes, tensor_map)
    generated_sources = collect_generated_sources("cpu", nodes, tensor_map, generated_src_dir=tmp_path)
    launches = [render_launch("cpu", item, tensor_map) for item in nodes]
    manifest = generated_sources["manifest"]
    manifest_sources = manifest["sources"]

    assert len(kernels) == 1
    assert kernels[0].count(f"int {_function_name(node)}(") == 1
    assert generated_sources["kernels"] == kernels
    assert (tmp_path / "source_manifest.json").exists()
    assert len(manifest_sources) == 2
    assert manifest_sources[0]["node_id"] == "n0_n1_fused"
    assert manifest_sources[0]["op"] == "fused_elementwise"
    assert manifest_sources[0]["target"] == "cpu"
    assert manifest_sources[0]["generated_function_name"] == _function_name(node)
    assert manifest_sources[0]["emitted_new_source"] is True
    assert manifest_sources[1]["node_id"] == "duplicate_fused"
    assert manifest_sources[1]["emitted_new_source"] is False
    assert manifest_sources[1]["emitted_source_path"] == manifest_sources[0]["emitted_source_path"]
    assert (tmp_path / manifest_sources[0]["emitted_source_path"]).exists()
    assert len(launches) == 2
