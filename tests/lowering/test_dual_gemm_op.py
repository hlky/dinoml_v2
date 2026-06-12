from __future__ import annotations

import pytest

import dinoml as dml
from dinoml.compiler import _validate_mvp_runtime_contract
from dinoml.kernels.families.dual_gemm import DUAL_GEMM_OPS
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.cuda import render_cuda_module
from dinoml.lowering.ops import render_launch
from dinoml.lowering.rocm import render_rocm_module
from tests.dual_gemm_parity import DUAL_GEMM_CASES, trace_dual_gemm_spec


class _DualGemmLoweringModule(dml.Module):
    def __init__(self, op_name: str):
        self.op_name = op_name

    def forward(self, a, b0, b1, bias0=None, bias1=None):
        op = getattr(dml.ops, self.op_name)
        args = [a, b0, b1]
        if "_bias_" in self.op_name:
            args.extend([bias0, bias1])
        return dml.ops.output(op(*args), "y")


def test_dual_gemm_gpu_contract_accepts_dynamic_shape_specs():
    case = next(case for case in DUAL_GEMM_CASES if case.name == "dual_gemm_bias_fast_gelu_bf16_dynamic")
    spec = trace_dual_gemm_spec(case)

    _validate_mvp_runtime_contract(spec.ir, dml.Target("cuda"))
    _validate_mvp_runtime_contract(spec.ir, dml.Target("rocm"))


def test_dual_gemm_cuda_manifest_and_lowering_support_runtime_b1_extent():
    case = next(case for case in DUAL_GEMM_CASES if case.name == "dual_gemm_fast_gelu_f16_broadcast_dynamic")
    spec = trace_dual_gemm_spec(case)
    manifest = build_kernel_manifest(spec.ir, dml.Target("cuda").to_json())
    item = manifest["required_kernels"][0]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = spec.ir["nodes"][0]
    launch = render_launch("cuda", node, tensors, kernel_manifest=manifest)
    source = render_cuda_module(spec.ir, kernel_manifest=manifest)

    assert item["kernel_library"] == "cutlass_gemm"
    assert item["candidate_set"]["launch_abi"] == "dinoml_cutlass_dual_gemm_v1"
    assert item["candidate_set"]["epilogue_config"]["activation"] == "fast_gelu"
    assert item["candidate_set"]["layouts"] == {"a": "row", "b0": "column", "b1": "column", "c": "row"}
    assert 'extern "C" int ' + item["kernel_symbol"] in source
    assert "const half* b1" in source
    assert "shape_b1_0 != 1 && shape_b1_0 != shape_b0_0" in launch
    assert "ptr_b1" in launch
    assert "static_cast<int>(shape_b1_0)" in launch
    assert item["kernel_symbol"] in launch


def test_dual_gemm_rocm_manifest_and_lowering_support_bias_dual_abi():
    case = next(case for case in DUAL_GEMM_CASES if case.name == "dual_gemm_bias_fast_gelu_bf16_dynamic")
    spec = trace_dual_gemm_spec(case)
    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    item = manifest["required_kernels"][0]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = spec.ir["nodes"][0]
    launch = render_launch("rocm", node, tensors, kernel_manifest=manifest)
    source = render_rocm_module(spec.ir, kernel_manifest=manifest)

    assert item["kernel_library"] == "ck_gemm"
    assert item["candidate_set"]["launch_abi"] == "dinoml_ck_dual_gemm_bias_v1"
    assert item["candidate_set"]["epilogue_config"]["activation"] == "fast_gelu"
    assert item["candidate_set"]["layouts"] == {"a": "row", "b0": "column", "b1": "column", "c": "row"}
    assert 'extern "C" int ' + item["kernel_symbol"] in source
    assert "ptr_b1" in launch
    assert "ptr_bias0" in launch
    assert "ptr_bias1" in launch
    assert "static_cast<int>(shape_b1_0)" in launch
    assert item["kernel_symbol"] in launch


@pytest.mark.parametrize(
    ("target_name", "op_name", "expected_library", "expected_abi"),
    [
        ("cuda", "dual_gemm_rcr_silu", "cutlass_gemm", "dinoml_cutlass_dual_gemm_v1"),
        ("cuda", "dual_gemm_rcr_bias_hardswish", "cutlass_gemm", "dinoml_cutlass_dual_gemm_bias_v1"),
        ("rocm", "dual_gemm_rcr_quick_gelu", "ck_gemm", "dinoml_ck_dual_gemm_v1"),
        ("rocm", "dual_gemm_rcr_bias_elup1", "ck_gemm", "dinoml_ck_dual_gemm_bias_v1"),
    ],
)
def test_dual_gemm_manifest_family_representatives_use_dual_launch_abi(
    target_name: str,
    op_name: str,
    expected_library: str,
    expected_abi: str,
):
    spec = _trace_dual_gemm_lowering_spec(op_name, dtype="float16")
    manifest = build_kernel_manifest(spec.ir, dml.Target(target_name).to_json())
    item = manifest["required_kernels"][0]

    assert item["op"] == op_name
    assert item["kernel_library"] == expected_library
    assert item["candidate_set"]["launch_abi"] == expected_abi
    assert item["candidate_set"]["epilogue_config"]["activation"] == _expected_activation(op_name)


@pytest.mark.parametrize("target_name", ["cuda", "rocm"])
@pytest.mark.parametrize("op_name", DUAL_GEMM_OPS)
def test_dual_gemm_manifest_all_epilogues_stay_on_dual_provider_path(target_name: str, op_name: str):
    spec = _trace_dual_gemm_lowering_spec(op_name, dtype="float16")
    manifest = build_kernel_manifest(spec.ir, dml.Target(target_name).to_json())
    item = manifest["required_kernels"][0]

    is_bias = "_bias_" in op_name
    expected_library = "cutlass_gemm" if target_name == "cuda" else "ck_gemm"
    expected_abi = (
        "dinoml_cutlass_dual_gemm_bias_v1"
        if target_name == "cuda" and is_bias
        else "dinoml_cutlass_dual_gemm_v1"
        if target_name == "cuda"
        else "dinoml_ck_dual_gemm_bias_v1"
        if is_bias
        else "dinoml_ck_dual_gemm_v1"
    )

    assert item["op"] == op_name
    assert item["kernel_library"] == expected_library
    assert item["candidate_set"]["launch_abi"] == expected_abi
    assert item["candidate_set"]["epilogue_config"]["activation"] == _expected_activation(op_name)


def _trace_dual_gemm_lowering_spec(op_name: str, *, dtype: str):
    has_bias = "_bias_" in op_name
    b1_n = 1 if "quick_gelu" in op_name else 6
    inputs = {
        "a": dml.TensorSpec([2, 3, 8], dtype),
        "b0": dml.TensorSpec([6, 8], dtype),
        "b1": dml.TensorSpec([b1_n, 8], dtype),
    }
    if has_bias:
        inputs["bias0"] = dml.TensorSpec([6], dtype)
        inputs["bias1"] = dml.TensorSpec([b1_n], dtype)
    return dml.trace(_DualGemmLoweringModule(op_name), inputs=inputs, name=f"{op_name}_{dtype}_lowering")


def _expected_activation(op_name: str) -> str:
    activation = op_name.split("_bias_", 1)[1] if "_bias_" in op_name else op_name.split("dual_gemm_rcr_", 1)[1]
    return activation
