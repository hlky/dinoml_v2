import numpy as np
import pytest

import dinoml as dml
from dinoml.constant_sources import MaterializedConstant
from dinoml.ir import array_to_storage
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.cuda import render_cuda_module


CUDA_TARGET = {"name": "cuda", "arch": "sm_86"}


class MaterializingConstant:
    def __init__(self, value, storage):
        self.value = np.asarray(value)
        self.storage = storage

    def materialize(self, dtype, shape):
        return MaterializedConstant(array_to_storage(self.value, dtype), self.storage)


class GemmRrrEncodedRhs(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([32, 32], dtype="float32")

    def forward(self, x):
        return dml.ops.output(dml.ops.gemm_rrr(x, self.weight), "y")


def _encoded_rhs_spec(*, materialization):
    storage = {
        "kind": "gguf",
        "path": "weights.gguf",
        "tensor": "blk.0.ffn.weight",
        "logical_dtype": "float32",
        "shape": [32, 32],
        "qtype": "Q4_0",
        "qtype_value": 2,
        "encoded_nbytes": 576,
        "gguf_shape": [32, 32],
        "n_per_row": 32,
        "materialization": materialization,
        "residency": "manual_runtime_load",
    }
    return dml.trace(
        GemmRrrEncodedRhs(),
        inputs={"x": dml.TensorSpec([4, 32], "float32")},
        constants={"weight": MaterializingConstant(np.zeros((32, 32), dtype=np.float32), storage)},
        name=f"gemm_rrr_{materialization}",
    )


def _renderable_ir(spec):
    ir = spec.clone().ir
    for constant in ir["constants"]:
        if constant["offset"] is None:
            constant["offset"] = 0
    return ir


def test_cutlass_manifest_marks_gguf_runtime_dequant_rhs_plan():
    spec = _encoded_rhs_spec(materialization="dequantize_on_gpu_before_launch")

    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    item = manifest["required_kernels"][0]
    plan = item["gguf_runtime_dequant"]
    assert plan["kind"] == "gguf_runtime_dequant_before_cutlass_gemm"
    assert plan["status"] == "planned_not_lowered"
    assert plan["blocked_reason"] == "missing_native_libgguf_cuda_dequant_launcher_abi"
    assert plan["op"] == "gemm_rrr"
    assert plan["operand"] == "b"
    assert plan["constant"] == "weight"
    assert plan["materialization"] == "dequantize_on_gpu_before_launch"
    assert plan["residency"] == "manual_runtime_load"
    assert plan["qtype"] == "Q4_0"
    assert plan["encoded_nbytes"] == 576
    assert plan["logical_shape"] == [32, 32]
    assert plan["scratch_nbytes"] == 32 * 32 * 4
    assert plan["dequant_scratch"] == "session_temporary_dense_rhs"
    assert plan["dense_launcher"] == "existing_cutlass_gemm"


def test_cutlass_manifest_does_not_mark_existing_load_time_gguf_policy():
    spec = _encoded_rhs_spec(materialization="dequantize_full_before_launch")

    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    assert "gguf_runtime_dequant" not in manifest["required_kernels"][0]


def test_cuda_gemm_lowering_rejects_planned_gguf_runtime_dequant_rhs():
    spec = _encoded_rhs_spec(materialization="dequantize_on_gpu_before_launch")
    ir = _renderable_ir(spec)
    manifest = build_kernel_manifest(ir, CUDA_TARGET)

    with pytest.raises(NotImplementedError, match="native libgguf CUDA dequant launcher ABI"):
        render_cuda_module(ir, kernel_manifest=manifest)
