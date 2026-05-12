import numpy as np
import pytest

import dinoml as dml
from dinoml.compiler import _validate_gguf_runtime_dequant_admission
from dinoml.constant_sources import MaterializedConstant
from dinoml.ir import array_to_storage
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.profiling import build_profile_workloads
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


class MixedGemmRrrEncodedRhs(dml.Module):
    def __init__(self):
        self.dense_weight = dml.Parameter([32, 32], dtype="float32")
        self.gguf_weight = dml.Parameter([32, 32], dtype="float32")

    def forward(self, x):
        dense_y = dml.ops.output(dml.ops.gemm_rrr(x, self.dense_weight), "dense_y")
        gguf_y = dml.ops.output(dml.ops.gemm_rrr(x, self.gguf_weight), "gguf_y")
        return dense_y, gguf_y


class AddEncodedWeight(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([32, 32], dtype="float32")

    def forward(self, x):
        return dml.ops.output(x + self.weight, "y")


class GemmRcrEncodedRhs(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([24, 32], dtype="float32")

    def forward(self, x):
        return dml.ops.output(dml.ops.gemm_rcr(x, self.weight), "y")


class GemmRrrBiasEncodedRhs(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([32, 32], dtype="float32")
        self.bias = dml.Parameter([32], dtype="float32")

    def forward(self, x):
        return dml.ops.output(dml.ops.gemm_rrr_bias(x, self.weight, self.bias), "y")


class GemmRcrBiasEncodedRhs(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([24, 32], dtype="float32")
        self.bias = dml.Parameter([24], dtype="float32")

    def forward(self, x):
        return dml.ops.output(dml.ops.gemm_rcr_bias(x, self.weight, self.bias), "y")


class GemmRcrBiasReluEncodedRhs(dml.Module):
    def __init__(self):
        self.weight = dml.Parameter([24, 32], dtype="float32")
        self.bias = dml.Parameter([24], dtype="float32")

    def forward(self, x):
        return dml.ops.output(dml.ops.gemm_rcr_bias_relu(x, self.weight, self.bias), "y")


class MultiGemmRuntimeDequantEncodedRhs(dml.Module):
    def __init__(self):
        self.small_weight = dml.Parameter([24, 32], dtype="float32")
        self.large_weight = dml.Parameter([64, 64], dtype="float32")

    def forward(self, x_small, x_large):
        small_y = dml.ops.output(dml.ops.gemm_rcr(x_small, self.small_weight), "small_y")
        large_y = dml.ops.output(dml.ops.gemm_rrr(x_large, self.large_weight), "large_y")
        return small_y, large_y


def _encoded_rhs_spec(*, materialization, residency="manual_runtime_load"):
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
        "residency": residency,
    }
    return dml.trace(
        GemmRrrEncodedRhs(),
        inputs={"x": dml.TensorSpec([4, 32], "float32")},
        constants={"weight": MaterializingConstant(np.zeros((32, 32), dtype=np.float32), storage)},
        name=f"gemm_rrr_{materialization}",
    )


def _encoded_rcr_rhs_spec():
    storage = {
        "kind": "gguf",
        "path": "weights.gguf",
        "tensor": "blk.0.ffn.weight",
        "logical_dtype": "float32",
        "shape": [24, 32],
        "qtype": "Q4_0",
        "qtype_value": 2,
        "encoded_nbytes": 432,
        "gguf_shape": [32, 24],
        "n_per_row": 32,
        "materialization": "dequantize_on_gpu_before_launch",
        "residency": "manual_runtime_load",
    }
    return dml.trace(
        GemmRcrEncodedRhs(),
        inputs={"x": dml.TensorSpec([4, 32], "float32")},
        constants={"weight": MaterializingConstant(np.zeros((24, 32), dtype=np.float32), storage)},
        name="gemm_rcr_dequantize_on_gpu_before_launch",
    )


def _encoded_rrr_bias_rhs_spec():
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
        "materialization": "dequantize_on_gpu_before_launch",
        "residency": "manual_runtime_load",
    }
    return dml.trace(
        GemmRrrBiasEncodedRhs(),
        inputs={"x": dml.TensorSpec([4, 32], "float32")},
        constants={
            "weight": MaterializingConstant(np.zeros((32, 32), dtype=np.float32), storage),
            "bias": MaterializingConstant(np.zeros((32,), dtype=np.float32), None),
        },
        name="gemm_rrr_bias_dequantize_on_gpu_before_launch",
    )


def _encoded_rcr_bias_rhs_spec():
    storage = {
        "kind": "gguf",
        "path": "weights.gguf",
        "tensor": "blk.0.ffn.weight",
        "logical_dtype": "float32",
        "shape": [24, 32],
        "qtype": "Q4_0",
        "qtype_value": 2,
        "encoded_nbytes": 432,
        "gguf_shape": [32, 24],
        "n_per_row": 32,
        "materialization": "dequantize_on_gpu_before_launch",
        "residency": "manual_runtime_load",
    }
    return dml.trace(
        GemmRcrBiasEncodedRhs(),
        inputs={"x": dml.TensorSpec([4, 32], "float32")},
        constants={
            "weight": MaterializingConstant(np.zeros((24, 32), dtype=np.float32), storage),
            "bias": MaterializingConstant(np.zeros((24,), dtype=np.float32), None),
        },
        name="gemm_rcr_bias_dequantize_on_gpu_before_launch",
    )


def _unsupported_encoded_spec(module, *, weight_shape, input_shape, name):
    storage = {
        "kind": "gguf",
        "path": "weights.gguf",
        "tensor": "blk.0.ffn.weight",
        "logical_dtype": "float32",
        "shape": list(weight_shape),
        "qtype": "Q4_0",
        "qtype_value": 2,
        "encoded_nbytes": 576,
        "gguf_shape": list(reversed(weight_shape)),
        "n_per_row": int(weight_shape[-1]),
        "materialization": "dequantize_on_gpu_before_launch",
        "residency": "manual_runtime_load",
    }
    constants = {"weight": MaterializingConstant(np.zeros(weight_shape, dtype=np.float32), storage)}
    if hasattr(module, "bias"):
        constants["bias"] = MaterializingConstant(np.zeros((weight_shape[0],), dtype=np.float32), None)
    return dml.trace(
        module,
        inputs={"x": dml.TensorSpec(list(input_shape), "float32")},
        constants=constants,
        name=name,
    )


def _mixed_dense_and_gguf_spec():
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
        "materialization": "dequantize_on_gpu_before_launch",
        "residency": "manual_runtime_load",
    }
    return dml.trace(
        MixedGemmRrrEncodedRhs(),
        inputs={"x": dml.TensorSpec([4, 32], "float32")},
        constants={
            "dense_weight": MaterializingConstant(np.zeros((32, 32), dtype=np.float32), None),
            "gguf_weight": MaterializingConstant(np.zeros((32, 32), dtype=np.float32), storage),
        },
        name="mixed_gemm_rrr_dequant",
    )


def _multi_runtime_dequant_spec():
    small_storage = {
        "kind": "gguf",
        "path": "weights_small.gguf",
        "tensor": "blk.0.ffn.small_weight",
        "logical_dtype": "float32",
        "shape": [24, 32],
        "qtype": "Q4_0",
        "qtype_value": 2,
        "encoded_nbytes": 432,
        "gguf_shape": [32, 24],
        "n_per_row": 32,
        "materialization": "dequantize_on_gpu_before_launch",
        "residency": "manual_runtime_load",
    }
    large_storage = {
        "kind": "gguf",
        "path": "weights_large.gguf",
        "tensor": "blk.0.ffn.large_weight",
        "logical_dtype": "float32",
        "shape": [64, 64],
        "qtype": "Q4_0",
        "qtype_value": 2,
        "encoded_nbytes": 2304,
        "gguf_shape": [64, 64],
        "n_per_row": 64,
        "materialization": "dequantize_on_gpu_before_launch",
        "residency": "manual_runtime_load",
    }
    return dml.trace(
        MultiGemmRuntimeDequantEncodedRhs(),
        inputs={
            "x_small": dml.TensorSpec([4, 32], "float32"),
            "x_large": dml.TensorSpec([4, 64], "float32"),
        },
        constants={
            "small_weight": MaterializingConstant(np.zeros((24, 32), dtype=np.float32), small_storage),
            "large_weight": MaterializingConstant(np.zeros((64, 64), dtype=np.float32), large_storage),
        },
        name="multi_gemm_runtime_dequant",
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
    assert plan["status"] == "lowered_runtime_dequant_scratch"
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


def test_cutlass_manifest_marks_gguf_runtime_dequant_rcr_rhs_plan():
    spec = _encoded_rcr_rhs_spec()

    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    item = manifest["required_kernels"][0]
    plan = item["gguf_runtime_dequant"]
    assert plan["status"] == "lowered_runtime_dequant_scratch"
    assert plan["op"] == "gemm_rcr"
    assert plan["operand"] == "b"
    assert plan["constant"] == "weight"
    assert plan["n_per_row"] == 32
    assert plan["logical_shape"] == [24, 32]
    assert plan["scratch_nbytes"] == 24 * 32 * 4
    assert plan["dense_launcher"] == "existing_cutlass_gemm"


@pytest.mark.parametrize(
    ("spec_builder", "op_name", "logical_shape", "scratch_nbytes"),
    [
        pytest.param(_encoded_rrr_bias_rhs_spec, "gemm_rrr_bias", [32, 32], 32 * 32 * 4, id="rrr"),
        pytest.param(_encoded_rcr_bias_rhs_spec, "gemm_rcr_bias", [24, 32], 24 * 32 * 4, id="rcr"),
    ],
)
def test_cutlass_manifest_marks_gguf_runtime_dequant_bias_rhs_plan(
    spec_builder,
    op_name,
    logical_shape,
    scratch_nbytes,
):
    spec = spec_builder()

    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    item = manifest["required_kernels"][0]
    plan = item["gguf_runtime_dequant"]
    assert plan["status"] == "lowered_runtime_dequant_scratch"
    assert plan["op"] == op_name
    assert plan["operand"] == "b"
    assert plan["constant"] == "weight"
    assert plan["logical_shape"] == logical_shape
    assert plan["scratch_nbytes"] == scratch_nbytes
    assert plan["dense_launcher"] == "existing_cutlass_gemm"


def test_cutlass_manifest_does_not_mark_existing_load_time_gguf_policy():
    spec = _encoded_rhs_spec(materialization="dequantize_full_before_launch")

    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    assert "gguf_runtime_dequant" not in manifest["required_kernels"][0]


def test_cutlass_manifest_skips_native_eager_load_for_runtime_dequant_constants():
    spec = _mixed_dense_and_gguf_spec()
    ir = _renderable_ir(spec)
    manifest = build_kernel_manifest(ir, CUDA_TARGET)

    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert source.count("module->const_dense_weight_loaded = true;") == 2
    assert source.count("module->const_gguf_weight_loaded = true;") == 1
    assert (
        "// Constant gguf_weight uses explicit encoded runtime load; skip eager constants.bin materialization."
        in source
    )


def test_cuda_gemm_lowering_emits_runtime_gguf_dequant_before_cutlass_gemm():
    spec = _encoded_rhs_spec(materialization="dequantize_on_gpu_before_launch")
    ir = _renderable_ir(spec)
    manifest = build_kernel_manifest(ir, CUDA_TARGET)

    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert "void* gguf_dequant_scratch = nullptr;" in source
    assert "dino_module_set_encoded_constant" in source
    assert "dino_module_set_libgguf_cuda_dequantize_rows_on_stream" in source
    assert (
        'return dinoml::module::fail("gemm_rrr GGUF runtime dequant for constant weight requires native libgguf CUDA dequant launcher");'
        in source
    )
    dequant_call = "module->libgguf_cuda_dequantize_rows_on_stream(module->const_weight, 2, shape_weight_0, 32, 0, session->gguf_dequant_scratch, session->stream)"
    gemm_call = "ptr_x, ptr_weight_dequant"
    assert dequant_call in source
    assert gemm_call in source
    assert source.index(dequant_call) < source.index(gemm_call)


def test_cuda_gemm_rcr_lowering_emits_runtime_gguf_dequant_before_cutlass_gemm():
    spec = _encoded_rcr_rhs_spec()
    ir = _renderable_ir(spec)
    manifest = build_kernel_manifest(ir, CUDA_TARGET)

    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert "void* gguf_dequant_scratch = nullptr;" in source
    assert (
        'return dinoml::module::fail("gemm_rcr GGUF runtime dequant for constant weight requires native libgguf CUDA dequant launcher");'
        in source
    )
    dequant_call = "module->libgguf_cuda_dequantize_rows_on_stream(module->const_weight, 2, shape_weight_0, 32, 0, session->gguf_dequant_scratch, session->stream)"
    gemm_call = "ptr_x, ptr_weight_dequant"
    assert dequant_call in source
    assert gemm_call in source
    assert source.index(dequant_call) < source.index(gemm_call)


@pytest.mark.parametrize(
    ("spec_builder", "op_name", "dequant_call", "gemm_call"),
    [
        pytest.param(
            _encoded_rrr_bias_rhs_spec,
            "gemm_rrr_bias",
            "module->libgguf_cuda_dequantize_rows_on_stream(module->const_weight, 2, shape_weight_0, 32, 0, session->gguf_dequant_scratch, session->stream)",
            "ptr_x, ptr_weight_dequant, ptr_bias, ",
            id="rrr",
        ),
        pytest.param(
            _encoded_rcr_bias_rhs_spec,
            "gemm_rcr_bias",
            "module->libgguf_cuda_dequantize_rows_on_stream(module->const_weight, 2, shape_weight_0, 32, 0, session->gguf_dequant_scratch, session->stream)",
            "ptr_x, ptr_weight_dequant, ptr_bias, ",
            id="rcr",
        ),
    ],
)
def test_cuda_gemm_bias_lowering_emits_runtime_gguf_dequant_before_cutlass_gemm(
    spec_builder,
    op_name,
    dequant_call,
    gemm_call,
):
    spec = spec_builder()
    ir = _renderable_ir(spec)
    manifest = build_kernel_manifest(ir, CUDA_TARGET)

    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert "void* gguf_dequant_scratch = nullptr;" in source
    assert (
        f'return dinoml::module::fail("{op_name} GGUF runtime dequant for constant weight requires native libgguf CUDA dequant launcher");'
        in source
    )
    assert dequant_call in source
    assert gemm_call in source
    assert source.index(dequant_call) < source.index(gemm_call)


def test_cuda_gemm_lowering_shares_max_sized_runtime_gguf_dequant_scratch():
    spec = _multi_runtime_dequant_spec()
    ir = _renderable_ir(spec)
    manifest = build_kernel_manifest(ir, CUDA_TARGET)

    plans = {
        str(item["gguf_runtime_dequant"]["constant"]): item["gguf_runtime_dequant"]
        for item in manifest["required_kernels"]
        if "gguf_runtime_dequant" in item
    }
    assert plans["small_weight"]["scratch_nbytes"] == 24 * 32 * 4
    assert plans["large_weight"]["scratch_nbytes"] == 64 * 64 * 4

    source = render_cuda_module(ir, kernel_manifest=manifest)

    assert source.count("void* gguf_dequant_scratch = nullptr;") == 1
    assert "size_t gguf_dequant_scratch_nbytes = 16384;" in source
    assert "cudaMalloc(&session->gguf_dequant_scratch, 16384)" in source
    assert (
        'if (session->gguf_dequant_scratch_nbytes < 3072) return dinoml::module::fail("gemm_rcr GGUF runtime dequant scratch is too small");'
        in source
    )
    assert (
        'if (session->gguf_dequant_scratch_nbytes < 16384) return dinoml::module::fail("gemm_rrr GGUF runtime dequant scratch is too small");'
        in source
    )
    assert source.count("session->gguf_dequant_scratch, session->stream") == 2


@pytest.mark.parametrize("spec_builder", [_encoded_rrr_bias_rhs_spec, _encoded_rcr_bias_rhs_spec])
def test_gguf_runtime_dequant_admission_accepts_bias_rhs_uses(spec_builder):
    spec = spec_builder()
    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    _validate_gguf_runtime_dequant_admission(spec.ir, dml.Target("cuda", arch="sm_86"), manifest)


def test_gguf_runtime_dequant_requires_manual_runtime_load_residency():
    spec = _encoded_rhs_spec(
        materialization="dequantize_on_gpu_before_launch",
        residency="eager_dense_device",
    )
    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    plan = manifest["required_kernels"][0]["gguf_runtime_dequant"]
    assert plan["status"] == "planned_not_lowered"
    assert plan["blocked_reason"] == "unsupported_gguf_runtime_dequant_residency:eager_dense_device"
    assert plan["supported_residency"] == "manual_runtime_load"

    with pytest.raises(
        NotImplementedError,
        match=(
            r"with residency='manual_runtime_load'; unsupported uses: "
            r"weight: unsupported_residency:eager_dense_device, gemm_rrr\[input 1\], no_supported_lowered_use"
        ),
    ):
        _validate_gguf_runtime_dequant_admission(spec.ir, dml.Target("cuda", arch="sm_86"), manifest)


def test_build_profile_workloads_accepts_lowered_gguf_runtime_dequant_rhs_in_mixed_graph():
    spec = _mixed_dense_and_gguf_spec()
    manifest = build_kernel_manifest(spec.ir, CUDA_TARGET)

    workloads = build_profile_workloads(spec.ir, manifest)

    assert {workload.node_id for workload in workloads} == {"n0", "n1"}


@pytest.mark.parametrize(
    ("spec", "manifest_builder", "match"),
    [
        pytest.param(
            _unsupported_encoded_spec(
                AddEncodedWeight(),
                weight_shape=(32, 32),
                input_shape=(32, 32),
                name="add_runtime_dequant",
            ),
            lambda spec: {"required_kernels": []},
            r"unsupported uses: weight: add\[input 1\], no_supported_lowered_use",
            id="add",
        ),
        pytest.param(
            _unsupported_encoded_spec(
                GemmRcrBiasReluEncodedRhs(),
                weight_shape=(24, 32),
                input_shape=(4, 32),
                name="gemm_rcr_bias_relu_runtime_dequant",
            ),
            lambda spec: build_kernel_manifest(spec.ir, CUDA_TARGET),
            r"unsupported uses: weight: gemm_rcr_bias_relu\[input 1\], no_supported_lowered_use",
            id="gemm_rcr_bias_relu",
        ),
    ],
)
def test_gguf_runtime_dequant_admission_rejects_unsupported_uses(spec, manifest_builder, match):
    manifest = manifest_builder(spec)

    with pytest.raises(NotImplementedError, match=match):
        _validate_gguf_runtime_dequant_admission(spec.ir, dml.Target("cuda", arch="sm_86"), manifest)
