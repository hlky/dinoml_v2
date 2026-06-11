from __future__ import annotations

import dinoml as dml
from dinoml.compiler import _validate_mvp_runtime_contract
from dinoml.lowering.cuda import render_cuda_module
from dinoml.lowering.ops import render_generated_kernels, render_launch
from dinoml.lowering.rocm import render_rocm_module


class _MaskedSelectModule(dml.Module):
    def forward(self, x, mask):
        return dml.ops.output(dml.ops.masked_select(x, mask), "out")


def _masked_select_spec():
    rows = dml.Dim("rows", min=1, max=4, typical=2, buckets=(2, 4))
    width = dml.Dim("width", min=1, max=8, typical=3, buckets=(3, 8))
    return dml.trace(
        _MaskedSelectModule(),
        inputs={
            "x": dml.TensorSpec([rows, 1, width], "bfloat16"),
            "mask": dml.TensorSpec([1, 4, width], "bool"),
        },
        name="masked_select_gpu_lowering",
    )


def test_masked_select_generated_kernel_renders_for_cpu():
    spec = _masked_select_spec()
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_name = str(spec.ir["nodes"][0]["outputs"][0])
    source = render_generated_kernels("cpu", spec.ir["nodes"], tensor_map)[0]
    launch = render_launch("cpu", spec.ir["nodes"][0], tensor_map)

    assert "masked_select_" in source
    assert "out_shape[0] = count;" in source
    assert "const bool* DINO_RESTRICT mask" in source
    assert f"session->shape_{output_name}.data()" in launch


def test_masked_select_generated_kernel_renders_parallel_gpu_paths():
    spec = _masked_select_spec()
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    output_name = str(spec.ir["nodes"][0]["outputs"][0])

    for target in ("cuda", "rocm"):
        source = render_generated_kernels(target, spec.ir["nodes"], tensor_map)[0]
        launch = render_launch(target, spec.ir["nodes"][0], tensor_map)

        assert "DeviceSelect::Flagged" in source
        assert "masked_select scratch is too small" in source
        assert "<<<1, 1, 0, stream>>>" not in source
        assert f"session->shape_{output_name}" in launch
        assert "session->masked_select_scratch" in launch
        assert "session->masked_select_scratch_nbytes" in launch


def test_masked_select_gpu_modules_report_runtime_output_shape_from_shape_buffer():
    spec = _masked_select_spec()
    output_name = str(spec.ir["nodes"][0]["outputs"][0])

    cuda_source = render_cuda_module(spec.ir)
    rocm_source = render_rocm_module(spec.ir)

    for source, memcpy_name in ((cuda_source, "cudaMemcpyAsync"), (rocm_source, "hipMemcpyAsync")):
        assert "masked_select_" in source
        assert "void* masked_select_scratch = nullptr;" in source
        assert f"session->shape_{output_name}" in source
        assert "session->masked_select_scratch" in source
        assert memcpy_name in source
        assert "session->last_output_shapes[0].resize(1);" in source
        assert "session->last_output_shapes[0].data()" in source


def test_masked_select_gpu_contract_accepts_bool_mask_and_dynamic_capacity():
    spec = _masked_select_spec()

    _validate_mvp_runtime_contract(spec.ir, dml.Target("cuda"))
    _validate_mvp_runtime_contract(spec.ir, dml.Target("rocm"))
