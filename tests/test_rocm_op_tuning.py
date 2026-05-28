from __future__ import annotations

import dinoml as dml
from dinoml.lowering.ops import render_generated_kernels


class _ReduceSumModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.reduce_sum(x, dim=-1), "output")


class _ArgmaxModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.argmax(x, dim=-1), "output")


def test_rocm_reduce_sum_uses_more_warp_rows_than_cuda():
    spec = dml.trace(
        _ReduceSumModule(),
        inputs={"x": dml.TensorSpec([32, 128, 1024], "float32")},
        name="rocm_reduce_sum_tuning",
    )
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    rocm_source = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]
    cuda_source = render_generated_kernels("cuda", spec.ir["nodes"], tensor_map)[0]

    assert "dim3 block(32, 8)" in rocm_source
    assert "dim3 block(32, 4)" in cuda_source


def test_rocm_argmax_uses_warp_row_reduction():
    spec = dml.trace(
        _ArgmaxModule(),
        inputs={"x": dml.TensorSpec([32, 128, 1024], "float32")},
        name="rocm_argmax_tuning",
    )
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    source = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]

    assert "_warp_kernel" in source
    assert "dim3 block(32, 8)" in source
    assert "col < best_index" in source
