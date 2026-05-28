from __future__ import annotations

import dinoml as dml
from dinoml.benchmarks.ops import benchmark_cases
from dinoml.lowering.gpu import render_gpu_module
from dinoml.lowering.ops import render_generated_kernels


class _ReduceSumModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.reduce_sum(x, dim=-1), "output")


class _ArgmaxModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.argmax(x, dim=-1), "output")


class _LayerNormModule(dml.Module):
    def forward(self, x, weight, bias):
        return dml.ops.output(dml.ops.layer_norm(x, weight, bias, eps=1e-5), "output")


class _T5LayerNormModule(dml.Module):
    def forward(self, x, weight):
        return dml.ops.output(dml.ops.t5_layer_norm(x, weight, eps=1e-6), "output")


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


def test_rocm_topk_benchmark_shape_stays_on_no_scratch_warp_rows():
    case = next(case for case in benchmark_cases() if case.name == "topk")
    source = render_gpu_module("rocm", case.build_spec().ir)

    assert "dim3 topk_block(32, 8)" in source
    assert "void* topk_scratch = nullptr;" not in source
    assert "session->topk_scratch" not in source


def test_rocm_layer_norm_benchmark_shape_keeps_two_warp_rows():
    spec = dml.trace(
        _LayerNormModule(),
        inputs={
            "x": dml.TensorSpec([16, 128, 768], "float32"),
            "weight": dml.TensorSpec([768], "float32"),
            "bias": dml.TensorSpec([768], "float32"),
        },
        name="rocm_layer_norm_tuning",
    )
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    source = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]

    assert "dim3 block(32, 2)" in source


def test_rocm_t5_layer_norm_benchmark_shape_keeps_two_warp_rows():
    spec = dml.trace(
        _T5LayerNormModule(),
        inputs={
            "x": dml.TensorSpec([16, 128, 768], "float32"),
            "weight": dml.TensorSpec([768], "float32"),
        },
        name="rocm_t5_layer_norm_tuning",
    )
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    source = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]

    assert "dim3 block(32, 2)" in source
