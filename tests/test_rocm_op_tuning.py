from __future__ import annotations

from pathlib import Path

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


class _EmbeddingModule(dml.Module):
    def forward(self, table, indices):
        return dml.ops.output(dml.ops.embedding(table, indices), "output")


class _BatchGatherModule(dml.Module):
    def forward(self, x, indices):
        return dml.ops.output(dml.ops.batch_gather(x, indices), "output")


class _AvgPool1dModule(dml.Module):
    def forward(self, x):
        return dml.ops.output(dml.ops.avg_pool1d(x, kernel_size=3, stride=2, padding=1), "output")


def test_gpu_warp_templates_use_explicit_shuffle_width_for_rocm():
    template_dir = Path("src/dinoml/lowering/ops/templates")
    logical_warp_templates = [
        "argmax_gpu.j2",
        "layer_norm_gpu.j2",
        "reduction_gpu.j2",
        "softmax_gpu.j2",
        "t5_layer_norm_gpu.j2",
        "topk_gpu.j2",
    ]
    shuffle_lines = [
        (template_name, line.strip())
        for template_name in logical_warp_templates
        for line in (template_dir / template_name).read_text(encoding="utf-8").splitlines()
        if "__shfl" in line
    ]

    assert shuffle_lines
    assert [
        (template_name, line)
        for template_name, line in shuffle_lines
        if ", 32)" not in line and ", group_lanes)" not in line
    ] == []


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


def test_rocm_embedding_float32_benchmark_shape_uses_float4_copy():
    spec = dml.trace(
        _EmbeddingModule(),
        inputs={
            "table": dml.TensorSpec([32768, 256], "float32"),
            "indices": dml.TensorSpec([32, 128], "int64"),
        },
        name="rocm_embedding_tuning",
    )
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    source = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]

    assert "_float4_kernel" in source
    assert "reinterpret_cast<const float4*>(table)" in source
    assert "runtime_numel_vec_out = runtime_numel_out / 4" in source


def test_rocm_embedding_bfloat16_int32_benchmark_shape_uses_uint4_copy():
    case = next(case for case in benchmark_cases() if case.name == "embedding_bfloat16_int32")
    source = render_gpu_module("rocm", case.build_spec().ir)

    assert "_uint4_kernel" in source
    assert "reinterpret_cast<const uint4*>(table)" in source
    assert "constexpr int64_t hidden_vectors = 32;" in source
    assert "runtime_numel_vec_out = runtime_numel_out / 8" in source


def test_rocm_gather_benchmark_shape_keeps_per_element_scalar_indices():
    case = next(case for case in benchmark_cases() if case.name == "gather")
    source = render_gpu_module("rocm", case.build_spec().ir)

    assert "const int block = 256;" in source
    assert "const int64_t selected_index = static_cast<int64_t>(index[idx]);" in source
    assert "float4" not in source


def test_rocm_batch_gather_float32_benchmark_shape_uses_float4_slice_copy():
    spec = dml.trace(
        _BatchGatherModule(),
        inputs={
            "x": dml.TensorSpec([32, 256, 768], "float32"),
            "indices": dml.TensorSpec([32, 128], "int64"),
        },
        name="rocm_batch_gather_tuning",
    )
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    source = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]

    assert "_float4_kernel" in source
    assert "reinterpret_cast<const float4*>(x)" in source
    assert "constexpr int64_t slice_vectors = 192;" in source
    assert "runtime_numel_vec = runtime_numel / 4" in source


def test_rocm_batch_gather_bfloat16_int32_benchmark_shape_uses_uint4_slice_copy():
    case = next(case for case in benchmark_cases() if case.name == "batch_gather_bfloat16_int32")
    source = render_gpu_module("rocm", case.build_spec().ir)

    assert "_uint4_kernel" in source
    assert "reinterpret_cast<const uint4*>(x)" in source
    assert "constexpr int64_t slice_vectors = 96;" in source
    assert "runtime_numel_vec = runtime_numel / 8" in source


def test_rocm_avg_pool1d_benchmark_shape_uses_smaller_block_than_cuda():
    spec = dml.trace(
        _AvgPool1dModule(),
        inputs={"x": dml.TensorSpec([16, 64, 1024], "float32")},
        name="rocm_avg_pool1d_tuning",
    )
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    rocm_source = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]
    cuda_source = render_generated_kernels("cuda", spec.ir["nodes"], tensor_map)[0]

    assert "const int block = 128;" in rocm_source
    assert "const int block = 256;" in cuda_source
