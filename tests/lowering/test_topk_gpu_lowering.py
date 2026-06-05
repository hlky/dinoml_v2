from __future__ import annotations

import dinoml as dml
from dinoml.benchmarks.ops import benchmark_cases
from dinoml.lowering.gpu import render_gpu_module
from dinoml.lowering.ops import render_generated_kernels, render_launch


class _TopKModule(dml.Module):
    def __init__(self, k: int):
        self.k = k

    def forward(self, x):
        values, indices = dml.ops.topk(x, self.k, dim=-1)
        return {
            "values": dml.ops.output(values, "values"),
            "indices": dml.ops.output(indices, "indices"),
        }


def test_topk_gpu_pair_uses_single_fused_launch():
    case = next(case for case in benchmark_cases() if case.name == "topk")
    ir = case.build_spec().ir
    tensor_map = {tensor["name"]: tensor for tensor in ir["tensors"]}

    sources = render_generated_kernels("rocm", ir["nodes"], tensor_map)
    launches = [render_launch("rocm", node, tensor_map) for node in ir["nodes"]]

    assert len(sources) == 1
    assert "ptr_t0, ptr_t1" in launches[0]
    assert "paired topk_indices is produced by the topk_values launch" in launches[1]
    assert "dim3 topk_block(32, 8)" in sources[0]


def test_topk_gpu_path_selection_covers_k_bands():
    assert "y_values[row] = x[base + best_index]" in _topk_source([8, 17], 1)

    k64_source = _topk_source([8, 1024], 64)
    assert "__shared__ float sort_values[sort_width]" in k64_source
    k64_wide_source = _topk_source([4, 4096], 64)
    assert "__shared__ float sort_values[sort_width]" in k64_wide_source
    k64_many_rows_source = _topk_source([128, 4096], 64)
    assert "constexpr int candidate_width = 64" in k64_many_rows_source
    assert "const int block = 512;" in k64_many_rows_source
    assert "dim3 topk_block" not in k64_many_rows_source

    k1_wide_source = _topk_source([4, 32768], 1)
    assert "__shared__ float shared_values[512]" in k1_wide_source

    k4_wide_source = _topk_source([4, 32768], 4)
    assert "__shared__ float merge_values[64 * k]" in k4_wide_source

    k8_wide_source = _topk_source([4, 32768], 8)
    assert "__shared__ float merge_values[128 * k]" in k8_wide_source

    k2_router_source = _topk_source([4096, 64], 2)
    assert "constexpr int group_lanes = 16" in k2_router_source
    assert "constexpr int rows_per_warp = 32 / group_lanes" in k2_router_source
    assert "dim3 topk_block(32, 8)" in k2_router_source

    k4_tiny_source = _topk_source([4096, 8], 4)
    assert "constexpr int group_lanes = 8" in k4_tiny_source
    assert "dim3 topk_block(32, 4)" in k4_tiny_source

    k8_router_source = _topk_source([4096, 256], 8)
    assert "constexpr int group_lanes = 16" in k8_router_source
    assert "dim3 topk_block(32, 4)" in k8_router_source

    k16_wide_source = _topk_source([4, 4096], 16)
    assert "dim3 topk_block(32, 16)" in k16_wide_source
    k16_32768_few_rows_source = _topk_source([64, 32768], 16)
    assert "constexpr int candidate_width = 16" in k16_32768_few_rows_source
    assert "const int block = 1024;" in k16_32768_few_rows_source
    assert "value_threshold_key" in k16_32768_few_rows_source
    assert "dim3 topk_block" not in k16_32768_few_rows_source
    k16_32768_mid_rows_source = _topk_source([128, 32768], 16)
    assert "const int block = 512;" in k16_32768_mid_rows_source
    k16_32768_many_rows_source = _topk_source([512, 32768], 16)
    assert "const int block = 256;" in k16_32768_many_rows_source
    k16_128256_source = _topk_source([8, 128256], 16)
    assert "_tile_keys_fast_vec4" in k16_128256_source
    assert "constexpr int tile_cols = 8192;" in k16_128256_source
    assert "topk scratch is too small" in k16_128256_source

    k32_shared_sort_source = _topk_source([4, 4096], 32)
    assert "__shared__ float sort_values[sort_width]" in k32_shared_sort_source
    k32_32768_source = _topk_source([32, 32768], 32)
    assert "constexpr int candidate_width = 32" in k32_32768_source
    assert "const int block = 1024;" in k32_32768_source
    assert "value_threshold_key" in k32_32768_source
    assert "dim3 topk_block" not in k32_32768_source
    k32_32768_mid_rows_source = _topk_source([128, 32768], 32)
    assert "const int block = 512;" in k32_32768_mid_rows_source
    k32_32768_many_rows_source = _topk_source([256, 32768], 32)
    assert "const int block = 256;" in k32_32768_many_rows_source
    k32_128256_small_rows_source = _topk_source([32, 128256], 32)
    assert "_radix_prefilter_count" in k32_128256_small_rows_source
    assert "constexpr int tile_cols = 16384;" in k32_128256_small_rows_source
    assert "_tile_keys_pref_vec4" in k32_128256_small_rows_source
    k32_128256_many_rows_source = _topk_source([64, 128256], 32)
    assert "_radix_prefilter_count" not in k32_128256_many_rows_source
    assert "value_threshold_key" in k32_128256_many_rows_source

    k128_source = _topk_source([4, 2048], 128)
    assert "__shared__ float sort_values[sort_width]" in k128_source
    k128_radix_source = _topk_source([4, 4096], 128)
    assert "constexpr int candidate_width = 128" in k128_radix_source
    assert "threshold_key = prefix" in k128_radix_source
    assert "const int block = 512;" in k128_radix_source
    assert "value_threshold_key" not in k128_radix_source

    k64_65536_source = _topk_source([256, 65536], 64)
    assert "constexpr int candidate_width = 64" in k64_65536_source
    assert "value_threshold_key" in k64_65536_source
    assert "dim3 topk_block" not in k64_65536_source

    k256_radix_source = _topk_source([2, 4096], 256)
    assert "constexpr int candidate_width = 256" in k256_radix_source
    assert "const int block = 1024;" in k256_radix_source
    k256_131072_source = _topk_source([64, 131072], 256)
    assert "constexpr int candidate_width = 256" in k256_131072_source
    assert "const int block = 1024;" in k256_131072_source
    assert "value_threshold_key" in k256_131072_source

    k300_detector_source = _topk_source([4, 8400], 300)
    assert "constexpr int candidate_width = 512" in k300_detector_source
    assert "static __device__ uint64_t" in k300_detector_source
    assert "const int block = 1024;" in k300_detector_source

    k512_source = _topk_source([2, 512], 512)
    assert "__shared__ float sort_values[sort_width]" in k512_source

    k4097_source = _topk_source([1, 5000], 4097)
    assert "const int64_t stride =" in k4097_source
    assert "__shared__ float sort_values[sort_width]" not in k4097_source


def test_topk_gpu_two_pass_scratch_is_allocated_only_when_selected():
    two_pass_spec = dml.trace(
        _TopKModule(8),
        inputs={"x": dml.TensorSpec([256, 128256], "float32")},
        name="topk_two_pass",
    )
    two_pass_module = render_gpu_module("rocm", two_pass_spec.ir)
    assert "void* topk_scratch = nullptr;" in two_pass_module
    assert "session->topk_scratch" in two_pass_module

    prefilter_spec = dml.trace(
        _TopKModule(32),
        inputs={"x": dml.TensorSpec([32, 128256], "float32")},
        name="topk_prefilter",
    )
    prefilter_module = render_gpu_module("rocm", prefilter_spec.ir)
    assert "void* topk_scratch = nullptr;" in prefilter_module
    assert "_radix_prefilter_count" in prefilter_module

    block_merge_spec = dml.trace(
        _TopKModule(8),
        inputs={"x": dml.TensorSpec([4, 32768], "float32")},
        name="topk_block_merge",
    )
    block_merge_module = render_gpu_module("rocm", block_merge_spec.ir)
    assert "void* topk_scratch = nullptr;" not in block_merge_module


def _topk_source(shape: list[int], k: int, dtype: str = "float32") -> str:
    spec = dml.trace(_TopKModule(k), inputs={"x": dml.TensorSpec(shape, dtype)}, name=f"topk_k{k}")
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    return render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)[0]
