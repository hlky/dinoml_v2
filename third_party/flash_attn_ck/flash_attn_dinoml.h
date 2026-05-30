#pragma once
#include <cstdint>
#include <hip/hip_runtime.h>

enum class DataType { kFloat16 = 0, kBFloat16 };

enum class MaskType { kNone, kCausalFromTopLeft, kCausalFromBottomRight };

float FlashAttentionLauncher(
    void* output,
    int64_t output_batch_stride,
    int64_t output_row_stride,
    int64_t output_head_stride,
    void* q,
    int64_t q_batch_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    void* k,
    int64_t k_batch_stride,
    int64_t k_row_stride,
    int64_t k_head_stride,
    void* v,
    int64_t v_batch_stride,
    int64_t v_row_stride,
    int64_t v_head_stride,
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t seqlen_k,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    MaskType mask_type,
    DataType dtype,
    int window_size_left,
    int window_size_right,
    hipStream_t stream);
