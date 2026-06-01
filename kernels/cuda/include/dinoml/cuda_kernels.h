#pragma once

#include <cuda_runtime_api.h>

#include <dinoml/device.h>
#include <dinoml/runtime.h>

#include <cstdint>

extern "C" int dinoml_flash_attn_cuda_fwd_float16_v1(
    const void* q,
    const void* k,
    const void* v,
    void* output,
    int64_t batch_size,
    int64_t seqlen_q,
    int64_t seqlen_k,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    int causal,
    cudaStream_t stream);

extern "C" int dinoml_flash_attn_cuda_qkv_fwd_float16_v1(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    cudaStream_t stream);
