#include <cuda_runtime.h>

#include <cstdint>

#include "flash_attn_dinoml.h"

namespace {

constexpr int64_t kElementBytes = 2;

const void* byte_offset(const void* base, int64_t element_offset, int64_t element_size) {
  return static_cast<const void*>(static_cast<const char*>(base) + element_offset * element_size);
}

int launch_flash_attention(
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
    bool causal,
    flash::DataType dtype,
    cudaStream_t stream) {
  void* softmax_lse = nullptr;
  cudaError_t alloc_status = cudaMallocAsync(
      &softmax_lse,
      static_cast<size_t>(batch_size * num_heads_q * seqlen_q) * sizeof(float),
      stream);
  if (alloc_status != cudaSuccess) {
    return static_cast<int>(alloc_status);
  }
  const int64_t output_batch_stride = seqlen_q * num_heads_q * head_dim;
  const int64_t output_row_stride = num_heads_q * head_dim;
  const int64_t output_head_stride = head_dim;
  const int64_t q_batch_stride = seqlen_q * num_heads_q * head_dim;
  const int64_t q_row_stride = num_heads_q * head_dim;
  const int64_t q_head_stride = head_dim;
  const int64_t k_batch_stride = seqlen_k * num_heads_k * head_dim;
  const int64_t k_row_stride = num_heads_k * head_dim;
  const int64_t k_head_stride = head_dim;
  const flash::MaskType mask_type =
      causal ? flash::MaskType::kCausalFromTopLeft : flash::MaskType::kNone;
  flash::FlashAttentionLauncher(
      output,
      output_batch_stride,
      output_row_stride,
      output_head_stride,
      const_cast<void*>(q),
      q_batch_stride,
      q_row_stride,
      q_head_stride,
      const_cast<void*>(k),
      k_batch_stride,
      k_row_stride,
      k_head_stride,
      const_cast<void*>(v),
      k_batch_stride,
      k_row_stride,
      k_head_stride,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      mask_type,
      softmax_lse,
      dtype,
      -1,
      -1,
      1,
      nullptr,
      nullptr,
      stream);
  cudaError_t launch_status = cudaGetLastError();
  cudaError_t free_status = cudaFreeAsync(softmax_lse, stream);
  return static_cast<int>(launch_status != cudaSuccess ? launch_status : free_status);
}

}  // namespace

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
    cudaStream_t stream) {
  return launch_flash_attention(
      q,
      k,
      v,
      output,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      causal != 0,
      flash::DataType::kFloat16,
      stream);
}

extern "C" int dinoml_flash_attn_cuda_fwd_bfloat16_v1(
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
    cudaStream_t stream) {
  return launch_flash_attention(
      q,
      k,
      v,
      output,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      causal != 0,
      flash::DataType::kBFloat16,
      stream);
}

int launch_flash_attention_qkv(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    bool causal,
    flash::DataType dtype,
    cudaStream_t stream) {
  void* softmax_lse = nullptr;
  cudaError_t alloc_status = cudaMallocAsync(
      &softmax_lse,
      static_cast<size_t>(batch_size * num_heads * seqlen) * sizeof(float),
      stream);
  if (alloc_status != cudaSuccess) {
    return static_cast<int>(alloc_status);
  }
  const int64_t qkv_batch_stride = seqlen * 3 * num_heads * head_dim;
  const int64_t qkv_row_stride = 3 * num_heads * head_dim;
  const int64_t packed_axis_stride = num_heads * head_dim;
  const void* q = qkv;
  const void* k = byte_offset(qkv, packed_axis_stride, kElementBytes);
  const void* v = byte_offset(qkv, 2 * packed_axis_stride, kElementBytes);
  const int64_t output_batch_stride = seqlen * num_heads * head_dim;
  const int64_t output_row_stride = num_heads * head_dim;
  const int64_t output_head_stride = head_dim;
  const flash::MaskType mask_type =
      causal ? flash::MaskType::kCausalFromTopLeft : flash::MaskType::kNone;
  flash::FlashAttentionLauncher(
      output,
      output_batch_stride,
      output_row_stride,
      output_head_stride,
      const_cast<void*>(q),
      qkv_batch_stride,
      qkv_row_stride,
      head_dim,
      const_cast<void*>(k),
      qkv_batch_stride,
      qkv_row_stride,
      head_dim,
      const_cast<void*>(v),
      qkv_batch_stride,
      qkv_row_stride,
      head_dim,
      batch_size,
      seqlen,
      seqlen,
      num_heads,
      num_heads,
      head_dim,
      mask_type,
      softmax_lse,
      dtype,
      -1,
      -1,
      1,
      nullptr,
      nullptr,
      stream);
  cudaError_t launch_status = cudaGetLastError();
  cudaError_t free_status = cudaFreeAsync(softmax_lse, stream);
  return static_cast<int>(launch_status != cudaSuccess ? launch_status : free_status);
}

extern "C" int dinoml_flash_attn_cuda_qkv_fwd_float16_v1(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    cudaStream_t stream) {
  return launch_flash_attention_qkv(
      qkv,
      output,
      batch_size,
      seqlen,
      num_heads,
      head_dim,
      causal != 0,
      flash::DataType::kFloat16,
      stream);
}

extern "C" int dinoml_flash_attn_cuda_qkv_fwd_bfloat16_v1(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    cudaStream_t stream) {
  return launch_flash_attention_qkv(
      qkv,
      output,
      batch_size,
      seqlen,
      num_heads,
      head_dim,
      causal != 0,
      flash::DataType::kBFloat16,
      stream);
}

int launch_flash_attention_static_kv_cache(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const void* knew,
    const void* vnew,
    const int32_t* cache_seqlens,
    void* output,
    int64_t batch_size,
    int64_t max_cache_len,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    flash::DataType dtype,
    cudaStream_t stream) {
  void* softmax_lse = nullptr;
  cudaError_t alloc_status = cudaMallocAsync(
      &softmax_lse,
      static_cast<size_t>(batch_size * num_heads_q) * sizeof(float),
      stream);
  if (alloc_status != cudaSuccess) {
    return static_cast<int>(alloc_status);
  }
  const int64_t output_batch_stride = num_heads_q * head_dim;
  const int64_t output_row_stride = num_heads_q * head_dim;
  const int64_t output_head_stride = head_dim;
  const int64_t q_batch_stride = num_heads_q * head_dim;
  const int64_t q_row_stride = num_heads_q * head_dim;
  const int64_t q_head_stride = head_dim;
  const int64_t cache_batch_stride = num_heads_k * max_cache_len * head_dim;
  const int64_t cache_row_stride = head_dim;
  const int64_t cache_head_stride = max_cache_len * head_dim;
  const int64_t new_batch_stride = num_heads_k * head_dim;
  const int64_t new_row_stride = head_dim;
  const int64_t new_head_stride = head_dim;
  flash::FlashAttentionStaticKvCacheLauncher(
      output,
      output_batch_stride,
      output_row_stride,
      output_head_stride,
      const_cast<void*>(q),
      q_batch_stride,
      q_row_stride,
      q_head_stride,
      const_cast<void*>(k_cache),
      cache_batch_stride,
      cache_row_stride,
      cache_head_stride,
      const_cast<void*>(v_cache),
      cache_batch_stride,
      cache_row_stride,
      cache_head_stride,
      const_cast<void*>(knew),
      new_batch_stride,
      new_row_stride,
      new_head_stride,
      const_cast<void*>(vnew),
      new_batch_stride,
      new_row_stride,
      new_head_stride,
      batch_size,
      max_cache_len,
      num_heads_q,
      num_heads_k,
      head_dim,
      const_cast<int32_t*>(cache_seqlens),
      softmax_lse,
      dtype,
      stream);
  cudaError_t launch_status = cudaGetLastError();
  cudaError_t free_status = cudaFreeAsync(softmax_lse, stream);
  return static_cast<int>(launch_status != cudaSuccess ? launch_status : free_status);
}

extern "C" int dinoml_flash_attn_cuda_static_kv_cache_fwd_float16_v1(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const void* knew,
    const void* vnew,
    const int32_t* cache_seqlens,
    void* output,
    int64_t batch_size,
    int64_t max_cache_len,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    cudaStream_t stream) {
  return launch_flash_attention_static_kv_cache(
      q,
      k_cache,
      v_cache,
      knew,
      vnew,
      cache_seqlens,
      output,
      batch_size,
      max_cache_len,
      num_heads_q,
      num_heads_k,
      head_dim,
      flash::DataType::kFloat16,
      stream);
}

extern "C" int dinoml_flash_attn_cuda_static_kv_cache_fwd_bfloat16_v1(
    const void* q,
    const void* k_cache,
    const void* v_cache,
    const void* knew,
    const void* vnew,
    const int32_t* cache_seqlens,
    void* output,
    int64_t batch_size,
    int64_t max_cache_len,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    cudaStream_t stream) {
  return launch_flash_attention_static_kv_cache(
      q,
      k_cache,
      v_cache,
      knew,
      vnew,
      cache_seqlens,
      output,
      batch_size,
      max_cache_len,
      num_heads_q,
      num_heads_k,
      head_dim,
      flash::DataType::kBFloat16,
      stream);
}
