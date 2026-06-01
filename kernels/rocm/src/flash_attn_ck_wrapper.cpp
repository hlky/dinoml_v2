#include <dinoml/rocm_kernels.h>

#include "flash_attn_dinoml.h"

#include <cmath>
#include <cstdint>

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
    int causal,
    DataType dtype,
    hipStream_t stream) {
  if (q == nullptr || k == nullptr || v == nullptr || output == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (batch_size <= 0 || seqlen_q <= 0 || seqlen_k <= 0 || num_heads_q <= 0 ||
      num_heads_k <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const int64_t q_batch_stride = seqlen_q * num_heads_q * head_dim;
  const int64_t k_batch_stride = seqlen_k * num_heads_k * head_dim;
  const int64_t v_batch_stride = seqlen_k * num_heads_k * head_dim;
  const int64_t output_batch_stride = seqlen_q * num_heads_q * head_dim;
  const int64_t q_row_stride = num_heads_q * head_dim;
  const int64_t k_row_stride = num_heads_k * head_dim;
  const int64_t v_row_stride = num_heads_k * head_dim;
  const int64_t output_row_stride = num_heads_q * head_dim;
  const int64_t q_head_stride = head_dim;
  const int64_t k_head_stride = head_dim;
  const int64_t v_head_stride = head_dim;
  const int64_t output_head_stride = head_dim;
  const MaskType mask_type = causal != 0 ? MaskType::kCausalFromTopLeft : MaskType::kNone;

  const float elapsed_ms = FlashAttentionLauncher(
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
      v_batch_stride,
      v_row_stride,
      v_head_stride,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      mask_type,
      dtype,
      -1,
      -1,
      stream);
  if (elapsed_ms < 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return 0;
}

extern "C" int dinoml_flash_attn_ck_fwd_float16_v1(
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
    hipStream_t stream) {
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
      causal,
      DataType::kFloat16,
      stream);
}

extern "C" int dinoml_flash_attn_ck_fwd_bfloat16_v1(
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
    hipStream_t stream) {
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
      causal,
      DataType::kBFloat16,
      stream);
}

int launch_flash_attention_qkv(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    DataType dtype,
    hipStream_t stream) {
  if (qkv == nullptr || output == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (batch_size <= 0 || seqlen <= 0 || num_heads <= 0 || head_dim <= 0) {
    return static_cast<int>(hipErrorInvalidValue);
  }

  const int64_t packed_row_stride = 3 * num_heads * head_dim;
  const int64_t qkv_batch_stride = seqlen * packed_row_stride;
  const int64_t output_batch_stride = seqlen * num_heads * head_dim;
  const int64_t output_row_stride = num_heads * head_dim;
  const int64_t output_head_stride = head_dim;
  const int64_t q_row_stride = packed_row_stride;
  const int64_t k_row_stride = packed_row_stride;
  const int64_t v_row_stride = packed_row_stride;
  const int64_t q_head_stride = head_dim;
  const int64_t k_head_stride = head_dim;
  const int64_t v_head_stride = head_dim;
  const int64_t head_block = num_heads * head_dim;
  auto* q_ptr = const_cast<void*>(qkv);
  auto* k_ptr = static_cast<void*>(static_cast<char*>(const_cast<void*>(qkv)) + head_block * sizeof(uint16_t));
  auto* v_ptr = static_cast<void*>(static_cast<char*>(const_cast<void*>(qkv)) + 2 * head_block * sizeof(uint16_t));
  const MaskType mask_type = causal != 0 ? MaskType::kCausalFromTopLeft : MaskType::kNone;

  const float elapsed_ms = FlashAttentionLauncher(
      output,
      output_batch_stride,
      output_row_stride,
      output_head_stride,
      q_ptr,
      qkv_batch_stride,
      q_row_stride,
      q_head_stride,
      k_ptr,
      qkv_batch_stride,
      k_row_stride,
      k_head_stride,
      v_ptr,
      qkv_batch_stride,
      v_row_stride,
      v_head_stride,
      batch_size,
      seqlen,
      seqlen,
      num_heads,
      num_heads,
      head_dim,
      mask_type,
      dtype,
      -1,
      -1,
      stream);
  if (elapsed_ms < 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return 0;
}

extern "C" int dinoml_flash_attn_ck_qkv_fwd_float16_v1(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    hipStream_t stream) {
  return launch_flash_attention_qkv(
      qkv,
      output,
      batch_size,
      seqlen,
      num_heads,
      head_dim,
      causal,
      DataType::kFloat16,
      stream);
}

extern "C" int dinoml_flash_attn_ck_qkv_fwd_bfloat16_v1(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    hipStream_t stream) {
  return launch_flash_attention_qkv(
      qkv,
      output,
      batch_size,
      seqlen,
      num_heads,
      head_dim,
      causal,
      DataType::kBFloat16,
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
    DataType dtype,
    void* scratch,
    size_t scratch_nbytes,
    hipStream_t stream) {
  if (q == nullptr || k_cache == nullptr || v_cache == nullptr || knew == nullptr ||
      vnew == nullptr || cache_seqlens == nullptr || output == nullptr ||
      scratch == nullptr) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  if (batch_size <= 0 || max_cache_len <= 0 || num_heads_q <= 0 || num_heads_k <= 0 ||
      head_dim <= 0 || (num_heads_q % num_heads_k) != 0) {
    return static_cast<int>(hipErrorInvalidValue);
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

  const float elapsed_ms = FlashAttentionStaticKvCacheLauncher(
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
      cache_seqlens,
      dtype,
      scratch,
      scratch_nbytes,
      stream);
  if (elapsed_ms < 0.0f) {
    return static_cast<int>(hipErrorInvalidValue);
  }
  return 0;
}

extern "C" int dinoml_flash_attn_ck_static_kv_cache_fwd_float16_v1(
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
    void* scratch,
    size_t scratch_nbytes,
    hipStream_t stream) {
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
      DataType::kFloat16,
      scratch,
      scratch_nbytes,
      stream);
}

extern "C" int dinoml_flash_attn_ck_static_kv_cache_fwd_bfloat16_v1(
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
    void* scratch,
    size_t scratch_nbytes,
    hipStream_t stream) {
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
      DataType::kBFloat16,
      scratch,
      scratch_nbytes,
      stream);
}
