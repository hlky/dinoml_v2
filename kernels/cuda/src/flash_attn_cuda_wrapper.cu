#include <cuda_runtime.h>

#include <cstdint>

#include "flash_attn_dinoml.h"

namespace {

constexpr int64_t kFloat16Bytes = 2;

const void* byte_offset(const void* base, int64_t element_offset, int64_t element_size) {
  return static_cast<const void*>(static_cast<const char*>(base) + element_offset * element_size);
}

int launch_flash_attention_float16(
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
    cudaStream_t stream) {
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
      nullptr,
      flash::DataType::kFloat16,
      -1,
      -1,
      1,
      nullptr,
      nullptr,
      stream);
  return static_cast<int>(cudaGetLastError());
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
  return launch_flash_attention_float16(
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
      stream);
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
  const int64_t qkv_batch_stride = seqlen * 3 * num_heads * head_dim;
  const int64_t qkv_row_stride = 3 * num_heads * head_dim;
  const int64_t packed_axis_stride = num_heads * head_dim;
  const void* q = qkv;
  const void* k = byte_offset(qkv, packed_axis_stride, kFloat16Bytes);
  const void* v = byte_offset(qkv, 2 * packed_axis_stride, kFloat16Bytes);
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
      nullptr,
      flash::DataType::kFloat16,
      -1,
      -1,
      1,
      nullptr,
      nullptr,
      stream);
  return static_cast<int>(cudaGetLastError());
}
