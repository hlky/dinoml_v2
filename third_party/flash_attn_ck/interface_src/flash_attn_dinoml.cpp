#include "../flash_attn_dinoml.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

#include <cstdint>

#include <cassert>
#include <cmath>
#include <type_traits>

#include "fmha_fwd.hpp"

namespace {

constexpr size_t kStaticKvScratchAlignment = 16;

size_t align_nbytes(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

void* next_scratch_slice(char* base, size_t& offset, size_t nbytes) {
  offset = align_nbytes(offset, kStaticKvScratchAlignment);
  void* result = base + offset;
  offset += nbytes;
  return result;
}

std::string dtype_string(DataType dtype) {
  if (dtype == DataType::kFloat16) {
    return "fp16";
  }
  if (dtype == DataType::kBFloat16) {
    return "bf16";
  }
  return "";
}

__global__ void build_total_seqlens_kernel(
    const int32_t* cache_seqlens,
    int32_t* total_seqlens,
    int32_t batch_size,
    int32_t max_cache_len) {
  const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= batch_size) {
    return;
  }
  int32_t total = cache_seqlens[index] + 1;
  if (total < 0) {
    total = 0;
  }
  if (total > max_cache_len) {
    total = max_cache_len;
  }
  total_seqlens[index] = total;
}

__global__ void advance_cache_seqlens_kernel(
    int32_t* cache_seqlens,
    int32_t batch_size,
    int32_t max_cache_len) {
  const int32_t index = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
  if (index >= batch_size) {
    return;
  }
  int32_t next = cache_seqlens[index] + 1;
  if (next < 0) {
    next = 0;
  }
  if (next > max_cache_len) {
    next = max_cache_len;
  }
  cache_seqlens[index] = next;
}

float build_total_seqlens(
    const int32_t* cache_seqlens,
    int32_t* total_seqlens,
    int64_t batch_size,
    int64_t max_cache_len,
    hipStream_t stream) {
  if (batch_size > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
      max_cache_len > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return -1.0f;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
  hipLaunchKernelGGL(
      build_total_seqlens_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      cache_seqlens,
      total_seqlens,
      static_cast<int32_t>(batch_size),
      static_cast<int32_t>(max_cache_len));
  return hipGetLastError() == hipSuccess ? 0.0f : -1.0f;
}

float advance_cache_seqlens(
    int32_t* cache_seqlens,
    int64_t batch_size,
    int64_t max_cache_len,
    hipStream_t stream) {
  if (batch_size > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
      max_cache_len > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
    return -1.0f;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((batch_size + threads - 1) / threads);
  hipLaunchKernelGGL(
      advance_cache_seqlens_kernel,
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      cache_seqlens,
      static_cast<int32_t>(batch_size),
      static_cast<int32_t>(max_cache_len));
  return hipGetLastError() == hipSuccess ? 0.0f : -1.0f;
}

} // namespace

fmha_fwd_traits get_ck_fmha_fwd_traits(
    const mask_info& mask,
    const bias_info& bias,
    std::string dtype,
    int head_size,
    bool has_dropout,
    bool has_lse) {
  return fmha_fwd_traits{
      head_size,
      head_size,
      dtype,
      false, // is_group_mode
      true, // is_v_rowmajor
      false, // has_logits_soft_cap
      mask.type,
      bias.type,
      has_lse,
      has_dropout,
      false, // do_fp8_static_quant
      false}; // skip_min_seqlen_q
}

fmha_fwd_args get_ck_fmha_fwd_args(
    const mask_info& mask,
    // sizes
    const int b,
    const int seqlen_q,
    const int seqlen_k,
    const int h,
    const int h_k,
    const int d,
    // strides
    const int stride_q,
    const int stride_k,
    const int stride_v,
    const int stride_o,
    const int nhead_stride_q,
    const int nhead_stride_k,
    const int nhead_stride_v,
    const int nhead_stride_o,
    const int batch_stride_q,
    const int batch_stride_k,
    const int batch_stride_v,
    const int batch_stride_o,
    const int bias_row_stride,
    const int bias_head_stride,
    const int bias_batch_stride,
    // device pointers
    const void* q,
    const void* k,
    const void* v,
    const void* bias,
    void* out,
    float softmax_scale) {
  ck_tile::index_t stride_randval = 0;

  ck_tile::index_t nhead_stride_lse = 0;
  ck_tile::index_t nhead_stride_randval = 0;


  ck_tile::index_t batch_stride_lse = 0;
  ck_tile::index_t batch_stride_randval = 0;

  fmha_fwd_args args{};
  args.q_ptr = q;
  args.k_ptr = k;
  args.v_ptr = v;
  args.bias_ptr = bias;
  args.rand_val_ptr = nullptr;
  args.lse_ptr = nullptr;
  args.o_ptr = out;
  args.seqstart_q_ptr = nullptr;
  args.seqstart_k_ptr = nullptr;
  args.seqlen_q_ptr = nullptr;
  args.seqlen_k_ptr = nullptr;
  args.cu_seqlen_q_ptr = nullptr;
  args.cu_seqlen_k_ptr = nullptr;
  args.seqlen_q = seqlen_q;
  args.seqlen_k = seqlen_k;
  args.batch = b;
  args.max_seqlen_q = seqlen_q;
  args.hdim_q = d;
  args.hdim_v = d;
  args.nhead_q = h;
  args.nhead_k = h_k;
  args.scale_s = softmax_scale;
  args.scale_p = 1.0f;
  args.scale_o = 1.0f;
  args.logits_soft_cap = 0.0f;
  args.stride_q = stride_q;
  args.stride_k = stride_k;
  args.stride_v = stride_v;
  args.stride_bias = bias_row_stride;
  args.stride_randval = stride_randval;
  args.stride_o = stride_o;
  args.nhead_stride_q = nhead_stride_q;
  args.nhead_stride_k = nhead_stride_k;
  args.nhead_stride_v = nhead_stride_v;
  args.nhead_stride_bias = bias_head_stride;
  args.nhead_stride_randval = nhead_stride_randval;
  args.nhead_stride_lse = nhead_stride_lse;
  args.nhead_stride_o = nhead_stride_o;
  args.batch_stride_q = batch_stride_q;
  args.batch_stride_k = batch_stride_k;
  args.batch_stride_v = batch_stride_v;
  args.batch_stride_bias = bias_batch_stride;
  args.batch_stride_randval = batch_stride_randval;
  args.batch_stride_lse = batch_stride_lse;
  args.batch_stride_o = batch_stride_o;
  args.window_size_left = mask.left;
  args.window_size_right = mask.right;
  args.mask_type = static_cast<ck_tile::index_t>(mask.type);
  args.min_seqlen_q = 0;
  args.p_drop = 0.0f;
  args.s_randval = false;
  args.drop_seed_offset = std::pair<uint64_t, uint64_t>(0, 0);
  return args;
}

float run_mha_fwd(
    fmha_fwd_traits traits,
    fmha_fwd_args args,
    const ck_tile::stream_config& config) {
  return fmha_fwd(traits, args, config);
}

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
    hipStream_t stream) {
  ck_tile::stream_config stream_config{stream};

  std::string mask_type_str;
  if (mask_type == MaskType::kNone) {
    mask_type_str = "0";
  } else if (mask_type == MaskType::kCausalFromTopLeft) {
    mask_type_str = "1";
  } else if (mask_type == MaskType::kCausalFromBottomRight) {
    mask_type_str = "2";
  }
  std::string dtype_str;
  if (dtype == DataType::kFloat16) {
    dtype_str = "fp16";
  } else if (dtype == DataType::kBFloat16) {
    dtype_str = "bf16";
  }
  auto mask = mask_info::decode(mask_type_str, seqlen_q, seqlen_k);
  auto bias = bias_info::decode("0");
  auto traits =
      get_ck_fmha_fwd_traits(mask, bias, dtype_str, head_dim, false, false);

  auto args = get_ck_fmha_fwd_args(
      mask,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      q_row_stride,
      k_row_stride,
      v_row_stride,
      output_row_stride,
      q_head_stride,
      k_head_stride,
      v_head_stride,
      output_head_stride,
      q_batch_stride,
      k_batch_stride,
      v_batch_stride,
      output_batch_stride,
      0,
      0,
      0,
      q,
      k,
      v,
      nullptr,
      output,
      1.0f / std::sqrt(static_cast<float>(head_dim)));

  return run_mha_fwd(traits, args, stream_config);
}

float FlashAttentionBiasLauncher(
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
    void* bias_ptr,
    int64_t bias_batch_stride,
    int64_t bias_row_stride,
    int64_t bias_head_stride,
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
    hipStream_t stream) {
  if (bias_ptr == nullptr) {
    return -1.0f;
  }
  ck_tile::stream_config stream_config{stream};

  std::string mask_type_str;
  if (mask_type == MaskType::kNone) {
    mask_type_str = "0";
  } else if (mask_type == MaskType::kCausalFromTopLeft) {
    mask_type_str = "1";
  } else if (mask_type == MaskType::kCausalFromBottomRight) {
    mask_type_str = "2";
  }
  const std::string dtype_str = dtype_string(dtype);
  if (dtype_str.empty()) {
    return -1.0f;
  }
  auto mask = mask_info::decode(mask_type_str, seqlen_q, seqlen_k);
  auto bias = bias_info::decode("1");
  auto traits = get_ck_fmha_fwd_traits(mask, bias, dtype_str, head_dim, false, false);

  auto args = get_ck_fmha_fwd_args(
      mask,
      batch_size,
      seqlen_q,
      seqlen_k,
      num_heads_q,
      num_heads_k,
      head_dim,
      q_row_stride,
      k_row_stride,
      v_row_stride,
      output_row_stride,
      q_head_stride,
      k_head_stride,
      v_head_stride,
      output_head_stride,
      q_batch_stride,
      k_batch_stride,
      v_batch_stride,
      output_batch_stride,
      bias_row_stride,
      bias_head_stride,
      bias_batch_stride,
      q,
      k,
      v,
      bias_ptr,
      output,
      1.0f / std::sqrt(static_cast<float>(head_dim)));

  return run_mha_fwd(traits, args, stream_config);
}

static float run_static_kv_cache_launcher(
    void* output,
    int64_t output_batch_stride,
    int64_t output_row_stride,
    int64_t output_head_stride,
    void* q,
    int64_t q_batch_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    void* k_cache,
    int64_t k_cache_batch_stride,
    int64_t k_cache_row_stride,
    int64_t k_cache_head_stride,
    void* v_cache,
    int64_t v_cache_batch_stride,
    int64_t v_cache_row_stride,
    int64_t v_cache_head_stride,
    void* knew,
    int64_t knew_batch_stride,
    int64_t knew_row_stride,
    int64_t knew_head_stride,
    void* vnew,
    int64_t vnew_batch_stride,
    int64_t vnew_row_stride,
    int64_t vnew_head_stride,
    void* bias_ptr,
    int64_t bias_batch_stride,
    int64_t bias_row_stride,
    int64_t bias_head_stride,
    bias_enum split_bias_type,
    int64_t batch_size,
    int64_t max_cache_len,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    const int32_t* cache_seqlens,
    DataType dtype,
    int advance_cache_seqlens_after_launch,
    void* scratch,
    size_t scratch_nbytes,
    hipStream_t stream) {
  if (output == nullptr || q == nullptr || k_cache == nullptr || v_cache == nullptr || knew == nullptr ||
      vnew == nullptr || cache_seqlens == nullptr || scratch == nullptr) {
    return -1.0f;
  }
  if (split_bias_type == bias_enum::elementwise_bias && bias_ptr == nullptr) {
    return -1.0f;
  }
  if (batch_size <= 0 || max_cache_len <= 0 || num_heads_q <= 0 || num_heads_k <= 0 || head_dim <= 0) {
    return -1.0f;
  }
  if ((num_heads_q % num_heads_k) != 0) {
    return -1.0f;
  }
  const std::string dtype_str = dtype_string(dtype);
  if (dtype_str.empty()) {
    return -1.0f;
  }

  constexpr int64_t seqlen_q = 1;
  constexpr int64_t seqlen_knew = 1;
  constexpr int64_t num_splits = 1;
  size_t offset = 0;
  char* scratch_base = static_cast<char*>(scratch);
  auto* total_seqlens = static_cast<int32_t*>(
      next_scratch_slice(scratch_base, offset, static_cast<size_t>(batch_size) * sizeof(int32_t)));
  auto* lse_acc = static_cast<float*>(next_scratch_slice(
      scratch_base,
      offset,
      static_cast<size_t>(batch_size * num_heads_q * num_splits * seqlen_q) * sizeof(float)));
  auto* lse = static_cast<float*>(next_scratch_slice(
      scratch_base,
      offset,
      static_cast<size_t>(batch_size * num_heads_q * seqlen_q) * sizeof(float)));
  auto* o_acc = static_cast<float*>(next_scratch_slice(
      scratch_base,
      offset,
      static_cast<size_t>(batch_size * num_heads_q * num_splits * seqlen_q * head_dim) * sizeof(float)));
  if (offset > scratch_nbytes) {
    return -1.0f;
  }

  ck_tile::stream_config stream_config{stream};
  fmha_fwd_appendkv_traits append_traits{
      static_cast<int>(head_dim),
      static_cast<int>(head_dim),
      dtype_str,
      true,
      rope_enum::none};
  fmha_fwd_appendkv_args append_args{};
  append_args.q_ptr = q;
  append_args.k_ptr = k_cache;
  append_args.knew_ptr = knew;
  append_args.v_ptr = v_cache;
  append_args.vnew_ptr = vnew;
  append_args.seqlen_k_ptr = cache_seqlens;
  append_args.seqlen_q = seqlen_q;
  append_args.seqlen_knew = seqlen_knew;
  append_args.batch = batch_size;
  append_args.hdim_q = head_dim;
  append_args.hdim_v = head_dim;
  append_args.nhead_q = num_heads_q;
  append_args.nhead_k = num_heads_k;
  append_args.rotary_cos_ptr = nullptr;
  append_args.rotary_sin_ptr = nullptr;
  append_args.rotary_dim = 0;
  append_args.has_mask = false;
  append_args.block_table_ptr = nullptr;
  append_args.batch_stride_block_table = 0;
  append_args.page_block_size = 0;
  append_args.cache_batch_idx = nullptr;
  append_args.stride_q = q_row_stride;
  append_args.stride_k = k_cache_row_stride;
  append_args.stride_knew = knew_row_stride;
  append_args.stride_v = v_cache_row_stride;
  append_args.stride_vnew = vnew_row_stride;
  append_args.nhead_stride_q = q_head_stride;
  append_args.nhead_stride_k = k_cache_head_stride;
  append_args.nhead_stride_knew = knew_head_stride;
  append_args.nhead_stride_v = v_cache_head_stride;
  append_args.nhead_stride_vnew = vnew_head_stride;
  append_args.batch_stride_q = q_batch_stride;
  append_args.batch_stride_k = k_cache_batch_stride;
  append_args.batch_stride_knew = knew_batch_stride;
  append_args.batch_stride_v = v_cache_batch_stride;
  append_args.batch_stride_vnew = vnew_batch_stride;

  const float append_elapsed_ms = fmha_fwd_appendkv(append_traits, append_args, stream_config);
  if (append_elapsed_ms < 0.0f) {
    return -1.0f;
  }
  const float seqlens_elapsed_ms =
      build_total_seqlens(cache_seqlens, total_seqlens, batch_size, max_cache_len, stream);
  if (seqlens_elapsed_ms < 0.0f) {
    return -1.0f;
  }

  fmha_fwd_splitkv_traits split_traits{
      static_cast<int>(head_dim),
      static_cast<int>(head_dim),
      dtype_str,
      false,
      true,
      false,
      mask_enum::no_mask,
      split_bias_type,
      false,
      false};
  fmha_fwd_splitkv_args split_args{};
  split_args.q_ptr = q;
  split_args.k_ptr = k_cache;
  split_args.v_ptr = v_cache;
  split_args.bias_ptr = bias_ptr;
  split_args.lse_acc_ptr = lse_acc;
  split_args.o_acc_ptr = o_acc;
  split_args.lse_ptr = lse;
  split_args.o_ptr = output;
  split_args.block_table_ptr = nullptr;
  split_args.batch_stride_block_table = 0;
  split_args.page_block_size = 0;
  split_args.is_gappy = false;
  split_args.cache_batch_idx = nullptr;
  split_args.seqstart_q_ptr = nullptr;
  split_args.seqstart_k_ptr = nullptr;
  split_args.seqlen_k_ptr = total_seqlens;
  split_args.seqlen_q = seqlen_q;
  split_args.seqlen_k = max_cache_len;
  split_args.batch = batch_size;
  split_args.max_seqlen_q = seqlen_q;
  split_args.hdim_q = head_dim;
  split_args.hdim_v = head_dim;
  split_args.nhead_q = num_heads_q;
  split_args.nhead_k = num_heads_k;
  split_args.num_splits = num_splits;
  split_args.scale_s = 1.0f / std::sqrt(static_cast<float>(head_dim));
  split_args.scale_p = 1.0f;
  split_args.scale_o = 1.0f;
  split_args.logits_soft_cap = 0.0f;
  split_args.stride_q = q_row_stride;
  split_args.stride_k = k_cache_row_stride;
  split_args.stride_v = v_cache_row_stride;
  split_args.stride_bias = bias_row_stride;
  split_args.stride_o_acc = head_dim;
  split_args.stride_o = output_row_stride;
  split_args.nhead_stride_q = q_head_stride;
  split_args.nhead_stride_k = k_cache_head_stride;
  split_args.nhead_stride_v = v_cache_head_stride;
  split_args.nhead_stride_bias = bias_head_stride;
  split_args.nhead_stride_lse = seqlen_q;
  split_args.nhead_stride_lse_acc = num_splits * seqlen_q;
  split_args.nhead_stride_o_acc = num_splits * seqlen_q * head_dim;
  split_args.nhead_stride_o = output_head_stride;
  split_args.batch_stride_q = q_batch_stride;
  split_args.batch_stride_k = k_cache_batch_stride;
  split_args.batch_stride_v = v_cache_batch_stride;
  split_args.batch_stride_bias = bias_batch_stride;
  split_args.batch_stride_lse = num_heads_q * seqlen_q;
  split_args.batch_stride_lse_acc = num_heads_q * num_splits * seqlen_q;
  split_args.batch_stride_o_acc = num_heads_q * num_splits * seqlen_q * head_dim;
  split_args.batch_stride_o = output_batch_stride;
  split_args.split_stride_lse_acc = seqlen_q;
  split_args.split_stride_o_acc = seqlen_q * head_dim;
  split_args.window_size_left = -1;
  split_args.window_size_right = -1;
  split_args.mask_type = static_cast<ck_tile::index_t>(mask_enum::no_mask);

  const float split_elapsed_ms = fmha_fwd_splitkv(split_traits, split_args, stream_config);
  if (split_elapsed_ms < 0.0f) {
    return -1.0f;
  }
  if (advance_cache_seqlens_after_launch != 0) {
    const float advance_elapsed_ms =
        advance_cache_seqlens(const_cast<int32_t*>(cache_seqlens), batch_size, max_cache_len, stream);
    if (advance_elapsed_ms < 0.0f) {
      return -1.0f;
    }
    return append_elapsed_ms + seqlens_elapsed_ms + split_elapsed_ms + advance_elapsed_ms;
  }
  return append_elapsed_ms + seqlens_elapsed_ms + split_elapsed_ms;
}

float FlashAttentionStaticKvCacheLauncher(
    void* output,
    int64_t output_batch_stride,
    int64_t output_row_stride,
    int64_t output_head_stride,
    void* q,
    int64_t q_batch_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    void* k_cache,
    int64_t k_cache_batch_stride,
    int64_t k_cache_row_stride,
    int64_t k_cache_head_stride,
    void* v_cache,
    int64_t v_cache_batch_stride,
    int64_t v_cache_row_stride,
    int64_t v_cache_head_stride,
    void* knew,
    int64_t knew_batch_stride,
    int64_t knew_row_stride,
    int64_t knew_head_stride,
    void* vnew,
    int64_t vnew_batch_stride,
    int64_t vnew_row_stride,
    int64_t vnew_head_stride,
    int64_t batch_size,
    int64_t max_cache_len,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    const int32_t* cache_seqlens,
    DataType dtype,
    int advance_cache_seqlens_after_launch,
    void* scratch,
    size_t scratch_nbytes,
    hipStream_t stream) {
  return run_static_kv_cache_launcher(
      output,
      output_batch_stride,
      output_row_stride,
      output_head_stride,
      q,
      q_batch_stride,
      q_row_stride,
      q_head_stride,
      k_cache,
      k_cache_batch_stride,
      k_cache_row_stride,
      k_cache_head_stride,
      v_cache,
      v_cache_batch_stride,
      v_cache_row_stride,
      v_cache_head_stride,
      knew,
      knew_batch_stride,
      knew_row_stride,
      knew_head_stride,
      vnew,
      vnew_batch_stride,
      vnew_row_stride,
      vnew_head_stride,
      nullptr,
      0,
      0,
      0,
      bias_enum::no_bias,
      batch_size,
      max_cache_len,
      num_heads_q,
      num_heads_k,
      head_dim,
      cache_seqlens,
      dtype,
      advance_cache_seqlens_after_launch,
      scratch,
      scratch_nbytes,
      stream);
}

float FlashAttentionStaticKvCacheBiasLauncher(
    void* output,
    int64_t output_batch_stride,
    int64_t output_row_stride,
    int64_t output_head_stride,
    void* q,
    int64_t q_batch_stride,
    int64_t q_row_stride,
    int64_t q_head_stride,
    void* k_cache,
    int64_t k_cache_batch_stride,
    int64_t k_cache_row_stride,
    int64_t k_cache_head_stride,
    void* v_cache,
    int64_t v_cache_batch_stride,
    int64_t v_cache_row_stride,
    int64_t v_cache_head_stride,
    void* knew,
    int64_t knew_batch_stride,
    int64_t knew_row_stride,
    int64_t knew_head_stride,
    void* vnew,
    int64_t vnew_batch_stride,
    int64_t vnew_row_stride,
    int64_t vnew_head_stride,
    void* bias,
    int64_t bias_batch_stride,
    int64_t bias_row_stride,
    int64_t bias_head_stride,
    int64_t batch_size,
    int64_t max_cache_len,
    int64_t num_heads_q,
    int64_t num_heads_k,
    int64_t head_dim,
    const int32_t* cache_seqlens,
    DataType dtype,
    int advance_cache_seqlens_after_launch,
    void* scratch,
    size_t scratch_nbytes,
    hipStream_t stream) {
  return run_static_kv_cache_launcher(
      output,
      output_batch_stride,
      output_row_stride,
      output_head_stride,
      q,
      q_batch_stride,
      q_row_stride,
      q_head_stride,
      k_cache,
      k_cache_batch_stride,
      k_cache_row_stride,
      k_cache_head_stride,
      v_cache,
      v_cache_batch_stride,
      v_cache_row_stride,
      v_cache_head_stride,
      knew,
      knew_batch_stride,
      knew_row_stride,
      knew_head_stride,
      vnew,
      vnew_batch_stride,
      vnew_row_stride,
      vnew_head_stride,
      bias,
      bias_batch_stride,
      bias_row_stride,
      bias_head_stride,
      bias_enum::elementwise_bias,
      batch_size,
      max_cache_len,
      num_heads_q,
      num_heads_k,
      head_dim,
      cache_seqlens,
      dtype,
      advance_cache_seqlens_after_launch,
      scratch,
      scratch_nbytes,
      stream);
}
