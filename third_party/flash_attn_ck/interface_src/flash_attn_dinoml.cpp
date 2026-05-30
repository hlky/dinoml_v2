#include "../flash_attn_dinoml.h"

#include <algorithm>
#include <vector>

#include <cstdint>

#include <cassert>
#include <cmath>
#include <type_traits>

#include "fmha_fwd.hpp"

fmha_fwd_traits get_ck_fmha_fwd_traits(
    const mask_info& mask,
    std::string dtype,
    int head_size,
    bool has_dropout,
    bool has_lse,
    bool enable_alibi) {
  return fmha_fwd_traits{
      head_size,
      head_size,
      dtype,
      false, // is_group_mode
      true, // is_v_rowmajor
      false, // has_logits_soft_cap
      mask.type,
      enable_alibi ? bias_enum::alibi : bias_enum::no_bias,
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
    // device pointers
    const void* q,
    const void* k,
    const void* v,
    void* out,
    float softmax_scale) {
  ck_tile::index_t stride_randval = 0;

  ck_tile::index_t nhead_stride_lse = 0;
  ck_tile::index_t nhead_stride_randval = 0;


  ck_tile::index_t batch_stride_lse = 0;
  ck_tile::index_t batch_stride_randval = 0;

  void* alibi_slopes_ptr = nullptr;
  ck_tile::index_t stride_alibi_slopes = 0;

  fmha_fwd_args args{};
  args.q_ptr = q;
  args.k_ptr = k;
  args.v_ptr = v;
  args.bias_ptr = alibi_slopes_ptr;
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
  args.stride_bias = stride_alibi_slopes;
  args.stride_randval = stride_randval;
  args.stride_o = stride_o;
  args.nhead_stride_q = nhead_stride_q;
  args.nhead_stride_k = nhead_stride_k;
  args.nhead_stride_v = nhead_stride_v;
  args.nhead_stride_bias = 0;
  args.nhead_stride_randval = nhead_stride_randval;
  args.nhead_stride_lse = nhead_stride_lse;
  args.nhead_stride_o = nhead_stride_o;
  args.batch_stride_q = batch_stride_q;
  args.batch_stride_k = batch_stride_k;
  args.batch_stride_v = batch_stride_v;
  args.batch_stride_bias = 0;
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
  auto traits =
      get_ck_fmha_fwd_traits(mask, dtype_str, head_dim, false, false, false);

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
      q,
      k,
      v,
      output,
      1.0f / std::sqrt(static_cast<float>(head_dim)));

  return run_mha_fwd(traits, args, stream_config);
}
