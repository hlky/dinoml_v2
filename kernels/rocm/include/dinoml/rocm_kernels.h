#pragma once

#include <dinoml/device.h>
#include <dinoml/runtime_rocm.h>

namespace dinoml::rocm {
// Reusable ROCm kernels are declared below as op families are ported.
}

extern "C" {

#define DINOML_ROCM_TILE_GEMM_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* a, const float* b, float* c, int m, int n, int k, hipStream_t stream);

#define DINOML_ROCM_TILE_GEMM_BIAS_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* a, const float* b, const float* bias, float* c, int m, int n, int k, hipStream_t stream);

#define DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* a, const float* b, const float* bias, const float* d0, float* c, int m, int n, int k, hipStream_t stream);

#define DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* a, const float* b, const float* bias, const float* d0, const float* d1, \
      float* c, int m, int n, int k, hipStream_t stream);

DINOML_ROCM_TILE_GEMM_V1(gemm_rcr)
DINOML_ROCM_TILE_GEMM_V1(gemm_rrr)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_relu)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_relu)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_gelu)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_gelu)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_fast_gelu)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_fast_gelu)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_quick_gelu)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_sigmoid)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_sigmoid)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_tanh)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_tanh)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_swish)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_swish)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_hardswish)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_hardswish)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rcr_bias_elup1)
DINOML_ROCM_TILE_GEMM_BIAS_V1(gemm_rrr_bias_elup1)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rcr_bias_add)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rrr_bias_add)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1(gemm_rcr_bias_add_add)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1(gemm_rrr_bias_add_add)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rcr_bias_mul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rrr_bias_mul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1(gemm_rcr_bias_mul_add)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1(gemm_rrr_bias_mul_add)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rcr_bias_add_relu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rrr_bias_add_relu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1(gemm_rcr_bias_add_add_relu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1(gemm_rrr_bias_add_add_relu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rcr_bias_sigmoid_mul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rrr_bias_sigmoid_mul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rcr_bias_sigmoid_mul_tanh)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rrr_bias_sigmoid_mul_tanh)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rcr_bias_mul_tanh)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1(gemm_rrr_bias_mul_tanh)

#undef DINOML_ROCM_TILE_GEMM_V1
#undef DINOML_ROCM_TILE_GEMM_BIAS_V1
#undef DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_V1
#undef DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_V1

#define DINOML_ROCM_TILE_BMM_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* a, const float* b, float* c, int batch_count, int m, int n, int k, \
      int64_t batch_stride_a, int64_t batch_stride_b, int64_t batch_stride_c, \
      int lda, int ldb, int ldc, hipStream_t stream);

#define DINOML_ROCM_TILE_BMM_ADD_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* a, const float* b, const float* d0, float* c, int batch_count, int m, int n, int k, \
      int64_t batch_stride_a, int64_t batch_stride_b, int64_t batch_stride_d0, int64_t batch_stride_c, \
      int lda, int ldb, int ldd0, int ldc, hipStream_t stream);

DINOML_ROCM_TILE_BMM_V1(bmm_ccc)
DINOML_ROCM_TILE_BMM_V1(bmm_ccr)
DINOML_ROCM_TILE_BMM_V1(bmm_crc)
DINOML_ROCM_TILE_BMM_V1(bmm_crr)
DINOML_ROCM_TILE_BMM_V1(bmm_rcc)
DINOML_ROCM_TILE_BMM_V1(bmm_rcr)
DINOML_ROCM_TILE_BMM_V1(bmm_rrc)
DINOML_ROCM_TILE_BMM_V1(bmm_rrr)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_ccc_add)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_ccr_add)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_crc_add)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_crr_add)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_rcc_add)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_rcr_add)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_rrc_add)
DINOML_ROCM_TILE_BMM_ADD_V1(bmm_rrr_add)

#undef DINOML_ROCM_TILE_BMM_V1
#undef DINOML_ROCM_TILE_BMM_ADD_V1

#define DINOML_ROCM_TILE_CONV_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* x, const float* weight, const float* bias, float* output, \
      int batch, int in_channels, int in_height, int in_width, int out_channels, \
      int kernel_h, int kernel_w, int out_height, int out_width, int stride_h, int stride_w, \
      int pad_h, int pad_w, int dilation_h, int dilation_w, hipStream_t stream);

#define DINOML_ROCM_TILE_CONV_ADD_V1(OP) \
  int dinoml_rocm_tile_##OP##_float32_v1( \
      const float* x, const float* weight, const float* bias, const float* residual, float* output, \
      int batch, int in_channels, int in_height, int in_width, int out_channels, \
      int kernel_h, int kernel_w, int out_height, int out_width, int stride_h, int stride_w, \
      int pad_h, int pad_w, int dilation_h, int dilation_w, hipStream_t stream);

DINOML_ROCM_TILE_CONV_V1(conv2d_bias)
DINOML_ROCM_TILE_CONV_V1(conv2d_bias_relu)
DINOML_ROCM_TILE_CONV_ADD_V1(conv2d_bias_add)
DINOML_ROCM_TILE_CONV_ADD_V1(conv2d_bias_add_relu)

#undef DINOML_ROCM_TILE_CONV_V1
#undef DINOML_ROCM_TILE_CONV_ADD_V1

int dinoml_flash_attn_ck_fwd_float16_v1(
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
    hipStream_t stream);

int dinoml_flash_attn_ck_fwd_bfloat16_v1(
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
    hipStream_t stream);

int dinoml_flash_attn_ck_qkv_fwd_float16_v1(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    hipStream_t stream);

int dinoml_flash_attn_ck_qkv_fwd_bfloat16_v1(
    const void* qkv,
    void* output,
    int64_t batch_size,
    int64_t seqlen,
    int64_t num_heads,
    int64_t head_dim,
    int causal,
    hipStream_t stream);

}
