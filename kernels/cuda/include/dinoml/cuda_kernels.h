#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <dinoml/runtime.h>

extern "C" int dinoml_cutlass_gemm_rrr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_f16(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_f16(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_bias_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_bias_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_bias_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_bias_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_bias_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_bias_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_bias_relu_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_bias_relu_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_bias_relu_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_bias_relu_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rrr_bias_relu_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" int dinoml_cutlass_gemm_rcr_bias_relu_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_f32(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_f16(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_f16(
    const half* a,
    const half* b,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_relu_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_relu_f32(
    const float* a,
    const float* b,
    const float* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_relu_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_relu_f16(
    const half* a,
    const half* b,
    const half* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_relu_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_relu_bf16(
    const __nv_bfloat16* a,
    const __nv_bfloat16* b,
    const __nv_bfloat16* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream);
