#pragma once

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
