#pragma once

#if defined(DINOML_CUDA) || defined(__CUDACC__)
#define DINO_COMPILE_CUDA 1
#endif

#if defined(DINOML_HIP) || defined(__HIPCC__)
#define DINO_COMPILE_HIP 1
#endif

#if defined(DINO_COMPILE_CUDA) && defined(DINO_COMPILE_HIP)
#error "DinoML device headers cannot target CUDA and HIP in the same translation unit"
#endif

#if defined(DINO_COMPILE_CUDA)
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#define DINO_DEVICE_GPU 1
#define DINO_DEVICE_HD __host__ __device__
#define DINO_DEVICE_FORCEINLINE __forceinline__
#ifndef LDG
#define LDG(x) __ldg(x)
#endif
#ifndef HALF2DATA
#define HALF2DATA(x) (x)
#endif
namespace dinoml {
using float16 = half;
using float16_2 = half2;
using bfloat16 = __nv_bfloat16;
using bfloat162 = __nv_bfloat162;
using bfloat16_2 = __nv_bfloat162;
using DeviceStream = cudaStream_t;
}  // namespace dinoml
#elif defined(DINO_COMPILE_HIP)
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#define DINO_DEVICE_GPU 1
#define DINO_DEVICE_HD __host__ __device__
#define DINO_DEVICE_FORCEINLINE __forceinline__
#ifndef LDG
#define LDG(x) (*(x))
#endif
#ifndef HALF2DATA
#define HALF2DATA(x) ((x).data)
#endif
namespace dinoml {
using float16 = half;
using float16_2 = half2;
using bfloat16 = __hip_bfloat16;
using bfloat162 = __hip_bfloat162;
using bfloat16_2 = __hip_bfloat162;
using DeviceStream = hipStream_t;
}  // namespace dinoml
#else
#define DINO_DEVICE_HD
#define DINO_DEVICE_FORCEINLINE inline
#endif
