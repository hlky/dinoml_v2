#pragma once

#include <dinoml/abi.h>

#include <cuda_runtime.h>

extern "C" {

DINO_EXPORT int dino_runtime_cuda_check(
    cudaError_t err,
    const char* expr,
    const char* file,
    int line);

DINO_EXPORT int dino_device_malloc(void** ptr, size_t nbytes);
DINO_EXPORT int dino_device_free(void* ptr);
DINO_EXPORT int dino_copy_host_to_device(
    void* dst_device,
    const void* src_host,
    size_t nbytes);
DINO_EXPORT int dino_copy_device_to_host(
    void* dst_host,
    const void* src_device,
    size_t nbytes);
DINO_EXPORT int dino_copy_device_to_device(
    void* dst_device,
    const void* src_device,
    size_t nbytes);

}

#define DINO_CUDA_CHECK(expr)                                      \
  do {                                                             \
    int _dino_err =                                                \
        dino_runtime_cuda_check((expr), #expr, __FILE__, __LINE__); \
    if (_dino_err) {                                               \
      return _dino_err;                                            \
    }                                                              \
  } while (0)
