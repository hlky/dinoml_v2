#pragma once

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define DINO_EXPORT __declspec(dllexport)
#else
#define DINO_EXPORT __attribute__((visibility("default")))
#endif

#define DINO_RUNTIME_ABI_VERSION 4

enum DinoDtype {
  DINO_DTYPE_FLOAT16 = 1,
  DINO_DTYPE_FLOAT32 = 2,
  DINO_DTYPE_INT32 = 3,
  DINO_DTYPE_INT64 = 4,
  DINO_DTYPE_BOOL = 5,
  DINO_DTYPE_BFLOAT16 = 6,
  DINO_DTYPE_FLOAT8_E4M3 = 7,
  DINO_DTYPE_FLOAT8_E5M2 = 8,
};

struct DinoTensor {
  void* data;
  // Host pointer to an int64 shape array with ndim entries. The caller owns
  // this storage and it must remain valid for the duration of dino_session_run.
  // Generated modules validate these dimensions against static shapes or the
  // min/max/divisibility constraints serialized in artifact metadata.
  const int64_t* shape;
  size_t ndim;
  int dtype;
};

struct DinoModule;
struct DinoSession;

extern "C" {

DINO_EXPORT int dino_runtime_fail(const char* message);
DINO_EXPORT int dino_abi_version();
DINO_EXPORT const char* dino_get_last_error();
DINO_EXPORT int dino_session_set_stream(DinoSession* session, void* stream);
DINO_EXPORT int dino_session_get_output_shape(
    DinoSession* session,
    size_t output_index,
    int64_t* out_shape,
    size_t* inout_ndim);

}
