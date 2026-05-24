#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp"

#include <cstdint>
#include <exception>

namespace {

constexpr ck_tile::index_t kQ8BlockSize = 32;
constexpr ck_tile::index_t kQ4_0BlockSize = 32;
constexpr ck_tile::index_t kQKBlockSize = 256;
constexpr ck_tile::index_t kKScaleSize = 12;
constexpr ck_tile::index_t kMxFp4BlockSize = 32;
constexpr ck_tile::index_t kNvFp4BlockSize = 64;
constexpr ck_tile::index_t kBlockPerCu = 1;
constexpr ck_tile::index_t kWavesPerBlock = 4;

struct GgufQ8_0Block {
  uint16_t d;
  int8_t qs[kQ8BlockSize];
};

static_assert(sizeof(GgufQ8_0Block) == 34, "GGUF Q8_0 block layout must be scale + 32 int8s");

struct GgufQ4_0Block {
  uint16_t d;
  uint8_t qs[kQ4_0BlockSize / 2];
};

static_assert(sizeof(GgufQ4_0Block) == 18, "libgguf Q4_0 block layout must be scale + 16 packed nibbles");

struct GgufQ4_KBlock {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[kKScaleSize];
  uint8_t qs[kQKBlockSize / 2];
};

static_assert(sizeof(GgufQ4_KBlock) == 144, "libgguf Q4_K block layout must match vendored CPU/CUDA kernels");

struct GgufQ5_KBlock {
  uint16_t d;
  uint16_t dmin;
  uint8_t scales[kKScaleSize];
  uint8_t qh[kQKBlockSize / 8];
  uint8_t qs[kQKBlockSize / 2];
};

static_assert(sizeof(GgufQ5_KBlock) == 176, "libgguf Q5_K block layout must match vendored CPU/CUDA kernels");

struct GgufQ2_KBlock {
  uint8_t scales[kQKBlockSize / 16];
  uint8_t qs[kQKBlockSize / 4];
  uint16_t d;
  uint16_t dmin;
};

static_assert(sizeof(GgufQ2_KBlock) == 84, "libgguf Q2_K block layout must match vendored CPU/CUDA kernels");

struct GgufQ3_KBlock {
  uint8_t hmask[kQKBlockSize / 8];
  uint8_t qs[kQKBlockSize / 4];
  uint8_t scales[kKScaleSize];
  uint16_t d;
};

static_assert(sizeof(GgufQ3_KBlock) == 110, "libgguf Q3_K block layout must match vendored CPU/CUDA kernels");

struct GgufQ6_KBlock {
  uint8_t ql[kQKBlockSize / 2];
  uint8_t qh[kQKBlockSize / 4];
  int8_t scales[kQKBlockSize / 16];
  uint16_t d;
};

static_assert(sizeof(GgufQ6_KBlock) == 210, "libgguf Q6_K block layout must match vendored CPU/CUDA kernels");

struct GgufMxFp4Block {
  uint8_t e;
  uint8_t qs[kMxFp4BlockSize / 2];
};

static_assert(sizeof(GgufMxFp4Block) == 17, "libgguf MXFP4 block layout must match vendored CPU/CUDA kernels");

struct GgufNvFp4Block {
  uint8_t d[kNvFp4BlockSize / 16];
  uint8_t qs[kNvFp4BlockSize / 2];
};

static_assert(sizeof(GgufNvFp4Block) == 36, "libgguf NVFP4 block layout must match vendored CPU/CUDA kernels");

template <typename AStorage, typename CStorage>
struct GgufQ8_0GemmRcrArgs {
  const AStorage* a;
  const GgufQ8_0Block* b;
  CStorage* c;
  ck_tile::index_t m;
  ck_tile::index_t n;
  ck_tile::index_t k;
};

template <typename Storage>
struct GgufQ8_0DequantNkArgs {
  const GgufQ8_0Block* b;
  Storage* b_nk;
  ck_tile::index_t n;
  ck_tile::index_t k;
};

template <typename QuantBlock, typename AStorage, typename CStorage>
struct GgufQuantGemmRcrArgs {
  const AStorage* a;
  const QuantBlock* b;
  CStorage* c;
  ck_tile::index_t m;
  ck_tile::index_t n;
  ck_tile::index_t k;
};

template <typename QuantBlock, typename Storage>
struct GgufQuantDequantArgs {
  const QuantBlock* b;
  Storage* b_dense;
  ck_tile::index_t n;
  ck_tile::index_t k;
};

template <typename BStorage, typename ScaleStorage>
struct GgufQ8_0TranscodeBqArgs {
  const GgufQ8_0Block* b;
  BStorage* b_kn;
  ScaleStorage* bq_qk_n;
  ck_tile::index_t n;
  ck_tile::index_t k;
};

CK_TILE_DEVICE float gguf_half_to_float(uint16_t bits) {
  return __half2float(__ushort_as_half(bits));
}

CK_TILE_DEVICE void gguf_get_scale_min_k4(
    ck_tile::index_t j,
    const uint8_t* q,
    uint8_t& d,
    uint8_t& m) {
  if (j < 4) {
    d = q[j] & 63;
    m = q[j + 4] & 63;
  } else {
    d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
    m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

CK_TILE_DEVICE int gguf_q3_k_scale(const uint8_t* scales, ck_tile::index_t is) {
  if (is < 4) {
    return static_cast<int>(scales[is] & 0x0f) |
           (static_cast<int>((scales[is + 8] >> 0) & 0x03) << 4);
  }
  if (is < 8) {
    return static_cast<int>(scales[is] & 0x0f) |
           (static_cast<int>((scales[is + 4] >> 2) & 0x03) << 4);
  }
  if (is < 12) {
    return static_cast<int>(scales[is - 8] >> 4) |
           (static_cast<int>((scales[is] >> 4) & 0x03) << 4);
  }
  return static_cast<int>(scales[is - 8] >> 4) |
         (static_cast<int>((scales[is - 4] >> 6) & 0x03) << 4);
}

CK_TILE_DEVICE float gguf_e8m0_to_fp32_half(uint8_t x) {
  const uint32_t bits = x < 2 ? (0x00200000u << x) : (static_cast<uint32_t>(x - 1) << 23);
  union {
    uint32_t u;
    float f;
  } value{bits};
  return value.f;
}

CK_TILE_DEVICE float gguf_ue4m3_to_fp32(uint8_t x) {
  if (x == 0 || x == 0x7f) {
    return 0.0f;
  }
  const int exp = static_cast<int>((x >> 3) & 0x0f);
  const int man = static_cast<int>(x & 0x07);
  if (exp == 0) {
    return static_cast<float>(man) * 0.0009765625f;
  }
  const uint32_t bits = (static_cast<uint32_t>(exp + 119) << 23) |
                        (static_cast<uint32_t>(man) << 20);
  union {
    uint32_t u;
    float f;
  } value{bits};
  return value.f;
}

CK_TILE_DEVICE int gguf_fp4_quant_value(uint8_t q) {
  const int mag = static_cast<int>(q & 0x07);
  const int value = mag <= 4 ? mag : (mag == 5 ? 6 : (mag == 6 ? 8 : 12));
  return (q & 0x08) ? -value : value;
}

CK_TILE_DEVICE float storage_to_float(float value) {
  return value;
}

CK_TILE_DEVICE float storage_to_float(half value) {
  return __half2float(value);
}

template <typename T>
CK_TILE_DEVICE T float_to_storage(float value);

template <>
CK_TILE_DEVICE float float_to_storage<float>(float value) {
  return value;
}

template <>
CK_TILE_DEVICE half float_to_storage<half>(float value) {
  return __float2half(value);
}

template <typename BStorage>
CK_TILE_DEVICE BStorage q8_to_bstorage(int8_t value);

template <>
CK_TILE_DEVICE int8_t q8_to_bstorage<int8_t>(int8_t value) {
  return value;
}

template <>
CK_TILE_DEVICE ck_tile::fp8_t q8_to_bstorage<ck_tile::fp8_t>(int8_t value) {
  return ck_tile::float_to_fp8(static_cast<float>(value));
}

template <>
CK_TILE_DEVICE ck_tile::bf8_t q8_to_bstorage<ck_tile::bf8_t>(int8_t value) {
  return ck_tile::float_to_bf8(static_cast<float>(value));
}

template <typename ScaleStorage>
CK_TILE_DEVICE ScaleStorage scale_to_storage(float value);

template <typename Span, typename F>
CK_TILE_DEVICE void sweep_gguf_distributed_span(Span, const F& f) {
  if constexpr (Span::Impl::size() == 0) {
    f(ck_tile::tile_distributed_index<>{});
  } else {
    ck_tile::sweep_tile_span(Span{}, f);
  }
}

template <>
CK_TILE_DEVICE float scale_to_storage<float>(float value) {
  return value;
}

template <>
CK_TILE_DEVICE half scale_to_storage<half>(float value) {
  return __float2half(value);
}

template <typename AStorage, typename CStorage, ck_tile::index_t TileM, ck_tile::index_t TileN>
struct GgufQ8_0GemmRcrKernel {
  static constexpr ck_tile::index_t kBlockSize = TileM * TileN;

  CK_TILE_DEVICE void operator()(GgufQ8_0GemmRcrArgs<AStorage, CStorage> args) const {
    __shared__ float a_tile[TileM][kQ8BlockSize];
    __shared__ int8_t b_tile[TileN][kQ8BlockSize];
    __shared__ float b_scale[TileN];

    const ck_tile::index_t tid = static_cast<ck_tile::index_t>(threadIdx.x);
    const ck_tile::index_t local_m = tid / TileN;
    const ck_tile::index_t local_n = tid - local_m * TileN;
    const ck_tile::index_t row = static_cast<ck_tile::index_t>(blockIdx.y) * TileM + local_m;
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x) * TileN + local_n;
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;

    float acc = 0.0f;
    for (ck_tile::index_t qb = 0; qb < qblocks_per_row; ++qb) {
      for (ck_tile::index_t idx = tid; idx < TileM * kQ8BlockSize; idx += kBlockSize) {
        const ck_tile::index_t tile_m = idx / kQ8BlockSize;
        const ck_tile::index_t tile_k = idx - tile_m * kQ8BlockSize;
        const ck_tile::index_t global_m = static_cast<ck_tile::index_t>(blockIdx.y) * TileM + tile_m;
        const ck_tile::index_t global_k = qb * kQ8BlockSize + tile_k;
        a_tile[tile_m][tile_k] =
            global_m < args.m ? storage_to_float(args.a[global_m * args.k + global_k]) : 0.0f;
      }

      for (ck_tile::index_t tile_n = tid; tile_n < TileN; tile_n += kBlockSize) {
        const ck_tile::index_t global_n = static_cast<ck_tile::index_t>(blockIdx.x) * TileN + tile_n;
        b_scale[tile_n] =
            global_n < args.n ? gguf_half_to_float(args.b[global_n * qblocks_per_row + qb].d) : 0.0f;
      }

      for (ck_tile::index_t idx = tid; idx < TileN * kQ8BlockSize; idx += kBlockSize) {
        const ck_tile::index_t tile_n = idx / kQ8BlockSize;
        const ck_tile::index_t tile_k = idx - tile_n * kQ8BlockSize;
        const ck_tile::index_t global_n = static_cast<ck_tile::index_t>(blockIdx.x) * TileN + tile_n;
        b_tile[tile_n][tile_k] =
            global_n < args.n ? args.b[global_n * qblocks_per_row + qb].qs[tile_k] : 0;
      }
      __syncthreads();

      if (row < args.m && col < args.n) {
        const float scale = b_scale[local_n];
#pragma unroll
        for (ck_tile::index_t kk = 0; kk < kQ8BlockSize; ++kk) {
          acc += a_tile[local_m][kk] * (scale * static_cast<float>(b_tile[local_n][kk]));
        }
      }
      __syncthreads();
    }

    if (row < args.m && col < args.n) {
      args.c[row * args.n + col] = float_to_storage<CStorage>(acc);
    }
  }
};

template <typename AStorage, typename CStorage>
struct GgufQ8_0GemmRcrWaveKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0GemmRcrArgs<AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t row = static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;

    float partial = 0.0f;
    if (row < args.m && col < args.n) {
      for (ck_tile::index_t global_k = lane; global_k < args.k; global_k += warp_size) {
        const ck_tile::index_t qb = global_k / kQ8BlockSize;
        const ck_tile::index_t kk = global_k - qb * kQ8BlockSize;
        const GgufQ8_0Block& b_block = args.b[col * qblocks_per_row + qb];
        const float scale = gguf_half_to_float(b_block.d);
        partial += storage_to_float(args.a[row * args.k + global_k]) *
                   (scale * static_cast<float>(b_block.qs[kk]));
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const float remote = ck_tile::warp_shuffle_down(partial, static_cast<uint32_t>(offset));
      if (lane < offset) {
        partial += remote;
      }
    }

    if (lane == 0 && row < args.m && col < args.n) {
      args.c[row * args.n + col] = float_to_storage<CStorage>(partial);
    }
  }
};

template <typename AStorage, typename CStorage>
struct GgufQ8_0GemmRcrWaveM4Kernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0GemmRcrArgs<AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;

    float partial0 = 0.0f;
    float partial1 = 0.0f;
    float partial2 = 0.0f;
    float partial3 = 0.0f;
    if (col < args.n) {
      for (ck_tile::index_t global_k = lane; global_k < args.k; global_k += warp_size) {
        const ck_tile::index_t qb = global_k / kQ8BlockSize;
        const ck_tile::index_t kk = global_k - qb * kQ8BlockSize;
        const GgufQ8_0Block& b_block = args.b[col * qblocks_per_row + qb];
        const float b_value = gguf_half_to_float(b_block.d) * static_cast<float>(b_block.qs[kk]);
        if (args.m > 0) {
          partial0 += storage_to_float(args.a[global_k]) * b_value;
        }
        if (args.m > 1) {
          partial1 += storage_to_float(args.a[args.k + global_k]) * b_value;
        }
        if (args.m > 2) {
          partial2 += storage_to_float(args.a[2 * args.k + global_k]) * b_value;
        }
        if (args.m > 3) {
          partial3 += storage_to_float(args.a[3 * args.k + global_k]) * b_value;
        }
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const auto shuffle_offset = static_cast<uint32_t>(offset);
      const float remote0 = ck_tile::warp_shuffle_down(partial0, shuffle_offset);
      const float remote1 = ck_tile::warp_shuffle_down(partial1, shuffle_offset);
      const float remote2 = ck_tile::warp_shuffle_down(partial2, shuffle_offset);
      const float remote3 = ck_tile::warp_shuffle_down(partial3, shuffle_offset);
      if (lane < offset) {
        partial0 += remote0;
        partial1 += remote1;
        partial2 += remote2;
        partial3 += remote3;
      }
    }

    if (lane == 0 && col < args.n) {
      if (args.m > 0) {
        args.c[col] = float_to_storage<CStorage>(partial0);
      }
      if (args.m > 1) {
        args.c[args.n + col] = float_to_storage<CStorage>(partial1);
      }
      if (args.m > 2) {
        args.c[2 * args.n + col] = float_to_storage<CStorage>(partial2);
      }
      if (args.m > 3) {
        args.c[3 * args.n + col] = float_to_storage<CStorage>(partial3);
      }
    }
  }
};

template <typename AStorage, typename CStorage, ck_tile::index_t ColsPerWave>
struct GgufQ8_0GemmRcrWaveM4NKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0GemmRcrArgs<AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t group = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t col_base = group * ColsPerWave;
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;
    float partial[ColsPerWave][4] = {};

    auto accumulate_k = [&](ck_tile::index_t global_k, ck_tile::index_t qb, ck_tile::index_t kk) {
      const float a0 = args.m > 0 ? storage_to_float(args.a[global_k]) : 0.0f;
      const float a1 = args.m > 1 ? storage_to_float(args.a[args.k + global_k]) : 0.0f;
      const float a2 = args.m > 2 ? storage_to_float(args.a[2 * args.k + global_k]) : 0.0f;
      const float a3 = args.m > 3 ? storage_to_float(args.a[3 * args.k + global_k]) : 0.0f;
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const ck_tile::index_t col = col_base + col_offset;
        if (col < args.n) {
          const GgufQ8_0Block& b_block = args.b[col * qblocks_per_row + qb];
          const float b_value = gguf_half_to_float(b_block.d) * static_cast<float>(b_block.qs[kk]);
          partial[col_offset][0] += a0 * b_value;
          partial[col_offset][1] += a1 * b_value;
          partial[col_offset][2] += a2 * b_value;
          partial[col_offset][3] += a3 * b_value;
        }
      }
    };

    if (col_base < args.n) {
      if (warp_size == kQ8BlockSize) {
        for (ck_tile::index_t qb = 0; qb < qblocks_per_row; ++qb) {
          accumulate_k(qb * kQ8BlockSize + lane, qb, lane);
        }
      } else {
        for (ck_tile::index_t global_k = lane; global_k < args.k; global_k += warp_size) {
          const ck_tile::index_t qb = global_k / kQ8BlockSize;
          const ck_tile::index_t kk = global_k - qb * kQ8BlockSize;
          accumulate_k(global_k, qb, kk);
        }
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const auto shuffle_offset = static_cast<uint32_t>(offset);
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const float remote0 = ck_tile::warp_shuffle_down(partial[col_offset][0], shuffle_offset);
        const float remote1 = ck_tile::warp_shuffle_down(partial[col_offset][1], shuffle_offset);
        const float remote2 = ck_tile::warp_shuffle_down(partial[col_offset][2], shuffle_offset);
        const float remote3 = ck_tile::warp_shuffle_down(partial[col_offset][3], shuffle_offset);
        if (lane < offset) {
          partial[col_offset][0] += remote0;
          partial[col_offset][1] += remote1;
          partial[col_offset][2] += remote2;
          partial[col_offset][3] += remote3;
        }
      }
    }

    if (lane == 0) {
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const ck_tile::index_t col = col_base + col_offset;
        if (col < args.n) {
          if (args.m > 0) {
            args.c[col] = float_to_storage<CStorage>(partial[col_offset][0]);
          }
          if (args.m > 1) {
            args.c[args.n + col] = float_to_storage<CStorage>(partial[col_offset][1]);
          }
          if (args.m > 2) {
            args.c[2 * args.n + col] = float_to_storage<CStorage>(partial[col_offset][2]);
          }
          if (args.m > 3) {
            args.c[3 * args.n + col] = float_to_storage<CStorage>(partial[col_offset][3]);
          }
        }
      }
    }
  }
};

template <typename AStorage, typename CStorage>
struct GgufQ8_0GemmRcrWaveM4N2ExactKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0GemmRcrArgs<AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t group = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t col0 = group * 2;
    const ck_tile::index_t col1 = col0 + 1;
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;

    float c00 = 0.0f;
    float c01 = 0.0f;
    float c02 = 0.0f;
    float c03 = 0.0f;
    float c10 = 0.0f;
    float c11 = 0.0f;
    float c12 = 0.0f;
    float c13 = 0.0f;

    auto accumulate_k = [&](ck_tile::index_t global_k, ck_tile::index_t qb, ck_tile::index_t kk) {
      const float a0 = storage_to_float(args.a[global_k]);
      const float a1 = storage_to_float(args.a[args.k + global_k]);
      const float a2 = storage_to_float(args.a[2 * args.k + global_k]);
      const float a3 = storage_to_float(args.a[3 * args.k + global_k]);
      if (col0 < args.n) {
        const GgufQ8_0Block& b_block0 = args.b[col0 * qblocks_per_row + qb];
        const float b0 = gguf_half_to_float(b_block0.d) * static_cast<float>(b_block0.qs[kk]);
        c00 += a0 * b0;
        c01 += a1 * b0;
        c02 += a2 * b0;
        c03 += a3 * b0;
      }
      if (col1 < args.n) {
        const GgufQ8_0Block& b_block1 = args.b[col1 * qblocks_per_row + qb];
        const float b1 = gguf_half_to_float(b_block1.d) * static_cast<float>(b_block1.qs[kk]);
        c10 += a0 * b1;
        c11 += a1 * b1;
        c12 += a2 * b1;
        c13 += a3 * b1;
      }
    };

    if (col0 < args.n) {
      if (warp_size == kQ8BlockSize) {
        for (ck_tile::index_t qb = 0; qb < qblocks_per_row; ++qb) {
          accumulate_k(qb * kQ8BlockSize + lane, qb, lane);
        }
      } else {
        for (ck_tile::index_t global_k = lane; global_k < args.k; global_k += warp_size) {
          const ck_tile::index_t qb = global_k / kQ8BlockSize;
          const ck_tile::index_t kk = global_k - qb * kQ8BlockSize;
          accumulate_k(global_k, qb, kk);
        }
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const auto shuffle_offset = static_cast<uint32_t>(offset);
      const float r00 = ck_tile::warp_shuffle_down(c00, shuffle_offset);
      const float r01 = ck_tile::warp_shuffle_down(c01, shuffle_offset);
      const float r02 = ck_tile::warp_shuffle_down(c02, shuffle_offset);
      const float r03 = ck_tile::warp_shuffle_down(c03, shuffle_offset);
      const float r10 = ck_tile::warp_shuffle_down(c10, shuffle_offset);
      const float r11 = ck_tile::warp_shuffle_down(c11, shuffle_offset);
      const float r12 = ck_tile::warp_shuffle_down(c12, shuffle_offset);
      const float r13 = ck_tile::warp_shuffle_down(c13, shuffle_offset);
      if (lane < offset) {
        c00 += r00;
        c01 += r01;
        c02 += r02;
        c03 += r03;
        c10 += r10;
        c11 += r11;
        c12 += r12;
        c13 += r13;
      }
    }

    if (lane == 0) {
      if (col0 < args.n) {
        args.c[col0] = float_to_storage<CStorage>(c00);
        args.c[args.n + col0] = float_to_storage<CStorage>(c01);
        args.c[2 * args.n + col0] = float_to_storage<CStorage>(c02);
        args.c[3 * args.n + col0] = float_to_storage<CStorage>(c03);
      }
      if (col1 < args.n) {
        args.c[col1] = float_to_storage<CStorage>(c10);
        args.c[args.n + col1] = float_to_storage<CStorage>(c11);
        args.c[2 * args.n + col1] = float_to_storage<CStorage>(c12);
        args.c[3 * args.n + col1] = float_to_storage<CStorage>(c13);
      }
    }
  }
};

template <typename AStorage, typename CStorage, ck_tile::index_t ColsPerWave>
struct GgufQ8_0GemmRcrWaveM4NExactKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0GemmRcrArgs<AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t group = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t col_base = group * ColsPerWave;
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;
    float partial[ColsPerWave][4] = {};

    if (col_base < args.n) {
      if (warp_size == kQ8BlockSize) {
        for (ck_tile::index_t qb = 0; qb < qblocks_per_row; ++qb) {
          const ck_tile::index_t global_k = qb * kQ8BlockSize + lane;
          const float a0 = storage_to_float(args.a[global_k]);
          const float a1 = storage_to_float(args.a[args.k + global_k]);
          const float a2 = storage_to_float(args.a[2 * args.k + global_k]);
          const float a3 = storage_to_float(args.a[3 * args.k + global_k]);
#pragma unroll
          for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
            const ck_tile::index_t col = col_base + col_offset;
            if (col < args.n) {
              const GgufQ8_0Block& b_block = args.b[col * qblocks_per_row + qb];
              const float b_value = gguf_half_to_float(b_block.d) * static_cast<float>(b_block.qs[lane]);
              partial[col_offset][0] += a0 * b_value;
              partial[col_offset][1] += a1 * b_value;
              partial[col_offset][2] += a2 * b_value;
              partial[col_offset][3] += a3 * b_value;
            }
          }
        }
      } else {
        for (ck_tile::index_t global_k = lane; global_k < args.k; global_k += warp_size) {
          const ck_tile::index_t qb = global_k / kQ8BlockSize;
          const ck_tile::index_t kk = global_k - qb * kQ8BlockSize;
          const float a0 = storage_to_float(args.a[global_k]);
          const float a1 = storage_to_float(args.a[args.k + global_k]);
          const float a2 = storage_to_float(args.a[2 * args.k + global_k]);
          const float a3 = storage_to_float(args.a[3 * args.k + global_k]);
#pragma unroll
          for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
            const ck_tile::index_t col = col_base + col_offset;
            if (col < args.n) {
              const GgufQ8_0Block& b_block = args.b[col * qblocks_per_row + qb];
              const float b_value = gguf_half_to_float(b_block.d) * static_cast<float>(b_block.qs[kk]);
              partial[col_offset][0] += a0 * b_value;
              partial[col_offset][1] += a1 * b_value;
              partial[col_offset][2] += a2 * b_value;
              partial[col_offset][3] += a3 * b_value;
            }
          }
        }
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const auto shuffle_offset = static_cast<uint32_t>(offset);
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const float remote0 = ck_tile::warp_shuffle_down(partial[col_offset][0], shuffle_offset);
        const float remote1 = ck_tile::warp_shuffle_down(partial[col_offset][1], shuffle_offset);
        const float remote2 = ck_tile::warp_shuffle_down(partial[col_offset][2], shuffle_offset);
        const float remote3 = ck_tile::warp_shuffle_down(partial[col_offset][3], shuffle_offset);
        if (lane < offset) {
          partial[col_offset][0] += remote0;
          partial[col_offset][1] += remote1;
          partial[col_offset][2] += remote2;
          partial[col_offset][3] += remote3;
        }
      }
    }

    if (lane == 0) {
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const ck_tile::index_t col = col_base + col_offset;
        if (col < args.n) {
          args.c[col] = float_to_storage<CStorage>(partial[col_offset][0]);
          args.c[args.n + col] = float_to_storage<CStorage>(partial[col_offset][1]);
          args.c[2 * args.n + col] = float_to_storage<CStorage>(partial[col_offset][2]);
          args.c[3 * args.n + col] = float_to_storage<CStorage>(partial[col_offset][3]);
        }
      }
    }
  }
};

struct GgufQ8_0Traits {
  using Block = GgufQ8_0Block;
  static constexpr ck_tile::index_t BlockSize = kQ8BlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    return gguf_half_to_float(block.d) * static_cast<float>(block.qs[kk]);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const float d = gguf_half_to_float(block.d);
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      set_value(i, d * static_cast<float>(block.qs[kk_base + i]));
    });
  }
};

struct GgufQ4_0Traits {
  using Block = GgufQ4_0Block;
  static constexpr ck_tile::index_t BlockSize = kQ4_0BlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const ck_tile::index_t byte = kk & 15;
    const uint8_t packed = block.qs[byte];
    const int q = kk < 16 ? static_cast<int>(packed & 0x0f) : static_cast<int>(packed >> 4);
    return gguf_half_to_float(block.d) * static_cast<float>(q - 8);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const float d = gguf_half_to_float(block.d);
    const bool high_nibble = kk_base >= 16;
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const uint8_t packed = block.qs[(kk_base + i) & 15];
      const int q = high_nibble ? static_cast<int>(packed >> 4) : static_cast<int>(packed & 0x0f);
      set_value(i, d * static_cast<float>(q - 8));
    });
  }
};

struct GgufQ4_KTraits {
  using Block = GgufQ4_KBlock;
  static constexpr ck_tile::index_t BlockSize = kQKBlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const ck_tile::index_t group64 = kk >> 6;
    const ck_tile::index_t local64 = kk & 63;
    const ck_tile::index_t high_nibble = local64 >= 32 ? 1 : 0;
    const ck_tile::index_t local32 = local64 & 31;
    const ck_tile::index_t scale_index = group64 * 2 + high_nibble;
    uint8_t sc = 0;
    uint8_t m = 0;
    gguf_get_scale_min_k4(scale_index, block.scales, sc, m);
    const uint8_t packed = block.qs[group64 * 32 + local32];
    const int q = high_nibble ? static_cast<int>(packed >> 4) : static_cast<int>(packed & 0x0f);
    return gguf_half_to_float(block.d) * static_cast<float>(sc) * static_cast<float>(q) -
           gguf_half_to_float(block.dmin) * static_cast<float>(m);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const ck_tile::index_t group64 = kk_base >> 6;
    const ck_tile::index_t local64 = kk_base & 63;
    const ck_tile::index_t high_nibble = local64 >= 32 ? 1 : 0;
    const ck_tile::index_t local32_base = local64 & 31;
    const ck_tile::index_t scale_index = group64 * 2 + high_nibble;
    uint8_t sc = 0;
    uint8_t m = 0;
    gguf_get_scale_min_k4(scale_index, block.scales, sc, m);
    const float d = gguf_half_to_float(block.d) * static_cast<float>(sc);
    const float min = gguf_half_to_float(block.dmin) * static_cast<float>(m);
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const ck_tile::index_t local32 = local32_base + i;
      const uint8_t packed = block.qs[group64 * 32 + local32];
      const int q = high_nibble ? static_cast<int>(packed >> 4) : static_cast<int>(packed & 0x0f);
      set_value(i, d * static_cast<float>(q) - min);
    });
  }
};

struct GgufQ5_KTraits {
  using Block = GgufQ5_KBlock;
  static constexpr ck_tile::index_t BlockSize = kQKBlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const ck_tile::index_t group64 = kk >> 6;
    const ck_tile::index_t local64 = kk & 63;
    const ck_tile::index_t high_nibble = local64 >= 32 ? 1 : 0;
    const ck_tile::index_t local32 = local64 & 31;
    const ck_tile::index_t scale_index = group64 * 2 + high_nibble;
    uint8_t sc = 0;
    uint8_t m = 0;
    gguf_get_scale_min_k4(scale_index, block.scales, sc, m);
    const uint8_t packed = block.qs[group64 * 32 + local32];
    const uint8_t bit = static_cast<uint8_t>((high_nibble ? 2u : 1u) << (2 * group64));
    const int low = high_nibble ? static_cast<int>(packed >> 4) : static_cast<int>(packed & 0x0f);
    const int q = low + ((block.qh[local32] & bit) ? 16 : 0);
    return gguf_half_to_float(block.d) * static_cast<float>(sc) * static_cast<float>(q) -
           gguf_half_to_float(block.dmin) * static_cast<float>(m);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const ck_tile::index_t group64 = kk_base >> 6;
    const ck_tile::index_t local64 = kk_base & 63;
    const ck_tile::index_t high_nibble = local64 >= 32 ? 1 : 0;
    const ck_tile::index_t local32_base = local64 & 31;
    const ck_tile::index_t scale_index = group64 * 2 + high_nibble;
    uint8_t sc = 0;
    uint8_t m = 0;
    gguf_get_scale_min_k4(scale_index, block.scales, sc, m);
    const float d = gguf_half_to_float(block.d) * static_cast<float>(sc);
    const float min = gguf_half_to_float(block.dmin) * static_cast<float>(m);
    const uint8_t bit = static_cast<uint8_t>((high_nibble ? 2u : 1u) << (2 * group64));
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const ck_tile::index_t local32 = local32_base + i;
      const uint8_t packed = block.qs[group64 * 32 + local32];
      const int low = high_nibble ? static_cast<int>(packed >> 4) : static_cast<int>(packed & 0x0f);
      const int q = low + ((block.qh[local32] & bit) ? 16 : 0);
      set_value(i, d * static_cast<float>(q) - min);
    });
  }
};

struct GgufQ2_KTraits {
  using Block = GgufQ2_KBlock;
  static constexpr ck_tile::index_t BlockSize = kQKBlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const ck_tile::index_t half_block = kk >> 7;
    const ck_tile::index_t local128 = kk & 127;
    const ck_tile::index_t j = local128 >> 5;
    const ck_tile::index_t local32 = local128 & 31;
    const ck_tile::index_t scale_index = half_block * 8 + 2 * j + (local32 >= 16 ? 1 : 0);
    const int q = static_cast<int>((block.qs[half_block * 32 + local32] >> (2 * j)) & 0x03);
    return gguf_half_to_float(block.d) *
               static_cast<float>(block.scales[scale_index] & 0x0f) * static_cast<float>(q) -
           gguf_half_to_float(block.dmin) * static_cast<float>(block.scales[scale_index] >> 4);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const ck_tile::index_t half_block = kk_base >> 7;
    const ck_tile::index_t local128 = kk_base & 127;
    const ck_tile::index_t j = local128 >> 5;
    const ck_tile::index_t local32_base = local128 & 31;
    const ck_tile::index_t scale_index = half_block * 8 + 2 * j + (local32_base >= 16 ? 1 : 0);
    const float d = gguf_half_to_float(block.d) * static_cast<float>(block.scales[scale_index] & 0x0f);
    const float min = gguf_half_to_float(block.dmin) * static_cast<float>(block.scales[scale_index] >> 4);
    const ck_tile::index_t qs_base = half_block * 32 + local32_base;
    const ck_tile::index_t shift = 2 * j;
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const int q = static_cast<int>((block.qs[qs_base + i] >> shift) & 0x03);
      set_value(i, d * static_cast<float>(q) - min);
    });
  }
};

struct GgufQ3_KTraits {
  using Block = GgufQ3_KBlock;
  static constexpr ck_tile::index_t BlockSize = kQKBlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const ck_tile::index_t half_block = kk >> 7;
    const ck_tile::index_t local128 = kk & 127;
    const ck_tile::index_t j = local128 >> 5;
    const ck_tile::index_t local32 = local128 & 31;
    const ck_tile::index_t is0 = local32 >= 16 ? 1 : 0;
    const ck_tile::index_t is = 8 * half_block + 2 * j + is0;
    const int scale = gguf_q3_k_scale(block.scales, is) - 32;
    const uint8_t mask = static_cast<uint8_t>(1u << (4 * half_block + j));
    const int low = static_cast<int>((block.qs[half_block * 32 + local32] >> (2 * j)) & 0x03);
    const int q = low - ((block.hmask[local32] & mask) ? 0 : 4);
    return gguf_half_to_float(block.d) * static_cast<float>(scale) * static_cast<float>(q);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const ck_tile::index_t half_block = kk_base >> 7;
    const ck_tile::index_t local128 = kk_base & 127;
    const ck_tile::index_t j = local128 >> 5;
    const ck_tile::index_t local32_base = local128 & 31;
    const ck_tile::index_t is = 8 * half_block + 2 * j + (local32_base >= 16 ? 1 : 0);
    const float d = gguf_half_to_float(block.d) * static_cast<float>(gguf_q3_k_scale(block.scales, is) - 32);
    const uint8_t mask = static_cast<uint8_t>(1u << (4 * half_block + j));
    const ck_tile::index_t qs_base = half_block * 32 + local32_base;
    const ck_tile::index_t shift = 2 * j;
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const ck_tile::index_t local32 = local32_base + i;
      const int low = static_cast<int>((block.qs[qs_base + i] >> shift) & 0x03);
      const int q = low - ((block.hmask[local32] & mask) ? 0 : 4);
      set_value(i, d * static_cast<float>(q));
    });
  }
};

struct GgufQ6_KTraits {
  using Block = GgufQ6_KBlock;
  static constexpr ck_tile::index_t BlockSize = kQKBlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const ck_tile::index_t half_block = kk >> 7;
    const ck_tile::index_t local128 = kk & 127;
    const ck_tile::index_t quad = local128 >> 5;
    const ck_tile::index_t local32 = local128 & 31;
    const ck_tile::index_t ql_index = half_block * 64 + ((quad & 1) ? 32 : 0) + local32;
    const ck_tile::index_t qh_index = half_block * 32 + local32;
    const ck_tile::index_t scale_index = half_block * 8 + 2 * quad + (local32 >= 16 ? 1 : 0);
    const int low = quad >= 2 ? static_cast<int>(block.ql[ql_index] >> 4)
                              : static_cast<int>(block.ql[ql_index] & 0x0f);
    const int high = static_cast<int>((block.qh[qh_index] >> (2 * quad)) & 0x03) << 4;
    const int q = (low | high) - 32;
    return gguf_half_to_float(block.d) *
           static_cast<float>(block.scales[scale_index]) * static_cast<float>(q);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const ck_tile::index_t half_block = kk_base >> 7;
    const ck_tile::index_t local128 = kk_base & 127;
    const ck_tile::index_t quad = local128 >> 5;
    const ck_tile::index_t local32_base = local128 & 31;
    const ck_tile::index_t ql_base = half_block * 64 + ((quad & 1) ? 32 : 0) + local32_base;
    const ck_tile::index_t qh_base = half_block * 32 + local32_base;
    const ck_tile::index_t scale_index = half_block * 8 + 2 * quad + (local32_base >= 16 ? 1 : 0);
    const float d = gguf_half_to_float(block.d) * static_cast<float>(block.scales[scale_index]);
    const ck_tile::index_t qh_shift = 2 * quad;
    const bool high_nibble = quad >= 2;
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const int low = high_nibble ? static_cast<int>(block.ql[ql_base + i] >> 4)
                                  : static_cast<int>(block.ql[ql_base + i] & 0x0f);
      const int high = static_cast<int>((block.qh[qh_base + i] >> qh_shift) & 0x03) << 4;
      const int q = (low | high) - 32;
      set_value(i, d * static_cast<float>(q));
    });
  }
};

struct GgufMxFp4Traits {
  using Block = GgufMxFp4Block;
  static constexpr ck_tile::index_t BlockSize = kMxFp4BlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const uint8_t packed = block.qs[kk & 15];
    const int q = gguf_fp4_quant_value(kk < 16 ? (packed & 0x0f) : (packed >> 4));
    return gguf_e8m0_to_fp32_half(block.e) * static_cast<float>(q);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const float d = gguf_e8m0_to_fp32_half(block.e);
    const bool high_nibble = kk_base >= 16;
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const uint8_t packed = block.qs[(kk_base + i) & 15];
      const int q = gguf_fp4_quant_value(high_nibble ? (packed >> 4) : (packed & 0x0f));
      set_value(i, d * static_cast<float>(q));
    });
  }
};

struct GgufNvFp4Traits {
  using Block = GgufNvFp4Block;
  static constexpr ck_tile::index_t BlockSize = kNvFp4BlockSize;

  CK_TILE_DEVICE static float value(const Block& block, ck_tile::index_t kk) {
    const ck_tile::index_t sub = kk >> 4;
    const ck_tile::index_t local16 = kk & 15;
    const uint8_t packed = block.qs[sub * 8 + (local16 & 7)];
    const int q = gguf_fp4_quant_value(local16 < 8 ? (packed & 0x0f) : (packed >> 4));
    return gguf_ue4m3_to_fp32(block.d[sub]) * static_cast<float>(q);
  }

  template <typename SetValue>
  CK_TILE_DEVICE static void values8(const Block& block, ck_tile::index_t kk_base, const SetValue& set_value) {
    const ck_tile::index_t sub = kk_base >> 4;
    const bool high_nibble = (kk_base & 15) >= 8;
    const float d = gguf_ue4m3_to_fp32(block.d[sub]);
    ck_tile::static_for<0, 8, 1>{}([&](auto i) {
      const uint8_t packed = block.qs[sub * 8 + ((kk_base + i) & 7)];
      const int q = gguf_fp4_quant_value(high_nibble ? (packed >> 4) : (packed & 0x0f));
      set_value(i, d * static_cast<float>(q));
    });
  }
};

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t RowsPerWave,
    ck_tile::index_t ColsPerWave>
struct GgufQuantGemmRcrWaveMNKernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t group = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t col_base = group * ColsPerWave;
    const ck_tile::index_t row_base = static_cast<ck_tile::index_t>(blockIdx.y) * RowsPerWave;
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;
    float partial[ColsPerWave][RowsPerWave] = {};

    if (row_base < args.m && col_base < args.n) {
      for (ck_tile::index_t qb = 0; qb < qblocks_per_row; ++qb) {
        for (ck_tile::index_t kk = lane; kk < Traits::BlockSize; kk += warp_size) {
          const ck_tile::index_t global_k = qb * Traits::BlockSize + kk;
          float a_values[RowsPerWave];
#pragma unroll
          for (ck_tile::index_t row_offset = 0; row_offset < RowsPerWave; ++row_offset) {
            const ck_tile::index_t row = row_base + row_offset;
            a_values[row_offset] =
                row < args.m ? storage_to_float(args.a[row * args.k + global_k]) : 0.0f;
          }
#pragma unroll
          for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
            const ck_tile::index_t col = col_base + col_offset;
            if (col < args.n) {
              const Block& b_block = args.b[col * qblocks_per_row + qb];
              const float b_value = Traits::value(b_block, kk);
#pragma unroll
              for (ck_tile::index_t row_offset = 0; row_offset < RowsPerWave; ++row_offset) {
                partial[col_offset][row_offset] += a_values[row_offset] * b_value;
              }
            }
          }
        }
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const auto shuffle_offset = static_cast<uint32_t>(offset);
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
#pragma unroll
        for (ck_tile::index_t row_offset = 0; row_offset < RowsPerWave; ++row_offset) {
          const float remote =
              ck_tile::warp_shuffle_down(partial[col_offset][row_offset], shuffle_offset);
          if (lane < offset) {
            partial[col_offset][row_offset] += remote;
          }
        }
      }
    }

    if (lane == 0) {
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const ck_tile::index_t col = col_base + col_offset;
        if (col < args.n) {
#pragma unroll
          for (ck_tile::index_t row_offset = 0; row_offset < RowsPerWave; ++row_offset) {
            const ck_tile::index_t row = row_base + row_offset;
            if (row < args.m) {
              args.c[row * args.n + col] =
                  float_to_storage<CStorage>(partial[col_offset][row_offset]);
            }
          }
        }
      }
    }
  }
};

template <typename Traits, typename AStorage, typename CStorage, ck_tile::index_t ColsPerWave>
struct GgufQuantGemmRcrWaveM4NExactKernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t group = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t col_base = group * ColsPerWave;
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;
    float partial[ColsPerWave][4] = {};

    if (col_base < args.n) {
      for (ck_tile::index_t qb = 0; qb < qblocks_per_row; ++qb) {
        for (ck_tile::index_t kk = lane; kk < Traits::BlockSize; kk += warp_size) {
          const ck_tile::index_t global_k = qb * Traits::BlockSize + kk;
          const float a0 = storage_to_float(args.a[global_k]);
          const float a1 = storage_to_float(args.a[args.k + global_k]);
          const float a2 = storage_to_float(args.a[2 * args.k + global_k]);
          const float a3 = storage_to_float(args.a[3 * args.k + global_k]);
#pragma unroll
          for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
            const ck_tile::index_t col = col_base + col_offset;
            if (col < args.n) {
              const Block& b_block = args.b[col * qblocks_per_row + qb];
              const float b_value = Traits::value(b_block, kk);
              partial[col_offset][0] += a0 * b_value;
              partial[col_offset][1] += a1 * b_value;
              partial[col_offset][2] += a2 * b_value;
              partial[col_offset][3] += a3 * b_value;
            }
          }
        }
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const auto shuffle_offset = static_cast<uint32_t>(offset);
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const float remote0 = ck_tile::warp_shuffle_down(partial[col_offset][0], shuffle_offset);
        const float remote1 = ck_tile::warp_shuffle_down(partial[col_offset][1], shuffle_offset);
        const float remote2 = ck_tile::warp_shuffle_down(partial[col_offset][2], shuffle_offset);
        const float remote3 = ck_tile::warp_shuffle_down(partial[col_offset][3], shuffle_offset);
        if (lane < offset) {
          partial[col_offset][0] += remote0;
          partial[col_offset][1] += remote1;
          partial[col_offset][2] += remote2;
          partial[col_offset][3] += remote3;
        }
      }
    }

    if (lane == 0) {
#pragma unroll
      for (ck_tile::index_t col_offset = 0; col_offset < ColsPerWave; ++col_offset) {
        const ck_tile::index_t col = col_base + col_offset;
        if (col < args.n) {
          args.c[col] = float_to_storage<CStorage>(partial[col_offset][0]);
          args.c[args.n + col] = float_to_storage<CStorage>(partial[col_offset][1]);
          args.c[2 * args.n + col] = float_to_storage<CStorage>(partial[col_offset][2]);
          args.c[3 * args.n + col] = float_to_storage<CStorage>(partial[col_offset][3]);
        }
      }
    }
  }
};

template <typename Traits, typename AStorage, typename CStorage>
struct GgufQuantGemmRcrWaveKernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t row = static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id;
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;

    float partial = 0.0f;
    if (row < args.m && col < args.n) {
      for (ck_tile::index_t qb = 0; qb < qblocks_per_row; ++qb) {
        const Block& b_block = args.b[col * qblocks_per_row + qb];
        for (ck_tile::index_t kk = lane; kk < Traits::BlockSize; kk += warp_size) {
          const ck_tile::index_t global_k = qb * Traits::BlockSize + kk;
          partial += storage_to_float(args.a[row * args.k + global_k]) * Traits::value(b_block, kk);
        }
      }
    }

    for (ck_tile::index_t offset = warp_size / 2; offset > 0; offset /= 2) {
      const float remote = ck_tile::warp_shuffle_down(partial, static_cast<uint32_t>(offset));
      if (lane < offset) {
        partial += remote;
      }
    }

    if (lane == 0 && row < args.m && col < args.n) {
      args.c[row * args.n + col] = float_to_storage<CStorage>(partial);
    }
  }
};

template <typename Traits, typename AStorage, typename CStorage, bool DirectFill>
struct GgufQuantGemmRcrWmma16x16Kernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;
  static constexpr ck_tile::index_t kMPerWave = 16;
  static constexpr ck_tile::index_t kNPerWave = 16;
  static constexpr ck_tile::index_t kKPerWave = 16;
  using WarpGemm = ck_tile::WarpGemmDispatcher<
      ck_tile::half_t,
      ck_tile::half_t,
      float,
      kMPerWave,
      kNPerWave,
      kKPerWave,
      false>;

  CK_TILE_DEVICE void operator()(GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= kWavesPerBlock) {
      return;
    }

    const ck_tile::index_t row_base = static_cast<ck_tile::index_t>(blockIdx.y) * kMPerWave;
    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t col_base =
        (static_cast<ck_tile::index_t>(blockIdx.x) * kWavesPerBlock + wave_id) * kNPerWave;
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;

    auto c_warp = ck_tile::make_static_distributed_tensor<float>(typename WarpGemm::CWarpDstr{});
    ck_tile::static_for<0, decltype(c_warp)::get_thread_buffer_size(), 1>{}(
        [&](auto i) { c_warp.get_thread_buffer()(i) = 0.0f; });

    for (ck_tile::index_t k_base = 0; k_base < args.k; k_base += kKPerWave) {
      auto a_warp =
          ck_tile::make_static_distributed_tensor<ck_tile::half_t>(typename WarpGemm::AWarpDstr{});
      auto b_warp =
          ck_tile::make_static_distributed_tensor<ck_tile::half_t>(typename WarpGemm::BWarpDstr{});

      if constexpr (DirectFill) {
        static_assert(decltype(a_warp)::get_thread_buffer_size() == 8);
        static_assert(decltype(b_warp)::get_thread_buffer_size() == 8);
        const ck_tile::index_t lane_mn = lane & 15;
        const ck_tile::index_t lane_k_base = (lane >> 4) * 8;
        ck_tile::static_for<0, 8, 1>{}([&](auto i) {
          const ck_tile::index_t row = row_base + lane_mn;
          const ck_tile::index_t global_k = k_base + lane_k_base + i;
          const float a_value =
              row < args.m ? storage_to_float(args.a[row * args.k + global_k]) : 0.0f;
          a_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(a_value);
        });
        const ck_tile::index_t qb = k_base / Traits::BlockSize;
        const ck_tile::index_t kk_base = k_base - qb * Traits::BlockSize + lane_k_base;
        const ck_tile::index_t col = col_base + lane_mn;
        if (col < args.n) {
          Traits::values8(args.b[col * qblocks_per_row + qb], kk_base, [&](auto i, float b_value) {
            b_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(b_value);
          });
        } else {
          ck_tile::static_for<0, 8, 1>{}([&](auto i) {
            b_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(0.0f);
          });
        }
      } else {
        constexpr auto a_spans = decltype(a_warp)::get_distributed_spans();
        sweep_gguf_distributed_span(a_spans[ck_tile::number<0>{}], [&](auto idx0) {
          sweep_gguf_distributed_span(a_spans[ck_tile::number<1>{}], [&](auto idx1) {
            constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
            const auto x_indices =
                ck_tile::get_x_indices_from_distributed_indices(a_warp.get_tile_distribution(), idx);
            const ck_tile::index_t row = row_base + x_indices.at(ck_tile::number<0>{});
            const ck_tile::index_t global_k = k_base + x_indices.at(ck_tile::number<1>{});
            const float value =
                row < args.m ? storage_to_float(args.a[row * args.k + global_k]) : 0.0f;
            a_warp(idx) = ck_tile::type_convert<ck_tile::half_t>(value);
          });
        });

        constexpr auto b_spans = decltype(b_warp)::get_distributed_spans();
        const ck_tile::index_t qb = k_base / Traits::BlockSize;
        const ck_tile::index_t kk_base = k_base - qb * Traits::BlockSize;
        sweep_gguf_distributed_span(b_spans[ck_tile::number<0>{}], [&](auto idx0) {
          sweep_gguf_distributed_span(b_spans[ck_tile::number<1>{}], [&](auto idx1) {
            constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
            const auto x_indices =
                ck_tile::get_x_indices_from_distributed_indices(b_warp.get_tile_distribution(), idx);
            const ck_tile::index_t col = col_base + x_indices.at(ck_tile::number<0>{});
            float value = 0.0f;
            if (col < args.n) {
              value =
                  Traits::value(args.b[col * qblocks_per_row + qb], kk_base + x_indices.at(ck_tile::number<1>{}));
            }
            b_warp(idx) = ck_tile::type_convert<ck_tile::half_t>(value);
          });
        });
      }

      WarpGemm{}(c_warp, a_warp, b_warp);
    }

    constexpr auto c_spans = decltype(c_warp)::get_distributed_spans();
    sweep_gguf_distributed_span(c_spans[ck_tile::number<0>{}], [&](auto idx0) {
      sweep_gguf_distributed_span(c_spans[ck_tile::number<1>{}], [&](auto idx1) {
        constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
        const auto x_indices =
            ck_tile::get_x_indices_from_distributed_indices(c_warp.get_tile_distribution(), idx);
        const ck_tile::index_t row = row_base + x_indices.at(ck_tile::number<0>{});
        const ck_tile::index_t col = col_base + x_indices.at(ck_tile::number<1>{});
        if (row < args.m && col < args.n) {
          args.c[row * args.n + col] = float_to_storage<CStorage>(c_warp[idx]);
        }
      });
    });
  }
};

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t MBlocksPerWave,
    ck_tile::index_t WavesPerBlock = kWavesPerBlock>
struct GgufQuantGemmRcrWmma16x16MBlocksKernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;
  static constexpr ck_tile::index_t kMPerWave = 16;
  static constexpr ck_tile::index_t kNPerWave = 16;
  static constexpr ck_tile::index_t kKPerWave = 16;
  using WarpGemm = ck_tile::WarpGemmDispatcher<
      ck_tile::half_t,
      ck_tile::half_t,
      float,
      kMPerWave,
      kNPerWave,
      kKPerWave,
      false>;

  CK_TILE_DEVICE void operator()(GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= WavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t lane_mn = lane & 15;
    const ck_tile::index_t lane_k_base = (lane >> 4) * 8;
    const ck_tile::index_t row_base =
        static_cast<ck_tile::index_t>(blockIdx.y) * (MBlocksPerWave * kMPerWave);
    const ck_tile::index_t col_base =
        (static_cast<ck_tile::index_t>(blockIdx.x) * WavesPerBlock + wave_id) * kNPerWave;
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;

    using CWarp = typename WarpGemm::CWarpTensor;
    CWarp c_warps[MBlocksPerWave];
    ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
      ck_tile::static_for<0, CWarp::get_thread_buffer_size(), 1>{}(
          [&](auto i) { c_warps[mb].get_thread_buffer()(i) = 0.0f; });
    });

    for (ck_tile::index_t k_base = 0; k_base < args.k; k_base += kKPerWave) {
      auto b_warp =
          ck_tile::make_static_distributed_tensor<ck_tile::half_t>(typename WarpGemm::BWarpDstr{});
      const ck_tile::index_t qb = k_base / Traits::BlockSize;
      const ck_tile::index_t kk_base = k_base - qb * Traits::BlockSize + lane_k_base;
      const ck_tile::index_t col = col_base + lane_mn;
      if (col < args.n) {
        Traits::values8(args.b[col * qblocks_per_row + qb], kk_base, [&](auto i, float b_value) {
          b_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(b_value);
        });
      } else {
        ck_tile::static_for<0, 8, 1>{}([&](auto i) {
          b_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(0.0f);
        });
      }

      ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
        auto a_warp =
            ck_tile::make_static_distributed_tensor<ck_tile::half_t>(typename WarpGemm::AWarpDstr{});
        ck_tile::static_for<0, 8, 1>{}([&](auto i) {
          const ck_tile::index_t row = row_base + mb * kMPerWave + lane_mn;
          const ck_tile::index_t global_k = k_base + lane_k_base + i;
          const float a_value =
              row < args.m ? storage_to_float(args.a[row * args.k + global_k]) : 0.0f;
          a_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(a_value);
        });
        WarpGemm{}(c_warps[mb], a_warp, b_warp);
      });
    }

    constexpr auto c_spans = CWarp::get_distributed_spans();
    ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
      sweep_gguf_distributed_span(c_spans[ck_tile::number<0>{}], [&](auto idx0) {
        sweep_gguf_distributed_span(c_spans[ck_tile::number<1>{}], [&](auto idx1) {
          constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
          const auto x_indices =
              ck_tile::get_x_indices_from_distributed_indices(c_warps[mb].get_tile_distribution(), idx);
          const ck_tile::index_t row =
              row_base + mb * kMPerWave + x_indices.at(ck_tile::number<0>{});
          const ck_tile::index_t col = col_base + x_indices.at(ck_tile::number<1>{});
          if (row < args.m && col < args.n) {
            args.c[row * args.n + col] = float_to_storage<CStorage>(c_warps[mb][idx]);
          }
        });
      });
    });
  }
};

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t MBlocksPerWave,
    ck_tile::index_t NBlocksPerWave,
    ck_tile::index_t WavesPerBlock = kWavesPerBlock>
struct GgufQuantGemmRcrWmma16x16MNBlocksKernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;
  static constexpr ck_tile::index_t kMPerWave = 16;
  static constexpr ck_tile::index_t kNPerWave = 16;
  static constexpr ck_tile::index_t kKPerWave = 16;
  using WarpGemm = ck_tile::WarpGemmDispatcher<
      ck_tile::half_t,
      ck_tile::half_t,
      float,
      kMPerWave,
      kNPerWave,
      kKPerWave,
      false>;

  CK_TILE_DEVICE void operator()(GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args) const {
    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t wave_id = static_cast<ck_tile::index_t>(threadIdx.x) / warp_size;
    if (wave_id >= WavesPerBlock) {
      return;
    }

    const ck_tile::index_t lane = static_cast<ck_tile::index_t>(threadIdx.x) - wave_id * warp_size;
    const ck_tile::index_t lane_mn = lane & 15;
    const ck_tile::index_t lane_k_base = (lane >> 4) * 8;
    const ck_tile::index_t row_base =
        static_cast<ck_tile::index_t>(blockIdx.y) * (MBlocksPerWave * kMPerWave);
    const ck_tile::index_t col_base =
        (static_cast<ck_tile::index_t>(blockIdx.x) * WavesPerBlock + wave_id) *
        (NBlocksPerWave * kNPerWave);
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;

    using CWarp = typename WarpGemm::CWarpTensor;
    CWarp c_warps[MBlocksPerWave][NBlocksPerWave];
    ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
      ck_tile::static_for<0, NBlocksPerWave, 1>{}([&](auto nb) {
        ck_tile::static_for<0, CWarp::get_thread_buffer_size(), 1>{}(
            [&](auto i) { c_warps[mb][nb].get_thread_buffer()(i) = 0.0f; });
      });
    });

    for (ck_tile::index_t k_base = 0; k_base < args.k; k_base += kKPerWave) {
      typename WarpGemm::BWarpTensor b_warps[NBlocksPerWave];
      const ck_tile::index_t qb = k_base / Traits::BlockSize;
      const ck_tile::index_t kk_base = k_base - qb * Traits::BlockSize + lane_k_base;
      ck_tile::static_for<0, NBlocksPerWave, 1>{}([&](auto nb) {
        const ck_tile::index_t col = col_base + nb * kNPerWave + lane_mn;
        if (col < args.n) {
          Traits::values8(args.b[col * qblocks_per_row + qb], kk_base, [&](auto i, float b_value) {
            b_warps[nb].get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(b_value);
          });
        } else {
          ck_tile::static_for<0, 8, 1>{}([&](auto i) {
            b_warps[nb].get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(0.0f);
          });
        }
      });

      ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
        auto a_warp =
            ck_tile::make_static_distributed_tensor<ck_tile::half_t>(typename WarpGemm::AWarpDstr{});
        ck_tile::static_for<0, 8, 1>{}([&](auto i) {
          const ck_tile::index_t row = row_base + mb * kMPerWave + lane_mn;
          const ck_tile::index_t global_k = k_base + lane_k_base + i;
          const float a_value =
              row < args.m ? storage_to_float(args.a[row * args.k + global_k]) : 0.0f;
          a_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(a_value);
        });
        ck_tile::static_for<0, NBlocksPerWave, 1>{}([&](auto nb) {
          WarpGemm{}(c_warps[mb][nb], a_warp, b_warps[nb]);
        });
      });
    }

    constexpr auto c_spans = CWarp::get_distributed_spans();
    ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
      ck_tile::static_for<0, NBlocksPerWave, 1>{}([&](auto nb) {
        sweep_gguf_distributed_span(c_spans[ck_tile::number<0>{}], [&](auto idx0) {
          sweep_gguf_distributed_span(c_spans[ck_tile::number<1>{}], [&](auto idx1) {
            constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
            const auto x_indices =
                ck_tile::get_x_indices_from_distributed_indices(c_warps[mb][nb].get_tile_distribution(), idx);
            const ck_tile::index_t row =
                row_base + mb * kMPerWave + x_indices.at(ck_tile::number<0>{});
            const ck_tile::index_t col =
                col_base + nb * kNPerWave + x_indices.at(ck_tile::number<1>{});
            if (row < args.m && col < args.n) {
              args.c[row * args.n + col] = float_to_storage<CStorage>(c_warps[mb][nb][idx]);
            }
          });
        });
      });
    });
  }
};

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t MBlocksPerWave,
    ck_tile::index_t WavesPerBlock = kWavesPerBlock>
struct GgufQuantGemmRcrWmma16x16SharedBKernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;
  static constexpr ck_tile::index_t kMPerWave = 16;
  static constexpr ck_tile::index_t kNPerBlock = 16;
  static constexpr ck_tile::index_t kKPerWave = 16;
  using WarpGemm = ck_tile::WarpGemmDispatcher<
      ck_tile::half_t,
      ck_tile::half_t,
      float,
      kMPerWave,
      kNPerBlock,
      kKPerWave,
      false>;

  CK_TILE_DEVICE void operator()(GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args) const {
    __shared__ ck_tile::half_t b_shared[kNPerBlock][kKPerWave];

    const ck_tile::index_t warp_size = ck_tile::get_warp_size();
    const ck_tile::index_t tid = static_cast<ck_tile::index_t>(threadIdx.x);
    const ck_tile::index_t wave_id = tid / warp_size;
    const ck_tile::index_t lane = tid - wave_id * warp_size;
    const ck_tile::index_t lane_mn = lane & 15;
    const ck_tile::index_t lane_k_base = (lane >> 4) * 8;
    const ck_tile::index_t row_base =
        static_cast<ck_tile::index_t>(blockIdx.y) * (WavesPerBlock * MBlocksPerWave * kMPerWave) +
        wave_id * MBlocksPerWave * kMPerWave;
    const ck_tile::index_t col_base = static_cast<ck_tile::index_t>(blockIdx.x) * kNPerBlock;
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;

    using CWarp = typename WarpGemm::CWarpTensor;
    CWarp c_warps[MBlocksPerWave];
    ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
      ck_tile::static_for<0, CWarp::get_thread_buffer_size(), 1>{}(
          [&](auto i) { c_warps[mb].get_thread_buffer()(i) = 0.0f; });
    });

    for (ck_tile::index_t k_base = 0; k_base < args.k; k_base += kKPerWave) {
      for (ck_tile::index_t idx = tid; idx < kNPerBlock * kKPerWave; idx += static_cast<ck_tile::index_t>(blockDim.x)) {
        const ck_tile::index_t local_n = idx / kKPerWave;
        const ck_tile::index_t local_k = idx - local_n * kKPerWave;
        const ck_tile::index_t col = col_base + local_n;
        const ck_tile::index_t global_k = k_base + local_k;
        float value = 0.0f;
        if (col < args.n) {
          const ck_tile::index_t qb = global_k / Traits::BlockSize;
          const ck_tile::index_t kk = global_k - qb * Traits::BlockSize;
          value = Traits::value(args.b[col * qblocks_per_row + qb], kk);
        }
        b_shared[local_n][local_k] = ck_tile::type_convert<ck_tile::half_t>(value);
      }
      __syncthreads();

      auto b_warp =
          ck_tile::make_static_distributed_tensor<ck_tile::half_t>(typename WarpGemm::BWarpDstr{});
      ck_tile::static_for<0, 8, 1>{}([&](auto i) {
        b_warp.get_thread_buffer()(i) = b_shared[lane_mn][lane_k_base + i];
      });

      ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
        auto a_warp =
            ck_tile::make_static_distributed_tensor<ck_tile::half_t>(typename WarpGemm::AWarpDstr{});
        ck_tile::static_for<0, 8, 1>{}([&](auto i) {
          const ck_tile::index_t row = row_base + mb * kMPerWave + lane_mn;
          const ck_tile::index_t global_k = k_base + lane_k_base + i;
          const float a_value =
              row < args.m ? storage_to_float(args.a[row * args.k + global_k]) : 0.0f;
          a_warp.get_thread_buffer()(i) = ck_tile::type_convert<ck_tile::half_t>(a_value);
        });
        WarpGemm{}(c_warps[mb], a_warp, b_warp);
      });
      __syncthreads();
    }

    constexpr auto c_spans = CWarp::get_distributed_spans();
    ck_tile::static_for<0, MBlocksPerWave, 1>{}([&](auto mb) {
      sweep_gguf_distributed_span(c_spans[ck_tile::number<0>{}], [&](auto idx0) {
        sweep_gguf_distributed_span(c_spans[ck_tile::number<1>{}], [&](auto idx1) {
          constexpr auto idx = ck_tile::make_tuple(idx0, idx1);
          const auto x_indices =
              ck_tile::get_x_indices_from_distributed_indices(c_warps[mb].get_tile_distribution(), idx);
          const ck_tile::index_t row =
              row_base + mb * kMPerWave + x_indices.at(ck_tile::number<0>{});
          const ck_tile::index_t col = col_base + x_indices.at(ck_tile::number<1>{});
          if (row < args.m && col < args.n) {
            args.c[row * args.n + col] = float_to_storage<CStorage>(c_warps[mb][idx]);
          }
        });
      });
    });
  }
};

template <typename Traits, typename Storage, bool KnLayout>
struct GgufQuantDequantKernel {
  using Block = typename Traits::Block;
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQuantDequantArgs<Block, Storage> args) const {
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x);
    const ck_tile::index_t qb = static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t qblocks_per_row = args.k / Traits::BlockSize;
    if (col >= args.n || qb >= qblocks_per_row) {
      return;
    }

    const Block& block = args.b[col * qblocks_per_row + qb];
    for (ck_tile::index_t kk = static_cast<ck_tile::index_t>(threadIdx.x); kk < Traits::BlockSize;
         kk += kBlockSize) {
      const ck_tile::index_t global_k = qb * Traits::BlockSize + kk;
      const float value = Traits::value(block, kk);
      if constexpr (KnLayout) {
        args.b_dense[global_k * args.n + col] = float_to_storage<Storage>(value);
      } else {
        args.b_dense[col * args.k + global_k] = float_to_storage<Storage>(value);
      }
    }
  }
};

template <typename BStorage, typename ScaleStorage>
struct GgufQ8_0TranscodeBqKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0TranscodeBqArgs<BStorage, ScaleStorage> args) const {
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x);
    const ck_tile::index_t qb = static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;
    if (col >= args.n || qb >= qblocks_per_row) {
      return;
    }

    const GgufQ8_0Block& block = args.b[col * qblocks_per_row + qb];
    if (threadIdx.x == 0) {
      args.bq_qk_n[qb + col * qblocks_per_row] =
          scale_to_storage<ScaleStorage>(gguf_half_to_float(block.d));
    }
    for (ck_tile::index_t kk = static_cast<ck_tile::index_t>(threadIdx.x); kk < kQ8BlockSize;
         kk += kBlockSize) {
      const ck_tile::index_t global_k = qb * kQ8BlockSize + kk;
      args.b_kn[global_k * args.n + col] = q8_to_bstorage<BStorage>(block.qs[kk]);
    }
  }
};

template <typename Storage>
struct GgufQ8_0DequantNkKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0DequantNkArgs<Storage> args) const {
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x);
    const ck_tile::index_t qb = static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;
    if (col >= args.n || qb >= qblocks_per_row) {
      return;
    }

    const GgufQ8_0Block& block = args.b[col * qblocks_per_row + qb];
    const float scale = gguf_half_to_float(block.d);
    for (ck_tile::index_t kk = static_cast<ck_tile::index_t>(threadIdx.x); kk < kQ8BlockSize;
         kk += kBlockSize) {
      const ck_tile::index_t global_k = qb * kQ8BlockSize + kk;
      const float value = scale * static_cast<float>(block.qs[kk]);
      args.b_nk[col * args.k + global_k] = float_to_storage<Storage>(value);
    }
  }
};

template <typename Storage>
struct GgufQ8_0DequantKnKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(GgufQ8_0DequantNkArgs<Storage> args) const {
    const ck_tile::index_t col = static_cast<ck_tile::index_t>(blockIdx.x);
    const ck_tile::index_t qb = static_cast<ck_tile::index_t>(blockIdx.y);
    const ck_tile::index_t qblocks_per_row = args.k / kQ8BlockSize;
    if (col >= args.n || qb >= qblocks_per_row) {
      return;
    }

    const GgufQ8_0Block& block = args.b[col * qblocks_per_row + qb];
    const float scale = gguf_half_to_float(block.d);
    for (ck_tile::index_t kk = static_cast<ck_tile::index_t>(threadIdx.x); kk < kQ8BlockSize;
         kk += kBlockSize) {
      const ck_tile::index_t global_k = qb * kQ8BlockSize + kk;
      const float value = scale * static_cast<float>(block.qs[kk]);
      args.b_nk[global_k * args.n + col] = float_to_storage<Storage>(value);
    }
  }
};

template <typename AStorage, typename CStorage>
int launch_gguf_q8_0_gemm_rcr_wave(
    const AStorage* a,
    const void* b_q8_0,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  const auto n_tile = static_cast<unsigned int>((n + kWavesPerBlock - 1) / kWavesPerBlock);
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile, static_cast<unsigned int>(m));
  const dim3 block(block_size);
  const GgufQ8_0GemmRcrArgs<AStorage, CStorage> args{
      a,
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  ck_tile::launch_kernel(
      ck_tile::stream_config{stream},
      ck_tile::make_kernel<kBlockPerCu>(GgufQ8_0GemmRcrWaveKernel<AStorage, CStorage>{}, grid, block, 0, args));
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename AStorage, typename CStorage>
int launch_gguf_q8_0_gemm_rcr_wave_m4(
    const AStorage* a,
    const void* b_q8_0,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (a == nullptr || b_q8_0 == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || m > 4 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }
  const auto n_tile = static_cast<unsigned int>((n + kWavesPerBlock - 1) / kWavesPerBlock);
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile);
  const dim3 block(block_size);
  const GgufQ8_0GemmRcrArgs<AStorage, CStorage> args{
      a,
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  ck_tile::launch_kernel(
      ck_tile::stream_config{stream},
      ck_tile::make_kernel<kBlockPerCu>(
          GgufQ8_0GemmRcrWaveM4Kernel<AStorage, CStorage>{}, grid, block, 0, args));
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename AStorage, typename CStorage, ck_tile::index_t ColsPerWave>
int launch_gguf_q8_0_gemm_rcr_wave_m4n(
    const AStorage* a,
    const void* b_q8_0,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (a == nullptr || b_q8_0 == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || m > 4 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }
  const auto n_tile =
      static_cast<unsigned int>((n + kWavesPerBlock * ColsPerWave - 1) / (kWavesPerBlock * ColsPerWave));
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile);
  const dim3 block(block_size);
  const GgufQ8_0GemmRcrArgs<AStorage, CStorage> args{
      a,
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  ck_tile::launch_kernel(
      ck_tile::stream_config{stream},
      ck_tile::make_kernel<kBlockPerCu>(
          GgufQ8_0GemmRcrWaveM4NKernel<AStorage, CStorage, ColsPerWave>{}, grid, block, 0, args));
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename AStorage, typename CStorage>
int launch_gguf_q8_0_gemm_rcr_wave_m4n2_exact(
    const AStorage* a,
    const void* b_q8_0,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (a == nullptr || b_q8_0 == nullptr || c == nullptr) {
    return 1;
  }
  if (m != 4 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }
  const auto n_tile = static_cast<unsigned int>((n + kWavesPerBlock * 2 - 1) / (kWavesPerBlock * 2));
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile);
  const dim3 block(block_size);
  const GgufQ8_0GemmRcrArgs<AStorage, CStorage> args{
      a,
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  ck_tile::launch_kernel(
      ck_tile::stream_config{stream},
      ck_tile::make_kernel<kBlockPerCu>(
          GgufQ8_0GemmRcrWaveM4N2ExactKernel<AStorage, CStorage>{}, grid, block, 0, args));
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename AStorage, typename CStorage, ck_tile::index_t ColsPerWave>
int launch_gguf_q8_0_gemm_rcr_wave_m4n_exact(
    const AStorage* a,
    const void* b_q8_0,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (a == nullptr || b_q8_0 == nullptr || c == nullptr) {
    return 1;
  }
  if (m != 4 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }
  const auto n_tile =
      static_cast<unsigned int>((n + kWavesPerBlock * ColsPerWave - 1) / (kWavesPerBlock * ColsPerWave));
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile);
  const dim3 block(block_size);
  const GgufQ8_0GemmRcrArgs<AStorage, CStorage> args{
      a,
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  ck_tile::launch_kernel(
      ck_tile::stream_config{stream},
      ck_tile::make_kernel<kBlockPerCu>(
          GgufQ8_0GemmRcrWaveM4NExactKernel<AStorage, CStorage, ColsPerWave>{}, grid, block, 0, args));
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename Traits, typename AStorage, typename CStorage>
int launch_gguf_quant_gemm_rcr_wave(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  if (a == nullptr || b_quant == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0) {
    return 3;
  }

  const auto n_tile = static_cast<unsigned int>((n + kWavesPerBlock - 1) / kWavesPerBlock);
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile, static_cast<unsigned int>(m));
  const dim3 block(block_size);
  const GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args{
      a,
      static_cast<const Block*>(b_quant),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(GgufQuantGemmRcrWaveKernel<Traits, AStorage, CStorage>{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 4;
  }
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t RowsPerWave,
    ck_tile::index_t ColsPerWave>
int launch_gguf_quant_gemm_rcr_wave_mn(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  if (a == nullptr || b_quant == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0) {
    return 3;
  }

  const auto m_tile = static_cast<unsigned int>((m + RowsPerWave - 1) / RowsPerWave);
  const auto n_tile =
      static_cast<unsigned int>((n + kWavesPerBlock * ColsPerWave - 1) / (kWavesPerBlock * ColsPerWave));
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile, m_tile);
  const dim3 block(block_size);
  const GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args{
      a,
      static_cast<const Block*>(b_quant),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(
            GgufQuantGemmRcrWaveMNKernel<Traits, AStorage, CStorage, RowsPerWave, ColsPerWave>{},
            grid,
            block,
            0,
            args));
  } catch (const std::exception&) {
    return 4;
  }
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename Traits, typename AStorage, typename CStorage, ck_tile::index_t ColsPerWave>
int launch_gguf_quant_gemm_rcr_wave_m4n_exact(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  if (a == nullptr || b_quant == nullptr || c == nullptr) {
    return 1;
  }
  if (m != 4 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0) {
    return 3;
  }

  const auto n_tile =
      static_cast<unsigned int>((n + kWavesPerBlock * ColsPerWave - 1) / (kWavesPerBlock * ColsPerWave));
  const auto block_size = static_cast<unsigned int>((ck_tile::is_wave32() ? 32 : 64) * kWavesPerBlock);
  const dim3 grid(n_tile);
  const dim3 block(block_size);
  const GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args{
      a,
      static_cast<const Block*>(b_quant),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(
            GgufQuantGemmRcrWaveM4NExactKernel<Traits, AStorage, CStorage, ColsPerWave>{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 4;
  }
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename Traits, typename AStorage, typename CStorage, bool DirectFill>
int launch_gguf_quant_gemm_rcr_wmma16x16(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  if (a == nullptr || b_quant == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0 ||
      k % GgufQuantGemmRcrWmma16x16Kernel<Traits, AStorage, CStorage, DirectFill>::kKPerWave != 0) {
    return 3;
  }
  if (!ck_tile::is_wave32()) {
    return 4;
  }

  const auto m_tile = static_cast<unsigned int>(
      (m + GgufQuantGemmRcrWmma16x16Kernel<Traits, AStorage, CStorage, DirectFill>::kMPerWave - 1) /
      GgufQuantGemmRcrWmma16x16Kernel<Traits, AStorage, CStorage, DirectFill>::kMPerWave);
  const auto n_tile = static_cast<unsigned int>(
      (n + kWavesPerBlock * GgufQuantGemmRcrWmma16x16Kernel<Traits, AStorage, CStorage, DirectFill>::kNPerWave - 1) /
      (kWavesPerBlock * GgufQuantGemmRcrWmma16x16Kernel<Traits, AStorage, CStorage, DirectFill>::kNPerWave));
  const auto block_size = static_cast<unsigned int>(ck_tile::get_warp_size() * kWavesPerBlock);
  const dim3 grid(n_tile, m_tile);
  const dim3 block(block_size);
  const GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args{
      a,
      static_cast<const Block*>(b_quant),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(
            GgufQuantGemmRcrWmma16x16Kernel<Traits, AStorage, CStorage, DirectFill>{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 5;
  }
  return hipGetLastError() == hipSuccess ? 0 : 6;
}

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t MBlocksPerWave,
    ck_tile::index_t WavesPerBlock = kWavesPerBlock>
int launch_gguf_quant_gemm_rcr_wmma16x16_mblocks(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  if (a == nullptr || b_quant == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0 ||
      k %
              GgufQuantGemmRcrWmma16x16MBlocksKernel<
                  Traits,
                  AStorage,
                  CStorage,
                  MBlocksPerWave,
                  WavesPerBlock>::kKPerWave !=
          0) {
    return 3;
  }
  if (!ck_tile::is_wave32()) {
    return 4;
  }

  const auto m_tile = static_cast<unsigned int>(
      (m + MBlocksPerWave *
               GgufQuantGemmRcrWmma16x16MBlocksKernel<
                   Traits,
                   AStorage,
                   CStorage,
                   MBlocksPerWave,
                   WavesPerBlock>::kMPerWave -
       1) /
      (MBlocksPerWave *
       GgufQuantGemmRcrWmma16x16MBlocksKernel<
           Traits,
           AStorage,
           CStorage,
           MBlocksPerWave,
           WavesPerBlock>::kMPerWave));
  const auto n_tile = static_cast<unsigned int>(
      (n + WavesPerBlock *
               GgufQuantGemmRcrWmma16x16MBlocksKernel<
                   Traits,
                   AStorage,
                   CStorage,
                   MBlocksPerWave,
                   WavesPerBlock>::kNPerWave -
       1) /
      (WavesPerBlock *
       GgufQuantGemmRcrWmma16x16MBlocksKernel<
           Traits,
           AStorage,
           CStorage,
           MBlocksPerWave,
           WavesPerBlock>::kNPerWave));
  const auto block_size = static_cast<unsigned int>(32 * WavesPerBlock);
  const dim3 grid(n_tile, m_tile);
  const dim3 block(block_size);
  const GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args{
      a,
      static_cast<const Block*>(b_quant),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(
            GgufQuantGemmRcrWmma16x16MBlocksKernel<
                Traits,
                AStorage,
                CStorage,
                MBlocksPerWave,
                WavesPerBlock>{},
            grid,
            block,
            0,
            args));
  } catch (const std::exception&) {
    return 5;
  }
  return hipGetLastError() == hipSuccess ? 0 : 6;
}

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t MBlocksPerWave,
    ck_tile::index_t NBlocksPerWave,
    ck_tile::index_t WavesPerBlock = kWavesPerBlock>
int launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  using Kernel = GgufQuantGemmRcrWmma16x16MNBlocksKernel<
      Traits,
      AStorage,
      CStorage,
      MBlocksPerWave,
      NBlocksPerWave,
      WavesPerBlock>;
  if (a == nullptr || b_quant == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0 || k % Kernel::kKPerWave != 0) {
    return 3;
  }
  if (!ck_tile::is_wave32()) {
    return 4;
  }

  const auto m_tile = static_cast<unsigned int>(
      (m + MBlocksPerWave * Kernel::kMPerWave - 1) / (MBlocksPerWave * Kernel::kMPerWave));
  const auto n_tile = static_cast<unsigned int>(
      (n + WavesPerBlock * NBlocksPerWave * Kernel::kNPerWave - 1) /
      (WavesPerBlock * NBlocksPerWave * Kernel::kNPerWave));
  const auto block_size = static_cast<unsigned int>(32 * WavesPerBlock);
  const dim3 grid(n_tile, m_tile);
  const dim3 block(block_size);
  const GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args{
      a,
      static_cast<const Block*>(b_quant),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 5;
  }
  return hipGetLastError() == hipSuccess ? 0 : 6;
}

template <
    typename Traits,
    typename AStorage,
    typename CStorage,
    ck_tile::index_t MBlocksPerWave,
    ck_tile::index_t WavesPerBlock = kWavesPerBlock>
int launch_gguf_quant_gemm_rcr_wmma16x16_shared_b(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  using Kernel = GgufQuantGemmRcrWmma16x16SharedBKernel<
      Traits,
      AStorage,
      CStorage,
      MBlocksPerWave,
      WavesPerBlock>;
  if (a == nullptr || b_quant == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0 || k % Kernel::kKPerWave != 0) {
    return 3;
  }
  if (!ck_tile::is_wave32()) {
    return 4;
  }

  const auto m_tile = static_cast<unsigned int>(
      (m + WavesPerBlock * MBlocksPerWave * Kernel::kMPerWave - 1) /
      (WavesPerBlock * MBlocksPerWave * Kernel::kMPerWave));
  const auto n_tile = static_cast<unsigned int>((n + Kernel::kNPerBlock - 1) / Kernel::kNPerBlock);
  const auto block_size = static_cast<unsigned int>(ck_tile::get_warp_size() * WavesPerBlock);
  const dim3 grid(n_tile, m_tile);
  const dim3 block(block_size);
  const GgufQuantGemmRcrArgs<Block, AStorage, CStorage> args{
      a,
      static_cast<const Block*>(b_quant),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  ck_tile::launch_kernel(
      ck_tile::stream_config{stream},
      ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grid, block, 0, args));
  return hipGetLastError() == hipSuccess ? 0 : 6;
}

template <typename Traits, typename Storage, bool KnLayout>
int launch_gguf_quant_dequant_b_ck_tile(
    const void* b_quant,
    Storage* b_dense,
    int n,
    int k,
    hipStream_t stream) {
  using Block = typename Traits::Block;
  if (b_quant == nullptr || b_dense == nullptr) {
    return 1;
  }
  if (n <= 0 || k <= 0) {
    return 2;
  }
  if (k % Traits::BlockSize != 0) {
    return 3;
  }

  const auto qblocks_per_row = static_cast<unsigned int>(k / Traits::BlockSize);
  const dim3 grid(static_cast<unsigned int>(n), qblocks_per_row);
  const dim3 block(static_cast<unsigned int>(GgufQuantDequantKernel<Traits, Storage, KnLayout>::kBlockSize));
  const GgufQuantDequantArgs<Block, Storage> args{
      static_cast<const Block*>(b_quant),
      b_dense,
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(GgufQuantDequantKernel<Traits, Storage, KnLayout>{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 4;
  }
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename AStorage, typename CStorage, ck_tile::index_t TileM, ck_tile::index_t TileN>
int launch_gguf_q8_0_gemm_rcr_tile(
    const AStorage* a,
    const void* b_q8_0,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  const auto m_tile = static_cast<unsigned int>((m + TileM - 1) / TileM);
  const auto n_tile = static_cast<unsigned int>((n + TileN - 1) / TileN);
  const dim3 grid(n_tile, m_tile);
  const dim3 block(static_cast<unsigned int>(TileM * TileN));
  const GgufQ8_0GemmRcrArgs<AStorage, CStorage> args{
      a,
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      c,
      static_cast<ck_tile::index_t>(m),
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  ck_tile::launch_kernel(
      ck_tile::stream_config{stream},
      ck_tile::make_kernel<kBlockPerCu>(GgufQ8_0GemmRcrKernel<AStorage, CStorage, TileM, TileN>{}, grid, block, 0, args));
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename AStorage, typename CStorage>
int launch_gguf_q8_0_gemm_rcr_ck_tile(
    const AStorage* a,
    const void* b_q8_0,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (a == nullptr || b_q8_0 == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }

  try {
    if (m == 4 && k >= 512) {
      return launch_gguf_q8_0_gemm_rcr_wave_m4n_exact<AStorage, CStorage, 3>(a, b_q8_0, c, m, n, k, stream);
    }
    if (m > 4 && k >= 512) {
      return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ8_0Traits, AStorage, CStorage, 8, 4>(
          a, b_q8_0, c, m, n, k, stream);
    }
    if (m <= 4 && k >= 512) {
      return launch_gguf_q8_0_gemm_rcr_wave_m4n<AStorage, CStorage, 4>(a, b_q8_0, c, m, n, k, stream);
    }
    if (m <= 4) {
      return launch_gguf_q8_0_gemm_rcr_tile<AStorage, CStorage, 4, 64>(a, b_q8_0, c, m, n, k, stream);
    }
    return launch_gguf_q8_0_gemm_rcr_tile<AStorage, CStorage, 16, 16>(a, b_q8_0, c, m, n, k, stream);
  } catch (const std::exception&) {
    return 4;
  }
}

template <typename BStorage, typename ScaleStorage>
int launch_gguf_q8_0_transcode_bq_ck_tile(
    const void* b_q8_0,
    BStorage* b_kn,
    ScaleStorage* bq_qk_n,
    int n,
    int k,
    hipStream_t stream) {
  if (b_q8_0 == nullptr || b_kn == nullptr || bq_qk_n == nullptr) {
    return 1;
  }
  if (n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }

  const auto qblocks_per_row = static_cast<unsigned int>(k / kQ8BlockSize);
  const dim3 grid(static_cast<unsigned int>(n), qblocks_per_row);
  const dim3 block(static_cast<unsigned int>(GgufQ8_0TranscodeBqKernel<BStorage, ScaleStorage>::kBlockSize));
  const GgufQ8_0TranscodeBqArgs<BStorage, ScaleStorage> args{
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      b_kn,
      bq_qk_n,
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(
            GgufQ8_0TranscodeBqKernel<BStorage, ScaleStorage>{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 4;
  }
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename Storage>
int launch_gguf_q8_0_dequant_b_nk_ck_tile(
    const void* b_q8_0,
    Storage* b_nk,
    int n,
    int k,
    hipStream_t stream) {
  if (b_q8_0 == nullptr || b_nk == nullptr) {
    return 1;
  }
  if (n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }

  const auto qblocks_per_row = static_cast<unsigned int>(k / kQ8BlockSize);
  const dim3 grid(static_cast<unsigned int>(n), qblocks_per_row);
  const dim3 block(static_cast<unsigned int>(GgufQ8_0DequantNkKernel<Storage>::kBlockSize));
  const GgufQ8_0DequantNkArgs<Storage> args{
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      b_nk,
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(GgufQ8_0DequantNkKernel<Storage>{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 4;
  }
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

template <typename Storage>
int launch_gguf_q8_0_dequant_b_kn_ck_tile(
    const void* b_q8_0,
    Storage* b_kn,
    int n,
    int k,
    hipStream_t stream) {
  if (b_q8_0 == nullptr || b_kn == nullptr) {
    return 1;
  }
  if (n <= 0 || k <= 0) {
    return 2;
  }
  if (k % kQ8BlockSize != 0) {
    return 3;
  }

  const auto qblocks_per_row = static_cast<unsigned int>(k / kQ8BlockSize);
  const dim3 grid(static_cast<unsigned int>(n), qblocks_per_row);
  const dim3 block(static_cast<unsigned int>(GgufQ8_0DequantKnKernel<Storage>::kBlockSize));
  const GgufQ8_0DequantNkArgs<Storage> args{
      static_cast<const GgufQ8_0Block*>(b_q8_0),
      b_kn,
      static_cast<ck_tile::index_t>(n),
      static_cast<ck_tile::index_t>(k),
  };

  try {
    ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(GgufQ8_0DequantKnKernel<Storage>{}, grid, block, 0, args));
  } catch (const std::exception&) {
    return 4;
  }
  return hipGetLastError() == hipSuccess ? 0 : 5;
}

enum class GgufProfiledFusedOpt {
  kQ8_0,
  kQ4_0,
  kQ4_K,
  kQ5_K,
  kQ2_K,
  kQ3_K,
  kQ6_K,
  kMxFp4,
  kNvFp4,
};

template <typename Traits, typename AStorage, typename CStorage>
int launch_gguf_quant_gemm_rcr_wmma16x16_m4n2(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, AStorage, CStorage, 4, 2>(
      a, b_quant, c, m, n, k, stream);
}

template <typename Traits, typename AStorage, typename CStorage>
int launch_gguf_quant_gemm_rcr_wmma16x16_m6n2(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, AStorage, CStorage, 6, 2>(
      a, b_quant, c, m, n, k, stream);
}

template <typename Traits, typename AStorage, typename CStorage>
int launch_gguf_quant_gemm_rcr_wmma16x16_m8n2(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, AStorage, CStorage, 8, 2>(
      a, b_quant, c, m, n, k, stream);
}

template <typename Traits, typename AStorage, typename CStorage>
int launch_gguf_quant_gemm_rcr_wmma16x16_m4n3(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, AStorage, CStorage, 4, 3>(
      a, b_quant, c, m, n, k, stream);
}

template <typename Traits, typename AStorage, typename CStorage>
int launch_gguf_quant_gemm_rcr_profiled_fallback(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (m >= 128 && k >= 512) {
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, AStorage, CStorage, 2>(
        a, b_quant, c, m, n, k, stream);
  }
  if (m >= 8 && k >= 512) {
    return launch_gguf_quant_gemm_rcr_wmma16x16<Traits, AStorage, CStorage, true>(
        a, b_quant, c, m, n, k, stream);
  }
  if (m == 4 && k >= Traits::BlockSize) {
    return launch_gguf_quant_gemm_rcr_wave_m4n_exact<Traits, AStorage, CStorage, 3>(
        a, b_quant, c, m, n, k, stream);
  }
  if (m == 2 && k >= Traits::BlockSize) {
    return launch_gguf_quant_gemm_rcr_wave_mn<Traits, AStorage, CStorage, 8, 2>(
        a, b_quant, c, m, n, k, stream);
  }
  return launch_gguf_quant_gemm_rcr_wave<Traits, AStorage, CStorage>(a, b_quant, c, m, n, k, stream);
}

template <typename Traits, typename AStorage, typename CStorage>
int launch_gguf_quant_gemm_rcr_profiled_opt(
    const AStorage* a,
    const void* b_quant,
    CStorage* c,
    int m,
    int n,
    int k,
    hipStream_t stream,
    GgufProfiledFusedOpt opt) {
  if (m < 256 || k < 512) {
    return launch_gguf_quant_gemm_rcr_profiled_fallback<Traits>(a, b_quant, c, m, n, k, stream);
  }

  // DiT sweeps on gfx1201: m4n3 wins balanced shapes; m8n2 wins most wide/large shapes.
  switch (opt) {
  case GgufProfiledFusedOpt::kQ8_0:
    if (m == 2048 && n >= 8192 && k >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n >= 8192 || m >= 8192 || (k >= 8192 && m >= 2048)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kQ4_0:
    if (m == 256 && n <= 3072 && k >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m >= 8192 && n >= 4096 && k >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n <= 4096 && k <= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if ((n >= 8192 && (n < 12288 || m >= 1024)) || m >= 4096 ||
        (k >= 8192 && m >= 1024 && m != 2048)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kQ4_K:
    if (m == 256 && n <= 3072) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m >= 8192 && n <= 4096 && k >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m >= 2048 && n >= 4096 && k >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n <= 4096 && k <= 4096 && m < 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if ((n >= 8192 && (n < 12288 || m >= 1024)) || (m >= 2048 && k < 8192) || m >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kQ5_K:
    if ((m <= 256 && n <= 3072 && k <= 3072) || (m == 512 && n == 4096 && k == 4096) ||
        (m == 512 && n <= 3072 && k >= 8192)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m >= 2048 && n >= 4096 && k >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n <= 4096 && k <= 4096 && m < 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n >= 8192 || m >= 2048) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kQ2_K:
    if (m == 256 && n <= 3072 && k <= 3072) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m >= 8192 && n >= 4096 && k >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n <= 4096 && k <= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if ((n >= 12288 && m >= 1024) || (n >= 8192 && m >= 2048) || m >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kQ3_K:
    if (m <= 512 && n <= 3072) {
      if (m == 512 && k >= 8192) {
        return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
      }
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m == 2048 && n >= 8192 && k >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, AStorage, CStorage, 12>(
          a, b_quant, c, m, n, k, stream);
    }
    if ((n >= 8192 && (n < 12288 || m >= 1024)) || m >= 8192 || (m >= 2048 && n >= 4096)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kQ6_K:
    if (m >= 8192 && k >= 8192 && n <= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, AStorage, CStorage, 12>(
          a, b_quant, c, m, n, k, stream);
    }
    if (m >= 2048 && n >= 4096 && k >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if ((n >= 8192 && m >= 512) || m >= 4096 || (k >= 8192 && m >= 1024)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kMxFp4:
    if (m == 256 && n <= 3072) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n >= 12288 && (m == 512 || m == 1024)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m == 4096 && n <= 4096 && k >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m >= 8192 && n >= 4096 && k >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (n <= 4096 && k <= 4096 && m >= 4096) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if ((n >= 8192 && (n < 12288 || m >= 2048 || m == 256)) || m >= 8192 ||
        (k >= 8192 && m >= 1024) || (m >= 2048 && n >= 12288)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m == 2048 && n <= 3072 && k <= 3072) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);

  case GgufProfiledFusedOpt::kNvFp4:
    if (m == 256 && n <= 3072) {
      if (k >= 8192) {
        return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
      }
      return launch_gguf_quant_gemm_rcr_wmma16x16_m4n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m == 512 && n <= 3072 && k >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m == 256 && n >= 12288) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m == 2048 && n >= 8192 && k >= 8192) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if (m >= 8192 && n <= 3072 && k <= 3072) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m6n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    if ((n >= 8192 && m >= 512) || m >= 4096 || (k >= 8192 && m >= 4096)) {
      return launch_gguf_quant_gemm_rcr_wmma16x16_m8n2<Traits>(a, b_quant, c, m, n, k, stream);
    }
    return launch_gguf_quant_gemm_rcr_wmma16x16_m4n3<Traits>(a, b_quant, c, m, n, k, stream);
  }

  return launch_gguf_quant_gemm_rcr_profiled_fallback<Traits>(a, b_quant, c, m, n, k, stream);
}

} // namespace

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n<half, float, 2>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n2_exact(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n_exact<half, float, 3>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n_exact<half, float, 4>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n5_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n_exact<half, float, 5>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n6_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n_exact<half, float, 6>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n_exact<half, float, 8>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n<half, float, 4>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_wave_m4n<half, float, 8>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile4x64_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_tile<half, float, 4, 64>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile16x16_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_tile<half, float, 16, 16>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_opt_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_profiled_opt<GgufQ8_0Traits>(
      a, b_q8_0, c, m, n, k, stream, GgufProfiledFusedOpt::kQ8_0);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_float16_opt_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_ck_tile(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_float32_opt_v1(
    const float* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_gemm_rcr_ck_tile(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_float16_ref_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return dinoml_ck_gguf_q8_0_gemm_rcr_float16_opt_v1(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_float32_ref_v1(
    const float* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return dinoml_ck_gguf_q8_0_gemm_rcr_float32_opt_v1(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_transcode_bq_i8_float_v1(
    const void* b_q8_0,
    int8_t* b_kn,
    float* bq_qk_n,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_transcode_bq_ck_tile(b_q8_0, b_kn, bq_qk_n, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_dequant_b_f16_nk_v1(
    const void* b_q8_0,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_dequant_b_nk_ck_tile(b_q8_0, b_nk, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_dequant_b_f16_kn_v1(
    const void* b_q8_0,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_dequant_b_kn_ck_tile(b_q8_0, b_kn, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ8_0Traits, half, float, 8, 1>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ8_0Traits, half, float, 8, 2>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ8_0Traits, half, float, 8, 3>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ8_0Traits, half, float, 8, 4>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ8_0Traits, half, float, 16, 1>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ8_0Traits, half, float, 16, 2>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ8_0Traits, half, float, false>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ8_0Traits, half, float, true>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, float, 2>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, float, 3>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, float, 4>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, float, 8>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, float, 12>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, float, 16>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<GgufQ8_0Traits, half, float, 4, 2>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<GgufQ8_0Traits, half, float, 6, 2>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<GgufQ8_0Traits, half, float, 8, 2>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<GgufQ8_0Traits, half, float, 4, 3>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ8_0Traits, half, half, true>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, half, 2>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, half, 3>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, half, 4>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, half, 8>(a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, half, 12>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ8_0Traits, half, half, 16>(
      a, b_q8_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave<GgufQ4_0Traits>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_0Traits, half, float, 8, 1>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_0Traits, half, float, 8, 2>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_0Traits, half, float, 8, 3>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_0Traits, half, float, 8, 4>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_0Traits, half, float, 16, 1>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_0Traits, half, float, 16, 2>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ4_0Traits, half, float, false>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ4_0Traits, half, float, true>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, float, 2>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, float, 3>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, float, 4>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, float, 8>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, float, 12>(
      a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, float, 16>(
      a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ4_0Traits, half, half, true>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, half, 2>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, half, 3>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, half, 4>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, half, 8>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, half, 12>(
      a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_0Traits, half, half, 16>(
      a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ4_0Traits, half, float, 2>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ4_0Traits, half, float, 3>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ4_0Traits, half, float, 4>(a, b_q4_0, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_opt_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_profiled_opt<GgufQ4_0Traits>(
      a, b_q4_0, c, m, n, k, stream, GgufProfiledFusedOpt::kQ4_0);
}

extern "C" int dinoml_ck_gguf_q4_0_dequant_b_f16_nk_v1(
    const void* b_q4_0,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_dequant_b_ck_tile<GgufQ4_0Traits, half, false>(b_q4_0, b_nk, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_0_dequant_b_f16_kn_v1(
    const void* b_q4_0,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_dequant_b_ck_tile<GgufQ4_0Traits, half, true>(b_q4_0, b_kn, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave<GgufQ4_KTraits>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_KTraits, half, float, 8, 1>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_KTraits, half, float, 8, 2>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_KTraits, half, float, 8, 3>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_KTraits, half, float, 8, 4>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_KTraits, half, float, 16, 1>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ4_KTraits, half, float, 16, 2>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ4_KTraits, half, float, false>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ4_KTraits, half, float, true>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, float, 2>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, float, 3>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, float, 4>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, float, 8>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, float, 12>(
      a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, float, 16>(
      a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ4_KTraits, half, half, true>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, half, 2>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, half, 3>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, half, 4>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, half, 8>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, half, 12>(
      a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ4_KTraits, half, half, 16>(
      a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ4_KTraits, half, float, 2>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ4_KTraits, half, float, 3>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ4_KTraits, half, float, 4>(a, b_q4_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_opt_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_profiled_opt<GgufQ4_KTraits>(
      a, b_q4_k, c, m, n, k, stream, GgufProfiledFusedOpt::kQ4_K);
}

extern "C" int dinoml_ck_gguf_q4_k_dequant_b_f16_nk_v1(
    const void* b_q4_k,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_dequant_b_ck_tile<GgufQ4_KTraits, half, false>(b_q4_k, b_nk, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q4_k_dequant_b_f16_kn_v1(
    const void* b_q4_k,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_dequant_b_ck_tile<GgufQ4_KTraits, half, true>(b_q4_k, b_kn, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave<GgufQ5_KTraits>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ5_KTraits, half, float, 8, 1>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ5_KTraits, half, float, 8, 2>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ5_KTraits, half, float, 8, 3>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ5_KTraits, half, float, 8, 4>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ5_KTraits, half, float, 16, 1>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_mn<GgufQ5_KTraits, half, float, 16, 2>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ5_KTraits, half, float, false>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ5_KTraits, half, float, true>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, float, 2>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, float, 3>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, float, 4>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, float, 8>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, float, 12>(
      a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, float, 16>(
      a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16<GgufQ5_KTraits, half, half, true>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, half, 2>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, half, 3>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, half, 4>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, half, 8>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, half, 12>(
      a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<GgufQ5_KTraits, half, half, 16>(
      a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ5_KTraits, half, float, 2>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ5_KTraits, half, float, 3>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_wave_m4n_exact<GgufQ5_KTraits, half, float, 4>(a, b_q5_k, c, m, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_opt_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_gemm_rcr_profiled_opt<GgufQ5_KTraits>(
      a, b_q5_k, c, m, n, k, stream, GgufProfiledFusedOpt::kQ5_K);
}

extern "C" int dinoml_ck_gguf_q5_k_dequant_b_f16_nk_v1(
    const void* b_q5_k,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_dequant_b_ck_tile<GgufQ5_KTraits, half, false>(b_q5_k, b_nk, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q5_k_dequant_b_f16_kn_v1(
    const void* b_q5_k,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_quant_dequant_b_ck_tile<GgufQ5_KTraits, half, true>(b_q5_k, b_kn, n, k, stream);
}

#define DINOML_DEFINE_GGUF_MNBLOCK_F32_EXPORTS(prefix, Traits)                                           \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {           \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 4, 2>(a, b_quant, c, m,    \
                                                                                    n, k, stream);       \
  }                                                                                                      \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {           \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 6, 2>(a, b_quant, c, m,    \
                                                                                    n, k, stream);       \
  }                                                                                                      \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {           \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 8, 2>(a, b_quant, c, m,    \
                                                                                    n, k, stream);       \
  }                                                                                                      \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {           \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 4, 3>(a, b_quant, c, m,    \
                                                                                    n, k, stream);       \
  }

DINOML_DEFINE_GGUF_MNBLOCK_F32_EXPORTS(q4_0, GgufQ4_0Traits)
DINOML_DEFINE_GGUF_MNBLOCK_F32_EXPORTS(q4_k, GgufQ4_KTraits)
DINOML_DEFINE_GGUF_MNBLOCK_F32_EXPORTS(q5_k, GgufQ5_KTraits)

#undef DINOML_DEFINE_GGUF_MNBLOCK_F32_EXPORTS

#define DINOML_DEFINE_GGUF_QUANT_EXPORTS(prefix, Traits, OptKind)                                      \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_v1(                                 \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave<Traits, half, float>(a, b_quant, c, m, n, k, stream);       \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n1_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_mn<Traits, half, float, 8, 1>(a, b_quant, c, m, n, k,       \
                                                                         stream);                      \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n2_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_mn<Traits, half, float, 8, 2>(a, b_quant, c, m, n, k,       \
                                                                         stream);                      \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n4_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_mn<Traits, half, float, 8, 4>(a, b_quant, c, m, n, k,       \
                                                                         stream);                      \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m16n1_v1(                           \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_mn<Traits, half, float, 16, 1>(a, b_quant, c, m, n, k,      \
                                                                          stream);                     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m16n2_v1(                           \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_mn<Traits, half, float, 16, 2>(a, b_quant, c, m, n, k,      \
                                                                          stream);                     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_direct_v1(                     \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16<Traits, half, float, true>(a, b_quant, c, m, n, k,     \
                                                                           stream);                    \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m2_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 2>(a, b_quant, c, m, n,   \
                                                                                k, stream);            \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 4>(a, b_quant, c, m, n,   \
                                                                                k, stream);            \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 6>(a, b_quant, c, m, n,   \
                                                                                k, stream);            \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8_v1(                         \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 8>(a, b_quant, c, m, n,   \
                                                                                k, stream);            \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m10_v1(                        \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 10>(a, b_quant, c, m, n,  \
                                                                                 k, stream);           \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m12_v1(                        \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 12>(a, b_quant, c, m, n,  \
                                                                                 k, stream);           \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m16_v1(                        \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 16>(a, b_quant, c, m, n,  \
                                                                                 k, stream);           \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m2_v1(                 \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_shared_b<Traits, half, float, 2>(a, b_quant, c, m, n,  \
                                                                                 k, stream);           \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m4_v1(                 \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_shared_b<Traits, half, float, 4>(a, b_quant, c, m, n,  \
                                                                                 k, stream);           \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m8_v1(                 \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_shared_b<Traits, half, float, 8>(a, b_quant, c, m, n,  \
                                                                                 k, stream);           \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8w2_v1(                       \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 8, 2>(a, b_quant, c, m,   \
                                                                                   n, k, stream);      \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m12w2_v1(                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, float, 12, 2>(a, b_quant, c, m,  \
                                                                                    n, k, stream);     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1(                       \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 4, 2>(a, b_quant, c, m,  \
                                                                                    n, k, stream);     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1(                       \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 6, 2>(a, b_quant, c, m,  \
                                                                                    n, k, stream);     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1(                       \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 8, 2>(a, b_quant, c, m,  \
                                                                                    n, k, stream);     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1(                       \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mnblocks<Traits, half, float, 4, 3>(a, b_quant, c, m,  \
                                                                                    n, k, stream);     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_opt_v1(                                  \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_profiled_opt<Traits>(a, b_quant, c, m, n, k, stream,             \
                                                           GgufProfiledFusedOpt::OptKind);             \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_m4n_exact<Traits, half, float, 2>(a, b_quant, c, m, n, k,   \
                                                                             stream);                  \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_m4n_exact<Traits, half, float, 3>(a, b_quant, c, m, n, k,   \
                                                                             stream);                  \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream) {         \
    return launch_gguf_quant_gemm_rcr_wave_m4n_exact<Traits, half, float, 4>(a, b_quant, c, m, n, k,   \
                                                                             stream);                  \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_direct_v1(                     \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream) {          \
    return launch_gguf_quant_gemm_rcr_wmma16x16<Traits, half, half, true>(a, b_quant, c, m, n, k,      \
                                                                          stream);                     \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m2_v1(                         \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream) {          \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, half, 2>(a, b_quant, c, m, n, k, \
                                                                               stream);                \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m4_v1(                         \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream) {          \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, half, 4>(a, b_quant, c, m, n, k, \
                                                                               stream);                \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m8_v1(                         \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream) {          \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, half, 8>(a, b_quant, c, m, n, k, \
                                                                               stream);                \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m12_v1(                        \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream) {          \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, half, 12>(a, b_quant, c, m, n,   \
                                                                                k, stream);            \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m16_v1(                        \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream) {          \
    return launch_gguf_quant_gemm_rcr_wmma16x16_mblocks<Traits, half, half, 16>(a, b_quant, c, m, n,   \
                                                                                k, stream);            \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_dequant_b_f16_nk_v1(                                       \
      const void* b_quant, half* b_nk, int n, int k, hipStream_t stream) {                             \
    return launch_gguf_quant_dequant_b_ck_tile<Traits, half, false>(b_quant, b_nk, n, k, stream);      \
  }                                                                                                    \
  extern "C" int dinoml_ck_gguf_##prefix##_dequant_b_f16_kn_v1(                                       \
      const void* b_quant, half* b_kn, int n, int k, hipStream_t stream) {                             \
    return launch_gguf_quant_dequant_b_ck_tile<Traits, half, true>(b_quant, b_kn, n, k, stream);       \
  }

DINOML_DEFINE_GGUF_QUANT_EXPORTS(q2_k, GgufQ2_KTraits, kQ2_K)
DINOML_DEFINE_GGUF_QUANT_EXPORTS(q3_k, GgufQ3_KTraits, kQ3_K)
DINOML_DEFINE_GGUF_QUANT_EXPORTS(q6_k, GgufQ6_KTraits, kQ6_K)
DINOML_DEFINE_GGUF_QUANT_EXPORTS(mxfp4, GgufMxFp4Traits, kMxFp4)
DINOML_DEFINE_GGUF_QUANT_EXPORTS(nvfp4, GgufNvFp4Traits, kNvFp4)

#undef DINOML_DEFINE_GGUF_QUANT_EXPORTS

extern "C" int dinoml_ck_gguf_q8_0_transcode_bq_fp8_float_v1(
    const void* b_q8_0,
    ck_tile::fp8_t* b_kn,
    float* bq_qk_n,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_transcode_bq_ck_tile(b_q8_0, b_kn, bq_qk_n, n, k, stream);
}

extern "C" int dinoml_ck_gguf_q8_0_transcode_bq_bf8_float_v1(
    const void* b_q8_0,
    ck_tile::bf8_t* b_kn,
    float* bq_qk_n,
    int n,
    int k,
    hipStream_t stream) {
  return launch_gguf_q8_0_transcode_bq_ck_tile(b_q8_0, b_kn, bq_qk_n, n, k, stream);
}
