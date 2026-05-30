#include <dinoml/abi.h>
#include <dinoml/math.h>

#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"

#include <cstdint>
#include <exception>

namespace {

constexpr int kBlockPerCu = 1;
constexpr int kTileM = 16;
constexpr int kTileN = 16;

enum class MatrixLayout {
  kRow,
  kColumn,
};

enum class GemmEpilogue {
  kLinearCombination,
  kBias,
  kBiasRelu,
  kBiasGelu,
  kBiasFastGelu,
  kBiasQuickGelu,
  kBiasSigmoid,
  kBiasTanh,
  kBiasSwish,
  kBiasHardSwish,
  kBiasElup1,
  kBiasAdd,
  kBiasAddAdd,
  kBiasMul,
  kBiasMulAdd,
  kBiasAddRelu,
  kBiasAddAddRelu,
  kBiasSigmoidMul,
  kBiasSigmoidMulTanh,
  kBiasMulTanh,
};

struct GemmArgs {
  const float* a;
  const float* b;
  const float* bias;
  const float* d0;
  const float* d1;
  float* c;
  int m;
  int n;
  int k;
};

struct BmmArgs {
  const float* a;
  const float* b;
  const float* d0;
  float* c;
  int batch_count;
  int m;
  int n;
  int k;
  int64_t batch_stride_a;
  int64_t batch_stride_b;
  int64_t batch_stride_d0;
  int64_t batch_stride_c;
  int lda;
  int ldb;
  int ldd0;
  int ldc;
};

struct Conv2dArgs {
  const float* x;
  const float* weight;
  const float* bias;
  const float* residual;
  float* output;
  int batch;
  int in_channels;
  int in_height;
  int in_width;
  int out_channels;
  int kernel_h;
  int kernel_w;
  int out_height;
  int out_width;
  int stride_h;
  int stride_w;
  int pad_h;
  int pad_w;
  int dilation_h;
  int dilation_w;
};

CK_TILE_DEVICE float sigmoid(float value) {
  return 1.0f / (1.0f + __expf(-value));
}

CK_TILE_DEVICE float gelu(float value) {
  return 0.5f * value *
      (1.0f + tanhf(0.7978845608028654f * (value + 0.044715f * value * value * value)));
}

CK_TILE_DEVICE float fast_gelu(float value) {
  return 0.5f * value *
      (1.0f + tanhf(0.7978845608f * value * (1.0f + 0.044715f * value * value)));
}

CK_TILE_DEVICE float quick_gelu(float value) {
  return value * sigmoid(1.702f * value);
}

CK_TILE_DEVICE float hard_swish(float value) {
  return value * fminf(fmaxf(value + 3.0f, 0.0f), 6.0f) / 6.0f;
}

CK_TILE_DEVICE float elup1(float value) {
  return value >= 0.0f ? value + 1.0f : expf(value);
}

template <GemmEpilogue Epilogue>
CK_TILE_DEVICE float apply_gemm_epilogue(float value, const GemmArgs& args, int row, int col) {
  if constexpr (Epilogue != GemmEpilogue::kLinearCombination) {
    value += args.bias[col];
  }

  if constexpr (
      Epilogue == GemmEpilogue::kBiasSigmoidMul ||
      Epilogue == GemmEpilogue::kBiasSigmoidMulTanh) {
    value = sigmoid(value);
  }

  if constexpr (
      Epilogue == GemmEpilogue::kBiasAdd ||
      Epilogue == GemmEpilogue::kBiasAddRelu) {
    value += args.d0[row * args.n + col];
  } else if constexpr (
      Epilogue == GemmEpilogue::kBiasAddAdd ||
      Epilogue == GemmEpilogue::kBiasAddAddRelu) {
    const int index = row * args.n + col;
    value += args.d0[index] + args.d1[index];
  } else if constexpr (
      Epilogue == GemmEpilogue::kBiasMul ||
      Epilogue == GemmEpilogue::kBiasMulTanh ||
      Epilogue == GemmEpilogue::kBiasSigmoidMul ||
      Epilogue == GemmEpilogue::kBiasSigmoidMulTanh) {
    value *= args.d0[row * args.n + col];
  } else if constexpr (Epilogue == GemmEpilogue::kBiasMulAdd) {
    const int index = row * args.n + col;
    value = value * args.d0[index] + args.d1[index];
  }

  if constexpr (
      Epilogue == GemmEpilogue::kBiasRelu ||
      Epilogue == GemmEpilogue::kBiasAddRelu ||
      Epilogue == GemmEpilogue::kBiasAddAddRelu) {
    value = fmaxf(value, 0.0f);
  } else if constexpr (Epilogue == GemmEpilogue::kBiasGelu) {
    value = gelu(value);
  } else if constexpr (Epilogue == GemmEpilogue::kBiasFastGelu) {
    value = fast_gelu(value);
  } else if constexpr (Epilogue == GemmEpilogue::kBiasQuickGelu) {
    value = quick_gelu(value);
  } else if constexpr (Epilogue == GemmEpilogue::kBiasSigmoid) {
    value = sigmoid(value);
  } else if constexpr (
      Epilogue == GemmEpilogue::kBiasTanh ||
      Epilogue == GemmEpilogue::kBiasSigmoidMulTanh ||
      Epilogue == GemmEpilogue::kBiasMulTanh) {
    value = tanhf(value);
  } else if constexpr (Epilogue == GemmEpilogue::kBiasSwish) {
    value *= sigmoid(value);
  } else if constexpr (Epilogue == GemmEpilogue::kBiasHardSwish) {
    value = hard_swish(value);
  } else if constexpr (Epilogue == GemmEpilogue::kBiasElup1) {
    value = elup1(value);
  }
  return value;
}

template <MatrixLayout BLayout, GemmEpilogue Epilogue>
struct GemmRxxKernel {
  static constexpr ck_tile::index_t kBlockSize = kTileM * kTileN;

  CK_TILE_DEVICE void operator()(GemmArgs args) const {
    const int row = static_cast<int>(blockIdx.y) * kTileM + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * kTileN + static_cast<int>(threadIdx.x);
    if (row >= args.m || col >= args.n) {
      return;
    }
    float value = 0.0f;
    for (int kk = 0; kk < args.k; ++kk) {
      const float a_value = args.a[row * args.k + kk];
      const float b_value =
          BLayout == MatrixLayout::kRow ? args.b[kk * args.n + col] : args.b[col * args.k + kk];
      value += a_value * b_value;
    }
    args.c[row * args.n + col] = apply_gemm_epilogue<Epilogue>(value, args, row, col);
  }
};

template <MatrixLayout Layout>
CK_TILE_DEVICE float bmm_load(const float* ptr, int row, int col, int ld) {
  if constexpr (Layout == MatrixLayout::kRow) {
    return ptr[row * ld + col];
  } else {
    return ptr[col * ld + row];
  }
}

template <MatrixLayout Layout>
CK_TILE_DEVICE int64_t bmm_offset(int row, int col, int ld) {
  if constexpr (Layout == MatrixLayout::kRow) {
    return static_cast<int64_t>(row) * ld + col;
  } else {
    return static_cast<int64_t>(col) * ld + row;
  }
}

template <MatrixLayout ALayout, MatrixLayout BLayout, MatrixLayout CLayout, bool Add>
struct BmmKernel {
  static constexpr ck_tile::index_t kBlockSize = kTileM * kTileN;

  CK_TILE_DEVICE void operator()(BmmArgs args) const {
    const int row = static_cast<int>(blockIdx.y) * kTileM + static_cast<int>(threadIdx.y);
    const int col = static_cast<int>(blockIdx.x) * kTileN + static_cast<int>(threadIdx.x);
    const int batch = static_cast<int>(blockIdx.z);
    if (row >= args.m || col >= args.n || batch >= args.batch_count) {
      return;
    }
    const float* a = args.a + static_cast<int64_t>(batch) * args.batch_stride_a;
    const float* b = args.b + static_cast<int64_t>(batch) * args.batch_stride_b;
    float value = 0.0f;
    for (int kk = 0; kk < args.k; ++kk) {
      value += bmm_load<ALayout>(a, row, kk, args.lda) * bmm_load<BLayout>(b, kk, col, args.ldb);
    }
    const int64_t c_index = bmm_offset<CLayout>(row, col, args.ldc);
    if constexpr (Add) {
      const float* d0 = args.d0 + static_cast<int64_t>(batch) * args.batch_stride_d0;
      if (args.ldd0 == 0) {
        value += CLayout == MatrixLayout::kRow ? d0[col] : d0[row];
      } else {
        value += d0[c_index];
      }
    }
    args.c[static_cast<int64_t>(batch) * args.batch_stride_c + c_index] = value;
  }
};

template <bool Add, bool Relu>
struct Conv2dBiasKernel {
  static constexpr ck_tile::index_t kBlockSize = 256;

  CK_TILE_DEVICE void operator()(Conv2dArgs args) const {
    const int64_t total = static_cast<int64_t>(args.batch) * args.out_channels * args.out_height * args.out_width;
    for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < total;
         index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
      const int out_w = static_cast<int>(index % args.out_width);
      int64_t tmp = index / args.out_width;
      const int out_h = static_cast<int>(tmp % args.out_height);
      tmp /= args.out_height;
      const int out_channel = static_cast<int>(tmp % args.out_channels);
      const int batch = static_cast<int>(tmp / args.out_channels);

      float value = args.bias[out_channel];
      for (int in_channel = 0; in_channel < args.in_channels; ++in_channel) {
        for (int kernel_h = 0; kernel_h < args.kernel_h; ++kernel_h) {
          const int in_h = out_h * args.stride_h - args.pad_h + kernel_h * args.dilation_h;
          if (in_h < 0 || in_h >= args.in_height) {
            continue;
          }
          for (int kernel_w = 0; kernel_w < args.kernel_w; ++kernel_w) {
            const int in_w = out_w * args.stride_w - args.pad_w + kernel_w * args.dilation_w;
            if (in_w < 0 || in_w >= args.in_width) {
              continue;
            }
            const int64_t x_index =
                ((static_cast<int64_t>(batch) * args.in_channels + in_channel) * args.in_height + in_h) *
                    args.in_width +
                in_w;
            const int64_t weight_index =
                ((static_cast<int64_t>(out_channel) * args.in_channels + in_channel) * args.kernel_h + kernel_h) *
                    args.kernel_w +
                kernel_w;
            value += args.x[x_index] * args.weight[weight_index];
          }
        }
      }
      if constexpr (Add) {
        value += args.residual[index];
      }
      if constexpr (Relu) {
        value = fmaxf(value, 0.0f);
      }
      args.output[index] = value;
    }
  }
};

template <typename Kernel, typename Args>
int launch_ck_tile_kernel(const Args& args, dim3 grids, dim3 blocks, hipStream_t stream) {
  try {
    (void)ck_tile::launch_kernel(
        ck_tile::stream_config{stream},
        ck_tile::make_kernel<kBlockPerCu>(Kernel{}, grids, blocks, 0, args));
  } catch (const std::exception&) {
    return 3;
  } catch (...) {
    return 3;
  }
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <MatrixLayout BLayout, GemmEpilogue Epilogue>
int launch_gemm(const float* a, const float* b, const float* bias, const float* d0, const float* d1, float* c, int m, int n, int k, hipStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 || k <= 0) {
    return 4;
  }
  if constexpr (Epilogue != GemmEpilogue::kLinearCombination) {
    if (bias == nullptr) {
      return 4;
    }
  }
  if constexpr (
      Epilogue == GemmEpilogue::kBiasAdd ||
      Epilogue == GemmEpilogue::kBiasMul ||
      Epilogue == GemmEpilogue::kBiasAddRelu ||
      Epilogue == GemmEpilogue::kBiasSigmoidMul ||
      Epilogue == GemmEpilogue::kBiasSigmoidMulTanh ||
      Epilogue == GemmEpilogue::kBiasMulTanh) {
    if (d0 == nullptr) {
      return 4;
    }
  }
  if constexpr (
      Epilogue == GemmEpilogue::kBiasAddAdd ||
      Epilogue == GemmEpilogue::kBiasMulAdd ||
      Epilogue == GemmEpilogue::kBiasAddAddRelu) {
    if (d0 == nullptr || d1 == nullptr) {
      return 4;
    }
  }
  const GemmArgs args{a, b, bias, d0, d1, c, m, n, k};
  const dim3 blocks(kTileN, kTileM);
  const dim3 grids((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM);
  return launch_ck_tile_kernel<GemmRxxKernel<BLayout, Epilogue>>(args, grids, blocks, stream);
}

template <MatrixLayout ALayout, MatrixLayout BLayout, MatrixLayout CLayout, bool Add>
int launch_bmm(
    const float* a,
    const float* b,
    const float* d0,
    float* c,
    int batch_count,
    int m,
    int n,
    int k,
    int64_t batch_stride_a,
    int64_t batch_stride_b,
    int64_t batch_stride_d0,
    int64_t batch_stride_c,
    int lda,
    int ldb,
    int ldd0,
    int ldc,
    hipStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr || batch_count <= 0 || m <= 0 || n <= 0 || k <= 0) {
    return 4;
  }
  if constexpr (Add) {
    if (d0 == nullptr) {
      return 4;
    }
  }
  const BmmArgs args{
      a, b, d0, c, batch_count, m, n, k, batch_stride_a, batch_stride_b,
      batch_stride_d0, batch_stride_c, lda, ldb, ldd0, ldc};
  const dim3 blocks(kTileN, kTileM);
  const dim3 grids((n + kTileN - 1) / kTileN, (m + kTileM - 1) / kTileM, batch_count);
  return launch_ck_tile_kernel<BmmKernel<ALayout, BLayout, CLayout, Add>>(args, grids, blocks, stream);
}

template <bool Add, bool Relu>
int launch_conv2d_bias(
    const float* x,
    const float* weight,
    const float* bias,
    const float* residual,
    float* output,
    int batch,
    int in_channels,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_h,
    int kernel_w,
    int out_height,
    int out_width,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || bias == nullptr || output == nullptr || batch <= 0 ||
      in_channels <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || out_height <= 0 || out_width <= 0 ||
      stride_h <= 0 || stride_w <= 0 || dilation_h <= 0 || dilation_w <= 0) {
    return 4;
  }
  if constexpr (Add) {
    if (residual == nullptr) {
      return 4;
    }
  }
  const Conv2dArgs args{
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels,
      kernel_h, kernel_w, out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w};
  const int64_t total = static_cast<int64_t>(batch) * out_channels * out_height * out_width;
  const int blocks = 256;
  const int grid = static_cast<int>((total + blocks - 1) / blocks);
  return launch_ck_tile_kernel<Conv2dBiasKernel<Add, Relu>>(args, dim3(grid), dim3(blocks), stream);
}

}  // namespace

#define DINOML_ROCM_TILE_GEMM_EXPORT(OP, LAYOUT_B, EPILOGUE) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* a, const float* b, float* c, int m, int n, int k, hipStream_t stream) { \
  return launch_gemm<MatrixLayout::LAYOUT_B, GemmEpilogue::EPILOGUE>(a, b, nullptr, nullptr, nullptr, c, m, n, k, stream); \
}

#define DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(OP, LAYOUT_B, EPILOGUE) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* a, const float* b, const float* bias, float* c, int m, int n, int k, hipStream_t stream) { \
  return launch_gemm<MatrixLayout::LAYOUT_B, GemmEpilogue::EPILOGUE>(a, b, bias, nullptr, nullptr, c, m, n, k, stream); \
}

#define DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(OP, LAYOUT_B, EPILOGUE) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* a, const float* b, const float* bias, const float* d0, float* c, int m, int n, int k, hipStream_t stream) { \
  return launch_gemm<MatrixLayout::LAYOUT_B, GemmEpilogue::EPILOGUE>(a, b, bias, d0, nullptr, c, m, n, k, stream); \
}

#define DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT(OP, LAYOUT_B, EPILOGUE) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* a, const float* b, const float* bias, const float* d0, const float* d1, float* c, int m, int n, int k, hipStream_t stream) { \
  return launch_gemm<MatrixLayout::LAYOUT_B, GemmEpilogue::EPILOGUE>(a, b, bias, d0, d1, c, m, n, k, stream); \
}

DINOML_ROCM_TILE_GEMM_EXPORT(gemm_rcr, kColumn, kLinearCombination)
DINOML_ROCM_TILE_GEMM_EXPORT(gemm_rrr, kRow, kLinearCombination)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias, kColumn, kBias)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias, kRow, kBias)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_relu, kColumn, kBiasRelu)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_relu, kRow, kBiasRelu)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_gelu, kColumn, kBiasGelu)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_gelu, kRow, kBiasGelu)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_fast_gelu, kColumn, kBiasFastGelu)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_fast_gelu, kRow, kBiasFastGelu)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_quick_gelu, kColumn, kBiasQuickGelu)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_sigmoid, kColumn, kBiasSigmoid)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_sigmoid, kRow, kBiasSigmoid)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_tanh, kColumn, kBiasTanh)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_tanh, kRow, kBiasTanh)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_swish, kColumn, kBiasSwish)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_swish, kRow, kBiasSwish)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_hardswish, kColumn, kBiasHardSwish)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_hardswish, kRow, kBiasHardSwish)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rcr_bias_elup1, kColumn, kBiasElup1)
DINOML_ROCM_TILE_GEMM_BIAS_EXPORT(gemm_rrr_bias_elup1, kRow, kBiasElup1)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rcr_bias_add, kColumn, kBiasAdd)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rrr_bias_add, kRow, kBiasAdd)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT(gemm_rcr_bias_add_add, kColumn, kBiasAddAdd)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT(gemm_rrr_bias_add_add, kRow, kBiasAddAdd)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rcr_bias_mul, kColumn, kBiasMul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rrr_bias_mul, kRow, kBiasMul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT(gemm_rcr_bias_mul_add, kColumn, kBiasMulAdd)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT(gemm_rrr_bias_mul_add, kRow, kBiasMulAdd)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rcr_bias_add_relu, kColumn, kBiasAddRelu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rrr_bias_add_relu, kRow, kBiasAddRelu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT(gemm_rcr_bias_add_add_relu, kColumn, kBiasAddAddRelu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT(gemm_rrr_bias_add_add_relu, kRow, kBiasAddAddRelu)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rcr_bias_sigmoid_mul, kColumn, kBiasSigmoidMul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rrr_bias_sigmoid_mul, kRow, kBiasSigmoidMul)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rcr_bias_sigmoid_mul_tanh, kColumn, kBiasSigmoidMulTanh)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rrr_bias_sigmoid_mul_tanh, kRow, kBiasSigmoidMulTanh)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rcr_bias_mul_tanh, kColumn, kBiasMulTanh)
DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rrr_bias_mul_tanh, kRow, kBiasMulTanh)

#undef DINOML_ROCM_TILE_GEMM_EXPORT
#undef DINOML_ROCM_TILE_GEMM_BIAS_EXPORT
#undef DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL_EXPORT
#undef DINOML_ROCM_TILE_GEMM_BIAS_RESIDUAL2_EXPORT

#define DINOML_ROCM_TILE_BMM_EXPORT(OP, A_LAYOUT, B_LAYOUT, C_LAYOUT) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* a, const float* b, float* c, int batch_count, int m, int n, int k, \
    int64_t batch_stride_a, int64_t batch_stride_b, int64_t batch_stride_c, \
    int lda, int ldb, int ldc, hipStream_t stream) { \
  return launch_bmm<MatrixLayout::A_LAYOUT, MatrixLayout::B_LAYOUT, MatrixLayout::C_LAYOUT, false>( \
      a, b, nullptr, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, 0, batch_stride_c, lda, ldb, 0, ldc, stream); \
}

#define DINOML_ROCM_TILE_BMM_ADD_EXPORT(OP, A_LAYOUT, B_LAYOUT, C_LAYOUT) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* a, const float* b, const float* d0, float* c, int batch_count, int m, int n, int k, \
    int64_t batch_stride_a, int64_t batch_stride_b, int64_t batch_stride_d0, int64_t batch_stride_c, \
    int lda, int ldb, int ldd0, int ldc, hipStream_t stream) { \
  return launch_bmm<MatrixLayout::A_LAYOUT, MatrixLayout::B_LAYOUT, MatrixLayout::C_LAYOUT, true>( \
      a, b, d0, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_d0, batch_stride_c, lda, ldb, ldd0, ldc, stream); \
}

DINOML_ROCM_TILE_BMM_EXPORT(bmm_ccc, kColumn, kColumn, kColumn)
DINOML_ROCM_TILE_BMM_EXPORT(bmm_ccr, kColumn, kColumn, kRow)
DINOML_ROCM_TILE_BMM_EXPORT(bmm_crc, kColumn, kRow, kColumn)
DINOML_ROCM_TILE_BMM_EXPORT(bmm_crr, kColumn, kRow, kRow)
DINOML_ROCM_TILE_BMM_EXPORT(bmm_rcc, kRow, kColumn, kColumn)
DINOML_ROCM_TILE_BMM_EXPORT(bmm_rcr, kRow, kColumn, kRow)
DINOML_ROCM_TILE_BMM_EXPORT(bmm_rrc, kRow, kRow, kColumn)
DINOML_ROCM_TILE_BMM_EXPORT(bmm_rrr, kRow, kRow, kRow)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_ccc_add, kColumn, kColumn, kColumn)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_ccr_add, kColumn, kColumn, kRow)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_crc_add, kColumn, kRow, kColumn)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_crr_add, kColumn, kRow, kRow)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_rcc_add, kRow, kColumn, kColumn)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_rcr_add, kRow, kColumn, kRow)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_rrc_add, kRow, kRow, kColumn)
DINOML_ROCM_TILE_BMM_ADD_EXPORT(bmm_rrr_add, kRow, kRow, kRow)

#undef DINOML_ROCM_TILE_BMM_EXPORT
#undef DINOML_ROCM_TILE_BMM_ADD_EXPORT

#define DINOML_ROCM_TILE_CONV_EXPORT(OP, ADD, RELU) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* x, const float* weight, const float* bias, float* output, \
    int batch, int in_channels, int in_height, int in_width, int out_channels, \
    int kernel_h, int kernel_w, int out_height, int out_width, int stride_h, int stride_w, \
    int pad_h, int pad_w, int dilation_h, int dilation_w, hipStream_t stream) { \
  return launch_conv2d_bias<ADD, RELU>( \
      x, weight, bias, nullptr, output, batch, in_channels, in_height, in_width, out_channels, \
      kernel_h, kernel_w, out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
}

#define DINOML_ROCM_TILE_CONV_ADD_EXPORT(OP, ADD, RELU) \
extern "C" DINO_EXPORT int dinoml_rocm_tile_##OP##_float32_v1( \
    const float* x, const float* weight, const float* bias, const float* residual, float* output, \
    int batch, int in_channels, int in_height, int in_width, int out_channels, \
    int kernel_h, int kernel_w, int out_height, int out_width, int stride_h, int stride_w, \
    int pad_h, int pad_w, int dilation_h, int dilation_w, hipStream_t stream) { \
  return launch_conv2d_bias<ADD, RELU>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, \
      kernel_h, kernel_w, out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
}

DINOML_ROCM_TILE_CONV_EXPORT(conv2d_bias, false, false)
DINOML_ROCM_TILE_CONV_EXPORT(conv2d_bias_relu, false, true)
DINOML_ROCM_TILE_CONV_ADD_EXPORT(conv2d_bias_add, true, false)
DINOML_ROCM_TILE_CONV_ADD_EXPORT(conv2d_bias_add_relu, true, true)

#undef DINOML_ROCM_TILE_CONV_EXPORT
#undef DINOML_ROCM_TILE_CONV_ADD_EXPORT
