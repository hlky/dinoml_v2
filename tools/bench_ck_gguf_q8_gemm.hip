#include <hip/hip_runtime.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_xdl_skip_b_lds.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kQ8BlockSize = 32;
constexpr int kQ4_0BlockSize = 32;
constexpr int kQKBlockSize = 256;
constexpr int kKScaleSize = 12;
constexpr int kMxFp4BlockSize = 32;
constexpr int kNvFp4BlockSize = 64;

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

struct Options {
  int m = 4;
  int n = 4096;
  int k = 4096;
  int warmup = 10;
  int iters = 50;
  std::string qtype = "q8_0";
  std::string sweep;
  std::string fused_candidate;
  std::vector<std::string> fused_candidate_names;
  std::string dense_candidate;
  std::vector<std::string> dense_candidate_names;
  bool verify = true;
  bool profile_fused_f16 = true;
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using EmptyTuple = ck::Tuple<>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
static constexpr auto GemmMNKPadding =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

enum class DenseCkConfig {
  kBaseline,
  kWideM,
  kSmall64x64,
  kSmall64x32,
};

template <DenseCkConfig Config>
struct DenseCkTraits;

template <>
struct DenseCkTraits<DenseCkConfig::kBaseline> {
  static constexpr const char* Name = "ck_xdl_baseline_128x128x32";
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 1;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 1;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <>
struct DenseCkTraits<DenseCkConfig::kWideM> {
  static constexpr const char* Name = "ck_xdl_wide_m_256x128x32";
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 256;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 8;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 2;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct DenseCkTraits<DenseCkConfig::kSmall64x64> {
  static constexpr const char* Name = "ck_xdl_small_64x64x32";
  static constexpr ck::index_t BlockSize = 64;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 64;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 1;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 8;
  using CDEBlockTransferClusterLengths = S<1, 16, 1, 4>;
};

template <>
struct DenseCkTraits<DenseCkConfig::kSmall64x32> {
  static constexpr const char* Name = "ck_xdl_small_64x32x64";
  static constexpr ck::index_t BlockSize = 64;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 32;
  static constexpr ck::index_t KPerBlock = 64;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 1;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 8;
  using CDEBlockTransferClusterLengths = S<1, 16, 1, 4>;
};

template <DenseCkConfig Config, typename BLayout>
using DenseCkDeviceGemm = ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<
    Row,
    BLayout,
    EmptyTuple,
    Row,
    ck::half_t,
    ck::half_t,
    float,
    float,
    EmptyTuple,
    float,
    PassThrough,
    PassThrough,
    PassThrough,
    GemmMNKPadding,
    1,
    DenseCkTraits<Config>::BlockSize,
    DenseCkTraits<Config>::MPerBlock,
    DenseCkTraits<Config>::NPerBlock,
    DenseCkTraits<Config>::KPerBlock,
    DenseCkTraits<Config>::AK1,
    DenseCkTraits<Config>::BK1,
    DenseCkTraits<Config>::MPerXDL,
    DenseCkTraits<Config>::NPerXDL,
    DenseCkTraits<Config>::MXdlPerWave,
    DenseCkTraits<Config>::NXdlPerWave,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    DenseCkTraits<Config>::ABlockTransferSrcScalarPerVector,
    DenseCkTraits<Config>::ABlockTransferDstScalarPerVector_AK1,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    DenseCkTraits<Config>::BBlockTransferSrcScalarPerVector,
    DenseCkTraits<Config>::BBlockTransferDstScalarPerVector_BK1,
    1,
    1,
    1,
    typename DenseCkTraits<Config>::CDEBlockTransferClusterLengths,
    DenseCkTraits<Config>::CDEBlockTransferScalarPerVector_NPerBlock,
    ck::LoopScheduler::Default,
    ck::PipelineVersion::v1>;

using DenseCkSkinnySkipBLdsGemm = ck::tensor_operation::device::DeviceGemmXdlSkipBLds<
    ck::half_t,
    ck::half_t,
    float,
    float,
    Row,
    Col,
    Row,
    PassThrough,
    PassThrough,
    PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNPadding,
    256,
    16,
    64,
    4,
    8,
    16,
    16,
    1,
    1,
    S<16, 16, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    8,
    8,
    true,
    8,
    8,
    7,
    1>;

using DenseCkDlSkinnyGemm = ck::tensor_operation::device::DeviceGemmDl<
    ck::half_t,
    ck::half_t,
    float,
    float,
    Row,
    Col,
    Row,
    PassThrough,
    PassThrough,
    PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNPadding,
    64,
    16,
    64,
    16,
    2,
    1,
    4,
    1,
    S<2, 4>,
    S<2, 4>,
    S<4, 1, 1, 2>,
    S<4, 1, 16, 1>,
    S<1, 2, 0, 3>,
    S<1, 2, 0, 3>,
    S<4, 1, 1, 2>,
    S<1, 2, 0, 3>,
    S<1, 1, 1, 2>,
    S<4, 1, 4, 2>,
    S<4, 1, 16, 1>,
    S<1, 2, 0, 3>,
    S<1, 2, 0, 3>,
    S<4, 1, 1, 2>,
    S<1, 2, 0, 3>,
    S<1, 1, 1, 2>,
    S<0, 1, 2, 3, 4, 5>,
    5,
    1>;

using DenseCkDlSkinnyGemmKn = ck::tensor_operation::device::DeviceGemmDl<
    ck::half_t,
    ck::half_t,
    float,
    float,
    Row,
    Row,
    Row,
    PassThrough,
    PassThrough,
    PassThrough,
    ck::tensor_operation::device::GemmSpecialization::MNPadding,
    64,
    16,
    64,
    16,
    2,
    1,
    4,
    1,
    S<2, 4>,
    S<2, 4>,
    S<4, 1, 1, 2>,
    S<4, 1, 16, 1>,
    S<0, 3, 1, 2>,
    S<0, 3, 1, 2>,
    S<4, 1, 1, 2>,
    S<0, 3, 1, 2>,
    S<1, 1, 1, 2>,
    S<4, 1, 4, 2>,
    S<4, 1, 16, 1>,
    S<0, 3, 1, 2>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 1>,
    S<0, 3, 1, 2>,
    S<1, 1, 4, 2>,
    S<0, 1, 2, 3, 4, 5>,
    5,
    1>;

void check_hip(hipError_t status, const char* what) {
  if (status != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(status));
  }
}

template <typename T>
void hip_malloc(T** ptr, std::size_t count) {
  check_hip(hipMalloc(reinterpret_cast<void**>(ptr), count * sizeof(T)), "hipMalloc");
}

uint32_t float_bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

float bits_float(uint32_t bits) {
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

uint16_t float_to_half_bits(float value) {
  const uint32_t bits = float_bits(value);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;

  if (exp <= 0) {
    if (exp < -10) {
      return static_cast<uint16_t>(sign);
    }
    mant = (mant | 0x800000u) >> (1 - exp);
    return static_cast<uint16_t>(sign | ((mant + 0x1000u) >> 13));
  }
  if (exp >= 31) {
    return static_cast<uint16_t>(sign | 0x7c00u);
  }

  uint32_t half_mant = (mant + 0x1000u) >> 13;
  if (half_mant == 0x400u) {
    half_mant = 0;
    ++exp;
    if (exp >= 31) {
      return static_cast<uint16_t>(sign | 0x7c00u);
    }
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | half_mant);
}

float half_bits_to_float(uint16_t half_bits) {
  const uint32_t sign = (static_cast<uint32_t>(half_bits & 0x8000u)) << 16;
  uint32_t exp = (half_bits >> 10) & 0x1fu;
  uint32_t mant = half_bits & 0x03ffu;

  if (exp == 0) {
    if (mant == 0) {
      return bits_float(sign);
    }
    while ((mant & 0x0400u) == 0) {
      mant <<= 1;
      --exp;
    }
    ++exp;
    mant &= ~0x0400u;
  } else if (exp == 31) {
    return bits_float(sign | 0x7f800000u | (mant << 13));
  }

  exp = exp + (127 - 15);
  return bits_float(sign | (exp << 23) | (mant << 13));
}

std::vector<std::string> split_csv(std::string value) {
  std::vector<std::string> parts;
  std::size_t start = 0;
  while (start <= value.size()) {
    const std::size_t end = value.find(',', start);
    std::string part = value.substr(start, end == std::string::npos ? std::string::npos : end - start);
    const std::size_t first = part.find_first_not_of(" \t");
    const std::size_t last = part.find_last_not_of(" \t");
    if (first != std::string::npos) {
      parts.push_back(part.substr(first, last - first + 1));
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return parts;
}

bool candidate_selected(
    const std::string& name,
    const std::string& single_filter,
    const std::vector<std::string>& filter_names) {
  if (!single_filter.empty()) {
    return name == single_filter;
  }
  if (filter_names.empty()) {
    return true;
  }
  return std::find(filter_names.begin(), filter_names.end(), name) != filter_names.end();
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto require_value = [&](const char* name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::stoi(argv[++i]);
    };
    auto require_string = [&](const char* name) {
      if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
      }
      return std::string(argv[++i]);
    };

    if (arg == "--m") {
      options.m = require_value("--m");
    } else if (arg == "--n") {
      options.n = require_value("--n");
    } else if (arg == "--k") {
      options.k = require_value("--k");
    } else if (arg == "--warmup") {
      options.warmup = require_value("--warmup");
    } else if (arg == "--iters") {
      options.iters = require_value("--iters");
    } else if (arg == "--qtype") {
      options.qtype = require_string("--qtype");
    } else if (arg == "--fused-candidate") {
      options.fused_candidate = require_string("--fused-candidate");
    } else if (arg == "--fused-candidates") {
      options.fused_candidate_names = split_csv(require_string("--fused-candidates"));
    } else if (arg == "--dense-candidate") {
      options.dense_candidate = require_string("--dense-candidate");
    } else if (arg == "--dense-candidates") {
      options.dense_candidate_names = split_csv(require_string("--dense-candidates"));
    } else if (arg == "--no-f16out" || arg == "--f32-only") {
      options.profile_fused_f16 = false;
    } else if (arg == "--sweep") {
      options.sweep = require_string("--sweep");
      options.verify = false;
    } else if (arg == "--no-verify") {
      options.verify = false;
    } else if (arg == "--verify") {
      options.verify = true;
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }
  return options;
}

float time_repeated(const std::string& label, int warmup, int iters, const std::function<void()>& launch) {
  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  check_hip(hipDeviceSynchronize(), (label + " warmup").c_str());

  hipEvent_t start{};
  hipEvent_t stop{};
  check_hip(hipEventCreate(&start), "hipEventCreate(start)");
  check_hip(hipEventCreate(&stop), "hipEventCreate(stop)");
  check_hip(hipEventRecord(start), "hipEventRecord(start)");
  for (int i = 0; i < iters; ++i) {
    launch();
  }
  check_hip(hipEventRecord(stop), "hipEventRecord(stop)");
  check_hip(hipEventSynchronize(stop), "hipEventSynchronize(stop)");
  float elapsed_ms = 0.0f;
  check_hip(hipEventElapsedTime(&elapsed_ms, start, stop), "hipEventElapsedTime");
  check_hip(hipEventDestroy(start), "hipEventDestroy(start)");
  check_hip(hipEventDestroy(stop), "hipEventDestroy(stop)");
  return elapsed_ms / static_cast<float>(iters);
}

template <DenseCkConfig Config, typename BLayout>
int launch_dense_ck_half_f32(
    const half* a,
    const half* b,
    float* c,
    int m,
    int n,
    int k,
    ck::index_t stride_b,
    hipStream_t stream) {
  using DeviceOp = DenseCkDeviceGemm<Config, BLayout>;
  const auto m_ck = static_cast<ck::index_t>(m);
  const auto n_ck = static_cast<ck::index_t>(n);
  const auto k_ck = static_cast<ck::index_t>(k);
  auto argument = DeviceOp::MakeArgument(
      reinterpret_cast<const ck::half_t*>(a),
      reinterpret_cast<const ck::half_t*>(b),
      std::array<const void*, 0>{},
      c,
      m_ck,
      n_ck,
      k_ck,
      k_ck,
      stride_b,
      std::array<ck::index_t, 0>{},
      n_ck,
      PassThrough{},
      PassThrough{},
      PassThrough{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  return hipGetLastError() == hipSuccess ? 0 : 3;
}

int launch_dense_ck_skip_b_lds_half_f32_nk(
    const half* a,
    const half* b,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  const auto m_ck = static_cast<ck::index_t>(m);
  const auto n_ck = static_cast<ck::index_t>(n);
  const auto k_ck = static_cast<ck::index_t>(k);
  auto argument = DenseCkSkinnySkipBLdsGemm::MakeArgument(
      reinterpret_cast<const ck::half_t*>(a),
      reinterpret_cast<const ck::half_t*>(b),
      c,
      m_ck,
      n_ck,
      k_ck,
      k_ck,
      k_ck,
      n_ck,
      PassThrough{},
      PassThrough{},
      PassThrough{});
  if (!DenseCkSkinnySkipBLdsGemm::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DenseCkSkinnySkipBLdsGemm::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  return hipGetLastError() == hipSuccess ? 0 : 3;
}

int launch_dense_ck_dl_half_f32_nk(
    const half* a,
    const half* b,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  const auto m_ck = static_cast<ck::index_t>(m);
  const auto n_ck = static_cast<ck::index_t>(n);
  const auto k_ck = static_cast<ck::index_t>(k);
  auto argument = DenseCkDlSkinnyGemm::MakeArgument(
      reinterpret_cast<const ck::half_t*>(a),
      reinterpret_cast<const ck::half_t*>(b),
      c,
      m_ck,
      n_ck,
      k_ck,
      k_ck,
      k_ck,
      n_ck,
      PassThrough{},
      PassThrough{},
      PassThrough{});
  if (!DenseCkDlSkinnyGemm::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DenseCkDlSkinnyGemm::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  return hipGetLastError() == hipSuccess ? 0 : 3;
}

int launch_dense_ck_dl_half_f32_kn(
    const half* a,
    const half* b,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  const auto m_ck = static_cast<ck::index_t>(m);
  const auto n_ck = static_cast<ck::index_t>(n);
  const auto k_ck = static_cast<ck::index_t>(k);
  auto argument = DenseCkDlSkinnyGemmKn::MakeArgument(
      reinterpret_cast<const ck::half_t*>(a),
      reinterpret_cast<const ck::half_t*>(b),
      c,
      m_ck,
      n_ck,
      k_ck,
      k_ck,
      n_ck,
      n_ck,
      PassThrough{},
      PassThrough{},
      PassThrough{});
  if (!DenseCkDlSkinnyGemmKn::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DenseCkDlSkinnyGemmKn::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  return hipGetLastError() == hipSuccess ? 0 : 3;
}

struct CandidateResult {
  std::string name;
  float ms = std::numeric_limits<float>::infinity();
  bool supported = false;
  std::string error;
};

CandidateResult profile_candidate(
    const std::string& name,
    int warmup,
    int iters,
    const std::function<int()>& launch) {
  CandidateResult result{name};
  try {
    result.ms = time_repeated(name, warmup, iters, [&]() {
      const int rc = launch();
      if (rc != 0) {
        throw std::runtime_error(name + " returned " + std::to_string(rc));
      }
    });
    result.supported = true;
  } catch (const std::exception& error) {
    result.supported = false;
    result.error = error.what();
  }
  return result;
}

const CandidateResult& best_supported(const std::vector<CandidateResult>& candidates, const char* label) {
  const CandidateResult* best = nullptr;
  for (const CandidateResult& candidate : candidates) {
    if (!candidate.supported) {
      continue;
    }
    if (best == nullptr || candidate.ms < best->ms) {
      best = &candidate;
    }
  }
  if (best == nullptr) {
    throw std::runtime_error(std::string("no supported candidates for ") + label);
  }
  return *best;
}

std::vector<float> reference_gemm_q8(
    const std::vector<float>& a,
    const std::vector<GgufQ8_0Block>& b,
    int m,
    int n,
    int k) {
  const int qblocks_per_row = k / kQ8BlockSize;
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int qb = 0; qb < qblocks_per_row; ++qb) {
        const GgufQ8_0Block& block = b[static_cast<std::size_t>(col) * qblocks_per_row + qb];
        const float scale = half_bits_to_float(block.d);
        for (int kk = 0; kk < kQ8BlockSize; ++kk) {
          acc += a[static_cast<std::size_t>(row) * k + qb * kQ8BlockSize + kk] *
                 (scale * static_cast<float>(block.qs[kk]));
        }
      }
      c[static_cast<std::size_t>(row) * n + col] = acc;
    }
  }
  return c;
}

std::vector<float> reference_gemm_dequant_f16(
    const std::vector<float>& a,
    const std::vector<GgufQ8_0Block>& b,
    int m,
    int n,
    int k) {
  const int qblocks_per_row = k / kQ8BlockSize;
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int qb = 0; qb < qblocks_per_row; ++qb) {
        const GgufQ8_0Block& block = b[static_cast<std::size_t>(col) * qblocks_per_row + qb];
        const float scale = half_bits_to_float(block.d);
        for (int kk = 0; kk < kQ8BlockSize; ++kk) {
          const float b_value = half_bits_to_float(float_to_half_bits(scale * static_cast<float>(block.qs[kk])));
          acc += a[static_cast<std::size_t>(row) * k + qb * kQ8BlockSize + kk] * b_value;
        }
      }
      c[static_cast<std::size_t>(row) * n + col] = acc;
    }
  }
  return c;
}

std::pair<float, float> max_errors(const std::vector<float>& expected, const std::vector<float>& actual) {
  float max_abs = 0.0f;
  float max_rel = 0.0f;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const float abs_err = std::abs(expected[i] - actual[i]);
    const float rel_err = abs_err / std::max(1.0f, std::abs(expected[i]));
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
  }
  return {max_abs, max_rel};
}

} // namespace

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n5_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n6_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_exact_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile4x64_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile16x16_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_dequant_b_f16_nk_v1(
    const void* b_q8_0,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_dequant_b_f16_kn_v1(
    const void* b_q8_0,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_opt_v1(
    const half* a,
    const void* b_q8_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q8_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_0,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q4_0,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_dequant_b_f16_nk_v1(
    const void* b_q4_0,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_0_dequant_b_f16_kn_v1(
    const void* b_q4_0,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q4_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q4_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_dequant_b_f16_nk_v1(
    const void* b_q4_k,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q4_k_dequant_b_f16_kn_v1(
    const void* b_q4_k,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n1_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n2_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n3_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n4_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m16n1_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m16n2_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_direct_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m3_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m4_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m8_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m12_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m16_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_direct_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m2_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m3_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m4_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m8_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m12_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m16_v1(
    const half* a,
    const void* b_q5_k,
    half* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
    const half* a,
    const void* b_q5_k,
    float* c,
    int m,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_dequant_b_f16_nk_v1(
    const void* b_q5_k,
    half* b_nk,
    int n,
    int k,
    hipStream_t stream);

extern "C" int dinoml_ck_gguf_q5_k_dequant_b_f16_kn_v1(
    const void* b_q5_k,
    half* b_kn,
    int n,
    int k,
    hipStream_t stream);

#define DINOML_DECLARE_GGUF_MNBLOCK_F32_EXPORTS(prefix)                                                   \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1(                          \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);              \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1(                          \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);              \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1(                          \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);              \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1(                          \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);              \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_opt_v1(                                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);

DINOML_DECLARE_GGUF_MNBLOCK_F32_EXPORTS(q4_0)
DINOML_DECLARE_GGUF_MNBLOCK_F32_EXPORTS(q4_k)
DINOML_DECLARE_GGUF_MNBLOCK_F32_EXPORTS(q5_k)

#undef DINOML_DECLARE_GGUF_MNBLOCK_F32_EXPORTS

#define DINOML_DECLARE_GGUF_QUANT_EXPORTS(prefix)                                                           \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_v1(                                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n1_v1(                                 \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n2_v1(                                 \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n4_v1(                                 \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m16n1_v1(                                \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m16n2_v1(                                \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_direct_v1(                          \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m2_v1(                              \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4_v1(                              \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6_v1(                              \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8_v1(                              \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m10_v1(                             \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m12_v1(                             \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m16_v1(                             \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m2_v1(                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m4_v1(                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m8_v1(                      \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8w2_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m12w2_v1(                           \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1(                            \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_opt_v1(                                        \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(                           \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(                           \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(                           \
      const half* a, const void* b_quant, float* c, int m, int n, int k, hipStream_t stream);               \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_direct_v1(                          \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream);                \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m2_v1(                              \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream);                \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m4_v1(                              \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream);                \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m8_v1(                              \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream);                \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m12_v1(                             \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream);                \
  extern "C" int dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m16_v1(                             \
      const half* a, const void* b_quant, half* c, int m, int n, int k, hipStream_t stream);                \
  extern "C" int dinoml_ck_gguf_##prefix##_dequant_b_f16_nk_v1(                                           \
      const void* b_quant, half* b_nk, int n, int k, hipStream_t stream);                                  \
  extern "C" int dinoml_ck_gguf_##prefix##_dequant_b_f16_kn_v1(                                           \
      const void* b_quant, half* b_kn, int n, int k, hipStream_t stream);

DINOML_DECLARE_GGUF_QUANT_EXPORTS(q2_k)
DINOML_DECLARE_GGUF_QUANT_EXPORTS(q3_k)
DINOML_DECLARE_GGUF_QUANT_EXPORTS(q6_k)
DINOML_DECLARE_GGUF_QUANT_EXPORTS(mxfp4)
DINOML_DECLARE_GGUF_QUANT_EXPORTS(nvfp4)

#undef DINOML_DECLARE_GGUF_QUANT_EXPORTS

namespace {

using FusedLaunch = int (*)(const half*, const void*, float*, int, int, int, hipStream_t);
using FusedF16Launch = int (*)(const half*, const void*, half*, int, int, int, hipStream_t);
using DequantLaunch = int (*)(const void*, half*, int, int, hipStream_t);
using FillBlocks = void (*)(std::vector<uint8_t>&, std::mt19937&);
using QuantValue = float (*)(const std::vector<uint8_t>&, std::size_t, int);

struct FusedCandidateSpec {
  const char* name;
  FusedLaunch launch;
};

struct FusedF16CandidateSpec {
  const char* name;
  FusedF16Launch launch;
};

struct QtypeSpec {
  const char* name;
  int block_size;
  std::size_t block_bytes;
  FillBlocks fill_blocks;
  QuantValue value;
  DequantLaunch dequant_nk;
  DequantLaunch dequant_kn;
  std::vector<FusedCandidateSpec> fused_candidates;
  std::vector<FusedF16CandidateSpec> fused_f16_candidates;
};

struct Shape {
  int m;
  int n;
  int k;
};

void get_scale_min_k4_host(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
  if (j < 4) {
    *d = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
  }
}

int q3_k_scale_host(const uint8_t* scales, int is) {
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

float e8m0_to_fp32_half_host(uint8_t x) {
  const uint32_t bits = x < 2 ? (0x00200000u << x) : (static_cast<uint32_t>(x - 1) << 23);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

float ue4m3_to_fp32_host(uint8_t x) {
  if (x == 0 || x == 0x7f) {
    return 0.0f;
  }
  const int exp = static_cast<int>((x >> 3) & 0x0f);
  const int man = static_cast<int>(x & 0x07);
  const float raw = exp == 0 ? std::ldexp(static_cast<float>(man), -9)
                             : std::ldexp(1.0f + static_cast<float>(man) * 0.125f, exp - 7);
  return 0.5f * raw;
}

int fp4_quant_value_host(uint8_t q) {
  static constexpr int kValues[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return kValues[q & 0x0f];
}

float q8_0_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufQ8_0Block*>(bytes.data());
  const GgufQ8_0Block& block = blocks[block_index];
  return half_bits_to_float(block.d) * static_cast<float>(block.qs[kk]);
}

float q4_0_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufQ4_0Block*>(bytes.data());
  const GgufQ4_0Block& block = blocks[block_index];
  const int byte = kk & 15;
  const uint8_t packed = block.qs[byte];
  const int q = kk < 16 ? static_cast<int>(packed & 0x0f) : static_cast<int>(packed >> 4);
  return half_bits_to_float(block.d) * static_cast<float>(q - 8);
}

float q4_k_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufQ4_KBlock*>(bytes.data());
  const GgufQ4_KBlock& block = blocks[block_index];
  const int group64 = kk >> 6;
  const int local64 = kk & 63;
  const int high_nibble = local64 >= 32 ? 1 : 0;
  const int local32 = local64 & 31;
  uint8_t sc = 0;
  uint8_t m = 0;
  get_scale_min_k4_host(group64 * 2 + high_nibble, block.scales, &sc, &m);
  const uint8_t packed = block.qs[group64 * 32 + local32];
  const int q = high_nibble ? static_cast<int>(packed >> 4) : static_cast<int>(packed & 0x0f);
  return half_bits_to_float(block.d) * static_cast<float>(sc) * static_cast<float>(q) -
         half_bits_to_float(block.dmin) * static_cast<float>(m);
}

float q5_k_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufQ5_KBlock*>(bytes.data());
  const GgufQ5_KBlock& block = blocks[block_index];
  const int group64 = kk >> 6;
  const int local64 = kk & 63;
  const int high_nibble = local64 >= 32 ? 1 : 0;
  const int local32 = local64 & 31;
  uint8_t sc = 0;
  uint8_t m = 0;
  get_scale_min_k4_host(group64 * 2 + high_nibble, block.scales, &sc, &m);
  const uint8_t packed = block.qs[group64 * 32 + local32];
  const uint8_t bit = static_cast<uint8_t>((high_nibble ? 2u : 1u) << (2 * group64));
  const int low = high_nibble ? static_cast<int>(packed >> 4) : static_cast<int>(packed & 0x0f);
  const int q = low + ((block.qh[local32] & bit) ? 16 : 0);
  return half_bits_to_float(block.d) * static_cast<float>(sc) * static_cast<float>(q) -
         half_bits_to_float(block.dmin) * static_cast<float>(m);
}

float q2_k_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufQ2_KBlock*>(bytes.data());
  const GgufQ2_KBlock& block = blocks[block_index];
  const int half_block = kk >> 7;
  const int local128 = kk & 127;
  const int j = local128 >> 5;
  const int local32 = local128 & 31;
  const int scale_index = half_block * 8 + 2 * j + (local32 >= 16 ? 1 : 0);
  const int q = static_cast<int>((block.qs[half_block * 32 + local32] >> (2 * j)) & 0x03);
  return half_bits_to_float(block.d) *
             static_cast<float>(block.scales[scale_index] & 0x0f) * static_cast<float>(q) -
         half_bits_to_float(block.dmin) * static_cast<float>(block.scales[scale_index] >> 4);
}

float q3_k_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufQ3_KBlock*>(bytes.data());
  const GgufQ3_KBlock& block = blocks[block_index];
  const int half_block = kk >> 7;
  const int local128 = kk & 127;
  const int j = local128 >> 5;
  const int local32 = local128 & 31;
  const int is = 8 * half_block + 2 * j + (local32 >= 16 ? 1 : 0);
  const int scale = q3_k_scale_host(block.scales, is) - 32;
  const uint8_t mask = static_cast<uint8_t>(1u << (4 * half_block + j));
  const int low = static_cast<int>((block.qs[half_block * 32 + local32] >> (2 * j)) & 0x03);
  const int q = low - ((block.hmask[local32] & mask) ? 0 : 4);
  return half_bits_to_float(block.d) * static_cast<float>(scale) * static_cast<float>(q);
}

float q6_k_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufQ6_KBlock*>(bytes.data());
  const GgufQ6_KBlock& block = blocks[block_index];
  const int half_block = kk >> 7;
  const int local128 = kk & 127;
  const int quad = local128 >> 5;
  const int local32 = local128 & 31;
  const int ql_index = half_block * 64 + ((quad & 1) ? 32 : 0) + local32;
  const int qh_index = half_block * 32 + local32;
  const int scale_index = half_block * 8 + 2 * quad + (local32 >= 16 ? 1 : 0);
  const int low = quad >= 2 ? static_cast<int>(block.ql[ql_index] >> 4)
                            : static_cast<int>(block.ql[ql_index] & 0x0f);
  const int high = static_cast<int>((block.qh[qh_index] >> (2 * quad)) & 0x03) << 4;
  const int q = (low | high) - 32;
  return half_bits_to_float(block.d) * static_cast<float>(block.scales[scale_index]) * static_cast<float>(q);
}

float mxfp4_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufMxFp4Block*>(bytes.data());
  const GgufMxFp4Block& block = blocks[block_index];
  const uint8_t packed = block.qs[kk & 15];
  const int q = fp4_quant_value_host(kk < 16 ? (packed & 0x0f) : (packed >> 4));
  return e8m0_to_fp32_half_host(block.e) * static_cast<float>(q);
}

float nvfp4_value_host(const std::vector<uint8_t>& bytes, std::size_t block_index, int kk) {
  const auto* blocks = reinterpret_cast<const GgufNvFp4Block*>(bytes.data());
  const GgufNvFp4Block& block = blocks[block_index];
  const int sub = kk >> 4;
  const int local16 = kk & 15;
  const uint8_t packed = block.qs[sub * 8 + (local16 & 7)];
  const int q = fp4_quant_value_host(local16 < 8 ? (packed & 0x0f) : (packed >> 4));
  return ue4m3_to_fp32_host(block.d[sub]) * static_cast<float>(q);
}

void fill_q8_0_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_real_distribution<float> scale_dist(0.0025f, 0.04f);
  std::uniform_int_distribution<int> q_dist(-127, 127);
  auto* blocks = reinterpret_cast<GgufQ8_0Block*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufQ8_0Block);
  for (std::size_t i = 0; i < count; ++i) {
    blocks[i].d = float_to_half_bits(scale_dist(rng));
    for (int kk = 0; kk < kQ8BlockSize; ++kk) {
      blocks[i].qs[kk] = static_cast<int8_t>(q_dist(rng));
    }
  }
}

void fill_q4_0_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_real_distribution<float> scale_dist(0.0025f, 0.04f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  auto* blocks = reinterpret_cast<GgufQ4_0Block*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufQ4_0Block);
  for (std::size_t i = 0; i < count; ++i) {
    blocks[i].d = float_to_half_bits(scale_dist(rng));
    for (uint8_t& q : blocks[i].qs) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
  }
}

void fill_q4_k_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_real_distribution<float> scale_dist(0.0002f, 0.002f);
  std::uniform_real_distribution<float> min_dist(0.0f, 0.001f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  auto* blocks = reinterpret_cast<GgufQ4_KBlock*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufQ4_KBlock);
  for (std::size_t i = 0; i < count; ++i) {
    blocks[i].d = float_to_half_bits(scale_dist(rng));
    blocks[i].dmin = float_to_half_bits(min_dist(rng));
    for (uint8_t& scale : blocks[i].scales) {
      scale = static_cast<uint8_t>(byte_dist(rng));
    }
    for (uint8_t& q : blocks[i].qs) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
  }
}

void fill_q5_k_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_real_distribution<float> scale_dist(0.00015f, 0.0015f);
  std::uniform_real_distribution<float> min_dist(0.0f, 0.001f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  auto* blocks = reinterpret_cast<GgufQ5_KBlock*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufQ5_KBlock);
  for (std::size_t i = 0; i < count; ++i) {
    blocks[i].d = float_to_half_bits(scale_dist(rng));
    blocks[i].dmin = float_to_half_bits(min_dist(rng));
    for (uint8_t& scale : blocks[i].scales) {
      scale = static_cast<uint8_t>(byte_dist(rng));
    }
    for (uint8_t& qh : blocks[i].qh) {
      qh = static_cast<uint8_t>(byte_dist(rng));
    }
    for (uint8_t& q : blocks[i].qs) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
  }
}

void fill_q2_k_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_real_distribution<float> scale_dist(0.0002f, 0.002f);
  std::uniform_real_distribution<float> min_dist(0.0f, 0.001f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  auto* blocks = reinterpret_cast<GgufQ2_KBlock*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufQ2_KBlock);
  for (std::size_t i = 0; i < count; ++i) {
    for (uint8_t& scale : blocks[i].scales) {
      scale = static_cast<uint8_t>(byte_dist(rng));
    }
    for (uint8_t& q : blocks[i].qs) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
    blocks[i].d = float_to_half_bits(scale_dist(rng));
    blocks[i].dmin = float_to_half_bits(min_dist(rng));
  }
}

void fill_q3_k_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_real_distribution<float> scale_dist(0.0002f, 0.002f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  auto* blocks = reinterpret_cast<GgufQ3_KBlock*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufQ3_KBlock);
  for (std::size_t i = 0; i < count; ++i) {
    for (uint8_t& h : blocks[i].hmask) {
      h = static_cast<uint8_t>(byte_dist(rng));
    }
    for (uint8_t& q : blocks[i].qs) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
    for (uint8_t& scale : blocks[i].scales) {
      scale = static_cast<uint8_t>(byte_dist(rng));
    }
    blocks[i].d = float_to_half_bits(scale_dist(rng));
  }
}

void fill_q6_k_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_real_distribution<float> scale_dist(0.00015f, 0.0015f);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  std::uniform_int_distribution<int> signed_scale_dist(-8, 8);
  auto* blocks = reinterpret_cast<GgufQ6_KBlock*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufQ6_KBlock);
  for (std::size_t i = 0; i < count; ++i) {
    for (uint8_t& q : blocks[i].ql) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
    for (uint8_t& q : blocks[i].qh) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
    for (int8_t& scale : blocks[i].scales) {
      scale = static_cast<int8_t>(signed_scale_dist(rng));
    }
    blocks[i].d = float_to_half_bits(scale_dist(rng));
  }
}

void fill_mxfp4_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_int_distribution<int> scale_dist(117, 122);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  auto* blocks = reinterpret_cast<GgufMxFp4Block*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufMxFp4Block);
  for (std::size_t i = 0; i < count; ++i) {
    blocks[i].e = static_cast<uint8_t>(scale_dist(rng));
    for (uint8_t& q : blocks[i].qs) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
  }
}

void fill_nvfp4_blocks(std::vector<uint8_t>& bytes, std::mt19937& rng) {
  std::uniform_int_distribution<int> scale_dist(32, 48);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  auto* blocks = reinterpret_cast<GgufNvFp4Block*>(bytes.data());
  const std::size_t count = bytes.size() / sizeof(GgufNvFp4Block);
  for (std::size_t i = 0; i < count; ++i) {
    for (uint8_t& d : blocks[i].d) {
      d = static_cast<uint8_t>(scale_dist(rng));
    }
    for (uint8_t& q : blocks[i].qs) {
      q = static_cast<uint8_t>(byte_dist(rng));
    }
  }
}

#define DINOML_GGUF_FUSED_F32_CANDIDATES(prefix)                                                         \
  {                                                                                                      \
      {"fused_opt", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_opt_v1},                                \
      {"fused_wave", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_v1},                              \
      {"fused_wave_m8n1", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n1_v1},                    \
      {"fused_wave_m8n2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n2_v1},                    \
      {"fused_wave_m8n4", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m8n4_v1},                    \
      {"fused_wave_m16n1", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m16n1_v1},                  \
      {"fused_wave_m16n2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m16n2_v1},                  \
      {"fused_wmma16x16_direct", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_direct_v1},      \
      {"fused_wmma16x16_m2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m2_v1},              \
      {"fused_wmma16x16_m4", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4_v1},              \
      {"fused_wmma16x16_m6", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6_v1},              \
      {"fused_wmma16x16_m8", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8_v1},              \
      {"fused_wmma16x16_m10", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m10_v1},            \
      {"fused_wmma16x16_m12", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m12_v1},            \
      {"fused_wmma16x16_m16", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m16_v1},            \
      {"fused_wmma16x16_sharedb_m2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m2_v1}, \
      {"fused_wmma16x16_sharedb_m4", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m4_v1}, \
      {"fused_wmma16x16_sharedb_m8", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_sharedb_m8_v1}, \
      {"fused_wmma16x16_m8w2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8w2_v1},          \
      {"fused_wmma16x16_m12w2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m12w2_v1},        \
      {"fused_wmma16x16_m4n2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1},          \
      {"fused_wmma16x16_m6n2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1},          \
      {"fused_wmma16x16_m8n2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1},          \
      {"fused_wmma16x16_m4n3", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1},          \
      {"fused_wave_m4n2_exact", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n2_exact_v1},        \
      {"fused_wave_m4n3_exact", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n3_exact_v1},        \
      {"fused_wave_m4n4_exact", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f32_wave_m4n4_exact_v1},        \
  }

#define DINOML_GGUF_FUSED_F16_CANDIDATES(prefix)                                                         \
  {                                                                                                      \
      {"wmma16x16_direct", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_direct_v1},            \
      {"wmma16x16_m2", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m2_v1},                    \
      {"wmma16x16_m4", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m4_v1},                    \
      {"wmma16x16_m8", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m8_v1},                    \
      {"wmma16x16_m12", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m12_v1},                  \
      {"wmma16x16_m16", dinoml_ck_gguf_##prefix##_gemm_rcr_fp16_f16_wmma16x16_m16_v1},                  \
  }

#define DINOML_GGUF_QTYPE_SPEC(name_literal, prefix, block_size_value, BlockType, fill_fn, value_fn)      \
  QtypeSpec {                                                                                            \
    name_literal, block_size_value, sizeof(BlockType), fill_fn, value_fn,                                \
        dinoml_ck_gguf_##prefix##_dequant_b_f16_nk_v1,                                                   \
        dinoml_ck_gguf_##prefix##_dequant_b_f16_kn_v1,                                                   \
        DINOML_GGUF_FUSED_F32_CANDIDATES(prefix), DINOML_GGUF_FUSED_F16_CANDIDATES(prefix)              \
  }

QtypeSpec make_qtype_spec(const std::string& qtype) {
  if (qtype == "q8_0") {
    return QtypeSpec{
        "q8_0",
        kQ8BlockSize,
        sizeof(GgufQ8_0Block),
        fill_q8_0_blocks,
        q8_0_value_host,
        dinoml_ck_gguf_q8_0_dequant_b_f16_nk_v1,
        dinoml_ck_gguf_q8_0_dequant_b_f16_kn_v1,
        {
            {"fused_opt", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_opt_v1},
            {"fused_wave4", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave4_v1},
            {"fused_wave_m8n1", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n1_v1},
            {"fused_wave_m8n2", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n2_v1},
            {"fused_wave_m8n3", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n3_v1},
            {"fused_wave_m8n4", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m8n4_v1},
            {"fused_wave_m16n1", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m16n1_v1},
            {"fused_wave_m16n2", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m16n2_v1},
            {"fused_wmma16x16", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_v1},
            {"fused_wmma16x16_direct", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_direct_v1},
            {"fused_wmma16x16_m2", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1},
            {"fused_wmma16x16_m3", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m3_v1},
            {"fused_wmma16x16_m4", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4_v1},
            {"fused_wmma16x16_m8", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m8_v1},
            {"fused_wmma16x16_m12", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m12_v1},
            {"fused_wmma16x16_m16", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m16_v1},
            {"fused_wmma16x16_m4n2", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1},
            {"fused_wmma16x16_m6n2", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1},
            {"fused_wmma16x16_m8n2", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1},
            {"fused_wmma16x16_m4n3", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1},
            {"fused_wave_m4n2_exact", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1},
            {"fused_wave_m4n3_exact", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1},
            {"fused_wave_m4n4_exact", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1},
        },
        {
            {"wmma16x16_direct", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_direct_v1},
            {"wmma16x16_m2", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m2_v1},
            {"wmma16x16_m3", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m3_v1},
            {"wmma16x16_m4", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m4_v1},
            {"wmma16x16_m8", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m8_v1},
            {"wmma16x16_m12", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m12_v1},
            {"wmma16x16_m16", dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f16_wmma16x16_m16_v1},
        }};
  }
  if (qtype == "q4_0") {
    return QtypeSpec{
        "q4_0",
        kQ4_0BlockSize,
        sizeof(GgufQ4_0Block),
        fill_q4_0_blocks,
        q4_0_value_host,
        dinoml_ck_gguf_q4_0_dequant_b_f16_nk_v1,
        dinoml_ck_gguf_q4_0_dequant_b_f16_kn_v1,
        {
            {"fused_opt", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_opt_v1},
            {"fused_wave", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_v1},
            {"fused_wave_m8n1", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n1_v1},
            {"fused_wave_m8n2", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n2_v1},
            {"fused_wave_m8n3", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n3_v1},
            {"fused_wave_m8n4", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m8n4_v1},
            {"fused_wave_m16n1", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m16n1_v1},
            {"fused_wave_m16n2", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m16n2_v1},
            {"fused_wmma16x16", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_v1},
            {"fused_wmma16x16_direct", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_direct_v1},
            {"fused_wmma16x16_m2", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m2_v1},
            {"fused_wmma16x16_m3", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m3_v1},
            {"fused_wmma16x16_m4", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m4_v1},
            {"fused_wmma16x16_m8", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m8_v1},
            {"fused_wmma16x16_m12", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m12_v1},
            {"fused_wmma16x16_m16", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m16_v1},
            {"fused_wmma16x16_m4n2", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1},
            {"fused_wmma16x16_m6n2", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1},
            {"fused_wmma16x16_m8n2", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1},
            {"fused_wmma16x16_m4n3", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1},
            {"fused_wave_m4n2_exact", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1},
            {"fused_wave_m4n3_exact", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1},
            {"fused_wave_m4n4_exact", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1},
        },
        {
            {"wmma16x16_direct", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_direct_v1},
            {"wmma16x16_m2", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m2_v1},
            {"wmma16x16_m3", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m3_v1},
            {"wmma16x16_m4", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m4_v1},
            {"wmma16x16_m8", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m8_v1},
            {"wmma16x16_m12", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m12_v1},
            {"wmma16x16_m16", dinoml_ck_gguf_q4_0_gemm_rcr_fp16_f16_wmma16x16_m16_v1},
        }};
  }
  if (qtype == "q4_k") {
    return QtypeSpec{
        "q4_k",
        kQKBlockSize,
        sizeof(GgufQ4_KBlock),
        fill_q4_k_blocks,
        q4_k_value_host,
        dinoml_ck_gguf_q4_k_dequant_b_f16_nk_v1,
        dinoml_ck_gguf_q4_k_dequant_b_f16_kn_v1,
        {
            {"fused_opt", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_opt_v1},
            {"fused_wave", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_v1},
            {"fused_wave_m8n1", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n1_v1},
            {"fused_wave_m8n2", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n2_v1},
            {"fused_wave_m8n3", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n3_v1},
            {"fused_wave_m8n4", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m8n4_v1},
            {"fused_wave_m16n1", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m16n1_v1},
            {"fused_wave_m16n2", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m16n2_v1},
            {"fused_wmma16x16", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_v1},
            {"fused_wmma16x16_direct", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_direct_v1},
            {"fused_wmma16x16_m2", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1},
            {"fused_wmma16x16_m3", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m3_v1},
            {"fused_wmma16x16_m4", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m4_v1},
            {"fused_wmma16x16_m8", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m8_v1},
            {"fused_wmma16x16_m12", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m12_v1},
            {"fused_wmma16x16_m16", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m16_v1},
            {"fused_wmma16x16_m4n2", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1},
            {"fused_wmma16x16_m6n2", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1},
            {"fused_wmma16x16_m8n2", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1},
            {"fused_wmma16x16_m4n3", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1},
            {"fused_wave_m4n2_exact", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n2_exact_v1},
            {"fused_wave_m4n3_exact", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1},
            {"fused_wave_m4n4_exact", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f32_wave_m4n4_exact_v1},
        },
        {
            {"wmma16x16_direct", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_direct_v1},
            {"wmma16x16_m2", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m2_v1},
            {"wmma16x16_m3", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m3_v1},
            {"wmma16x16_m4", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m4_v1},
            {"wmma16x16_m8", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m8_v1},
            {"wmma16x16_m12", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m12_v1},
            {"wmma16x16_m16", dinoml_ck_gguf_q4_k_gemm_rcr_fp16_f16_wmma16x16_m16_v1},
        }};
  }
  if (qtype == "q5_k") {
    return QtypeSpec{
        "q5_k",
        kQKBlockSize,
        sizeof(GgufQ5_KBlock),
        fill_q5_k_blocks,
        q5_k_value_host,
        dinoml_ck_gguf_q5_k_dequant_b_f16_nk_v1,
        dinoml_ck_gguf_q5_k_dequant_b_f16_kn_v1,
        {
            {"fused_opt", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_opt_v1},
            {"fused_wave", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_v1},
            {"fused_wave_m8n1", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n1_v1},
            {"fused_wave_m8n2", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n2_v1},
            {"fused_wave_m8n3", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n3_v1},
            {"fused_wave_m8n4", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m8n4_v1},
            {"fused_wave_m16n1", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m16n1_v1},
            {"fused_wave_m16n2", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m16n2_v1},
            {"fused_wmma16x16", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_v1},
            {"fused_wmma16x16_direct", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_direct_v1},
            {"fused_wmma16x16_m2", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m2_v1},
            {"fused_wmma16x16_m3", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m3_v1},
            {"fused_wmma16x16_m4", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m4_v1},
            {"fused_wmma16x16_m8", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m8_v1},
            {"fused_wmma16x16_m12", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m12_v1},
            {"fused_wmma16x16_m16", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m16_v1},
            {"fused_wmma16x16_m4n2", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m4n2_v1},
            {"fused_wmma16x16_m6n2", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m6n2_v1},
            {"fused_wmma16x16_m8n2", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m8n2_v1},
            {"fused_wmma16x16_m4n3", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wmma16x16_m4n3_v1},
            {"fused_wave_m4n2_exact", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n2_exact_v1},
            {"fused_wave_m4n3_exact", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n3_exact_v1},
            {"fused_wave_m4n4_exact", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f32_wave_m4n4_exact_v1},
        },
        {
            {"wmma16x16_direct", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_direct_v1},
            {"wmma16x16_m2", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m2_v1},
            {"wmma16x16_m3", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m3_v1},
            {"wmma16x16_m4", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m4_v1},
            {"wmma16x16_m8", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m8_v1},
            {"wmma16x16_m12", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m12_v1},
            {"wmma16x16_m16", dinoml_ck_gguf_q5_k_gemm_rcr_fp16_f16_wmma16x16_m16_v1},
        }};
  }
  if (qtype == "q2_k") {
    return DINOML_GGUF_QTYPE_SPEC("q2_k", q2_k, kQKBlockSize, GgufQ2_KBlock, fill_q2_k_blocks, q2_k_value_host);
  }
  if (qtype == "q3_k") {
    return DINOML_GGUF_QTYPE_SPEC("q3_k", q3_k, kQKBlockSize, GgufQ3_KBlock, fill_q3_k_blocks, q3_k_value_host);
  }
  if (qtype == "q6_k") {
    return DINOML_GGUF_QTYPE_SPEC("q6_k", q6_k, kQKBlockSize, GgufQ6_KBlock, fill_q6_k_blocks, q6_k_value_host);
  }
  if (qtype == "mxfp4") {
    return DINOML_GGUF_QTYPE_SPEC("mxfp4", mxfp4, kMxFp4BlockSize, GgufMxFp4Block, fill_mxfp4_blocks, mxfp4_value_host);
  }
  if (qtype == "nvfp4") {
    return DINOML_GGUF_QTYPE_SPEC("nvfp4", nvfp4, kNvFp4BlockSize, GgufNvFp4Block, fill_nvfp4_blocks, nvfp4_value_host);
  }
  throw std::runtime_error("unknown qtype: " + qtype);
}

#undef DINOML_GGUF_QTYPE_SPEC
#undef DINOML_GGUF_FUSED_F16_CANDIDATES
#undef DINOML_GGUF_FUSED_F32_CANDIDATES

std::vector<float> reference_gemm_quant(
    const std::vector<float>& a,
    const std::vector<uint8_t>& b,
    const QtypeSpec& spec,
    int m,
    int n,
    int k,
    bool f16_dequant) {
  const int qblocks_per_row = k / spec.block_size;
  std::vector<float> c(static_cast<std::size_t>(m) * n, 0.0f);
  for (int row = 0; row < m; ++row) {
    for (int col = 0; col < n; ++col) {
      float acc = 0.0f;
      for (int qb = 0; qb < qblocks_per_row; ++qb) {
        const std::size_t block_index = static_cast<std::size_t>(col) * qblocks_per_row + qb;
        for (int kk = 0; kk < spec.block_size; ++kk) {
          float b_value = spec.value(b, block_index, kk);
          if (f16_dequant) {
            b_value = half_bits_to_float(float_to_half_bits(b_value));
          }
          acc += a[static_cast<std::size_t>(row) * k + qb * spec.block_size + kk] * b_value;
        }
      }
      c[static_cast<std::size_t>(row) * n + col] = acc;
    }
  }
  return c;
}

std::vector<CandidateResult> profile_dense_candidates(
    half* d_a,
    half* d_b_dequant_nk,
    half* d_b_dequant_kn,
    float* d_c_dense,
    const Options& options) {
  std::vector<CandidateResult> dense_candidates;
  const auto add_candidate = [&](const std::string& name, const std::function<int()>& launch) {
    if (candidate_selected(name, options.dense_candidate, options.dense_candidate_names)) {
      dense_candidates.push_back(profile_candidate(name, options.warmup, options.iters, launch));
    }
  };
  add_candidate("dense_ck_xdl_baseline_nk", [&]() {
    return launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Col>(
        d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
  });
  add_candidate("dense_ck_xdl_skip_b_lds_16x64_nk", [&]() {
    return launch_dense_ck_skip_b_lds_half_f32_nk(
        d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
  });
  add_candidate("dense_ck_xdl_small_64x64_nk", [&]() {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Col>(
        d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
  });
  add_candidate("dense_ck_xdl_small_64x32_nk", [&]() {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Col>(
        d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
  });
  add_candidate("dense_ck_dl_16x64_nk", [&]() {
    return launch_dense_ck_dl_half_f32_nk(
        d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
  });
  add_candidate("dense_ck_dl_16x64_kn", [&]() {
    return launch_dense_ck_dl_half_f32_kn(
        d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, nullptr);
  });
  add_candidate("dense_ck_xdl_baseline_kn", [&]() {
    return launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Row>(
        d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
  });
  add_candidate("dense_ck_xdl_small_64x64_kn", [&]() {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Row>(
        d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
  });
  add_candidate("dense_ck_xdl_small_64x32_kn", [&]() {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Row>(
        d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
  });
  return dense_candidates;
}

int launch_dense_candidate_by_name(
    const std::string& name,
    half* d_a,
    half* d_b_dequant_nk,
    half* d_b_dequant_kn,
    float* d_c_dense,
    const Options& options) {
  if (name == "dense_ck_xdl_small_64x64_kn") {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Row>(
        d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
  }
  if (name == "dense_ck_xdl_small_64x32_kn") {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Row>(
        d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
  }
  if (name == "dense_ck_xdl_small_64x64_nk") {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Col>(
        d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
  }
  if (name == "dense_ck_xdl_small_64x32_nk") {
    return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Col>(
        d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
  }
  if (name == "dense_ck_dl_16x64_kn") {
    return launch_dense_ck_dl_half_f32_kn(d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, nullptr);
  }
  if (name == "dense_ck_dl_16x64_nk") {
    return launch_dense_ck_dl_half_f32_nk(d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
  }
  if (name == "dense_ck_xdl_skip_b_lds_16x64_nk") {
    return launch_dense_ck_skip_b_lds_half_f32_nk(d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
  }
  if (name == "dense_ck_xdl_baseline_kn") {
    return launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Row>(
        d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
  }
  return launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Col>(
      d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
}

int run_qtype_case(Options options, const QtypeSpec& spec) {
  if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.k % spec.block_size != 0) {
    throw std::runtime_error(std::string("m/n/k must be positive and k must be divisible by ") +
                             std::to_string(spec.block_size) + " for " + spec.name);
  }
  if (options.warmup < 0 || options.iters <= 0) {
    throw std::runtime_error("warmup must be non-negative and iters must be positive");
  }

  const int qblocks_per_row = options.k / spec.block_size;
  const std::size_t block_count = static_cast<std::size_t>(options.n) * qblocks_per_row;
  std::mt19937 rng(20260522 + options.m * 17 + options.n * 3 + options.k);
  std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);

  std::vector<float> h_a(static_cast<std::size_t>(options.m) * options.k);
  std::vector<uint16_t> h_a_half_bits(h_a.size());
  std::vector<uint8_t> h_b(block_count * spec.block_bytes);
  for (std::size_t i = 0; i < h_a.size(); ++i) {
    h_a_half_bits[i] = float_to_half_bits(a_dist(rng));
    h_a[i] = half_bits_to_float(h_a_half_bits[i]);
  }
  spec.fill_blocks(h_b, rng);

  half* d_a = nullptr;
  float* d_c_fused = nullptr;
  half* d_c_fused_f16 = nullptr;
  float* d_c_dense = nullptr;
  void* d_b = nullptr;
  half* d_b_dequant_nk = nullptr;
  half* d_b_dequant_kn = nullptr;
  hip_malloc(&d_a, h_a.size());
  hip_malloc(&d_c_fused, static_cast<std::size_t>(options.m) * options.n);
  if (options.profile_fused_f16) {
    hip_malloc(&d_c_fused_f16, static_cast<std::size_t>(options.m) * options.n);
  }
  hip_malloc(&d_c_dense, static_cast<std::size_t>(options.m) * options.n);
  check_hip(hipMalloc(&d_b, h_b.size()), "hipMalloc(B quant)");
  hip_malloc(&d_b_dequant_nk, static_cast<std::size_t>(options.n) * options.k);
  hip_malloc(&d_b_dequant_kn, static_cast<std::size_t>(options.n) * options.k);

  check_hip(hipMemcpy(d_a, h_a_half_bits.data(), h_a_half_bits.size() * sizeof(uint16_t), hipMemcpyHostToDevice), "copy A");
  check_hip(hipMemcpy(d_b, h_b.data(), h_b.size(), hipMemcpyHostToDevice), "copy B");

  const float dequant_nk_ms = time_repeated("dequant_b_f16_nk", options.warmup, options.iters, [&]() {
    const int rc = spec.dequant_nk(d_b, d_b_dequant_nk, options.n, options.k, nullptr);
    if (rc != 0) {
      throw std::runtime_error(std::string("dequant_nk returned ") + std::to_string(rc));
    }
  });
  const float dequant_kn_ms = time_repeated("dequant_b_f16_kn", options.warmup, options.iters, [&]() {
    const int rc = spec.dequant_kn(d_b, d_b_dequant_kn, options.n, options.k, nullptr);
    if (rc != 0) {
      throw std::runtime_error(std::string("dequant_kn returned ") + std::to_string(rc));
    }
  });

  std::vector<CandidateResult> fused_candidates;
  for (const FusedCandidateSpec& candidate : spec.fused_candidates) {
    if (!candidate_selected(candidate.name, options.fused_candidate, options.fused_candidate_names)) {
      continue;
    }
    fused_candidates.push_back(profile_candidate(candidate.name, options.warmup, options.iters, [&]() {
      return candidate.launch(d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
  }
  const CandidateResult& best_fused = best_supported(fused_candidates, spec.name);
  std::vector<CandidateResult> fused_f16_candidates;
  const CandidateResult* best_fused_f16 = nullptr;
  if (options.profile_fused_f16) {
    for (const FusedF16CandidateSpec& candidate : spec.fused_f16_candidates) {
      fused_f16_candidates.push_back(profile_candidate(candidate.name, options.warmup, options.iters, [&]() {
        return candidate.launch(d_a, d_b, d_c_fused_f16, options.m, options.n, options.k, nullptr);
      }));
    }
    best_fused_f16 = &best_supported(fused_f16_candidates, spec.name);
  }
  std::vector<CandidateResult> dense_candidates =
      profile_dense_candidates(d_a, d_b_dequant_nk, d_b_dequant_kn, d_c_dense, options);
  const CandidateResult& best_dense = best_supported(dense_candidates, "standard CK GEMM");

  const auto dense_dequant_ms = [&](const CandidateResult& candidate) {
    const bool uses_kn = candidate.name.size() >= 3 && candidate.name.rfind("_kn") == candidate.name.size() - 3;
    return uses_kn ? dequant_kn_ms : dequant_nk_ms;
  };
  const CandidateResult* best_dense_plus_dequant = nullptr;
  for (const CandidateResult& candidate : dense_candidates) {
    if (!candidate.supported) {
      continue;
    }
    if (best_dense_plus_dequant == nullptr ||
        candidate.ms + dense_dequant_ms(candidate) <
            best_dense_plus_dequant->ms + dense_dequant_ms(*best_dense_plus_dequant)) {
      best_dense_plus_dequant = &candidate;
    }
  }
  if (best_dense_plus_dequant == nullptr) {
    throw std::runtime_error("no supported candidates for dequant + standard CK GEMM");
  }

  if (options.verify) {
    for (const FusedCandidateSpec& candidate : spec.fused_candidates) {
      if (best_fused.name == candidate.name) {
        (void)candidate.launch(d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
        break;
      }
    }
    if (best_fused_f16 != nullptr) {
      for (const FusedF16CandidateSpec& candidate : spec.fused_f16_candidates) {
        if (best_fused_f16->name == candidate.name) {
          (void)candidate.launch(d_a, d_b, d_c_fused_f16, options.m, options.n, options.k, nullptr);
          break;
        }
      }
    }
    (void)launch_dense_candidate_by_name(best_dense.name, d_a, d_b_dequant_nk, d_b_dequant_kn, d_c_dense, options);
    check_hip(hipDeviceSynchronize(), "final candidate sync");
  }

  const double gflops = 2.0 * static_cast<double>(options.m) * options.n * options.k / 1.0e6;
  const float best_dequant_ms = dense_dequant_ms(best_dense);
  const float best_dequant_plus_ms =
      dense_dequant_ms(*best_dense_plus_dequant) + best_dense_plus_dequant->ms;
  const double dequant_nk_gbs =
      static_cast<double>(h_b.size() + static_cast<std::size_t>(options.n) * options.k * sizeof(uint16_t)) /
      (static_cast<double>(dequant_nk_ms) * 1.0e6);
  const double dequant_kn_gbs =
      static_cast<double>(h_b.size() + static_cast<std::size_t>(options.n) * options.k * sizeof(uint16_t)) /
      (static_cast<double>(dequant_kn_ms) * 1.0e6);

  std::cout << "qtype: " << spec.name << '\n';
  std::cout << "shape: m=" << options.m << " n=" << options.n << " k=" << options.k << '\n';
  std::cout << "dtype: A=f16 B_scratch=f16 accum=f32 C=f32\n";
  if (options.profile_fused_f16) {
    std::cout << "dtype_fused_f16out: A=f16 B_scratch=f16 accum=f32 C=f16\n";
  }
  std::cout << "verify: " << (options.verify ? "enabled" : "skipped") << '\n';
  std::cout << "dequant_b_f16_nk_ms: " << dequant_nk_ms << '\n';
  std::cout << "dequant_b_f16_nk_gb_s: " << dequant_nk_gbs << '\n';
  std::cout << "dequant_b_f16_kn_ms: " << dequant_kn_ms << '\n';
  std::cout << "dequant_b_f16_kn_gb_s: " << dequant_kn_gbs << '\n';
  for (const CandidateResult& candidate : fused_candidates) {
    std::cout << "candidate_fused_" << candidate.name << "_ms: "
              << (candidate.supported ? std::to_string(candidate.ms) : std::string("unsupported")) << '\n';
    if (!candidate.supported && !candidate.error.empty()) {
      std::cout << "candidate_fused_" << candidate.name << "_error: " << candidate.error << '\n';
    }
  }
  if (options.profile_fused_f16) {
    for (const CandidateResult& candidate : fused_f16_candidates) {
      std::cout << "candidate_fused_f16out_" << candidate.name << "_ms: "
                << (candidate.supported ? std::to_string(candidate.ms) : std::string("unsupported")) << '\n';
      if (!candidate.supported && !candidate.error.empty()) {
        std::cout << "candidate_fused_f16out_" << candidate.name << "_error: " << candidate.error << '\n';
      }
    }
  }
  for (const CandidateResult& candidate : dense_candidates) {
    std::cout << "candidate_dense_" << candidate.name << "_ms: "
              << (candidate.supported ? std::to_string(candidate.ms) : std::string("unsupported")) << '\n';
    if (!candidate.supported && !candidate.error.empty()) {
      std::cout << "candidate_dense_" << candidate.name << "_error: " << candidate.error << '\n';
    }
  }
  std::cout << "best_fused_candidate: " << best_fused.name << '\n';
  std::cout << "best_fused_ms: " << best_fused.ms << '\n';
  std::cout << "best_fused_gflops: " << (gflops / static_cast<double>(best_fused.ms)) << '\n';
  if (best_fused_f16 != nullptr) {
    std::cout << "best_fused_f16out_candidate: " << best_fused_f16->name << '\n';
    std::cout << "best_fused_f16out_ms: " << best_fused_f16->ms << '\n';
    std::cout << "best_fused_f16out_gflops: " << (gflops / static_cast<double>(best_fused_f16->ms)) << '\n';
    std::cout << "fused_f16out_vs_fused_f32out_speedup: " << (best_fused.ms / best_fused_f16->ms) << '\n';
  }
  std::cout << "best_dense_ck_candidate: " << best_dense.name << '\n';
  std::cout << "best_dense_ck_gemm_ms: " << best_dense.ms << '\n';
  std::cout << "best_dense_ck_gemm_gflops: " << (gflops / static_cast<double>(best_dense.ms)) << '\n';
  std::cout << "fused_vs_dense_gemm_only_speedup: " << (best_dense.ms / best_fused.ms) << '\n';
  if (best_fused_f16 != nullptr) {
    std::cout << "fused_f16out_vs_dense_gemm_only_speedup: " << (best_dense.ms / best_fused_f16->ms) << '\n';
  }
  std::cout << "best_dense_dequant_ms: " << best_dequant_ms << '\n';
  std::cout << "dequant_plus_best_dense_ck_ms: " << (best_dequant_ms + best_dense.ms) << '\n';
  std::cout << "best_dequant_plus_ck_candidate: " << best_dense_plus_dequant->name << '\n';
  std::cout << "best_dequant_plus_ck_gemm_ms: " << best_dense_plus_dequant->ms << '\n';
  std::cout << "best_dequant_plus_ck_dequant_ms: " << dense_dequant_ms(*best_dense_plus_dequant) << '\n';
  std::cout << "best_dequant_plus_ck_ms: " << best_dequant_plus_ms << '\n';
  std::cout << "fused_speedup_vs_dequant_plus_ck: " << (best_dequant_plus_ms / best_fused.ms) << '\n';
  if (best_fused_f16 != nullptr) {
    std::cout << "fused_f16out_speedup_vs_dequant_plus_ck: " << (best_dequant_plus_ms / best_fused_f16->ms) << '\n';
  }

  bool ok = true;
  if (options.verify) {
    std::vector<float> h_c_fused(static_cast<std::size_t>(options.m) * options.n);
    std::vector<float> h_c_dense(h_c_fused.size());
    check_hip(hipMemcpy(h_c_fused.data(), d_c_fused, h_c_fused.size() * sizeof(float), hipMemcpyDeviceToHost), "copy fused C");
    check_hip(hipMemcpy(h_c_dense.data(), d_c_dense, h_c_dense.size() * sizeof(float), hipMemcpyDeviceToHost), "copy dense C");
    std::vector<float> h_c_fused_f16;
    if (best_fused_f16 != nullptr) {
      std::vector<uint16_t> h_c_fused_f16_bits(h_c_fused.size());
      h_c_fused_f16.resize(h_c_fused.size());
      check_hip(
          hipMemcpy(h_c_fused_f16_bits.data(),
                    d_c_fused_f16,
                    h_c_fused_f16_bits.size() * sizeof(uint16_t),
                    hipMemcpyDeviceToHost),
          "copy fused C f16");
      for (std::size_t i = 0; i < h_c_fused_f16_bits.size(); ++i) {
        h_c_fused_f16[i] = half_bits_to_float(h_c_fused_f16_bits[i]);
      }
    }
    const std::vector<float> ref_quant = reference_gemm_quant(h_a, h_b, spec, options.m, options.n, options.k, false);
    const std::vector<float> ref_dequant = reference_gemm_quant(h_a, h_b, spec, options.m, options.n, options.k, true);
    const bool fused_uses_f16_b = best_fused.name.find("wmma") != std::string::npos;
    const std::vector<float>& fused_ref = fused_uses_f16_b ? ref_dequant : ref_quant;
    const auto [fused_max_abs, fused_max_rel] = max_errors(fused_ref, h_c_fused);
    const auto [fused_quant_max_abs, fused_quant_max_rel] = max_errors(ref_quant, h_c_fused);
    const auto [dense_max_abs, dense_max_rel] = max_errors(ref_dequant, h_c_dense);
    const auto [fused_vs_dense_max_abs, fused_vs_dense_max_rel] = max_errors(h_c_dense, h_c_fused);
    std::cout << "fused_max_abs_error_vs_" << (fused_uses_f16_b ? "f16_dequant_ref" : "quant_ref")
              << ": " << fused_max_abs << '\n';
    std::cout << "fused_max_rel_error_vs_" << (fused_uses_f16_b ? "f16_dequant_ref" : "quant_ref")
              << ": " << fused_max_rel << '\n';
    std::cout << "fused_max_abs_error_vs_quant_ref: " << fused_quant_max_abs << '\n';
    std::cout << "fused_max_rel_error_vs_quant_ref: " << fused_quant_max_rel << '\n';
    std::cout << "dense_max_abs_error_vs_f16_dequant_ref: " << dense_max_abs << '\n';
    std::cout << "dense_max_rel_error_vs_f16_dequant_ref: " << dense_max_rel << '\n';
    std::cout << "fused_vs_dense_max_abs_delta: " << fused_vs_dense_max_abs << '\n';
    std::cout << "fused_vs_dense_max_rel_delta: " << fused_vs_dense_max_rel << '\n';
    bool fused_f16_ok = true;
    if (best_fused_f16 != nullptr) {
      const auto [fused_f16_max_abs, fused_f16_max_rel] = max_errors(ref_dequant, h_c_fused_f16);
      const auto [fused_f16_vs_dense_max_abs, fused_f16_vs_dense_max_rel] = max_errors(h_c_dense, h_c_fused_f16);
      std::cout << "fused_f16out_max_abs_error_vs_f16_dequant_ref: " << fused_f16_max_abs << '\n';
      std::cout << "fused_f16out_max_rel_error_vs_f16_dequant_ref: " << fused_f16_max_rel << '\n';
      std::cout << "fused_f16out_vs_dense_max_abs_delta: " << fused_f16_vs_dense_max_abs << '\n';
      std::cout << "fused_f16out_vs_dense_max_rel_delta: " << fused_f16_vs_dense_max_rel << '\n';
      fused_f16_ok = fused_f16_max_abs <= 1.0e-1f || fused_f16_max_rel <= 3.0e-2f;
    }
    const bool fused_ok = fused_uses_f16_b ? (fused_max_abs <= 5.0e-2f || fused_max_rel <= 2.0e-2f)
                                           : (fused_max_abs <= 5.0e-4f || fused_max_rel <= 1.0e-3f);
    ok = fused_ok && fused_f16_ok &&
         (dense_max_abs <= 5.0e-2f || dense_max_rel <= 2.0e-2f);
  }

  check_hip(hipFree(d_a), "hipFree(A)");
  check_hip(hipFree(d_c_fused), "hipFree(C fused)");
  if (d_c_fused_f16 != nullptr) {
    check_hip(hipFree(d_c_fused_f16), "hipFree(C fused f16)");
  }
  check_hip(hipFree(d_c_dense), "hipFree(C dense)");
  check_hip(hipFree(d_b), "hipFree(B)");
  check_hip(hipFree(d_b_dequant_nk), "hipFree(B dequant NK)");
  check_hip(hipFree(d_b_dequant_kn), "hipFree(B dequant KN)");

  return ok ? 0 : 2;
}

std::vector<std::string> expand_qtypes(const std::string& qtype) {
  if (qtype == "all") {
    return {"q8_0", "q4_0", "q4_k", "q5_k", "q2_k", "q3_k", "q6_k", "mxfp4", "nvfp4"};
  }
  return {qtype};
}

std::vector<Shape> expand_shapes(const Options& options) {
  if (options.sweep.empty()) {
    return {{options.m, options.n, options.k}};
  }
  if (options.sweep == "decode_prefill") {
    return {
        {1, 4096, 4096},
        {2, 4096, 4096},
        {4, 4096, 4096},
        {8, 4096, 4096},
        {16, 4096, 4096},
        {32, 4096, 4096},
        {64, 4096, 4096},
        {128, 4096, 4096},
        {4, 2048, 2048},
        {4, 8192, 8192},
    };
  }
  if (options.sweep == "dit_balanced") {
    return {
        {256, 3072, 3072},
        {512, 3072, 3072},
        {1024, 3072, 3072},
        {2048, 3072, 3072},
        {4096, 3072, 3072},
        {8192, 3072, 3072},
        {256, 12288, 3072},
        {512, 12288, 3072},
        {1024, 12288, 3072},
        {2048, 12288, 3072},
        {4096, 12288, 3072},
        {8192, 12288, 3072},
        {256, 3072, 12288},
        {512, 3072, 12288},
        {1024, 3072, 12288},
        {2048, 3072, 12288},
        {4096, 3072, 12288},
        {8192, 3072, 12288},
        {512, 4096, 4096},
        {2048, 4096, 4096},
        {8192, 4096, 4096},
        {512, 8192, 8192},
        {2048, 8192, 8192},
        {8192, 8192, 8192},
    };
  }
  throw std::runtime_error("unknown sweep: " + options.sweep);
}

int run_qtype_benchmark(const Options& options) {
  int rc = 0;
  const std::vector<std::string> qtypes = expand_qtypes(options.qtype);
  const std::vector<Shape> shapes = expand_shapes(options);
  for (const std::string& qtype : qtypes) {
    const QtypeSpec spec = make_qtype_spec(qtype);
    for (const Shape& shape : shapes) {
      Options case_options = options;
      case_options.qtype = qtype;
      case_options.m = shape.m;
      case_options.n = shape.n;
      case_options.k = shape.k;
      if (!options.sweep.empty()) {
        case_options.verify = false;
      }
      if (qtype != qtypes.front() || shape.m != shapes.front().m || shape.n != shapes.front().n || shape.k != shapes.front().k) {
        std::cout << "---\n";
      }
      const int case_rc = run_qtype_case(case_options, spec);
      if (case_rc != 0) {
        rc = case_rc;
      }
    }
  }
  return rc;
}

} // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    return run_qtype_benchmark(options);
    if (options.m <= 0 || options.n <= 0 || options.k <= 0 || options.k % kQ8BlockSize != 0) {
      throw std::runtime_error("m/n/k must be positive and k must be divisible by 32");
    }
    if (options.warmup < 0 || options.iters <= 0) {
      throw std::runtime_error("warmup must be non-negative and iters must be positive");
    }

    const int qblocks_per_row = options.k / kQ8BlockSize;
    std::mt19937 rng(20260522);
    std::uniform_real_distribution<float> a_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.0025f, 0.04f);
    std::uniform_int_distribution<int> q_dist(-127, 127);

    std::vector<float> h_a(static_cast<std::size_t>(options.m) * options.k);
    std::vector<uint16_t> h_a_half_bits(h_a.size());
    std::vector<GgufQ8_0Block> h_b(static_cast<std::size_t>(options.n) * qblocks_per_row);
    for (std::size_t i = 0; i < h_a.size(); ++i) {
      h_a_half_bits[i] = float_to_half_bits(a_dist(rng));
      h_a[i] = half_bits_to_float(h_a_half_bits[i]);
    }
    for (GgufQ8_0Block& block : h_b) {
      block.d = float_to_half_bits(scale_dist(rng));
      for (int kk = 0; kk < kQ8BlockSize; ++kk) {
        block.qs[kk] = static_cast<int8_t>(q_dist(rng));
      }
    }

    half* d_a = nullptr;
    float* d_c_fused = nullptr;
    float* d_c_dense = nullptr;
    GgufQ8_0Block* d_b = nullptr;
    half* d_b_dequant_nk = nullptr;
    half* d_b_dequant_kn = nullptr;
    hip_malloc(&d_a, h_a.size());
    hip_malloc(&d_c_fused, static_cast<std::size_t>(options.m) * options.n);
    hip_malloc(&d_c_dense, static_cast<std::size_t>(options.m) * options.n);
    hip_malloc(&d_b, h_b.size());
    hip_malloc(&d_b_dequant_nk, static_cast<std::size_t>(options.n) * options.k);
    hip_malloc(&d_b_dequant_kn, static_cast<std::size_t>(options.n) * options.k);

    check_hip(hipMemcpy(d_a, h_a_half_bits.data(), h_a_half_bits.size() * sizeof(uint16_t), hipMemcpyHostToDevice), "copy A");
    check_hip(hipMemcpy(d_b, h_b.data(), h_b.size() * sizeof(GgufQ8_0Block), hipMemcpyHostToDevice), "copy B");

    auto launch_dequant = [&]() {
      return dinoml_ck_gguf_q8_0_dequant_b_f16_nk_v1(d_b, d_b_dequant_nk, options.n, options.k, nullptr);
    };
    auto launch_dequant_kn = [&]() {
      return dinoml_ck_gguf_q8_0_dequant_b_f16_kn_v1(d_b, d_b_dequant_kn, options.n, options.k, nullptr);
    };

    const float dequant_nk_ms = time_repeated("dequant_b_f16_nk", options.warmup, options.iters, [&]() {
      const int rc = launch_dequant();
      if (rc != 0) {
        throw std::runtime_error("dequant_nk returned " + std::to_string(rc));
      }
    });
    const float dequant_kn_ms = time_repeated("dequant_b_f16_kn", options.warmup, options.iters, [&]() {
      const int rc = launch_dequant_kn();
      if (rc != 0) {
        throw std::runtime_error("dequant_kn returned " + std::to_string(rc));
      }
    });

    std::vector<CandidateResult> fused_candidates;
    fused_candidates.push_back(profile_candidate("fused_wave4", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave4_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n2", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n2_exact", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n3_exact", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n4_exact", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n5_exact", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n5_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n6_exact", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n6_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n8_exact", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n4", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_wave_m4n8", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_tile4x64", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile4x64_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    fused_candidates.push_back(profile_candidate("fused_tile16x16", options.warmup, options.iters, [&]() {
      return dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile16x16_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }));
    const CandidateResult& best_fused = best_supported(fused_candidates, "fused Q8_0");

    std::vector<CandidateResult> dense_candidates;
    dense_candidates.push_back(profile_candidate("dense_ck_xdl_baseline_nk",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Col>(
                                                       d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_xdl_skip_b_lds_16x64_nk",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_skip_b_lds_half_f32_nk(
                                                       d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_xdl_small_64x64_nk",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Col>(
                                                       d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_xdl_small_64x32_nk",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Col>(
                                                       d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_dl_16x64_nk",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_dl_half_f32_nk(
                                                       d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_dl_16x64_kn",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_dl_half_f32_kn(
                                                       d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_xdl_baseline_kn",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Row>(
                                                       d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_xdl_small_64x64_kn",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Row>(
                                                       d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
                                                 }));
    dense_candidates.push_back(profile_candidate("dense_ck_xdl_small_64x32_kn",
                                                 options.warmup,
                                                 options.iters,
                                                 [&]() {
                                                   return launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Row>(
                                                       d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
                                                 }));
    const CandidateResult& best_dense = best_supported(dense_candidates, "standard CK GEMM");
    const auto dense_dequant_ms = [&](const CandidateResult& candidate) {
      const bool uses_kn = candidate.name.size() >= 3 && candidate.name.rfind("_kn") == candidate.name.size() - 3;
      return uses_kn ? dequant_kn_ms : dequant_nk_ms;
    };
    const CandidateResult* best_dense_plus_dequant = nullptr;
    for (const CandidateResult& candidate : dense_candidates) {
      if (!candidate.supported) {
        continue;
      }
      if (best_dense_plus_dequant == nullptr ||
          candidate.ms + dense_dequant_ms(candidate) <
              best_dense_plus_dequant->ms + dense_dequant_ms(*best_dense_plus_dequant)) {
        best_dense_plus_dequant = &candidate;
      }
    }
    if (best_dense_plus_dequant == nullptr) {
      throw std::runtime_error("no supported candidates for dequant + standard CK GEMM");
    }

    if (best_fused.name == "fused_wave_m4n2_exact") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n3_exact") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n3_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n4_exact") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n5_exact") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n5_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n6_exact") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n6_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n8_exact") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_exact_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n2") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n2_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n4") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n4_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4n8") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4n8_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave_m4") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave_m4_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_wave4") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_wave4_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else if (best_fused.name == "fused_tile4x64") {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile4x64_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    } else {
      (void)dinoml_ck_gguf_q8_0_gemm_rcr_fp16_f32_tile16x16_v1(
          d_a, d_b, d_c_fused, options.m, options.n, options.k, nullptr);
    }
    if (best_dense.name == "dense_ck_xdl_small_64x64_kn") {
      (void)launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Row>(
          d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
    } else if (best_dense.name == "dense_ck_xdl_small_64x32_kn") {
      (void)launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Row>(
          d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
    } else if (best_dense.name == "dense_ck_xdl_small_64x64_nk") {
      (void)launch_dense_ck_half_f32<DenseCkConfig::kSmall64x64, Col>(
          d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
    } else if (best_dense.name == "dense_ck_xdl_small_64x32_nk") {
      (void)launch_dense_ck_half_f32<DenseCkConfig::kSmall64x32, Col>(
          d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
    } else if (best_dense.name == "dense_ck_dl_16x64_kn") {
      (void)launch_dense_ck_dl_half_f32_kn(
          d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, nullptr);
    } else if (best_dense.name == "dense_ck_dl_16x64_nk") {
      (void)launch_dense_ck_dl_half_f32_nk(
          d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
    } else if (best_dense.name == "dense_ck_xdl_skip_b_lds_16x64_nk") {
      (void)launch_dense_ck_skip_b_lds_half_f32_nk(
          d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, nullptr);
    } else if (best_dense.name == "dense_ck_xdl_baseline_kn") {
      (void)launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Row>(
          d_a, d_b_dequant_kn, d_c_dense, options.m, options.n, options.k, options.n, nullptr);
    } else {
      (void)launch_dense_ck_half_f32<DenseCkConfig::kBaseline, Col>(
          d_a, d_b_dequant_nk, d_c_dense, options.m, options.n, options.k, options.k, nullptr);
    }
    check_hip(hipDeviceSynchronize(), "final candidate sync");

    std::vector<float> h_c_fused(static_cast<std::size_t>(options.m) * options.n);
    std::vector<float> h_c_dense(h_c_fused.size());
    check_hip(hipMemcpy(h_c_fused.data(), d_c_fused, h_c_fused.size() * sizeof(float), hipMemcpyDeviceToHost), "copy fused C");
    check_hip(hipMemcpy(h_c_dense.data(), d_c_dense, h_c_dense.size() * sizeof(float), hipMemcpyDeviceToHost), "copy dense C");

    const std::vector<float> ref_q8 = reference_gemm_q8(h_a, h_b, options.m, options.n, options.k);
    const std::vector<float> ref_dequant = reference_gemm_dequant_f16(h_a, h_b, options.m, options.n, options.k);
    const auto [fused_max_abs, fused_max_rel] = max_errors(ref_q8, h_c_fused);
    const auto [dense_max_abs, dense_max_rel] = max_errors(ref_dequant, h_c_dense);
    const auto [fused_vs_dense_max_abs, fused_vs_dense_max_rel] = max_errors(h_c_dense, h_c_fused);

    const double gflops = 2.0 * static_cast<double>(options.m) * options.n * options.k / 1.0e6;
    const float best_dequant_ms = dense_dequant_ms(best_dense);
    const float best_dequant_plus_ms =
        dense_dequant_ms(*best_dense_plus_dequant) + best_dense_plus_dequant->ms;
    const double dequant_nk_gbs =
        static_cast<double>(h_b.size() * sizeof(GgufQ8_0Block) + static_cast<std::size_t>(options.n) * options.k * sizeof(uint16_t)) /
        (static_cast<double>(dequant_nk_ms) * 1.0e6);
    const double dequant_kn_gbs =
        static_cast<double>(h_b.size() * sizeof(GgufQ8_0Block) + static_cast<std::size_t>(options.n) * options.k * sizeof(uint16_t)) /
        (static_cast<double>(dequant_kn_ms) * 1.0e6);

    std::cout << "shape: m=" << options.m << " n=" << options.n << " k=" << options.k << '\n';
    std::cout << "dtype: A=f16 B_scratch=f16 C=f32\n";
    std::cout << "dequant_b_f16_nk_ms: " << dequant_nk_ms << '\n';
    std::cout << "dequant_b_f16_nk_gb_s: " << dequant_nk_gbs << '\n';
    std::cout << "dequant_b_f16_kn_ms: " << dequant_kn_ms << '\n';
    std::cout << "dequant_b_f16_kn_gb_s: " << dequant_kn_gbs << '\n';
    for (const CandidateResult& candidate : fused_candidates) {
      std::cout << "candidate_fused_" << candidate.name << "_ms: "
                << (candidate.supported ? std::to_string(candidate.ms) : std::string("unsupported")) << '\n';
    }
    for (const CandidateResult& candidate : dense_candidates) {
      std::cout << "candidate_dense_" << candidate.name << "_ms: "
                << (candidate.supported ? std::to_string(candidate.ms) : std::string("unsupported")) << '\n';
    }
    std::cout << "best_fused_candidate: " << best_fused.name << '\n';
    std::cout << "best_fused_ms: " << best_fused.ms << '\n';
    std::cout << "best_fused_gflops: " << (gflops / static_cast<double>(best_fused.ms)) << '\n';
    std::cout << "best_dense_ck_candidate: " << best_dense.name << '\n';
    std::cout << "best_dense_ck_gemm_ms: " << best_dense.ms << '\n';
    std::cout << "best_dense_ck_gemm_gflops: " << (gflops / static_cast<double>(best_dense.ms)) << '\n';
    std::cout << "best_dense_dequant_ms: " << best_dequant_ms << '\n';
    std::cout << "dequant_plus_best_dense_ck_ms: " << (best_dequant_ms + best_dense.ms) << '\n';
    std::cout << "best_dequant_plus_ck_candidate: " << best_dense_plus_dequant->name << '\n';
    std::cout << "best_dequant_plus_ck_gemm_ms: " << best_dense_plus_dequant->ms << '\n';
    std::cout << "best_dequant_plus_ck_dequant_ms: " << dense_dequant_ms(*best_dense_plus_dequant) << '\n';
    std::cout << "best_dequant_plus_ck_ms: " << best_dequant_plus_ms << '\n';
    std::cout << "fused_speedup_vs_dequant_plus_ck: " << (best_dequant_plus_ms / best_fused.ms) << '\n';
    std::cout << "fused_max_abs_error_vs_q8_ref: " << fused_max_abs << '\n';
    std::cout << "fused_max_rel_error_vs_q8_ref: " << fused_max_rel << '\n';
    std::cout << "dense_max_abs_error_vs_f16_dequant_ref: " << dense_max_abs << '\n';
    std::cout << "dense_max_rel_error_vs_f16_dequant_ref: " << dense_max_rel << '\n';
    std::cout << "fused_vs_dense_max_abs_delta: " << fused_vs_dense_max_abs << '\n';
    std::cout << "fused_vs_dense_max_rel_delta: " << fused_vs_dense_max_rel << '\n';

    check_hip(hipFree(d_a), "hipFree(A)");
    check_hip(hipFree(d_c_fused), "hipFree(C fused)");
    check_hip(hipFree(d_c_dense), "hipFree(C dense)");
    check_hip(hipFree(d_b), "hipFree(B)");
    check_hip(hipFree(d_b_dequant_nk), "hipFree(B dequant)");
    check_hip(hipFree(d_b_dequant_kn), "hipFree(B dequant KN)");

    const bool fused_ok = fused_max_abs <= 5.0e-4f || fused_max_rel <= 1.0e-3f;
    const bool dense_ok = dense_max_abs <= 5.0e-2f || dense_max_rel <= 2.0e-2f;
    return (fused_ok && dense_ok) ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
