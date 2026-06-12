#include <dinoml/abi.h>
#include <dinoml/math.h>

#include <hip/hip_runtime.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/convolution_backward_data_specialization.hpp>
#include <ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>
#include <ck/utility/type_convert.hpp>

#include <array>
#include <cstdint>

namespace {

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

namespace conv_layout = ck::tensor_layout::convolution;

using In1dLayout = conv_layout::G_NW_C;
using Wei1dLayout = conv_layout::G_K_X_C;
using Out1dLayout = conv_layout::G_NW_K;
using In3dLegacyLayout = conv_layout::G_NDHW_C;
using Wei3dLegacyLayout = conv_layout::G_K_ZYX_C;
using Out3dLegacyLayout = conv_layout::G_NDHW_K;
using In3dLayout = conv_layout::GNDHWC;
using Wei3dLayout = conv_layout::GKZYXC;
using Out3dLayout = conv_layout::GNDHWK;
using InLayout = conv_layout::G_NHW_C;
using WeiLayout = conv_layout::G_K_YX_C;
using BiasLayout = conv_layout::G_K;
using OutLayout = conv_layout::G_NHW_K;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

enum class ConvConfig {
  kBaseline,
  kWideN,
  kWideM,
  kSquare,
  kSkinnyM,
  kSkinnyN,
  kSmall,
};

static constexpr ck::index_t NDimSpatial1 = 1;
static constexpr ck::index_t NDimSpatial3 = 3;
static constexpr ck::index_t NDimSpatial = 2;
static constexpr auto ConvSpec =
    ck::tensor_operation::device::ConvolutionForwardSpecialization::Default;
static constexpr auto GemmSpec =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <ConvConfig Config, typename Storage>
struct ConvKernelConfig;

template <typename Storage, ck::index_t DefaultValue, ck::index_t FloatValue>
struct StorageTunedValue {
  static constexpr ck::index_t value = DefaultValue;
};

template <ck::index_t DefaultValue, ck::index_t FloatValue>
struct StorageTunedValue<float, DefaultValue, FloatValue> {
  static constexpr ck::index_t value = FloatValue;
};

template <typename Storage>
struct ConvKernelConfig<ConvConfig::kBaseline, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 4;
  static constexpr ck::index_t BK1 = 4;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 1;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 1;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <typename Storage>
struct ConvKernelConfig<ConvConfig::kWideN, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 256;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 8;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 8;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 8;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 8;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct ConvKernelConfig<ConvConfig::kWideN, float> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 256;
  static constexpr ck::index_t KPerBlock = 16;
  static constexpr ck::index_t AK1 = 4;
  static constexpr ck::index_t BK1 = 4;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 4;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 4;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 4;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 4;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using CDEBlockTransferClusterLengths = S<1, 16, 1, 16>;
};

template <typename Storage>
struct ConvKernelConfig<ConvConfig::kWideM, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 256;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = StorageTunedValue<Storage, 32, 32>::value;
  static constexpr ck::index_t AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t MPerXDL = StorageTunedValue<Storage, 16, 32>::value;
  static constexpr ck::index_t NPerXDL = StorageTunedValue<Storage, 16, 32>::value;
  static constexpr ck::index_t MXdlPerWave = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t NXdlPerWave = StorageTunedValue<Storage, 4, 2>::value;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = StorageTunedValue<Storage, 4, 4>::value;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <typename Storage>
struct ConvKernelConfig<ConvConfig::kSquare, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = StorageTunedValue<Storage, 4, 4>::value;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct ConvKernelConfig<ConvConfig::kSkinnyM, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 1;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = StorageTunedValue<Storage, 4, 4>::value;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct ConvKernelConfig<ConvConfig::kSkinnyN, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 64;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 1;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = StorageTunedValue<Storage, 4, 4>::value;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct ConvKernelConfig<ConvConfig::kSmall, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 64;
  static constexpr ck::index_t KPerBlock = StorageTunedValue<Storage, 32, 16>::value;
  static constexpr ck::index_t AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 4, 2>::value;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 4, 2>::value;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = StorageTunedValue<Storage, 4, 2>::value;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = StorageTunedValue<Storage, 4, 2>::value;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = StorageTunedValue<Storage, 4, 4>::value;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct CkElement;

template <>
struct CkElement<float> {
  using Type = float;
};

template <>
struct CkElement<half> {
  using Type = ck::half_t;
};

template <>
struct CkElement<dinoml::bfloat16> {
  using Type = ck::bhalf_t;
};

template <typename Storage, typename Element>
Element const* ck_ptr(Storage const* ptr) {
  static_assert(sizeof(Storage) == sizeof(Element), "CK element storage size mismatch");
  return reinterpret_cast<Element const*>(ptr);
}

template <typename Storage, typename Element>
Element* ck_ptr(Storage* ptr) {
  static_assert(sizeof(Storage) == sizeof(Element), "CK element storage size mismatch");
  return reinterpret_cast<Element*>(ptr);
}

template <typename Storage>
__global__ void dino_ck_pack_conv1d_weight_kcx_to_kxc(
    Storage const* source,
    Storage* packed,
    int out_channels,
    int in_channels,
    int kernel_w) {
  const auto linear_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = static_cast<std::size_t>(out_channels) * in_channels * kernel_w;
  if (linear_idx >= total) {
    return;
  }
  const auto w = linear_idx % static_cast<std::size_t>(kernel_w);
  const auto c = (linear_idx / static_cast<std::size_t>(kernel_w)) % static_cast<std::size_t>(in_channels);
  const auto k = linear_idx / (static_cast<std::size_t>(kernel_w) * in_channels);
  const auto packed_idx =
      (k * static_cast<std::size_t>(kernel_w) + w) * static_cast<std::size_t>(in_channels) + c;
  packed[packed_idx] = source[linear_idx];
}

template <typename Storage>
int pack_conv1d_weight_kcx_to_kxc(
    Storage const* source,
    Storage** packed_out,
    int out_channels,
    int in_channels,
    int kernel_w,
    hipStream_t stream) {
  if (source == nullptr || packed_out == nullptr || out_channels <= 0 || in_channels <= 0 || kernel_w <= 0) {
    return 5;
  }
  *packed_out = nullptr;
  const auto count = static_cast<std::size_t>(out_channels) * in_channels * kernel_w;
  const auto nbytes = count * sizeof(Storage);
  if (hipMalloc(reinterpret_cast<void**>(packed_out), nbytes) != hipSuccess) {
    return 5;
  }
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((count + block.x - 1) / block.x));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(dino_ck_pack_conv1d_weight_kcx_to_kxc<Storage>),
      grid,
      block,
      0,
      stream,
      source,
      *packed_out,
      out_channels,
      in_channels,
      kernel_w);
  if (hipGetLastError() != hipSuccess) {
    (void)hipFree(*packed_out);
    *packed_out = nullptr;
    return 5;
  }
  return 0;
}

template <typename Storage>
__global__ void dino_ck_pack_conv1d_ncw_to_nwc(
    Storage const* source,
    Storage* packed,
    int batch,
    int channels,
    int width) {
  const auto linear_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = static_cast<std::size_t>(batch) * channels * width;
  if (linear_idx >= total) {
    return;
  }
  const auto w = linear_idx % static_cast<std::size_t>(width);
  const auto c = (linear_idx / static_cast<std::size_t>(width)) % static_cast<std::size_t>(channels);
  const auto n = linear_idx / (static_cast<std::size_t>(width) * channels);
  const auto packed_idx = (n * static_cast<std::size_t>(width) + w) * static_cast<std::size_t>(channels) + c;
  packed[packed_idx] = source[linear_idx];
}

template <typename Storage>
int launch_ck_conv1d_ncw_to_nwc(
    Storage const* source,
    Storage* packed,
    int batch,
    int channels,
    int width,
    hipStream_t stream) {
  if (source == nullptr || packed == nullptr || batch <= 0 || channels <= 0 || width <= 0) {
    return 1;
  }
  const auto total = static_cast<std::size_t>(batch) * channels * width;
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((total + block.x - 1) / block.x));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(dino_ck_pack_conv1d_ncw_to_nwc<Storage>),
      grid,
      block,
      0,
      stream,
      source,
      packed,
      batch,
      channels,
      width);
  return hipGetLastError() == hipSuccess ? 0 : 3;
}

template <typename Storage>
__global__ void dino_ck_unpack_conv1d_nwk_to_ncw(
    Storage const* source,
    Storage* unpacked,
    int batch,
    int channels,
    int width) {
  const auto linear_idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto total = static_cast<std::size_t>(batch) * width * channels;
  if (linear_idx >= total) {
    return;
  }
  const auto c = linear_idx % static_cast<std::size_t>(channels);
  const auto w = (linear_idx / static_cast<std::size_t>(channels)) % static_cast<std::size_t>(width);
  const auto n = linear_idx / (static_cast<std::size_t>(channels) * width);
  const auto unpacked_idx = (n * static_cast<std::size_t>(channels) + c) * static_cast<std::size_t>(width) + w;
  unpacked[unpacked_idx] = source[linear_idx];
}

template <typename Storage>
int launch_ck_conv1d_nwk_to_ncw(
    Storage const* source,
    Storage* unpacked,
    int batch,
    int channels,
    int width,
    hipStream_t stream) {
  if (source == nullptr || unpacked == nullptr || batch <= 0 || channels <= 0 || width <= 0) {
    return 1;
  }
  const auto total = static_cast<std::size_t>(batch) * width * channels;
  const dim3 block(256);
  const dim3 grid(static_cast<unsigned int>((total + block.x - 1) / block.x));
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(dino_ck_unpack_conv1d_nwk_to_ncw<Storage>),
      grid,
      block,
      0,
      stream,
      source,
      unpacked,
      batch,
      channels,
      width);
  return hipGetLastError() == hipSuccess ? 0 : 3;
}

template <typename T>
__host__ __device__ float to_float(const T& value) {
  return ck::type_convert<float>(value);
}

template <typename T>
__global__ void dinoml_ck_conv_nchw_to_nhwc_kernel(
    const T* src,
    T* dst,
    int n,
    int c,
    int h,
    int w) {
  int linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  int total = n * c * h * w;
  if (linear >= total) {
    return;
  }
  int x = linear % w;
  int tmp = linear / w;
  int y = tmp % h;
  tmp /= h;
  int channel = tmp % c;
  int batch = tmp / c;
  int dst_index = ((batch * h + y) * w + x) * c + channel;
  dst[dst_index] = src[linear];
}

template <typename Storage>
int launch_ck_conv_nchw_to_nhwc(
    Storage const* src,
    Storage* dst,
    int n,
    int c,
    int h,
    int w,
    hipStream_t stream) {
  if (src == nullptr || dst == nullptr || n <= 0 || c <= 0 || h <= 0 || w <= 0) {
    return 1;
  }
  int total = n * c * h * w;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  hipLaunchKernelGGL(
      HIP_KERNEL_NAME(dinoml_ck_conv_nchw_to_nhwc_kernel<Storage>),
      dim3(blocks),
      dim3(threads),
      0,
      stream,
      src,
      dst,
      n,
      c,
      h,
      w);
  return hipGetLastError() == hipSuccess ? 0 : 3;
}

struct DinoConvBiasEpilogue {
  template <typename E, typename C, typename D0>
  __host__ __device__ void operator()(E& e, const C& c, const D0& bias) const {
    const float value = to_float(c) + to_float(bias);
    e = ck::type_convert<E>(value);
  }
};

struct DinoConvBiasReluEpilogue {
  template <typename E, typename C, typename D0>
  __host__ __device__ void operator()(E& e, const C& c, const D0& bias) const {
    const float value = to_float(c) + to_float(bias);
    e = ck::type_convert<E>(value > 0.0f ? value : 0.0f);
  }
};

struct DinoConvBiasAddEpilogue {
  template <typename E, typename C, typename D0, typename D1>
  __host__ __device__ void operator()(E& e, const C& c, const D0& bias, const D1& residual) const {
    const float value = to_float(c) + to_float(bias) + to_float(residual);
    e = ck::type_convert<E>(value);
  }
};

struct DinoConvBiasAddReluEpilogue {
  template <typename E, typename C, typename D0, typename D1>
  __host__ __device__ void operator()(E& e, const C& c, const D0& bias, const D1& residual) const {
    const float value = to_float(c) + to_float(bias) + to_float(residual);
    e = ck::type_convert<E>(value > 0.0f ? value : 0.0f);
  }
};

template <
    ConvConfig Config,
    typename Storage,
    typename Element,
    typename DLayouts,
    typename DDataTypes,
    typename CDEElementOp>
using DeviceConv2dXdl = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial,
    InLayout,
    WeiLayout,
    DLayouts,
    OutLayout,
    Element,
    Element,
    float,
    Element,
    DDataTypes,
    Element,
    PassThrough,
    PassThrough,
    CDEElementOp,
    ConvSpec,
    GemmSpec,
    1,
    ConvKernelConfig<Config, Storage>::BlockSize,
    ConvKernelConfig<Config, Storage>::MPerBlock,
    ConvKernelConfig<Config, Storage>::NPerBlock,
    ConvKernelConfig<Config, Storage>::KPerBlock,
    ConvKernelConfig<Config, Storage>::AK1,
    ConvKernelConfig<Config, Storage>::BK1,
    ConvKernelConfig<Config, Storage>::MPerXDL,
    ConvKernelConfig<Config, Storage>::NPerXDL,
    ConvKernelConfig<Config, Storage>::MXdlPerWave,
    ConvKernelConfig<Config, Storage>::NXdlPerWave,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::ABlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::ABlockTransferDstScalarPerVector_AK1,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::BBlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::BBlockTransferDstScalarPerVector_BK1,
    1,
    1,
    1,
    typename ConvKernelConfig<Config, Storage>::CDEBlockTransferClusterLengths,
    ConvKernelConfig<Config, Storage>::CDEBlockTransferScalarPerVector_NPerBlock>;

template <
    ConvConfig Config,
    typename Storage,
    typename Element,
    typename DLayouts,
    typename DDataTypes,
    typename CDEElementOp>
using DeviceConv3dXdl = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial3,
    In3dLayout,
    Wei3dLayout,
    DLayouts,
    Out3dLayout,
    Element,
    Element,
    float,
    Element,
    DDataTypes,
    Element,
    PassThrough,
    PassThrough,
    CDEElementOp,
    ConvSpec,
    GemmSpec,
    1,
    ConvKernelConfig<Config, Storage>::BlockSize,
    ConvKernelConfig<Config, Storage>::MPerBlock,
    ConvKernelConfig<Config, Storage>::NPerBlock,
    ConvKernelConfig<Config, Storage>::KPerBlock,
    ConvKernelConfig<Config, Storage>::AK1,
    ConvKernelConfig<Config, Storage>::BK1,
    ConvKernelConfig<Config, Storage>::MPerXDL,
    ConvKernelConfig<Config, Storage>::NPerXDL,
    ConvKernelConfig<Config, Storage>::MXdlPerWave,
    ConvKernelConfig<Config, Storage>::NXdlPerWave,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::ABlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::ABlockTransferDstScalarPerVector_AK1,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::BBlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::BBlockTransferDstScalarPerVector_BK1,
    1,
    1,
    1,
    typename ConvKernelConfig<Config, Storage>::CDEBlockTransferClusterLengths,
    ConvKernelConfig<Config, Storage>::CDEBlockTransferScalarPerVector_NPerBlock>;

template <
    ConvConfig Config,
    typename Storage,
    typename Element,
    typename DLayouts,
    typename DDataTypes,
    typename CDEElementOp>
using DeviceConv3dLegacyXdl = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial3,
    In3dLegacyLayout,
    Wei3dLegacyLayout,
    DLayouts,
    Out3dLegacyLayout,
    Element,
    Element,
    float,
    Element,
    DDataTypes,
    Element,
    PassThrough,
    PassThrough,
    CDEElementOp,
    ConvSpec,
    GemmSpec,
    1,
    ConvKernelConfig<Config, Storage>::BlockSize,
    ConvKernelConfig<Config, Storage>::MPerBlock,
    ConvKernelConfig<Config, Storage>::NPerBlock,
    ConvKernelConfig<Config, Storage>::KPerBlock,
    ConvKernelConfig<Config, Storage>::AK1,
    ConvKernelConfig<Config, Storage>::BK1,
    ConvKernelConfig<Config, Storage>::MPerXDL,
    ConvKernelConfig<Config, Storage>::NPerXDL,
    ConvKernelConfig<Config, Storage>::MXdlPerWave,
    ConvKernelConfig<Config, Storage>::NXdlPerWave,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::ABlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::ABlockTransferDstScalarPerVector_AK1,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::BBlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::BBlockTransferDstScalarPerVector_BK1,
    1,
    1,
    1,
    typename ConvKernelConfig<Config, Storage>::CDEBlockTransferClusterLengths,
    ConvKernelConfig<Config, Storage>::CDEBlockTransferScalarPerVector_NPerBlock>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv2dBiasXdl =
    DeviceConv2dXdl<Config, Storage, Element, ck::Tuple<BiasLayout>, ck::Tuple<Element>, DinoConvBiasEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv3dBiasXdl =
    DeviceConv3dXdl<Config, Storage, Element, ck::Tuple<BiasLayout>, ck::Tuple<Element>, DinoConvBiasEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv3dBiasLegacyXdl =
    DeviceConv3dLegacyXdl<Config, Storage, Element, ck::Tuple<BiasLayout>, ck::Tuple<Element>, DinoConvBiasEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv2dBiasReluXdl =
    DeviceConv2dXdl<Config, Storage, Element, ck::Tuple<BiasLayout>, ck::Tuple<Element>, DinoConvBiasReluEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv2dBiasAddXdl = DeviceConv2dXdl<
    Config,
    Storage,
    Element,
    ck::Tuple<BiasLayout, OutLayout>,
    ck::Tuple<Element, Element>,
    DinoConvBiasAddEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv2dBiasAddReluXdl = DeviceConv2dXdl<
    Config,
    Storage,
    Element,
    ck::Tuple<BiasLayout, OutLayout>,
    ck::Tuple<Element, Element>,
    DinoConvBiasAddReluEpilogue>;

template <typename Storage, ConvConfig Config>
int launch_ck_conv2d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
  if (x == nullptr || weight == nullptr || bias == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || out_height <= 0 || out_width <= 0 ||
      stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0 || dilation_h <= 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv2dBiasXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto hi = static_cast<ck::index_t>(in_height);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto y = static_cast<ck::index_t>(kernel_h);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto ho = static_cast<ck::index_t>(out_height);
  const auto wo = static_cast<ck::index_t>(out_width);

  const std::array<ck::index_t, NDimSpatial + 3> a_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> a_strides{
      n * c * hi * wi,
      c * hi * wi,
      hi * wi,
      wi,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> b_lengths{g, k, c, y, x_filter};
  const std::array<ck::index_t, NDimSpatial + 3> b_strides{
      k * c * y * x_filter,
      c * y * x_filter,
      y * x_filter,
      x_filter,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial + 3> e_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> e_strides{
      n * k * ho * wo,
      k * ho * wo,
      ho * wo,
      wo,
      ck::index_t{1}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_lengths{{bias_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_strides{{bias_strides}};
  const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
  const std::array<ck::index_t, NDimSpatial> conv_strides{
      static_cast<ck::index_t>(stride_h),
      static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial> conv_dilations{
      static_cast<ck::index_t>(dilation_h),
      static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial> left_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial> right_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(x),
      ck_ptr<Storage, Element>(weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <typename Storage, ConvConfig Config>
int launch_ck_conv3d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
    int batch,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || bias == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_depth <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 ||
      kernel_d <= 0 || kernel_h <= 0 || kernel_w <= 0 || out_depth <= 0 || out_height <= 0 || out_width <= 0 ||
      stride_d <= 0 || stride_h <= 0 || stride_w <= 0 || pad_d < 0 || pad_h < 0 || pad_w < 0 ||
      dilation_d <= 0 || dilation_h <= 0 || dilation_w <= 0 || groups <= 0 ||
      (in_channels % groups) != 0 || (out_channels % groups) != 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  const auto g = static_cast<ck::index_t>(groups);
  const auto n = static_cast<ck::index_t>(batch);
  const auto di = static_cast<ck::index_t>(in_depth);
  const auto hi = static_cast<ck::index_t>(in_height);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto z = static_cast<ck::index_t>(kernel_d);
  const auto y = static_cast<ck::index_t>(kernel_h);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto do_ = static_cast<ck::index_t>(out_depth);
  const auto ho = static_cast<ck::index_t>(out_height);
  const auto wo = static_cast<ck::index_t>(out_width);
  const std::array<ck::index_t, NDimSpatial3> conv_strides{
      static_cast<ck::index_t>(stride_d),
      static_cast<ck::index_t>(stride_h),
      static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial3> conv_dilations{
      static_cast<ck::index_t>(dilation_d),
      static_cast<ck::index_t>(dilation_h),
      static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial3> left_pads{
      static_cast<ck::index_t>(pad_d),
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial3> right_pads{
      static_cast<ck::index_t>(pad_d),
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};

  if(groups == 1)
  {
    using DeviceOp = DeviceConv3dBiasLegacyXdl<Config, Storage, Element>;
    const auto c = static_cast<ck::index_t>(in_channels);
    const auto k = static_cast<ck::index_t>(out_channels);
    const std::array<ck::index_t, NDimSpatial3 + 3> a_lengths{g, n, c, di, hi, wi};
    const std::array<ck::index_t, NDimSpatial3 + 3> a_strides{
        n * c * di * hi * wi,
        c * di * hi * wi,
        di * hi * wi,
        hi * wi,
        wi,
        ck::index_t{1}};
    const std::array<ck::index_t, NDimSpatial3 + 3> b_lengths{g, k, c, z, y, x_filter};
    const std::array<ck::index_t, NDimSpatial3 + 3> b_strides{
        k * c * z * y * x_filter,
        c * z * y * x_filter,
        z * y * x_filter,
        y * x_filter,
        x_filter,
        ck::index_t{1}};
    const std::array<ck::index_t, NDimSpatial3 + 3> bias_lengths{g, n, k, do_, ho, wo};
    const std::array<ck::index_t, NDimSpatial3 + 3> bias_strides{
        k,
        ck::index_t{0},
        ck::index_t{1},
        ck::index_t{0},
        ck::index_t{0},
        ck::index_t{0}};
    const std::array<ck::index_t, NDimSpatial3 + 3> e_lengths{g, n, k, do_, ho, wo};
    const std::array<ck::index_t, NDimSpatial3 + 3> e_strides{
        n * k * do_ * ho * wo,
        k * do_ * ho * wo,
        do_ * ho * wo,
        ho * wo,
        wo,
        ck::index_t{1}};
    const std::array<std::array<ck::index_t, NDimSpatial3 + 3>, 1> ds_lengths{{bias_lengths}};
    const std::array<std::array<ck::index_t, NDimSpatial3 + 3>, 1> ds_strides{{bias_strides}};
    const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
    auto argument = DeviceOp::MakeArgument(
        ck_ptr<Storage, Element>(x),
        ck_ptr<Storage, Element>(weight),
        ds_ptrs,
        ck_ptr<Storage, Element>(output),
        a_lengths,
        a_strides,
        b_lengths,
        b_strides,
        ds_lengths,
        ds_strides,
        e_lengths,
        e_strides,
        conv_strides,
        conv_dilations,
        left_pads,
        right_pads,
        PassThrough{},
        PassThrough{},
        DinoConvBiasEpilogue{});
    if(!DeviceOp::IsSupportedArgument(argument))
    {
      return 4;
    }
    auto invoker = DeviceOp::MakeInvoker();
    (void)invoker.Run(argument, StreamConfig{stream, false});
  }
  else
  {
    using DeviceOp = DeviceConv3dBiasXdl<Config, Storage, Element>;
    const auto c_per_group = static_cast<ck::index_t>(in_channels / groups);
    const auto k_per_group = static_cast<ck::index_t>(out_channels / groups);
    const auto in_channel_stride = di * hi * wi;
    const auto out_channel_stride = do_ * ho * wo;
    const std::array<ck::index_t, NDimSpatial3 + 3> a_lengths{g, n, c_per_group, di, hi, wi};
    const std::array<ck::index_t, NDimSpatial3 + 3> a_strides{
        c_per_group * in_channel_stride,
        static_cast<ck::index_t>(in_channels) * in_channel_stride,
        in_channel_stride,
        hi * wi,
        wi,
        ck::index_t{1}};
    const std::array<ck::index_t, NDimSpatial3 + 3> b_lengths{g, k_per_group, c_per_group, z, y, x_filter};
    const std::array<ck::index_t, NDimSpatial3 + 3> b_strides{
        k_per_group * c_per_group * z * y * x_filter,
        c_per_group * z * y * x_filter,
        z * y * x_filter,
        y * x_filter,
        x_filter,
        ck::index_t{1}};
    const std::array<ck::index_t, NDimSpatial3 + 3> bias_lengths{g, n, k_per_group, do_, ho, wo};
    const std::array<ck::index_t, NDimSpatial3 + 3> bias_strides{
        k_per_group,
        ck::index_t{0},
        ck::index_t{1},
        ck::index_t{0},
        ck::index_t{0},
        ck::index_t{0}};
    const std::array<ck::index_t, NDimSpatial3 + 3> e_lengths{g, n, k_per_group, do_, ho, wo};
    const std::array<ck::index_t, NDimSpatial3 + 3> e_strides{
        k_per_group * out_channel_stride,
        static_cast<ck::index_t>(out_channels) * out_channel_stride,
        out_channel_stride,
        ho * wo,
        wo,
        ck::index_t{1}};
    const std::array<std::array<ck::index_t, NDimSpatial3 + 3>, 1> ds_lengths{{bias_lengths}};
    const std::array<std::array<ck::index_t, NDimSpatial3 + 3>, 1> ds_strides{{bias_strides}};
    const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
    auto argument = DeviceOp::MakeArgument(
        ck_ptr<Storage, Element>(x),
        ck_ptr<Storage, Element>(weight),
        ds_ptrs,
        ck_ptr<Storage, Element>(output),
        a_lengths,
        a_strides,
        b_lengths,
        b_strides,
        ds_lengths,
        ds_strides,
        e_lengths,
        e_strides,
        conv_strides,
        conv_dilations,
        left_pads,
        right_pads,
        PassThrough{},
        PassThrough{},
        DinoConvBiasEpilogue{});
    if(!DeviceOp::IsSupportedArgument(argument))
    {
      return 4;
    }
    auto invoker = DeviceOp::MakeInvoker();
    (void)invoker.Run(argument, StreamConfig{stream, false});
  }
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <typename Storage, ConvConfig Config>
int launch_ck_conv2d_bias_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
  if (x == nullptr || weight == nullptr || bias == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || out_height <= 0 || out_width <= 0 ||
      stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0 || dilation_h <= 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv2dBiasReluXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto hi = static_cast<ck::index_t>(in_height);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto y = static_cast<ck::index_t>(kernel_h);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto ho = static_cast<ck::index_t>(out_height);
  const auto wo = static_cast<ck::index_t>(out_width);

  const std::array<ck::index_t, NDimSpatial + 3> a_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> a_strides{
      n * c * hi * wi,
      c * hi * wi,
      hi * wi,
      wi,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> b_lengths{g, k, c, y, x_filter};
  const std::array<ck::index_t, NDimSpatial + 3> b_strides{
      k * c * y * x_filter,
      c * y * x_filter,
      y * x_filter,
      x_filter,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial + 3> e_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> e_strides{
      n * k * ho * wo,
      k * ho * wo,
      ho * wo,
      wo,
      ck::index_t{1}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_lengths{{bias_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_strides{{bias_strides}};
  const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
  const std::array<ck::index_t, NDimSpatial> conv_strides{
      static_cast<ck::index_t>(stride_h),
      static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial> conv_dilations{
      static_cast<ck::index_t>(dilation_h),
      static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial> left_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial> right_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(x),
      ck_ptr<Storage, Element>(weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasReluEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <typename Storage, ConvConfig Config>
int launch_ck_conv2d_bias_add(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
  if (x == nullptr || weight == nullptr || bias == nullptr || residual == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || out_height <= 0 || out_width <= 0 ||
      stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0 || dilation_h <= 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv2dBiasAddXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto hi = static_cast<ck::index_t>(in_height);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto y = static_cast<ck::index_t>(kernel_h);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto ho = static_cast<ck::index_t>(out_height);
  const auto wo = static_cast<ck::index_t>(out_width);

  const std::array<ck::index_t, NDimSpatial + 3> a_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> a_strides{
      n * c * hi * wi,
      c * hi * wi,
      hi * wi,
      wi,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> b_lengths{g, k, c, y, x_filter};
  const std::array<ck::index_t, NDimSpatial + 3> b_strides{
      k * c * y * x_filter,
      c * y * x_filter,
      y * x_filter,
      x_filter,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial + 3> e_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> e_strides{
      n * k * ho * wo,
      k * ho * wo,
      ho * wo,
      wo,
      ck::index_t{1}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_lengths{{bias_lengths, e_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_strides{{bias_strides, e_strides}};
  const std::array<const void*, 2> ds_ptrs{
      ck_ptr<Storage, Element>(bias),
      ck_ptr<Storage, Element>(residual)};
  const std::array<ck::index_t, NDimSpatial> conv_strides{
      static_cast<ck::index_t>(stride_h),
      static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial> conv_dilations{
      static_cast<ck::index_t>(dilation_h),
      static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial> left_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial> right_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(x),
      ck_ptr<Storage, Element>(weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasAddEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <typename Storage, ConvConfig Config>
int launch_ck_conv2d_bias_add_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
  if (x == nullptr || weight == nullptr || bias == nullptr || residual == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || out_height <= 0 || out_width <= 0 ||
      stride_h <= 0 || stride_w <= 0 || pad_h < 0 || pad_w < 0 || dilation_h <= 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv2dBiasAddReluXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto hi = static_cast<ck::index_t>(in_height);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto y = static_cast<ck::index_t>(kernel_h);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto ho = static_cast<ck::index_t>(out_height);
  const auto wo = static_cast<ck::index_t>(out_width);

  const std::array<ck::index_t, NDimSpatial + 3> a_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> a_strides{
      n * c * hi * wi,
      c * hi * wi,
      hi * wi,
      wi,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> b_lengths{g, k, c, y, x_filter};
  const std::array<ck::index_t, NDimSpatial + 3> b_strides{
      k * c * y * x_filter,
      c * y * x_filter,
      y * x_filter,
      x_filter,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial + 3> e_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> e_strides{
      n * k * ho * wo,
      k * ho * wo,
      ho * wo,
      wo,
      ck::index_t{1}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_lengths{{bias_lengths, e_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_strides{{bias_strides, e_strides}};
  const std::array<const void*, 2> ds_ptrs{
      ck_ptr<Storage, Element>(bias),
      ck_ptr<Storage, Element>(residual)};
  const std::array<ck::index_t, NDimSpatial> conv_strides{
      static_cast<ck::index_t>(stride_h),
      static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial> conv_dilations{
      static_cast<ck::index_t>(dilation_h),
      static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial> left_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial> right_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(x),
      ck_ptr<Storage, Element>(weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasAddReluEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv2d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv2d_bias<Storage, Config>(
            x,
            weight,
            bias,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv3d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
    int batch,
    int in_channels,
    int in_depth,
    int in_height,
    int in_width,
    int out_channels,
    int kernel_d,
    int kernel_h,
    int kernel_w,
    int out_depth,
    int out_height,
    int out_width,
    int stride_d,
    int stride_h,
    int stride_w,
    int pad_d,
    int pad_h,
    int pad_w,
    int dilation_d,
    int dilation_h,
    int dilation_w,
    int groups,
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv3d_bias<Storage, Config>(
        x, weight, bias, output, batch, in_channels, in_depth, in_height, in_width, out_channels, kernel_d, kernel_h,
        kernel_w, out_depth, out_height, out_width, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d,
        dilation_h, dilation_w, groups, stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv2d_bias_add(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv2d_bias_add<Storage, Config>(
            x,
            weight,
            bias,
            residual,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv2d_bias_add_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv2d_bias_add_relu<Storage, Config>(
            x,
            weight,
            bias,
            residual,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv2d_bias_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv2d_bias_relu<Storage, Config>(
            x,
            weight,
            bias,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation_h,
            dilation_w,
            stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <
    ConvConfig Config,
    typename Storage,
    typename Element,
    typename DLayouts,
    typename DDataTypes,
    typename CDEElementOp>
using DeviceConv1dXdl = ck::tensor_operation::device::DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<
    NDimSpatial1,
    In1dLayout,
    Wei1dLayout,
    DLayouts,
    Out1dLayout,
    Element,
    Element,
    float,
    Element,
    DDataTypes,
    Element,
    PassThrough,
    PassThrough,
    CDEElementOp,
    ConvSpec,
    GemmSpec,
    1,
    ConvKernelConfig<Config, Storage>::BlockSize,
    ConvKernelConfig<Config, Storage>::MPerBlock,
    ConvKernelConfig<Config, Storage>::NPerBlock,
    ConvKernelConfig<Config, Storage>::KPerBlock,
    ConvKernelConfig<Config, Storage>::AK1,
    ConvKernelConfig<Config, Storage>::BK1,
    ConvKernelConfig<Config, Storage>::MPerXDL,
    ConvKernelConfig<Config, Storage>::NPerXDL,
    ConvKernelConfig<Config, Storage>::MXdlPerWave,
    ConvKernelConfig<Config, Storage>::NXdlPerWave,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::ABlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::ABlockTransferDstScalarPerVector_AK1,
    1,
    S<4, 64, 1>,
    S<1, 0, 2>,
    S<1, 0, 2>,
    2,
    ConvKernelConfig<Config, Storage>::BBlockTransferSrcScalarPerVector,
    ConvKernelConfig<Config, Storage>::BBlockTransferDstScalarPerVector_BK1,
    1,
    1,
    1,
    typename ConvKernelConfig<Config, Storage>::CDEBlockTransferClusterLengths,
    ConvKernelConfig<Config, Storage>::CDEBlockTransferScalarPerVector_NPerBlock>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv1dBiasXdl =
    DeviceConv1dXdl<Config, Storage, Element, ck::Tuple<BiasLayout>, ck::Tuple<Element>, DinoConvBiasEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv1dBiasReluXdl =
    DeviceConv1dXdl<Config, Storage, Element, ck::Tuple<BiasLayout>, ck::Tuple<Element>, DinoConvBiasReluEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv1dBiasAddXdl = DeviceConv1dXdl<
    Config,
    Storage,
    Element,
    ck::Tuple<BiasLayout, Out1dLayout>,
    ck::Tuple<Element, Element>,
    DinoConvBiasAddEpilogue>;

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv1dBiasAddReluXdl = DeviceConv1dXdl<
    Config,
    Storage,
    Element,
    ck::Tuple<BiasLayout, Out1dLayout>,
    ck::Tuple<Element, Element>,
    DinoConvBiasAddReluEpilogue>;

template <typename Storage, ConvConfig Config>
int launch_ck_conv1d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || bias == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_width <= 0 || out_channels <= 0 || kernel_w <= 0 ||
      out_width <= 0 || stride_w <= 0 || pad_w < 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv1dBiasXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto wo = static_cast<ck::index_t>(out_width);
  const std::size_t packed_input_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(in_width) * static_cast<std::size_t>(in_channels) *
      sizeof(Storage);
  const std::size_t packed_output_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(out_width) * static_cast<std::size_t>(out_channels) *
      sizeof(Storage);
  Storage* packed_input = nullptr;
  Storage* packed_weight = nullptr;
  const Storage* kernel_weight = nullptr;
  bool owns_packed_weight = false;
  Storage* packed_output = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&packed_input), packed_input_nbytes) != hipSuccess) {
    return 5;
  }
  if (hipMalloc(reinterpret_cast<void**>(&packed_output), packed_output_nbytes) != hipSuccess) {
    (void)hipFree(packed_input);
    return 5;
  }
  int pack_status = launch_ck_conv1d_ncw_to_nwc(x, packed_input, batch, in_channels, in_width, stream);
  if (pack_status != 0) {
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return pack_status;
  }
  if (weight_is_kxc != 0) {
    kernel_weight = weight;
  } else {
    pack_status = pack_conv1d_weight_kcx_to_kxc(weight, &packed_weight, out_channels, in_channels, kernel_w, stream);
    if (pack_status != 0) {
      (void)hipFree(packed_output);
      (void)hipFree(packed_input);
      return pack_status;
    }
    kernel_weight = packed_weight;
    owns_packed_weight = true;
  }

  const std::array<ck::index_t, NDimSpatial1 + 3> a_lengths{g, n, c, wi};
  const std::array<ck::index_t, NDimSpatial1 + 3> a_strides{
      c,
      wi * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_lengths{g, k, c, x_filter};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_strides{
      k * x_filter * c,
      x_filter * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_strides{
      k,
      wo * k,
      ck::index_t{1},
      k};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 1> ds_lengths{{bias_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 1> ds_strides{{bias_strides}};
  const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
  const std::array<ck::index_t, NDimSpatial1> conv_strides{static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial1> conv_dilations{static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial1> left_pads{static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial1> right_pads{static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(packed_input),
      ck_ptr<Storage, Element>(kernel_weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(packed_output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    if (owns_packed_weight) {
      (void)hipFree(packed_weight);
    }
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  int unpack_status = 0;
  if (err == hipSuccess) {
    unpack_status = launch_ck_conv1d_nwk_to_ncw(packed_output, output, batch, out_channels, out_width, stream);
  }
  (void)hipFree(packed_output);
  (void)hipFree(packed_input);
  if (owns_packed_weight) {
    (void)hipFree(packed_weight);
  }
  if (err != hipSuccess) {
    return 3;
  }
  return unpack_status;
}

template <typename Storage, ConvConfig Config>
int launch_ck_conv1d_bias_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || bias == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_width <= 0 || out_channels <= 0 || kernel_w <= 0 ||
      out_width <= 0 || stride_w <= 0 || pad_w < 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv1dBiasReluXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto wo = static_cast<ck::index_t>(out_width);
  const std::size_t packed_input_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(in_width) * static_cast<std::size_t>(in_channels) *
      sizeof(Storage);
  const std::size_t packed_output_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(out_width) * static_cast<std::size_t>(out_channels) *
      sizeof(Storage);
  Storage* packed_input = nullptr;
  Storage* packed_weight = nullptr;
  const Storage* kernel_weight = nullptr;
  bool owns_packed_weight = false;
  Storage* packed_output = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&packed_input), packed_input_nbytes) != hipSuccess) {
    return 5;
  }
  if (hipMalloc(reinterpret_cast<void**>(&packed_output), packed_output_nbytes) != hipSuccess) {
    (void)hipFree(packed_input);
    return 5;
  }
  int pack_status = launch_ck_conv1d_ncw_to_nwc(x, packed_input, batch, in_channels, in_width, stream);
  if (pack_status != 0) {
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return pack_status;
  }
  if (weight_is_kxc != 0) {
    kernel_weight = weight;
  } else {
    pack_status = pack_conv1d_weight_kcx_to_kxc(weight, &packed_weight, out_channels, in_channels, kernel_w, stream);
    if (pack_status != 0) {
      (void)hipFree(packed_output);
      (void)hipFree(packed_input);
      return pack_status;
    }
    kernel_weight = packed_weight;
    owns_packed_weight = true;
  }

  const std::array<ck::index_t, NDimSpatial1 + 3> a_lengths{g, n, c, wi};
  const std::array<ck::index_t, NDimSpatial1 + 3> a_strides{
      c,
      wi * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_lengths{g, k, c, x_filter};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_strides{
      k * x_filter * c,
      x_filter * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_strides{
      k,
      wo * k,
      ck::index_t{1},
      k};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 1> ds_lengths{{bias_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 1> ds_strides{{bias_strides}};
  const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
  const std::array<ck::index_t, NDimSpatial1> conv_strides{static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial1> conv_dilations{static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial1> left_pads{static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial1> right_pads{static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(packed_input),
      ck_ptr<Storage, Element>(kernel_weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(packed_output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasReluEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    if (owns_packed_weight) {
      (void)hipFree(packed_weight);
    }
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  int unpack_status = 0;
  if (err == hipSuccess) {
    unpack_status = launch_ck_conv1d_nwk_to_ncw(packed_output, output, batch, out_channels, out_width, stream);
  }
  (void)hipFree(packed_output);
  (void)hipFree(packed_input);
  if (owns_packed_weight) {
    (void)hipFree(packed_weight);
  }
  if (err != hipSuccess) {
    return 3;
  }
  return unpack_status;
}

template <typename Storage, ConvConfig Config>
int launch_ck_conv1d_bias_add(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || bias == nullptr || residual == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_width <= 0 || out_channels <= 0 || kernel_w <= 0 ||
      out_width <= 0 || stride_w <= 0 || pad_w < 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv1dBiasAddXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto wo = static_cast<ck::index_t>(out_width);
  const std::size_t packed_input_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(in_width) * static_cast<std::size_t>(in_channels) *
      sizeof(Storage);
  const std::size_t packed_output_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(out_width) * static_cast<std::size_t>(out_channels) *
      sizeof(Storage);
  Storage* packed_input = nullptr;
  Storage* packed_weight = nullptr;
  const Storage* kernel_weight = nullptr;
  bool owns_packed_weight = false;
  Storage* packed_output = nullptr;
  Storage* packed_residual = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&packed_input), packed_input_nbytes) != hipSuccess) {
    return 5;
  }
  if (hipMalloc(reinterpret_cast<void**>(&packed_output), packed_output_nbytes) != hipSuccess) {
    (void)hipFree(packed_input);
    return 5;
  }
  if (hipMalloc(reinterpret_cast<void**>(&packed_residual), packed_output_nbytes) != hipSuccess) {
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return 5;
  }
  int pack_status = launch_ck_conv1d_ncw_to_nwc(x, packed_input, batch, in_channels, in_width, stream);
  if (pack_status != 0) {
    (void)hipFree(packed_residual);
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return pack_status;
  }
  pack_status = launch_ck_conv1d_ncw_to_nwc(residual, packed_residual, batch, out_channels, out_width, stream);
  if (pack_status != 0) {
    (void)hipFree(packed_residual);
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return pack_status;
  }
  if (weight_is_kxc != 0) {
    kernel_weight = weight;
  } else {
    pack_status = pack_conv1d_weight_kcx_to_kxc(weight, &packed_weight, out_channels, in_channels, kernel_w, stream);
    if (pack_status != 0) {
      (void)hipFree(packed_residual);
      (void)hipFree(packed_output);
      (void)hipFree(packed_input);
      return pack_status;
    }
    kernel_weight = packed_weight;
    owns_packed_weight = true;
  }

  const std::array<ck::index_t, NDimSpatial1 + 3> a_lengths{g, n, c, wi};
  const std::array<ck::index_t, NDimSpatial1 + 3> a_strides{
      c,
      wi * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_lengths{g, k, c, x_filter};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_strides{
      k * x_filter * c,
      x_filter * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_strides{
      k,
      wo * k,
      ck::index_t{1},
      k};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 2> ds_lengths{{bias_lengths, e_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 2> ds_strides{{bias_strides, e_strides}};
  const std::array<const void*, 2> ds_ptrs{
      ck_ptr<Storage, Element>(bias),
      ck_ptr<Storage, Element>(packed_residual)};
  const std::array<ck::index_t, NDimSpatial1> conv_strides{static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial1> conv_dilations{static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial1> left_pads{static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial1> right_pads{static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(packed_input),
      ck_ptr<Storage, Element>(kernel_weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(packed_output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasAddEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    (void)hipFree(packed_residual);
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    if (owns_packed_weight) {
      (void)hipFree(packed_weight);
    }
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  int unpack_status = 0;
  if (err == hipSuccess) {
    unpack_status = launch_ck_conv1d_nwk_to_ncw(packed_output, output, batch, out_channels, out_width, stream);
  }
  (void)hipFree(packed_residual);
  (void)hipFree(packed_output);
  (void)hipFree(packed_input);
  if (owns_packed_weight) {
    (void)hipFree(packed_weight);
  }
  if (err != hipSuccess) {
    return 3;
  }
  return unpack_status;
}

template <typename Storage, ConvConfig Config>
int launch_ck_conv1d_bias_add_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || bias == nullptr || residual == nullptr || output == nullptr) {
    return 1;
  }
  if (batch <= 0 || in_channels <= 0 || in_width <= 0 || out_channels <= 0 || kernel_w <= 0 ||
      out_width <= 0 || stride_w <= 0 || pad_w < 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceConv1dBiasAddReluXdl<Config, Storage, Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(in_channels);
  const auto wi = static_cast<ck::index_t>(in_width);
  const auto k = static_cast<ck::index_t>(out_channels);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto wo = static_cast<ck::index_t>(out_width);
  const std::size_t packed_input_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(in_width) * static_cast<std::size_t>(in_channels) *
      sizeof(Storage);
  const std::size_t packed_output_nbytes =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(out_width) * static_cast<std::size_t>(out_channels) *
      sizeof(Storage);
  Storage* packed_input = nullptr;
  Storage* packed_weight = nullptr;
  const Storage* kernel_weight = nullptr;
  bool owns_packed_weight = false;
  Storage* packed_output = nullptr;
  Storage* packed_residual = nullptr;
  if (hipMalloc(reinterpret_cast<void**>(&packed_input), packed_input_nbytes) != hipSuccess) {
    return 5;
  }
  if (hipMalloc(reinterpret_cast<void**>(&packed_output), packed_output_nbytes) != hipSuccess) {
    (void)hipFree(packed_input);
    return 5;
  }
  if (hipMalloc(reinterpret_cast<void**>(&packed_residual), packed_output_nbytes) != hipSuccess) {
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return 5;
  }
  int pack_status = launch_ck_conv1d_ncw_to_nwc(x, packed_input, batch, in_channels, in_width, stream);
  if (pack_status != 0) {
    (void)hipFree(packed_residual);
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return pack_status;
  }
  pack_status = launch_ck_conv1d_ncw_to_nwc(residual, packed_residual, batch, out_channels, out_width, stream);
  if (pack_status != 0) {
    (void)hipFree(packed_residual);
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    return pack_status;
  }
  if (weight_is_kxc != 0) {
    kernel_weight = weight;
  } else {
    pack_status = pack_conv1d_weight_kcx_to_kxc(weight, &packed_weight, out_channels, in_channels, kernel_w, stream);
    if (pack_status != 0) {
      (void)hipFree(packed_residual);
      (void)hipFree(packed_output);
      (void)hipFree(packed_input);
      return pack_status;
    }
    kernel_weight = packed_weight;
    owns_packed_weight = true;
  }

  const std::array<ck::index_t, NDimSpatial1 + 3> a_lengths{g, n, c, wi};
  const std::array<ck::index_t, NDimSpatial1 + 3> a_strides{
      c,
      wi * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_lengths{g, k, c, x_filter};
  const std::array<ck::index_t, NDimSpatial1 + 3> b_strides{
      k * x_filter * c,
      x_filter * c,
      ck::index_t{1},
      c};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> bias_strides{
      k,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_lengths{g, n, k, wo};
  const std::array<ck::index_t, NDimSpatial1 + 3> e_strides{
      k,
      wo * k,
      ck::index_t{1},
      k};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 2> ds_lengths{{bias_lengths, e_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial1 + 3>, 2> ds_strides{{bias_strides, e_strides}};
  const std::array<const void*, 2> ds_ptrs{
      ck_ptr<Storage, Element>(bias),
      ck_ptr<Storage, Element>(packed_residual)};
  const std::array<ck::index_t, NDimSpatial1> conv_strides{static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial1> conv_dilations{static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial1> left_pads{static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial1> right_pads{static_cast<ck::index_t>(pad_w)};

  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(packed_input),
      ck_ptr<Storage, Element>(kernel_weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(packed_output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      DinoConvBiasAddReluEpilogue{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    (void)hipFree(packed_residual);
    (void)hipFree(packed_output);
    (void)hipFree(packed_input);
    if (owns_packed_weight) {
      (void)hipFree(packed_weight);
    }
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  int unpack_status = 0;
  if (err == hipSuccess) {
    unpack_status = launch_ck_conv1d_nwk_to_ncw(packed_output, output, batch, out_channels, out_width, stream);
  }
  (void)hipFree(packed_residual);
  (void)hipFree(packed_output);
  (void)hipFree(packed_input);
  if (owns_packed_weight) {
    (void)hipFree(packed_weight);
  }
  if (err != hipSuccess) {
    return 3;
  }
  return unpack_status;
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv1d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv1d_bias<Storage, Config>(
        x, weight, bias, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv1d_bias_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv1d_bias_relu<Storage, Config>(
        x, weight, bias, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv1d_bias_add(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv1d_bias_add<Storage, Config>(
        x, weight, bias, residual, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, ConvConfig Config>
float profile_ck_conv1d_bias_add_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
    int batch,
    int in_channels,
    int in_width,
    int out_channels,
    int kernel_w,
    int out_width,
    int stride_w,
    int pad_w,
    int dilation_w,
    int weight_is_kxc,
    int iterations,
    hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch_ck_conv1d_bias_add_relu<Storage, Config>(
        x, weight, bias, residual, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

using TransposedInputLayout = conv_layout::NGKHW;
using TransposedWeightLayout = conv_layout::GKCYX;
using TransposedOutputLayout = conv_layout::NGCHW;
using TransposedBiasLayout = conv_layout::G_C;
using TransposedResidualLayout = conv_layout::G_NHW_C;

template <typename Element, typename DLayouts, typename DDataTypes, typename CDEElementOp>
using DeviceTransposedConv2dXdl =
    ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1<
        NDimSpatial,
        TransposedInputLayout,
        TransposedWeightLayout,
        DLayouts,
        TransposedOutputLayout,
        Element,
        Element,
        float,
        Element,
        DDataTypes,
        Element,
        PassThrough,
        PassThrough,
        CDEElementOp,
        ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Default,
        true,
        true,
        1,
        256,
        128,
        256,
        32,
        8,
        2,
        16,
        16,
        4,
        8,
        S<4, 64, 1>,
        S<1, 0, 2>,
        S<1, 0, 2>,
        2,
        8,
        8,
        1,
        S<4, 64, 1>,
        S<0, 2, 1>,
        S<0, 2, 1>,
        1,
        4,
        2,
        0,
        1,
        1,
        S<1, 32, 1, 8>,
        4>;

template <typename Element>
using DeviceTransposedConv2dIdentityXdl =
    DeviceTransposedConv2dXdl<Element, ck::Tuple<>, ck::Tuple<>, PassThrough>;

template <typename Element>
using DeviceTransposedConv2dBiasXdl =
    DeviceTransposedConv2dXdl<Element, ck::Tuple<TransposedBiasLayout>, ck::Tuple<Element>, DinoConvBiasEpilogue>;

template <typename Element>
using DeviceTransposedConv2dBiasReluXdl =
    DeviceTransposedConv2dXdl<Element, ck::Tuple<TransposedBiasLayout>, ck::Tuple<Element>, DinoConvBiasReluEpilogue>;

template <typename Element>
using DeviceTransposedConv2dBiasAddXdl = DeviceTransposedConv2dXdl<
    Element,
    ck::Tuple<TransposedBiasLayout, TransposedResidualLayout>,
    ck::Tuple<Element, Element>,
    DinoConvBiasAddEpilogue>;

template <typename Element>
using DeviceTransposedConv2dBiasAddReluXdl = DeviceTransposedConv2dXdl<
    Element,
    ck::Tuple<TransposedBiasLayout, TransposedResidualLayout>,
    ck::Tuple<Element, Element>,
    DinoConvBiasAddReluEpilogue>;

template <typename Storage, typename DeviceOp, std::size_t NumDTensor, typename CDEElementOp>
int launch_ck_transposed_conv2d_impl(
    Storage const* x,
    Storage const* weight,
    const std::array<const void*, NumDTensor>& ds_ptrs,
    Storage* output,
    const std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDTensor>& ds_lengths,
    const std::array<std::array<ck::index_t, NDimSpatial + 3>, NumDTensor>& ds_strides,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    const CDEElementOp& cde_element_op,
    hipStream_t stream) {
  if (x == nullptr || weight == nullptr || output == nullptr) {
    return 1;
  }
  for (const void* ptr : ds_ptrs) {
    if (ptr == nullptr) {
      return 1;
    }
  }
  if (batch <= 0 || in_channels <= 0 || in_height <= 0 || in_width <= 0 || out_channels <= 0 ||
      kernel_h <= 0 || kernel_w <= 0 || out_height <= 0 || out_width <= 0 || stride_h <= 0 ||
      stride_w <= 0 || pad_h < 0 || pad_w < 0 || output_pad_h < 0 || output_pad_w < 0 ||
      dilation_h <= 0 || dilation_w <= 0) {
    return 2;
  }

  using Element = typename CkElement<Storage>::Type;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto k = static_cast<ck::index_t>(in_channels);
  const auto ho = static_cast<ck::index_t>(in_height);
  const auto wo = static_cast<ck::index_t>(in_width);
  const auto c = static_cast<ck::index_t>(out_channels);
  const auto y = static_cast<ck::index_t>(kernel_h);
  const auto x_filter = static_cast<ck::index_t>(kernel_w);
  const auto hi = static_cast<ck::index_t>(out_height);
  const auto wi = static_cast<ck::index_t>(out_width);

  const std::array<ck::index_t, NDimSpatial + 3> a_lengths{g, n, k, ho, wo};
  const std::array<ck::index_t, NDimSpatial + 3> a_strides{
      n * k * ho * wo,
      k * ho * wo,
      ho * wo,
      wo,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> b_lengths{g, k, c, y, x_filter};
  const std::array<ck::index_t, NDimSpatial + 3> b_strides{
      k * c * y * x_filter,
      c * y * x_filter,
      y * x_filter,
      x_filter,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial + 3> e_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> e_strides{
      n * c * hi * wi,
      c * hi * wi,
      hi * wi,
      wi,
      ck::index_t{1}};
  const std::array<ck::index_t, NDimSpatial> conv_strides{
      static_cast<ck::index_t>(stride_h),
      static_cast<ck::index_t>(stride_w)};
  const std::array<ck::index_t, NDimSpatial> conv_dilations{
      static_cast<ck::index_t>(dilation_h),
      static_cast<ck::index_t>(dilation_w)};
  const std::array<ck::index_t, NDimSpatial> left_pads{
      static_cast<ck::index_t>(pad_h),
      static_cast<ck::index_t>(pad_w)};
  const std::array<ck::index_t, NDimSpatial> right_pads{
      static_cast<ck::index_t>(pad_h - output_pad_h),
      static_cast<ck::index_t>(pad_w - output_pad_w)};

  DeviceOp op{};
  auto argument = op.MakeArgumentPointer(
      ck_ptr<Storage, Element>(x),
      ck_ptr<Storage, Element>(weight),
      ds_ptrs,
      ck_ptr<Storage, Element>(output),
      a_lengths,
      a_strides,
      b_lengths,
      b_strides,
      ds_lengths,
      ds_strides,
      e_lengths,
      e_strides,
      conv_strides,
      conv_dilations,
      left_pads,
      right_pads,
      PassThrough{},
      PassThrough{},
      cde_element_op);
  void* workspace = nullptr;
  const std::size_t workspace_nbytes = op.GetWorkSpaceSize(argument.get());
  if (workspace_nbytes != 0) {
    if (hipMalloc(&workspace, workspace_nbytes) != hipSuccess) {
      return 5;
    }
    op.SetWorkSpacePointer(argument.get(), workspace, StreamConfig{stream, false});
  }
  const bool supported = op.IsSupportedArgument(argument.get());
  if (!supported) {
    if (workspace != nullptr) {
      (void)hipFree(workspace);
    }
    return 4;
  }
  auto invoker = op.MakeInvokerPointer();
  (void)invoker->Run(argument.get(), StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  if (workspace != nullptr) {
    (void)hipFree(workspace);
  }
  return err == hipSuccess ? 0 : 3;
}

template <typename LaunchFn>
float profile_ck_transposed_conv2d_impl(LaunchFn&& launch, int iterations, hipStream_t stream) {
  if (iterations <= 0) {
    return -1.0f;
  }
  hipEvent_t start = nullptr;
  hipEvent_t end = nullptr;
  if (hipEventCreate(&start) != hipSuccess || hipEventCreate(&end) != hipSuccess) {
    if (start != nullptr) {
      (void)hipEventDestroy(start);
    }
    if (end != nullptr) {
      (void)hipEventDestroy(end);
    }
    return -1.0f;
  }
  (void)hipEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    const int launch_status = launch(stream);
    if (launch_status != 0) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -static_cast<float>(launch_status);
    }
  }
  (void)hipEventRecord(end, stream);
  (void)hipEventSynchronize(end);
  float ms = 0.0f;
  (void)hipEventElapsedTime(&ms, start, end);
  (void)hipEventDestroy(start);
  (void)hipEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage>
int launch_ck_transposed_conv2d(
    Storage const* x,
    Storage const* weight,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    hipStream_t stream) {
  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceTransposedConv2dIdentityXdl<Element>;
  const std::array<const void*, 0> ds_ptrs{};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 0> ds_lengths{};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 0> ds_strides{};
  return launch_ck_transposed_conv2d_impl<Storage, DeviceOp>(
      x,
      weight,
      ds_ptrs,
      output,
      ds_lengths,
      ds_strides,
      batch,
      in_channels,
      in_height,
      in_width,
      out_channels,
      kernel_h,
      kernel_w,
      out_height,
      out_width,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      output_pad_h,
      output_pad_w,
      dilation_h,
      dilation_w,
      PassThrough{},
      stream);
}

template <typename Storage>
int launch_ck_transposed_conv2d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    hipStream_t stream) {
  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceTransposedConv2dBiasXdl<Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(out_channels);
  const auto hi = static_cast<ck::index_t>(out_height);
  const auto wi = static_cast<ck::index_t>(out_width);
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      c,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_lengths{{bias_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_strides{{bias_strides}};
  return launch_ck_transposed_conv2d_impl<Storage, DeviceOp>(
      x,
      weight,
      ds_ptrs,
      output,
      ds_lengths,
      ds_strides,
      batch,
      in_channels,
      in_height,
      in_width,
      out_channels,
      kernel_h,
      kernel_w,
      out_height,
      out_width,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      output_pad_h,
      output_pad_w,
      dilation_h,
      dilation_w,
      DinoConvBiasEpilogue{},
      stream);
}

template <typename Storage>
int launch_ck_transposed_conv2d_bias_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    hipStream_t stream) {
  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceTransposedConv2dBiasReluXdl<Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(out_channels);
  const auto hi = static_cast<ck::index_t>(out_height);
  const auto wi = static_cast<ck::index_t>(out_width);
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      c,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<const void*, 1> ds_ptrs{ck_ptr<Storage, Element>(bias)};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_lengths{{bias_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 1> ds_strides{{bias_strides}};
  return launch_ck_transposed_conv2d_impl<Storage, DeviceOp>(
      x,
      weight,
      ds_ptrs,
      output,
      ds_lengths,
      ds_strides,
      batch,
      in_channels,
      in_height,
      in_width,
      out_channels,
      kernel_h,
      kernel_w,
      out_height,
      out_width,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      output_pad_h,
      output_pad_w,
      dilation_h,
      dilation_w,
      DinoConvBiasReluEpilogue{},
      stream);
}

template <typename Storage>
int launch_ck_transposed_conv2d_bias_add(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    hipStream_t stream) {
  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceTransposedConv2dBiasAddXdl<Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(out_channels);
  const auto hi = static_cast<ck::index_t>(out_height);
  const auto wi = static_cast<ck::index_t>(out_width);
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      c,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial + 3> residual_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> residual_strides{
      n * c * hi * wi,
      hi * wi * c,
      ck::index_t{1},
      wi * c,
      c};
  void* residual_packed = nullptr;
  const std::size_t residual_nbytes =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(c) * static_cast<std::size_t>(hi) * static_cast<std::size_t>(wi) *
      sizeof(Storage);
  if (hipMalloc(&residual_packed, residual_nbytes) != hipSuccess) {
    return 5;
  }
  const int pack_status = launch_ck_conv_nchw_to_nhwc(
      residual,
      static_cast<Storage*>(residual_packed),
      batch,
      out_channels,
      out_height,
      out_width,
      stream);
  if (pack_status != 0) {
    (void)hipFree(residual_packed);
    return pack_status;
  }
  const std::array<const void*, 2> ds_ptrs{
      ck_ptr<Storage, Element>(bias),
      ck_ptr<Storage, Element>(static_cast<Storage*>(residual_packed))};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_lengths{{bias_lengths, residual_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_strides{{bias_strides, residual_strides}};
  const int status = launch_ck_transposed_conv2d_impl<Storage, DeviceOp>(
      x,
      weight,
      ds_ptrs,
      output,
      ds_lengths,
      ds_strides,
      batch,
      in_channels,
      in_height,
      in_width,
      out_channels,
      kernel_h,
      kernel_w,
      out_height,
      out_width,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      output_pad_h,
      output_pad_w,
      dilation_h,
      dilation_w,
      DinoConvBiasAddEpilogue{},
      stream);
  (void)hipFree(residual_packed);
  return status;
}

template <typename Storage>
int launch_ck_transposed_conv2d_bias_add_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    hipStream_t stream) {
  using Element = typename CkElement<Storage>::Type;
  using DeviceOp = DeviceTransposedConv2dBiasAddReluXdl<Element>;
  const auto g = ck::index_t{1};
  const auto n = static_cast<ck::index_t>(batch);
  const auto c = static_cast<ck::index_t>(out_channels);
  const auto hi = static_cast<ck::index_t>(out_height);
  const auto wi = static_cast<ck::index_t>(out_width);
  const std::array<ck::index_t, NDimSpatial + 3> bias_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> bias_strides{
      c,
      ck::index_t{0},
      ck::index_t{1},
      ck::index_t{0},
      ck::index_t{0}};
  const std::array<ck::index_t, NDimSpatial + 3> residual_lengths{g, n, c, hi, wi};
  const std::array<ck::index_t, NDimSpatial + 3> residual_strides{
      n * c * hi * wi,
      hi * wi * c,
      ck::index_t{1},
      wi * c,
      c};
  void* residual_packed = nullptr;
  const std::size_t residual_nbytes =
      static_cast<std::size_t>(n) * static_cast<std::size_t>(c) * static_cast<std::size_t>(hi) * static_cast<std::size_t>(wi) *
      sizeof(Storage);
  if (hipMalloc(&residual_packed, residual_nbytes) != hipSuccess) {
    return 5;
  }
  const int pack_status = launch_ck_conv_nchw_to_nhwc(
      residual,
      static_cast<Storage*>(residual_packed),
      batch,
      out_channels,
      out_height,
      out_width,
      stream);
  if (pack_status != 0) {
    (void)hipFree(residual_packed);
    return pack_status;
  }
  const std::array<const void*, 2> ds_ptrs{
      ck_ptr<Storage, Element>(bias),
      ck_ptr<Storage, Element>(static_cast<Storage*>(residual_packed))};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_lengths{{bias_lengths, residual_lengths}};
  const std::array<std::array<ck::index_t, NDimSpatial + 3>, 2> ds_strides{{bias_strides, residual_strides}};
  const int status = launch_ck_transposed_conv2d_impl<Storage, DeviceOp>(
      x,
      weight,
      ds_ptrs,
      output,
      ds_lengths,
      ds_strides,
      batch,
      in_channels,
      in_height,
      in_width,
      out_channels,
      kernel_h,
      kernel_w,
      out_height,
      out_width,
      stride_h,
      stride_w,
      pad_h,
      pad_w,
      output_pad_h,
      output_pad_w,
      dilation_h,
      dilation_w,
      DinoConvBiasAddReluEpilogue{},
      stream);
  (void)hipFree(residual_packed);
  return status;
}

template <typename Storage>
float profile_ck_transposed_conv2d(
    Storage const* x,
    Storage const* weight,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    int iterations,
    hipStream_t stream) {
  return profile_ck_transposed_conv2d_impl(
      [&](hipStream_t profile_stream) {
        return launch_ck_transposed_conv2d<Storage>(
            x,
            weight,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            output_pad_h,
            output_pad_w,
            dilation_h,
            dilation_w,
            profile_stream);
      },
      iterations,
      stream);
}

template <typename Storage>
float profile_ck_transposed_conv2d_bias(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    int iterations,
    hipStream_t stream) {
  return profile_ck_transposed_conv2d_impl(
      [&](hipStream_t profile_stream) {
        return launch_ck_transposed_conv2d_bias<Storage>(
            x,
            weight,
            bias,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            output_pad_h,
            output_pad_w,
            dilation_h,
            dilation_w,
            profile_stream);
      },
      iterations,
      stream);
}

template <typename Storage>
float profile_ck_transposed_conv2d_bias_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    int iterations,
    hipStream_t stream) {
  return profile_ck_transposed_conv2d_impl(
      [&](hipStream_t profile_stream) {
        return launch_ck_transposed_conv2d_bias_relu<Storage>(
            x,
            weight,
            bias,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            output_pad_h,
            output_pad_w,
            dilation_h,
            dilation_w,
            profile_stream);
      },
      iterations,
      stream);
}

template <typename Storage>
float profile_ck_transposed_conv2d_bias_add(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    int iterations,
    hipStream_t stream) {
  return profile_ck_transposed_conv2d_impl(
      [&](hipStream_t profile_stream) {
        return launch_ck_transposed_conv2d_bias_add<Storage>(
            x,
            weight,
            bias,
            residual,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            output_pad_h,
            output_pad_w,
            dilation_h,
            dilation_w,
            profile_stream);
      },
      iterations,
      stream);
}

template <typename Storage>
float profile_ck_transposed_conv2d_bias_add_relu(
    Storage const* x,
    Storage const* weight,
    Storage const* bias,
    Storage const* residual,
    Storage* output,
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
    int output_pad_h,
    int output_pad_w,
    int dilation_h,
    int dilation_w,
    int iterations,
    hipStream_t stream) {
  return profile_ck_transposed_conv2d_impl(
      [&](hipStream_t profile_stream) {
        return launch_ck_transposed_conv2d_bias_add_relu<Storage>(
            x,
            weight,
            bias,
            residual,
            output,
            batch,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_h,
            kernel_w,
            out_height,
            out_width,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            output_pad_h,
            output_pad_w,
            dilation_h,
            dilation_w,
            profile_stream);
      },
      iterations,
      stream);
}

}  // namespace

#define DINOML_CK_CONV1D_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    hipStream_t stream) { \
  return launch_ck_conv1d_bias<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv1d_bias<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, iterations, stream); \
}

#define DINOML_CK_CONV1D_BIAS_RELU_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    hipStream_t stream) { \
  return launch_ck_conv1d_bias_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv1d_bias_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, iterations, stream); \
}

#define DINOML_CK_CONV1D_BIAS_ADD_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    hipStream_t stream) { \
  return launch_ck_conv1d_bias_add<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv1d_bias_add<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, iterations, stream); \
}

#define DINOML_CK_CONV1D_BIAS_ADD_RELU_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    hipStream_t stream) { \
  return launch_ck_conv1d_bias_add_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_width, \
    int out_channels, \
    int kernel_w, \
    int out_width, \
    int stride_w, \
    int pad_w, \
    int dilation_w, \
    int weight_is_kxc, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv1d_bias_add_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_width, out_channels, kernel_w, out_width, stride_w, pad_w, dilation_w, weight_is_kxc, iterations, stream); \
}

#define DINOML_CK_CONV2D_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_conv2d_bias<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv2d_bias<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, iterations, stream); \
}

#define DINOML_CK_CONV3D_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_depth, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_d, \
    int kernel_h, \
    int kernel_w, \
    int out_depth, \
    int out_height, \
    int out_width, \
    int stride_d, \
    int stride_h, \
    int stride_w, \
    int pad_d, \
    int pad_h, \
    int pad_w, \
    int dilation_d, \
    int dilation_h, \
    int dilation_w, \
    int groups, \
    hipStream_t stream) { \
  return launch_ck_conv3d_bias<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_depth, in_height, in_width, out_channels, kernel_d, kernel_h, \
      kernel_w, out_depth, out_height, out_width, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, \
      dilation_h, dilation_w, groups, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_depth, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_d, \
    int kernel_h, \
    int kernel_w, \
    int out_depth, \
    int out_height, \
    int out_width, \
    int stride_d, \
    int stride_h, \
    int stride_w, \
    int pad_d, \
    int pad_h, \
    int pad_w, \
    int dilation_d, \
    int dilation_h, \
    int dilation_w, \
    int groups, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv3d_bias<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_depth, in_height, in_width, out_channels, kernel_d, kernel_h, \
      kernel_w, out_depth, out_height, out_width, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, \
      dilation_h, dilation_w, groups, iterations, stream); \
}

#define DINOML_CK_CONV2D_BIAS_RELU_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_conv2d_bias_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv2d_bias_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, iterations, stream); \
}

#define DINOML_CK_CONV2D_BIAS_ADD_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_conv2d_bias_add<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv2d_bias_add<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, iterations, stream); \
}

#define DINOML_CK_CONV2D_BIAS_ADD_RELU_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_conv2d_bias_add_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_conv2d_bias_add_relu<CTYPE, ConvConfig::CONFIG>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, iterations, stream); \
}

#define DINOML_CK_TRANSPOSED_CONV2D_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_transposed_conv2d<CTYPE>( \
      x, weight, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, out_height, \
      out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_transposed_conv2d<CTYPE>( \
      x, weight, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, out_height, \
      out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, iterations, \
      stream); \
}

#define DINOML_CK_TRANSPOSED_CONV2D_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_transposed_conv2d_bias<CTYPE>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_transposed_conv2d_bias<CTYPE>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      iterations, stream); \
}

#define DINOML_CK_TRANSPOSED_CONV2D_BIAS_RELU_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_transposed_conv2d_bias_relu<CTYPE>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_transposed_conv2d_bias_relu<CTYPE>( \
      x, weight, bias, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      iterations, stream); \
}

#define DINOML_CK_TRANSPOSED_CONV2D_BIAS_ADD_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_transposed_conv2d_bias_add<CTYPE>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_transposed_conv2d_bias_add<CTYPE>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      iterations, stream); \
}

#define DINOML_CK_TRANSPOSED_CONV2D_BIAS_ADD_RELU_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    hipStream_t stream) { \
  return launch_ck_transposed_conv2d_bias_add_relu<CTYPE>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* x, \
    CTYPE const* weight, \
    CTYPE const* bias, \
    CTYPE const* residual, \
    CTYPE* output, \
    int batch, \
    int in_channels, \
    int in_height, \
    int in_width, \
    int out_channels, \
    int kernel_h, \
    int kernel_w, \
    int out_height, \
    int out_width, \
    int stride_h, \
    int stride_w, \
    int pad_h, \
    int pad_w, \
    int output_pad_h, \
    int output_pad_w, \
    int dilation_h, \
    int dilation_w, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_transposed_conv2d_bias_add_relu<CTYPE>( \
      x, weight, bias, residual, output, batch, in_channels, in_height, in_width, out_channels, kernel_h, kernel_w, \
      out_height, out_width, stride_h, stride_w, pad_h, pad_w, output_pad_h, output_pad_w, dilation_h, dilation_w, \
      iterations, stream); \
}

// DINOML_CK_CONV_GENERATED_EXPORTS
