#include <dinoml/abi.h>
#include <dinoml/math.h>

#include <hip/hip_runtime.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/convolution_forward_specialization.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
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

template <typename T>
__host__ __device__ float to_float(const T& value) {
  return ck::type_convert<float>(value);
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

template <ConvConfig Config, typename Storage, typename Element>
using DeviceConv2dBiasXdl =
    DeviceConv2dXdl<Config, Storage, Element, ck::Tuple<BiasLayout>, ck::Tuple<Element>, DinoConvBiasEpilogue>;

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

}  // namespace

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

// DINOML_CK_CONV_GENERATED_EXPORTS
