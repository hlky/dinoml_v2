#include <dinoml/abi.h>
#include <dinoml/math.h>

#include <hip/hip_runtime.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>
#include <ck/utility/type_convert.hpp>

#include <array>
#include <cstdint>
#include <type_traits>

namespace {

enum class GemmBLayout {
  kRcr,
  kRrr,
};

enum class GemmConfig {
  kBaseline,
  kWideM,
  kWideN,
  kSquare,
  kSkinnyM,
  kSkinnyN,
  kCodegenT02DefaultV1,
  kCodegenT04DefaultV1,
  kCodegenT05DefaultV1,
  kCodegenT00InterwaveV1,
  kCodegenT01InterwaveV1,
  kCodegenT02InterwaveV1,
  kCodegenT03InterwaveV1,
  kCodegenT04InterwaveV1,
  kCodegenT05InterwaveV1,
  kCodegenT06InterwaveV1,
  kCodegenT07InterwaveV1,
  kCodegenT08InterwaveV1,
  kCodegenT00DefaultV2,
  kCodegenT01DefaultV2,
  kCodegenT02DefaultV2,
  kCodegenT03DefaultV2,
  kCodegenT04DefaultV2,
  kCodegenT05DefaultV2,
  kCodegenT06DefaultV2,
  kCodegenT07DefaultV2,
  kCodegenT08DefaultV2,
};

enum class EpilogueKind {
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

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using EmptyTuple = ck::Tuple<>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmMNKPadding =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <int TileIndex>
struct GemmCodegenTile;

template <>
struct GemmCodegenTile<0> {
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
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 64, 1>;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct GemmCodegenTile<1> {
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
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 64, 1>;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct GemmCodegenTile<2> {
  static constexpr ck::index_t BlockSize = 128;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 8;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 32, 1>;
  using CDEBlockTransferClusterLengths = S<1, 16, 1, 8>;
};

template <>
struct GemmCodegenTile<3> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 64, 1>;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct GemmCodegenTile<4> {
  static constexpr ck::index_t BlockSize = 128;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 64;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 32, 1>;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <>
struct GemmCodegenTile<5> {
  static constexpr ck::index_t BlockSize = 128;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 32, 1>;
  using CDEBlockTransferClusterLengths = S<1, 16, 1, 8>;
};

template <>
struct GemmCodegenTile<6> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 64;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 4;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 64, 1>;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct GemmCodegenTile<7> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 4;
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 4;
  using ABlockTransferThreadClusterLengths = S<4, 64, 1>;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct GemmCodegenTile<8> {
  static constexpr ck::index_t BlockSize = 64;
  static constexpr ck::index_t MPerBlock = 32;
  static constexpr ck::index_t NPerBlock = 32;
  static constexpr ck::index_t KPerBlock = 32;
  static constexpr ck::index_t AK1 = 8;
  static constexpr ck::index_t BK1 = 8;
  static constexpr ck::index_t MPerXDL = 16;
  static constexpr ck::index_t NPerXDL = 16;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t NumGemmKPrefetchStage = 1;
  static constexpr ck::index_t CShuffleMXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CShuffleNXdlPerWavePerShuffle = 1;
  static constexpr ck::index_t CDEBlockTransferScalarPerVector_NPerBlock = 1;
  using ABlockTransferThreadClusterLengths = S<4, 16, 1>;
  using CDEBlockTransferClusterLengths = S<1, 16, 1, 4>;
};

template <GemmBLayout LayoutB, int TileIndex>
struct GemmBBlockTransferTraits;

template <int TileIndex>
struct GemmBBlockTransferRowMajorSrcScalarPerVector;

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<0> {
  static constexpr ck::index_t value = 2;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<1> {
  static constexpr ck::index_t value = 4;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<2> {
  static constexpr ck::index_t value = 4;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<3> {
  static constexpr ck::index_t value = 2;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<4> {
  static constexpr ck::index_t value = 2;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<5> {
  static constexpr ck::index_t value = 4;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<6> {
  static constexpr ck::index_t value = 1;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<7> {
  static constexpr ck::index_t value = 2;
};

template <>
struct GemmBBlockTransferRowMajorSrcScalarPerVector<8> {
  static constexpr ck::index_t value = 1;
};

template <int TileIndex>
struct GemmBBlockTransferTraits<GemmBLayout::kRcr, TileIndex> {
  using ThreadClusterLengths = typename GemmCodegenTile<TileIndex>::ABlockTransferThreadClusterLengths;
  using ThreadClusterArrangeOrder = S<1, 0, 2>;
  using SrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t SrcVectorDim = 2;
  static constexpr ck::index_t SrcScalarPerVector = TileIndex == 8 ? 1 : 8;
  static constexpr ck::index_t DstScalarPerVector_BK1 = 8;
  static constexpr ck::index_t LdsExtraN = 1;
};

template <int TileIndex>
struct GemmBBlockTransferTraits<GemmBLayout::kRrr, TileIndex> {
  using ThreadClusterLengths = typename GemmCodegenTile<TileIndex>::ABlockTransferThreadClusterLengths;
  using ThreadClusterArrangeOrder = S<0, 2, 1>;
  using SrcAccessOrder = S<0, 2, 1>;
  static constexpr ck::index_t SrcVectorDim = 1;
  static constexpr ck::index_t SrcScalarPerVector =
      GemmBBlockTransferRowMajorSrcScalarPerVector<TileIndex>::value;
  static constexpr ck::index_t DstScalarPerVector_BK1 = 8;
  static constexpr ck::index_t LdsExtraN = 1;
};

template <int TileIndex>
struct GemmABlockTransferTraits {
  using ThreadClusterLengths = typename GemmCodegenTile<TileIndex>::ABlockTransferThreadClusterLengths;
  using ThreadClusterArrangeOrder = S<1, 0, 2>;
  using SrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t SrcVectorDim = 2;
  static constexpr ck::index_t SrcScalarPerVector = TileIndex == 8 ? 1 : 8;
  static constexpr ck::index_t DstScalarPerVector_AK1 = 8;
  static constexpr ck::index_t LdsExtraM = 1;
};

template <GemmConfig Config>
struct GemmConfigTraits;

template <int TileIndex, ck::LoopScheduler SchedulerValue, ck::PipelineVersion PipelineValue>
struct GemmConfigTraitsBase : GemmCodegenTile<TileIndex> {
  static constexpr int Tile = TileIndex;
  static constexpr ck::LoopScheduler LoopScheduler = SchedulerValue;
  static constexpr ck::PipelineVersion PipelineVersion = PipelineValue;
};

#define DINOML_CK_GEMM_CONFIG_TRAITS(CONFIG, TILE, SCHEDULER, PIPELINE) \
template <> \
struct GemmConfigTraits<GemmConfig::CONFIG> \
    : GemmConfigTraitsBase<TILE, ck::LoopScheduler::SCHEDULER, ck::PipelineVersion::PIPELINE> {};

DINOML_CK_GEMM_CONFIG_TRAITS(kWideM, 0, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kWideN, 1, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT02DefaultV1, 2, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kSquare, 3, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT04DefaultV1, 4, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT05DefaultV1, 5, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kSkinnyN, 6, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kSkinnyM, 7, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kBaseline, 8, Default, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT00InterwaveV1, 0, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT01InterwaveV1, 1, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT02InterwaveV1, 2, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT03InterwaveV1, 3, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT04InterwaveV1, 4, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT05InterwaveV1, 5, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT06InterwaveV1, 6, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT07InterwaveV1, 7, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT08InterwaveV1, 8, Interwave, v1)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT00DefaultV2, 0, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT01DefaultV2, 1, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT02DefaultV2, 2, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT03DefaultV2, 3, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT04DefaultV2, 4, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT05DefaultV2, 5, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT06DefaultV2, 6, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT07DefaultV2, 7, Default, v2)
DINOML_CK_GEMM_CONFIG_TRAITS(kCodegenT08DefaultV2, 8, Default, v2)

#undef DINOML_CK_GEMM_CONFIG_TRAITS

template <GemmConfig Config, typename Storage>
struct GemmKernelConfig : GemmConfigTraits<Config> {
  using StorageType = Storage;
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

template <GemmBLayout Layout>
struct CkBLayout;

template <>
struct CkBLayout<GemmBLayout::kRcr> {
  // DinoML rcr stores B as [N, K] row-major. CK's MultipleD B descriptor is
  // logical [N, K], where ColumnMajor gives address n * K + k.
  using Type = Col;
  static ck::index_t stride_b(ck::index_t, ck::index_t, ck::index_t k) { return k; }
};

template <>
struct CkBLayout<GemmBLayout::kRrr> {
  // DinoML rrr stores B as [K, N] row-major. CK consumes logical B[N, K], and
  // RowMajor with stride N gives address k * N + n.
  using Type = Row;
  static ck::index_t stride_b(ck::index_t, ck::index_t n, ck::index_t) { return n; }
};

template <int Count, typename Element>
struct DTensorTraits;

template <typename Element>
struct DTensorTraits<0, Element> {
  using Layouts = EmptyTuple;
  using Types = EmptyTuple;
};

template <typename Element>
struct DTensorTraits<1, Element> {
  using Layouts = ck::Tuple<Row>;
  using Types = ck::Tuple<Element>;
};

template <typename Element>
struct DTensorTraits<2, Element> {
  using Layouts = ck::Tuple<Row, Row>;
  using Types = ck::Tuple<Element, Element>;
};

template <typename Element>
struct DTensorTraits<3, Element> {
  using Layouts = ck::Tuple<Row, Row, Row>;
  using Types = ck::Tuple<Element, Element, Element>;
};

template <EpilogueKind Epilogue>
constexpr int epilogue_residual_count() {
  if constexpr (
      Epilogue == EpilogueKind::kBiasAddAdd ||
      Epilogue == EpilogueKind::kBiasMulAdd ||
      Epilogue == EpilogueKind::kBiasAddAddRelu) {
    return 2;
  } else if constexpr (
      Epilogue == EpilogueKind::kBiasAdd ||
      Epilogue == EpilogueKind::kBiasMul ||
      Epilogue == EpilogueKind::kBiasAddRelu ||
      Epilogue == EpilogueKind::kBiasSigmoidMul ||
      Epilogue == EpilogueKind::kBiasSigmoidMulTanh ||
      Epilogue == EpilogueKind::kBiasMulTanh) {
    return 1;
  } else {
    return 0;
  }
}

template <EpilogueKind Epilogue>
constexpr bool epilogue_has_bias() {
  return Epilogue != EpilogueKind::kLinearCombination;
}

template <EpilogueKind Epilogue>
constexpr int epilogue_d_count() {
  return epilogue_has_bias<Epilogue>() ? 1 + epilogue_residual_count<Epilogue>() : 0;
}

template <typename T>
__host__ __device__ float to_float(const T& value) {
  return ck::type_convert<float>(value);
}

__host__ __device__ float sigmoid(float value) {
  return 1.0f / (1.0f + ck::math::exp(-value));
}

__host__ __device__ float gelu(float value) {
  return 0.5f * value *
      (1.0f + ck::math::tanh(0.7978845608028654f * (value + 0.044715f * value * value * value)));
}

__host__ __device__ float fast_gelu(float value) {
  return 0.5f * value *
      (1.0f + ck::math::tanh(0.7978845608f * value * (1.0f + 0.044715f * value * value)));
}

__host__ __device__ float quick_gelu(float value) {
  return value * sigmoid(1.702f * value);
}

__host__ __device__ float hard_swish(float value) {
  return value * ck::math::min(ck::math::max(value + 3.0f, 0.0f), 6.0f) / 6.0f;
}

__host__ __device__ float elup1(float value) {
  return value >= 0.0f ? value + 1.0f : ck::math::exp(value);
}

template <EpilogueKind Epilogue>
struct DinoEpilogue {
  template <typename E, typename C>
  __host__ __device__ void operator()(E& e, const C& c) const {
    static_assert(epilogue_d_count<Epilogue>() == 0, "Unexpected CK GEMM D tensor count");
    e = ck::type_convert<E>(apply(to_float(c), 0.0f, 0.0f, 0.0f));
  }

  template <typename E, typename C, typename D0>
  __host__ __device__ void operator()(E& e, const C& c, const D0& d0) const {
    static_assert(epilogue_d_count<Epilogue>() == 1, "Unexpected CK GEMM D tensor count");
    e = ck::type_convert<E>(apply(to_float(c), to_float(d0), 0.0f, 0.0f));
  }

  template <typename E, typename C, typename D0, typename D1>
  __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1) const {
    static_assert(epilogue_d_count<Epilogue>() == 2, "Unexpected CK GEMM D tensor count");
    e = ck::type_convert<E>(apply(to_float(c), to_float(d0), to_float(d1), 0.0f));
  }

  template <typename E, typename C, typename D0, typename D1, typename D2>
  __host__ __device__ void operator()(E& e, const C& c, const D0& d0, const D1& d1, const D2& d2) const {
    static_assert(epilogue_d_count<Epilogue>() == 3, "Unexpected CK GEMM D tensor count");
    e = ck::type_convert<E>(apply(to_float(c), to_float(d0), to_float(d1), to_float(d2)));
  }

  __host__ __device__ static float apply(float value, float d0, float d1, float d2) {
    if constexpr (epilogue_has_bias<Epilogue>()) {
      value += d0;
    }

    if constexpr (
        Epilogue == EpilogueKind::kBiasSigmoidMul ||
        Epilogue == EpilogueKind::kBiasSigmoidMulTanh) {
      value = sigmoid(value);
    }

    if constexpr (
        Epilogue == EpilogueKind::kBiasAdd ||
        Epilogue == EpilogueKind::kBiasAddRelu) {
      value += d1;
    } else if constexpr (
        Epilogue == EpilogueKind::kBiasAddAdd ||
        Epilogue == EpilogueKind::kBiasAddAddRelu) {
      value += d1 + d2;
    } else if constexpr (
        Epilogue == EpilogueKind::kBiasMul ||
        Epilogue == EpilogueKind::kBiasMulTanh ||
        Epilogue == EpilogueKind::kBiasSigmoidMul ||
        Epilogue == EpilogueKind::kBiasSigmoidMulTanh) {
      value *= d1;
    } else if constexpr (Epilogue == EpilogueKind::kBiasMulAdd) {
      value = value * d1 + d2;
    }

    if constexpr (
        Epilogue == EpilogueKind::kBiasRelu ||
        Epilogue == EpilogueKind::kBiasAddRelu ||
        Epilogue == EpilogueKind::kBiasAddAddRelu) {
      value = ck::math::max(value, 0.0f);
    } else if constexpr (Epilogue == EpilogueKind::kBiasGelu) {
      value = gelu(value);
    } else if constexpr (Epilogue == EpilogueKind::kBiasFastGelu) {
      value = fast_gelu(value);
    } else if constexpr (Epilogue == EpilogueKind::kBiasQuickGelu) {
      value = quick_gelu(value);
    } else if constexpr (Epilogue == EpilogueKind::kBiasSigmoid) {
      value = sigmoid(value);
    } else if constexpr (
        Epilogue == EpilogueKind::kBiasTanh ||
        Epilogue == EpilogueKind::kBiasSigmoidMulTanh ||
        Epilogue == EpilogueKind::kBiasMulTanh) {
      value = ck::math::tanh(value);
    } else if constexpr (Epilogue == EpilogueKind::kBiasSwish) {
      value *= sigmoid(value);
    } else if constexpr (Epilogue == EpilogueKind::kBiasHardSwish) {
      value = hard_swish(value);
    } else if constexpr (Epilogue == EpilogueKind::kBiasElup1) {
      value = elup1(value);
    }

    return value;
  }
};

template <
    GemmConfig Config,
    typename Storage,
    typename Element,
    GemmBLayout LayoutBKind,
    typename DsLayout,
    typename DsDataType,
    typename CDEOp>
using DeviceGemmXdl = ck::tensor_operation::device::DeviceGemmMultipleD_Xdl_CShuffle<
    Row,
    typename CkBLayout<LayoutBKind>::Type,
    DsLayout,
    Row,
    Element,
    Element,
    float,
    float,
    DsDataType,
    Element,
    PassThrough,
    PassThrough,
    CDEOp,
    GemmMNKPadding,
    GemmKernelConfig<Config, Storage>::NumGemmKPrefetchStage,
    GemmKernelConfig<Config, Storage>::BlockSize,
    GemmKernelConfig<Config, Storage>::MPerBlock,
    GemmKernelConfig<Config, Storage>::NPerBlock,
    GemmKernelConfig<Config, Storage>::KPerBlock,
    GemmKernelConfig<Config, Storage>::AK1,
    GemmKernelConfig<Config, Storage>::BK1,
    GemmKernelConfig<Config, Storage>::MPerXDL,
    GemmKernelConfig<Config, Storage>::NPerXDL,
    GemmKernelConfig<Config, Storage>::MXdlPerWave,
    GemmKernelConfig<Config, Storage>::NXdlPerWave,
    typename GemmABlockTransferTraits<GemmKernelConfig<Config, Storage>::Tile>::ThreadClusterLengths,
    typename GemmABlockTransferTraits<GemmKernelConfig<Config, Storage>::Tile>::ThreadClusterArrangeOrder,
    typename GemmABlockTransferTraits<GemmKernelConfig<Config, Storage>::Tile>::SrcAccessOrder,
    GemmABlockTransferTraits<GemmKernelConfig<Config, Storage>::Tile>::SrcVectorDim,
    GemmABlockTransferTraits<GemmKernelConfig<Config, Storage>::Tile>::SrcScalarPerVector,
    GemmABlockTransferTraits<GemmKernelConfig<Config, Storage>::Tile>::DstScalarPerVector_AK1,
    GemmABlockTransferTraits<GemmKernelConfig<Config, Storage>::Tile>::LdsExtraM,
    typename GemmBBlockTransferTraits<LayoutBKind, GemmKernelConfig<Config, Storage>::Tile>::ThreadClusterLengths,
    typename GemmBBlockTransferTraits<LayoutBKind, GemmKernelConfig<Config, Storage>::Tile>::ThreadClusterArrangeOrder,
    typename GemmBBlockTransferTraits<LayoutBKind, GemmKernelConfig<Config, Storage>::Tile>::SrcAccessOrder,
    GemmBBlockTransferTraits<LayoutBKind, GemmKernelConfig<Config, Storage>::Tile>::SrcVectorDim,
    GemmBBlockTransferTraits<LayoutBKind, GemmKernelConfig<Config, Storage>::Tile>::SrcScalarPerVector,
    GemmBBlockTransferTraits<LayoutBKind, GemmKernelConfig<Config, Storage>::Tile>::DstScalarPerVector_BK1,
    GemmBBlockTransferTraits<LayoutBKind, GemmKernelConfig<Config, Storage>::Tile>::LdsExtraN,
    GemmKernelConfig<Config, Storage>::CShuffleMXdlPerWavePerShuffle,
    GemmKernelConfig<Config, Storage>::CShuffleNXdlPerWavePerShuffle,
    typename GemmKernelConfig<Config, Storage>::CDEBlockTransferClusterLengths,
    GemmKernelConfig<Config, Storage>::CDEBlockTransferScalarPerVector_NPerBlock,
    GemmKernelConfig<Config, Storage>::LoopScheduler,
    GemmKernelConfig<Config, Storage>::PipelineVersion>;

template <
    GemmConfig Config,
    typename Storage,
    typename Element,
    GemmBLayout LayoutBKind,
    typename DsLayout,
    typename DsDataType,
    typename CDEOp>
struct DeviceOpForElement {
  using Type = DeviceGemmXdl<Config, Storage, Element, LayoutBKind, DsLayout, DsDataType, CDEOp>;
};

template <typename Storage, GemmBLayout LayoutB, EpilogueKind Epilogue, GemmConfig Config>
struct DeviceOpTraits {
  using Element = typename CkElement<Storage>::Type;
  static constexpr int NumDTensor = epilogue_d_count<Epilogue>();
  using DsLayout = typename DTensorTraits<NumDTensor, Element>::Layouts;
  using DsDataType = typename DTensorTraits<NumDTensor, Element>::Types;
  using CDEOp = DinoEpilogue<Epilogue>;
  using DeviceOp = typename DeviceOpForElement<Config, Storage, Element, LayoutB, DsLayout, DsDataType, CDEOp>::Type;
};

template <typename Storage, typename Element, int NumD>
std::array<const void*, NumD> make_d_pointers(
    Storage const* bias,
    Storage const* d0,
    Storage const* d1) {
  if constexpr (NumD == 0) {
    return {};
  } else if constexpr (NumD == 1) {
    return {ck_ptr<Storage, Element>(bias)};
  } else if constexpr (NumD == 2) {
    return {ck_ptr<Storage, Element>(bias), ck_ptr<Storage, Element>(d0)};
  } else {
    return {ck_ptr<Storage, Element>(bias), ck_ptr<Storage, Element>(d0), ck_ptr<Storage, Element>(d1)};
  }
}

template <int NumD>
std::array<ck::index_t, NumD> make_d_strides(ck::index_t n) {
  if constexpr (NumD == 0) {
    return {};
  } else if constexpr (NumD == 1) {
    return {0};
  } else if constexpr (NumD == 2) {
    return {0, n};
  } else {
    return {0, n, n};
  }
}

template <typename Storage, GemmBLayout LayoutB, EpilogueKind Epilogue, GemmConfig Config>
int launch_ck_gemm(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage const* d0,
    Storage const* d1,
    Storage* c,
    int m,
    int n,
    int k,
    hipStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return 1;
  }
  if constexpr (epilogue_has_bias<Epilogue>()) {
    if (bias == nullptr) {
      return 1;
    }
  }
  if constexpr (epilogue_residual_count<Epilogue>() >= 1) {
    if (d0 == nullptr) {
      return 1;
    }
  }
  if constexpr (epilogue_residual_count<Epilogue>() >= 2) {
    if (d1 == nullptr) {
      return 1;
    }
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }

  using Traits = DeviceOpTraits<Storage, LayoutB, Epilogue, Config>;
  using Element = typename Traits::Element;
  using DeviceOp = typename Traits::DeviceOp;
  constexpr int NumD = Traits::NumDTensor;
  const auto m_ck = static_cast<ck::index_t>(m);
  const auto n_ck = static_cast<ck::index_t>(n);
  const auto k_ck = static_cast<ck::index_t>(k);
  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(a),
      ck_ptr<Storage, Element>(b),
      make_d_pointers<Storage, Element, NumD>(bias, d0, d1),
      ck_ptr<Storage, Element>(c),
      m_ck,
      n_ck,
      k_ck,
      k_ck,
      CkBLayout<LayoutB>::stride_b(m_ck, n_ck, k_ck),
      make_d_strides<NumD>(n_ck),
      n_ck,
      PassThrough{},
      PassThrough{},
      typename Traits::CDEOp{});
  if (!DeviceOp::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <typename Storage, GemmBLayout LayoutB, EpilogueKind Epilogue, GemmConfig Config>
float profile_ck_gemm(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage const* d0,
    Storage const* d1,
    Storage* c,
    int m,
    int n,
    int k,
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
    if (launch_ck_gemm<Storage, LayoutB, Epilogue, Config>(a, b, bias, d0, d1, c, m, n, k, stream)) {
      (void)hipEventDestroy(start);
      (void)hipEventDestroy(end);
      return -1.0f;
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

#define DINOML_CK_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, LAYOUT_B, EPILOGUE, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    hipStream_t stream) { \
  return launch_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, nullptr, nullptr, nullptr, c, m, n, k, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, nullptr, nullptr, nullptr, c, m, n, k, iterations, stream); \
}

#define DINOML_CK_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, LAYOUT_B, EPILOGUE, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    hipStream_t stream) { \
  return launch_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, bias, nullptr, nullptr, c, m, n, k, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, bias, nullptr, nullptr, c, m, n, k, iterations, stream); \
}

#define DINOML_CK_GEMM_BIAS_RESIDUAL_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, LAYOUT_B, EPILOGUE, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    hipStream_t stream) { \
  return launch_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, bias, d0, nullptr, c, m, n, k, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, bias, d0, nullptr, c, m, n, k, iterations, stream); \
}

#define DINOML_CK_GEMM_BIAS_RESIDUAL2_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, LAYOUT_B, EPILOGUE, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE const* d1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    hipStream_t stream) { \
  return launch_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, bias, d0, d1, c, m, n, k, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE const* d1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_gemm<CTYPE, GemmBLayout::LAYOUT_B, EpilogueKind::EPILOGUE, GemmConfig::CONFIG>(a, b, bias, d0, d1, c, m, n, k, iterations, stream); \
}

#define DINOML_CK_GENERATED_EXPORTS 1
