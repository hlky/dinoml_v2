#include <dinoml/abi.h>
#include <dinoml/math.h>

#include <hip/hip_runtime.h>

#include <ck/ck.hpp>
#include <ck/tensor_operation/gpu/device/gemm_specialization.hpp>
#include <ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp>
#include <ck/tensor_operation/gpu/device/tensor_layout.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <ck/utility/sequence.hpp>
#include <ck/utility/type_convert.hpp>

#include <array>
#include <cstdint>

namespace {

enum class BmmLayout {
  kCcc,
  kCcr,
  kCrc,
  kCrr,
  kRcc,
  kRcr,
  kRrc,
  kRrr,
};

enum class BmmConfig {
  kBaseline,
  kWideM,
  kWideN,
  kSquare,
  kSkinnyM,
  kSkinnyN,
  kSmall,
};

enum class EpilogueKind {
  kLinearCombination,
  kAdd,
};

template <ck::index_t... Is>
using S = ck::Sequence<Is...>;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
using EmptyTuple = ck::Tuple<>;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

static constexpr auto GemmMNKPadding =
    ck::tensor_operation::device::GemmSpecialization::MNKPadding;

template <BmmConfig Config, typename Storage>
struct BmmKernelConfig;

template <typename Storage, ck::index_t DefaultValue, ck::index_t FloatValue>
struct StorageTunedValue {
  static constexpr ck::index_t value = DefaultValue;
};

template <ck::index_t DefaultValue, ck::index_t FloatValue>
struct StorageTunedValue<float, DefaultValue, FloatValue> {
  static constexpr ck::index_t value = FloatValue;
};

template <typename Storage>
struct BmmKernelConfig<BmmConfig::kBaseline, Storage> {
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
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <>
struct BmmKernelConfig<BmmConfig::kBaseline, float> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = 16;
  static constexpr ck::index_t AK1 = 4;
  static constexpr ck::index_t BK1 = 4;
  static constexpr ck::index_t MPerXDL = 32;
  static constexpr ck::index_t NPerXDL = 32;
  static constexpr ck::index_t MXdlPerWave = 2;
  static constexpr ck::index_t NXdlPerWave = 2;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 1;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 1;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 1;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct BmmKernelConfig<BmmConfig::kWideM, Storage> {
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
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <>
struct BmmKernelConfig<BmmConfig::kWideM, float> {
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
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = 4;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = 4;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = 4;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = 4;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <typename Storage>
struct BmmKernelConfig<BmmConfig::kWideN, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 256;
  static constexpr ck::index_t KPerBlock = StorageTunedValue<Storage, 32, 32>::value;
  static constexpr ck::index_t AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t MPerXDL = StorageTunedValue<Storage, 16, 32>::value;
  static constexpr ck::index_t NPerXDL = StorageTunedValue<Storage, 16, 32>::value;
  static constexpr ck::index_t MXdlPerWave = StorageTunedValue<Storage, 4, 2>::value;
  static constexpr ck::index_t NXdlPerWave = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t ABlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferSrcScalarPerVector = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t ABlockTransferDstScalarPerVector_AK1 = StorageTunedValue<Storage, 8, 4>::value;
  static constexpr ck::index_t BBlockTransferDstScalarPerVector_BK1 = StorageTunedValue<Storage, 8, 4>::value;
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 8>;
};

template <typename Storage>
struct BmmKernelConfig<BmmConfig::kSquare, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = StorageTunedValue<Storage, 64, 32>::value;
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
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct BmmKernelConfig<BmmConfig::kSkinnyM, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 64;
  static constexpr ck::index_t NPerBlock = 128;
  static constexpr ck::index_t KPerBlock = StorageTunedValue<Storage, 64, 32>::value;
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
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct BmmKernelConfig<BmmConfig::kSkinnyN, Storage> {
  static constexpr ck::index_t BlockSize = 256;
  static constexpr ck::index_t MPerBlock = 128;
  static constexpr ck::index_t NPerBlock = 64;
  static constexpr ck::index_t KPerBlock = StorageTunedValue<Storage, 64, 32>::value;
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
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <typename Storage>
struct BmmKernelConfig<BmmConfig::kSmall, Storage> {
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
  using CDEBlockTransferClusterLengths = S<1, 32, 1, 4>;
};

template <BmmConfig Config, typename Storage, int Count>
struct BmmCDETransferTraits;

template <BmmConfig Config, typename Storage>
struct BmmCDETransferTraits<Config, Storage, 0> {
  using Type = S<1>;
};

template <BmmConfig Config, typename Storage>
struct BmmCDETransferTraits<Config, Storage, 1> {
  using Type = S<1, 1>;
};

template <typename Storage, int Count>
struct BmmVectorizedCDETransferTraits;

template <typename Storage>
struct BmmVectorizedCDETransferTraits<Storage, 0> {
  using Type = S<4>;
};

template <typename Storage>
struct BmmVectorizedCDETransferTraits<Storage, 1> {
  using Type = S<4, 4>;
};

template <>
struct BmmVectorizedCDETransferTraits<float, 0> {
  using Type = S<2>;
};

template <>
struct BmmVectorizedCDETransferTraits<float, 1> {
  using Type = S<2, 2>;
};

#define DINOML_BMM_VECTORIZED_CDE_TRAITS(CONFIG) \
template <typename Storage, int Count> \
struct BmmCDETransferTraits<BmmConfig::CONFIG, Storage, Count> \
    : BmmVectorizedCDETransferTraits<Storage, Count> {}

DINOML_BMM_VECTORIZED_CDE_TRAITS(kWideM);
DINOML_BMM_VECTORIZED_CDE_TRAITS(kWideN);
DINOML_BMM_VECTORIZED_CDE_TRAITS(kSquare);
DINOML_BMM_VECTORIZED_CDE_TRAITS(kSkinnyM);
DINOML_BMM_VECTORIZED_CDE_TRAITS(kSkinnyN);
DINOML_BMM_VECTORIZED_CDE_TRAITS(kSmall);

#undef DINOML_BMM_VECTORIZED_CDE_TRAITS

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

template <BmmLayout Layout>
struct BmmLayoutTraits;

#define DINOML_BMM_LAYOUT_TRAITS(NAME, A_LAYOUT, B_LAYOUT, C_LAYOUT) \
template <> \
struct BmmLayoutTraits<BmmLayout::NAME> { \
  using ALayout = A_LAYOUT; \
  using BLayout = B_LAYOUT; \
  using CLayout = C_LAYOUT; \
}

DINOML_BMM_LAYOUT_TRAITS(kCcc, Col, Col, Col);
DINOML_BMM_LAYOUT_TRAITS(kCcr, Col, Col, Row);
DINOML_BMM_LAYOUT_TRAITS(kCrc, Col, Row, Col);
DINOML_BMM_LAYOUT_TRAITS(kCrr, Col, Row, Row);
DINOML_BMM_LAYOUT_TRAITS(kRcc, Row, Col, Col);
DINOML_BMM_LAYOUT_TRAITS(kRcr, Row, Col, Row);
DINOML_BMM_LAYOUT_TRAITS(kRrc, Row, Row, Col);
DINOML_BMM_LAYOUT_TRAITS(kRrr, Row, Row, Row);

#undef DINOML_BMM_LAYOUT_TRAITS

template <typename Layout>
struct ATransferTraits;

template <>
struct ATransferTraits<Row> {
  using ArrangeOrder = S<1, 0, 2>;
  using SrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t SrcVectorDim = 2;
};

template <>
struct ATransferTraits<Col> {
  using ArrangeOrder = S<0, 2, 1>;
  using SrcAccessOrder = S<0, 2, 1>;
  static constexpr ck::index_t SrcVectorDim = 1;
};

template <typename Layout>
struct BTransferTraits;

template <>
struct BTransferTraits<Row> {
  using ArrangeOrder = S<0, 2, 1>;
  using SrcAccessOrder = S<0, 2, 1>;
  static constexpr ck::index_t SrcVectorDim = 1;
};

template <>
struct BTransferTraits<Col> {
  using ArrangeOrder = S<1, 0, 2>;
  using SrcAccessOrder = S<1, 0, 2>;
  static constexpr ck::index_t SrcVectorDim = 2;
};

template <typename Element, int Count, typename Layout>
struct DTensorTraits;

template <typename Element, typename Layout>
struct DTensorTraits<Element, 0, Layout> {
  using Layouts = EmptyTuple;
  using Types = EmptyTuple;
};

template <typename Element, typename Layout>
struct DTensorTraits<Element, 1, Layout> {
  using Layouts = ck::Tuple<Layout>;
  using Types = ck::Tuple<Element>;
};

template <EpilogueKind Epilogue>
constexpr int epilogue_d_count() {
  return Epilogue == EpilogueKind::kAdd ? 1 : 0;
}

template <typename T>
__host__ __device__ float to_float(const T& value) {
  return ck::type_convert<float>(value);
}

template <EpilogueKind Epilogue>
struct DinoBmmEpilogue {
  template <typename E, typename C>
  __host__ __device__ void operator()(E& e, const C& c) const {
    static_assert(epilogue_d_count<Epilogue>() == 0, "Unexpected CK BMM D tensor count");
    e = ck::type_convert<E>(to_float(c));
  }

  template <typename E, typename C, typename D0>
  __host__ __device__ void operator()(E& e, const C& c, const D0& d0) const {
    static_assert(epilogue_d_count<Epilogue>() == 1, "Unexpected CK BMM D tensor count");
    float value = to_float(c);
    if constexpr (Epilogue == EpilogueKind::kAdd) {
      value += to_float(d0);
    }
    e = ck::type_convert<E>(value);
  }
};

template <
    BmmConfig Config,
    typename Storage,
    typename Element,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename DsLayout,
    typename DsDataType,
    typename CDEOp,
    typename CDETransferScalars>
using DeviceBmmXdl = ck::tensor_operation::device::DeviceBatchedGemmMultiD_Xdl_CShuffle_V3<
    ALayout,
    BLayout,
    DsLayout,
    CLayout,
    Element,
    Element,
    DsDataType,
    Element,
    float,
    float,
    PassThrough,
    PassThrough,
    CDEOp,
    GemmMNKPadding,
    BmmKernelConfig<Config, Storage>::BlockSize,
    BmmKernelConfig<Config, Storage>::MPerBlock,
    BmmKernelConfig<Config, Storage>::NPerBlock,
    BmmKernelConfig<Config, Storage>::KPerBlock,
    BmmKernelConfig<Config, Storage>::AK1,
    BmmKernelConfig<Config, Storage>::BK1,
    BmmKernelConfig<Config, Storage>::MPerXDL,
    BmmKernelConfig<Config, Storage>::NPerXDL,
    BmmKernelConfig<Config, Storage>::MXdlPerWave,
    BmmKernelConfig<Config, Storage>::NXdlPerWave,
    S<4, 64, 1>,
    typename ATransferTraits<ALayout>::ArrangeOrder,
    typename ATransferTraits<ALayout>::SrcAccessOrder,
    ATransferTraits<ALayout>::SrcVectorDim,
    BmmKernelConfig<Config, Storage>::ABlockTransferSrcScalarPerVector,
    BmmKernelConfig<Config, Storage>::ABlockTransferDstScalarPerVector_AK1,
    1,
    S<4, 64, 1>,
    typename BTransferTraits<BLayout>::ArrangeOrder,
    typename BTransferTraits<BLayout>::SrcAccessOrder,
    BTransferTraits<BLayout>::SrcVectorDim,
    BmmKernelConfig<Config, Storage>::BBlockTransferSrcScalarPerVector,
    BmmKernelConfig<Config, Storage>::BBlockTransferDstScalarPerVector_BK1,
    1,
    1,
    1,
    typename BmmKernelConfig<Config, Storage>::CDEBlockTransferClusterLengths,
    CDETransferScalars,
    ck::BlockGemmPipelineScheduler::Intrawave,
    ck::BlockGemmPipelineVersion::v3>;

template <
    BmmConfig Config,
    typename Storage,
    typename Element,
    typename ALayout,
    typename BLayout,
    typename CLayout,
    typename DsLayout,
    typename DsDataType,
    typename CDEOp,
    typename CDETransferScalars>
struct DeviceOpForElement {
  using Type = DeviceBmmXdl<Config, Storage, Element, ALayout, BLayout, CLayout, DsLayout, DsDataType, CDEOp, CDETransferScalars>;
};

template <typename Storage, BmmLayout Layout, EpilogueKind Epilogue, BmmConfig Config>
struct DeviceOpTraits {
  using Element = typename CkElement<Storage>::Type;
  using Layouts = BmmLayoutTraits<Layout>;
  using ALayout = typename Layouts::ALayout;
  using BLayout = typename Layouts::BLayout;
  using CLayout = typename Layouts::CLayout;
  static constexpr int NumDTensor = epilogue_d_count<Epilogue>();
  using DTraits = DTensorTraits<Element, NumDTensor, CLayout>;
  using DsLayout = typename DTraits::Layouts;
  using DsDataType = typename DTraits::Types;
  using CDETransferScalars = typename BmmCDETransferTraits<Config, Storage, NumDTensor>::Type;
  using CDEOp = DinoBmmEpilogue<Epilogue>;
  using DeviceOp = typename DeviceOpForElement<
      Config,
      Storage,
      Element,
      ALayout,
      BLayout,
      CLayout,
      DsLayout,
      DsDataType,
      CDEOp,
      CDETransferScalars>::Type;
};

template <typename Storage, typename Element, int NumD>
std::array<const void*, NumD> make_d_pointers(Storage const* d0) {
  if constexpr (NumD == 0) {
    return {};
  } else {
    return {ck_ptr<Storage, Element>(d0)};
  }
}

template <int NumD>
std::array<ck::index_t, NumD> make_d_strides(ck::index_t ldd0) {
  if constexpr (NumD == 0) {
    return {};
  } else {
    return {ldd0};
  }
}

template <int NumD>
std::array<ck::index_t, NumD> make_d_batch_strides(ck::index_t batch_stride_d0) {
  if constexpr (NumD == 0) {
    return {};
  } else {
    return {batch_stride_d0};
  }
}

template <typename Storage, BmmLayout Layout, EpilogueKind Epilogue, BmmConfig Config>
int launch_ck_bmm(
    Storage const* a,
    Storage const* b,
    Storage const* d0,
    Storage* c,
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
  if (a == nullptr || b == nullptr || c == nullptr) {
    return 1;
  }
  if constexpr (Epilogue == EpilogueKind::kAdd) {
    if (d0 == nullptr) {
      return 1;
    }
  }
  if (batch_count <= 0 || m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (batch_stride_a < 0 || batch_stride_b < 0 || batch_stride_c <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return 2;
  }
  if constexpr (Epilogue == EpilogueKind::kAdd) {
    if (batch_stride_d0 < 0 || ldd0 < 0) {
      return 2;
    }
  }

  using Traits = DeviceOpTraits<Storage, Layout, Epilogue, Config>;
  using Element = typename Traits::Element;
  using DeviceOp = typename Traits::DeviceOp;
  constexpr int NumD = Traits::NumDTensor;
  auto argument = DeviceOp::MakeArgument(
      ck_ptr<Storage, Element>(a),
      ck_ptr<Storage, Element>(b),
      make_d_pointers<Storage, Element, NumD>(d0),
      ck_ptr<Storage, Element>(c),
      static_cast<ck::index_t>(m),
      static_cast<ck::index_t>(n),
      static_cast<ck::index_t>(k),
      static_cast<ck::index_t>(batch_count),
      static_cast<ck::index_t>(lda),
      static_cast<ck::index_t>(ldb),
      make_d_strides<NumD>(static_cast<ck::index_t>(ldd0)),
      static_cast<ck::index_t>(ldc),
      static_cast<ck::index_t>(batch_stride_a),
      static_cast<ck::index_t>(batch_stride_b),
      make_d_batch_strides<NumD>(static_cast<ck::index_t>(batch_stride_d0)),
      static_cast<ck::index_t>(batch_stride_c),
      PassThrough{},
      PassThrough{},
      typename Traits::CDEOp{},
      1);
  if (!DeviceOp::IsSupportedArgument(argument)) {
    return 4;
  }
  auto invoker = DeviceOp::MakeInvoker();
  (void)invoker.Run(argument, StreamConfig{stream, false});
  const hipError_t err = hipGetLastError();
  return err == hipSuccess ? 0 : 3;
}

template <typename Storage, BmmLayout Layout, EpilogueKind Epilogue, BmmConfig Config>
float profile_ck_bmm(
    Storage const* a,
    Storage const* b,
    Storage const* d0,
    Storage* c,
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
    if (launch_ck_bmm<Storage, Layout, Epilogue, Config>(
            a,
            b,
            d0,
            c,
            batch_count,
            m,
            n,
            k,
            batch_stride_a,
            batch_stride_b,
            batch_stride_d0,
            batch_stride_c,
            lda,
            ldb,
            ldd0,
            ldc,
            stream)) {
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

#define DINOML_CK_BMM_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, LAYOUT, EPILOGUE, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int batch_count, \
    int m, \
    int n, \
    int k, \
    int64_t batch_stride_a, \
    int64_t batch_stride_b, \
    int64_t batch_stride_c, \
    int lda, \
    int ldb, \
    int ldc, \
    hipStream_t stream) { \
  return launch_ck_bmm<CTYPE, BmmLayout::LAYOUT, EpilogueKind::EPILOGUE, BmmConfig::CONFIG>( \
      a, b, nullptr, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, 0, batch_stride_c, lda, ldb, 0, ldc, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int batch_count, \
    int m, \
    int n, \
    int k, \
    int64_t batch_stride_a, \
    int64_t batch_stride_b, \
    int64_t batch_stride_c, \
    int lda, \
    int ldb, \
    int ldc, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_bmm<CTYPE, BmmLayout::LAYOUT, EpilogueKind::EPILOGUE, BmmConfig::CONFIG>( \
      a, b, nullptr, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, 0, batch_stride_c, lda, ldb, 0, ldc, iterations, stream); \
}

#define DINOML_CK_BMM_ADD_EXPORT(OP, DTYPE_NAME, CTYPE, SYMBOL_ID, LAYOUT, EPILOGUE, CONFIG) \
extern "C" int dinoml_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* d0, \
    CTYPE* c, \
    int batch_count, \
    int m, \
    int n, \
    int k, \
    int64_t batch_stride_a, \
    int64_t batch_stride_b, \
    int64_t batch_stride_d0, \
    int64_t batch_stride_c, \
    int lda, \
    int ldb, \
    int ldd0, \
    int ldc, \
    hipStream_t stream) { \
  return launch_ck_bmm<CTYPE, BmmLayout::LAYOUT, EpilogueKind::EPILOGUE, BmmConfig::CONFIG>( \
      a, b, d0, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_d0, batch_stride_c, lda, ldb, ldd0, ldc, stream); \
} \
extern "C" DINO_EXPORT float dinoml_profile_ck_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* d0, \
    CTYPE* c, \
    int batch_count, \
    int m, \
    int n, \
    int k, \
    int64_t batch_stride_a, \
    int64_t batch_stride_b, \
    int64_t batch_stride_d0, \
    int64_t batch_stride_c, \
    int lda, \
    int ldb, \
    int ldd0, \
    int ldc, \
    int iterations, \
    hipStream_t stream) { \
  return profile_ck_bmm<CTYPE, BmmLayout::LAYOUT, EpilogueKind::EPILOGUE, BmmConfig::CONFIG>( \
      a, b, d0, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_d0, batch_stride_c, lda, ldb, ldd0, ldc, iterations, stream); \
}

#define DINOML_CK_BMM_GENERATED_EXPORTS 1
