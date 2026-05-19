#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include <cutlass/bfloat16.h>
#include <cutlass/arch/mma.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_bias_elementwise.h>
#include <cutlass/epilogue/thread/linear_combination_gelu.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/thread/linear_combination_hardswish.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include <cutlass/epilogue/thread/linear_combination_silu.h>
#include <cutlass/fast_math.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_with_broadcast.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

namespace cutlass {
namespace epilogue {
namespace thread {

template <typename T>
struct ELUp1 {
  CUTLASS_HOST_DEVICE
  T operator()(T const& scalar) const {
    return scalar >= T(0) ? scalar + T(1) : cutlass::fast_exp(scalar);
  }
};

template <typename T, int N>
struct ELUp1<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& value) const {
    Array<T, N> result;
    ELUp1<T> elup1;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = elup1(value[i]);
    }
    return result;
  }
};

template <typename T>
struct QuickGELU {
  CUTLASS_HOST_DEVICE
  T operator()(T const& scalar) const {
    float x = static_cast<float>(scalar);
    return T(x / (1.0f + cutlass::fast_exp(-1.702f * x)));
  }
};

template <typename T, int N>
struct QuickGELU<Array<T, N>> {
  CUTLASS_HOST_DEVICE
  Array<T, N> operator()(Array<T, N> const& value) const {
    Array<T, N> result;
    QuickGELU<T> quick_gelu;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N; ++i) {
      result[i] = quick_gelu(value[i]);
    }
    return result;
  }
};

template <
    typename ElementOutput,
    int Count,
    typename ElementAccumulator = ElementOutput,
    typename ElementCompute = ElementOutput,
    ScaleType::Kind Scale = ScaleType::Default,
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
using LinearCombinationELUp1 = LinearCombinationGeneric<
    ELUp1,
    ElementOutput,
    Count,
    ElementAccumulator,
    ElementCompute,
    Scale,
    Round,
    false>;

template <
    typename ElementOutput,
    int Count,
    typename ElementAccumulator = ElementOutput,
    typename ElementCompute = ElementOutput,
    ScaleType::Kind Scale = ScaleType::Default,
    FloatRoundStyle Round = FloatRoundStyle::round_to_nearest>
using LinearCombinationQuickGELU = LinearCombinationGeneric<
    QuickGELU,
    ElementOutput,
    Count,
    ElementAccumulator,
    ElementCompute,
    Scale,
    Round,
    false>;

}  // namespace thread
}  // namespace epilogue
}  // namespace cutlass


namespace {


template <typename Element, typename ElementAccumulator = float>
using BiasEpilogue = cutlass::epilogue::thread::LinearCombination<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <typename Element, typename ElementAccumulator = float>
using BiasReluEpilogue = cutlass::epilogue::thread::LinearCombinationRelu<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <typename Element, typename ElementAccumulator = float>
using BiasGeluEpilogue = cutlass::epilogue::thread::LinearCombinationGELU<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <typename Element, typename ElementAccumulator = float>
using BiasFastGeluEpilogue = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::GELU_taylor,
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling,
    cutlass::FloatRoundStyle::round_to_nearest,
    true>;

template <typename Element, typename ElementAccumulator = float>
using BiasQuickGeluEpilogue = cutlass::epilogue::thread::LinearCombinationQuickGELU<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <typename Element, typename ElementAccumulator = float>
using BiasSigmoidEpilogue = cutlass::epilogue::thread::LinearCombinationSigmoid<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <typename Element, typename ElementAccumulator = float>
using BiasTanhEpilogue = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::Tanh,
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling,
    cutlass::FloatRoundStyle::round_to_nearest,
    true>;

template <typename Element, typename ElementAccumulator = float>
using BiasSwishEpilogue = cutlass::epilogue::thread::LinearCombinationSilu<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <typename Element, typename ElementAccumulator = float>
using BiasHardSwishEpilogue = cutlass::epilogue::thread::LinearCombinationHardSwish<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;

template <typename Element, typename ElementAccumulator = float>
using BiasElup1Epilogue = cutlass::epilogue::thread::LinearCombinationELUp1<
    Element,
    1,
    ElementAccumulator,
    float,
    cutlass::epilogue::thread::ScaleType::NoBetaScaling,
    cutlass::FloatRoundStyle::round_to_nearest>;

enum class BiasResidualKind {
  kAdd,
  kAddAdd,
  kMul,
  kMulAdd,
};

template <
    typename Element_,
    typename ElementAccumulator_,
    BiasResidualKind Kind_,
    template <typename>
    class ElementwiseOp_ = cutlass::epilogue::thread::Identity,
    template <typename>
    class BaseElementwiseOp_ = cutlass::epilogue::thread::Identity,
    int ElementsPerAccess = 1>
class BiasResidualEpilogue {
 public:
  using ElementOutput = Element_;
  using ElementD = ElementOutput;
  using ElementC = Element_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementCompute = float;
  using ElementScalar = ElementCompute;
  using ElementZ = Element_;
  using ElementT = Element_;
  using ElementVector = Element_;
  using ElementSource = ElementC;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kCount = kElementsPerAccess;
  static bool const IsEltActSupported = true;
  static bool const kIsSingleSource = Kind_ == BiasResidualKind::kAdd || Kind_ == BiasResidualKind::kMul;
  static bool const kSupportsSerialSplitK = Kind_ == BiasResidualKind::kAdd || Kind_ == BiasResidualKind::kAddAdd;
  using ElementwiseOp = ElementwiseOp_<ElementCompute>;
  using BaseElementwiseOp = BaseElementwiseOp_<ElementCompute>;
  static bool const kIsHeavy = cutlass::epilogue::thread::kIsHeavy_member_or_false<ElementwiseOp>::value ||
      cutlass::epilogue::thread::kIsHeavy_member_or_false<BaseElementwiseOp>::value;
  static bool const kStoreZ = true;
  static bool const kStoreT = false;
  static constexpr bool IsPerChannelScalingSupported = false;
  static const cutlass::epilogue::thread::ScaleType::Kind kScale = cutlass::epilogue::thread::ScaleType::Default;

  using ActivationFn = ElementwiseOp;
  using FragmentAccumulator = cutlass::Array<ElementAccumulator, kElementsPerAccess>;
  using FragmentCompute = cutlass::Array<ElementCompute, kElementsPerAccess>;
  using FragmentC = cutlass::Array<ElementC, kElementsPerAccess>;
  using FragmentZ = cutlass::Array<ElementZ, kElementsPerAccess>;
  using FragmentT = cutlass::Array<ElementT, kElementsPerAccess>;
  using FragmentSource = FragmentC;
  using FragmentOutput = FragmentZ;
  using ElementBias = ElementVector;
  using FragmentBias = cutlass::Array<ElementBias, kElementsPerAccess>;

  struct Params {
    ElementCompute alpha;

    CUTLASS_HOST_DEVICE
    Params(ElementCompute alpha_ = ElementCompute(1)) : alpha(alpha_) {}
  };

 private:
  ElementCompute alpha_;
  int k_partition_;
  int k_partition_count_;

 public:
  CUTLASS_HOST_DEVICE
  BiasResidualEpilogue(Params const& params) : alpha_(params.alpha), k_partition_(0), k_partition_count_(1) {}

  CUTLASS_HOST_DEVICE
  bool is_source_needed() const {
    return true;
  }

  CUTLASS_HOST_DEVICE
  void set_k_partition(int k_partition, int k_partition_count) {
    k_partition_ = k_partition;
    k_partition_count_ = k_partition_count;
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& frag_z,
      FragmentT& frag_t,
      FragmentAccumulator const& accum,
      FragmentC const& d0,
      FragmentCompute const& bias) const {
    FragmentCompute base = base_fragment(accum, bias);
    FragmentCompute source0 = cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(d0);
    FragmentCompute result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      if constexpr (Kind_ == BiasResidualKind::kMul) {
        result[i] = base[i] * source0[i];
      } else {
        result[i] = base[i] + source0[i];
      }
    }
    store_result(frag_z, frag_t, result);
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& frag_z,
      FragmentT& frag_t,
      FragmentAccumulator const& accum,
      FragmentC const& d0,
      FragmentC const& d1,
      FragmentCompute const& bias) const {
    FragmentCompute base = base_fragment(accum, bias);
    FragmentCompute source0 = cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(d0);
    FragmentCompute source1 = cutlass::NumericArrayConverter<ElementCompute, ElementC, kElementsPerAccess>()(d1);
    FragmentCompute result;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      if constexpr (Kind_ == BiasResidualKind::kMulAdd) {
        result[i] = base[i] * source0[i] + source1[i];
      } else if constexpr (Kind_ == BiasResidualKind::kAddAdd) {
        result[i] = base[i] + source0[i] + (is_serial_split_k_reduction() && k_partition_ > 0 ? ElementCompute(0) : source1[i]);
      } else {
        result[i] = base[i] + source0[i] + source1[i];
      }
    }
    store_result(frag_z, frag_t, result);
  }

  CUTLASS_HOST_DEVICE
  void operator()(
      FragmentZ& frag_z,
      FragmentT& frag_t,
      FragmentAccumulator const& accum,
      FragmentCompute const& bias) const {
    store_result(frag_z, frag_t, base_fragment(accum, bias));
  }

 private:
  CUTLASS_HOST_DEVICE
  FragmentCompute base_fragment(FragmentAccumulator const& accum, FragmentCompute const& bias) const {
    FragmentCompute converted_accum =
        cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kElementsPerAccess>()(accum);
    FragmentCompute result;
    BaseElementwiseOp base_elementwise_op;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      ElementCompute biased = alpha_ * converted_accum[i] + (is_final_k_partition() ? bias[i] : ElementCompute(0));
      result[i] = is_final_k_partition() ? base_elementwise_op(biased) : biased;
    }
    return result;
  }

  CUTLASS_HOST_DEVICE
  void store_result(FragmentZ& frag_z, FragmentT& frag_t, FragmentCompute const& result) const {
    ElementwiseOp elementwise_op;
    FragmentCompute activated;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kElementsPerAccess; ++i) {
      activated[i] = is_final_k_partition() ? elementwise_op(result[i]) : result[i];
    }
    frag_z = cutlass::NumericArrayConverter<ElementZ, ElementCompute, kElementsPerAccess>()(activated);
    frag_t.clear();
  }

  CUTLASS_HOST_DEVICE
  bool is_serial_split_k_reduction() const {
    return k_partition_count_ > 1;
  }

  CUTLASS_HOST_DEVICE
  bool is_final_k_partition() const {
    return !is_serial_split_k_reduction() || k_partition_ == k_partition_count_ - 1;
  }
};

template <typename Element, typename ElementAccumulator = float>
using BiasAddEpilogue = BiasResidualEpilogue<Element, ElementAccumulator, BiasResidualKind::kAdd>;

template <typename Element, typename ElementAccumulator = float>
using BiasAddAddEpilogue = BiasResidualEpilogue<Element, ElementAccumulator, BiasResidualKind::kAddAdd>;

template <typename Element, typename ElementAccumulator = float>
using BiasAddReluEpilogue =
    BiasResidualEpilogue<Element, ElementAccumulator, BiasResidualKind::kAdd, cutlass::epilogue::thread::ReLu>;

template <typename Element, typename ElementAccumulator = float>
using BiasAddAddReluEpilogue =
    BiasResidualEpilogue<Element, ElementAccumulator, BiasResidualKind::kAddAdd, cutlass::epilogue::thread::ReLu>;

template <typename Element, typename ElementAccumulator = float>
using BiasMulEpilogue = BiasResidualEpilogue<Element, ElementAccumulator, BiasResidualKind::kMul>;

template <typename Element, typename ElementAccumulator = float>
using BiasMulAddEpilogue = BiasResidualEpilogue<Element, ElementAccumulator, BiasResidualKind::kMulAdd>;

template <typename Element, typename ElementAccumulator = float>
using BiasMulTanhEpilogue =
    BiasResidualEpilogue<Element, ElementAccumulator, BiasResidualKind::kMul, cutlass::epilogue::thread::Tanh>;

template <typename Element, typename ElementAccumulator = float>
using BiasSigmoidMulEpilogue = BiasResidualEpilogue<
    Element,
    ElementAccumulator,
    BiasResidualKind::kMul,
    cutlass::epilogue::thread::Identity,
    cutlass::epilogue::thread::Sigmoid>;

template <typename Element, typename ElementAccumulator = float>
using BiasSigmoidMulTanhEpilogue = BiasResidualEpilogue<
    Element,
    ElementAccumulator,
    BiasResidualKind::kMul,
    cutlass::epilogue::thread::Tanh,
    cutlass::epilogue::thread::Sigmoid>;

template <
    typename OperatorClass_,
    typename ArchTag_,
    typename ThreadblockShape_,
    typename WarpShape_,
    typename InstructionShape_,
    typename ElementAccumulator_,
    int Stages_,
    int AlignmentA_ = 1,
    int AlignmentB_ = 1,
    typename Operator_ = cutlass::arch::OpMultiplyAdd>
struct GemmPolicy {
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using ElementAccumulator = ElementAccumulator_;
  using Operator = Operator_;
  static int const kStages = Stages_;
  static int const kAlignmentA = AlignmentA_;
  static int const kAlignmentB = AlignmentB_;
};

template <typename BasePolicy, int Alignment>
struct AlignedGemmPolicy : BasePolicy {
  static int const kAlignmentA = Alignment;
  static int const kAlignmentB = Alignment;
};

template <typename Element>
struct BiasGemmPolicySelector {
  using Policy = Sm80TensorOp256x128x16S3W4x2x1TF32F32GemmPolicy;
};

template <>
struct BiasGemmPolicySelector<cutlass::half_t> {
  using Policy = Sm80TensorOp256x128x32S3W4x2x1F16F32GemmPolicy;
};

template <>
struct BiasGemmPolicySelector<cutlass::bfloat16_t> {
  using Policy = Sm80TensorOp256x128x32S3W4x2x1F16F32GemmPolicy;
};

template <typename Element>
using BiasGemmPolicy = typename BiasGemmPolicySelector<Element>::Policy;


template <typename Storage, typename Element>
Element const* cutlass_ptr(Storage const* ptr) {
  static_assert(sizeof(Storage) == sizeof(Element), "CUTLASS storage type size mismatch");
  return reinterpret_cast<Element const*>(ptr);
}

template <typename Storage, typename Element>
Element* cutlass_ptr(Storage* ptr) {
  static_assert(sizeof(Storage) == sizeof(Element), "CUTLASS storage type size mismatch");
  return reinterpret_cast<Element*>(ptr);
}

template <typename Element, typename LayoutB, typename EpilogueOp, typename Policy, bool SplitKSerial>
using PolicyDeviceGemm = cutlass::gemm::device::Gemm<
    Element,
    cutlass::layout::RowMajor,
    Element,
    LayoutB,
    Element,
    cutlass::layout::RowMajor,
    typename Policy::ElementAccumulator,
    typename Policy::OperatorClass,
    typename Policy::ArchTag,
    typename Policy::ThreadblockShape,
    typename Policy::WarpShape,
    typename Policy::InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
    Policy::kStages,
    Policy::kAlignmentA,
    Policy::kAlignmentB,
    SplitKSerial,
    typename Policy::Operator>;

template <
    typename Element,
    typename LayoutB,
    typename EpilogueOp,
    typename Policy,
    typename OperatorClass = typename Policy::OperatorClass>
struct BroadcastGemmKernelSelector {
  using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmWithBroadcast<
      Element,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      Policy::kAlignmentA,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      Policy::kAlignmentB,
      Element,
      cutlass::layout::RowMajor,
      typename Policy::ElementAccumulator,
      typename Policy::OperatorClass,
      typename Policy::ArchTag,
      typename Policy::ThreadblockShape,
      typename Policy::WarpShape,
      typename Policy::InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Policy::kStages,
      typename Policy::Operator>::GemmKernel;
};

template <typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
struct BroadcastGemmKernelSelector<Element, LayoutB, EpilogueOp, Policy, cutlass::arch::OpClassSimt> {
  using GemmBase = typename cutlass::gemm::kernel::DefaultGemmUniversal<
      Element,
      cutlass::layout::RowMajor,
      cutlass::ComplexTransform::kNone,
      Policy::kAlignmentA,
      Element,
      LayoutB,
      cutlass::ComplexTransform::kNone,
      Policy::kAlignmentB,
      Element,
      cutlass::layout::RowMajor,
      typename Policy::ElementAccumulator,
      cutlass::arch::OpClassSimt,
      typename Policy::ArchTag,
      typename Policy::ThreadblockShape,
      typename Policy::WarpShape,
      typename Policy::InstructionShape,
      EpilogueOp,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Policy::kStages,
      typename Policy::Operator>::GemmKernel;

  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWithBroadcastSimt<
      typename GemmBase::Epilogue::Shape,
      typename GemmBase::Epilogue::WarpMmaOperator,
      Element,
      typename EpilogueOp::ElementT,
      typename EpilogueOp::ElementVector,
      EpilogueOp,
      GemmBase::Epilogue::kElementsPerAccess>::Epilogue;

  using GemmKernel = cutlass::gemm::kernel::GemmWithFusedEpilogue<
      typename GemmBase::Mma,
      Epilogue,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>>;
};

template <typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
using BroadcastDeviceGemm =
    cutlass::gemm::device::GemmUniversalBase<
        typename BroadcastGemmKernelSelector<Element, LayoutB, EpilogueOp, Policy>::GemmKernel>;


}


#define DINOML_LAYOUT_B_gemm_rrr cutlass::layout::RowMajor
#define DINOML_LAYOUT_B_gemm_rcr cutlass::layout::ColumnMajor
#define DINOML_LAYOUT_B_gemm_rrr_bias cutlass::layout::RowMajor
#define DINOML_LAYOUT_B_gemm_rcr_bias cutlass::layout::ColumnMajor
#define DINOML_LAYOUT_B_gemm_rrr_bias_relu cutlass::layout::RowMajor
#define DINOML_LAYOUT_B_gemm_rcr_bias_relu cutlass::layout::ColumnMajor
#define DINOML_LDB_gemm_rrr n
#define DINOML_LDB_gemm_rcr k
#define DINOML_LDB_gemm_rrr_bias n
#define DINOML_LDB_gemm_rcr_bias k
#define DINOML_LDB_gemm_rrr_bias_relu n
#define DINOML_LDB_gemm_rcr_bias_relu k
#define DINOML_BIAS_EPILOGUE_gemm_rrr_bias BiasEpilogue
#define DINOML_BIAS_EPILOGUE_gemm_rcr_bias BiasEpilogue
#define DINOML_BIAS_EPILOGUE_gemm_rrr_bias_relu BiasReluEpilogue
#define DINOML_BIAS_EPILOGUE_gemm_rcr_bias_relu BiasReluEpilogue


using Sm80TensorOp128x128x32S3W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S3W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S3W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S4W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S4W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S4W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S5W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    5,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S5W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    5,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S5W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    5,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x64S3W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x64S3W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x64S3W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x64S4W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x64S4W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x64S4W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x160x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 160, 32>,
    cutlass::gemm::GemmShape<32, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x160x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 160, 32>,
    cutlass::gemm::GemmShape<32, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x160x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 160, 32>,
    cutlass::gemm::GemmShape<32, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x160x32S4W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 160, 32>,
    cutlass::gemm::GemmShape<32, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x160x32S4W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 160, 32>,
    cutlass::gemm::GemmShape<32, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x160x32S4W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 160, 32>,
    cutlass::gemm::GemmShape<32, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x192x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 192, 32>,
    cutlass::gemm::GemmShape<32, 96, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x192x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 192, 32>,
    cutlass::gemm::GemmShape<32, 96, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x192x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 192, 32>,
    cutlass::gemm::GemmShape<32, 96, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x192x32S4W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 192, 32>,
    cutlass::gemm::GemmShape<32, 96, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x192x32S4W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 192, 32>,
    cutlass::gemm::GemmShape<32, 96, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x192x32S4W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 192, 32>,
    cutlass::gemm::GemmShape<32, 96, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x224x32S3W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 224, 32>,
    cutlass::gemm::GemmShape<64, 56, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x224x32S3W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 224, 32>,
    cutlass::gemm::GemmShape<64, 56, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x224x32S3W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 224, 32>,
    cutlass::gemm::GemmShape<64, 56, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x224x32S4W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 224, 32>,
    cutlass::gemm::GemmShape<64, 56, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x224x32S4W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 224, 32>,
    cutlass::gemm::GemmShape<64, 56, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x224x32S4W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 224, 32>,
    cutlass::gemm::GemmShape<64, 56, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x32S3W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x32S3W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x32S3W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x64S3W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x64S3W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x64S3W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x32S6W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    6,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x32S6W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    6,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x32S6W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    6,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x64S3W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x64S3W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x64S3W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 64>,
    cutlass::gemm::GemmShape<64, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x128x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 128, 32>,
    cutlass::gemm::GemmShape<40, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x128x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 128, 32>,
    cutlass::gemm::GemmShape<40, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x128x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 128, 32>,
    cutlass::gemm::GemmShape<40, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x128x32S4W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 128, 32>,
    cutlass::gemm::GemmShape<40, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x128x32S4W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 128, 32>,
    cutlass::gemm::GemmShape<40, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x128x32S4W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 128, 32>,
    cutlass::gemm::GemmShape<40, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x192x32S3W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 192, 32>,
    cutlass::gemm::GemmShape<80, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x192x32S3W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 192, 32>,
    cutlass::gemm::GemmShape<80, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x192x32S3W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 192, 32>,
    cutlass::gemm::GemmShape<80, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x192x32S4W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 192, 32>,
    cutlass::gemm::GemmShape<80, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x192x32S4W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 192, 32>,
    cutlass::gemm::GemmShape<80, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp160x192x32S4W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<160, 192, 32>,
    cutlass::gemm::GemmShape<80, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x128x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 128, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x128x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 128, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x128x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 128, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x128x32S4W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 128, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x128x32S4W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 128, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x128x32S4W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 128, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x160x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 160, 32>,
    cutlass::gemm::GemmShape<48, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x160x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 160, 32>,
    cutlass::gemm::GemmShape<48, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x160x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 160, 32>,
    cutlass::gemm::GemmShape<48, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x160x32S4W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 160, 32>,
    cutlass::gemm::GemmShape<48, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x160x32S4W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 160, 32>,
    cutlass::gemm::GemmShape<48, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x160x32S4W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 160, 32>,
    cutlass::gemm::GemmShape<48, 80, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x96x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 96, 32>,
    cutlass::gemm::GemmShape<48, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x96x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 96, 32>,
    cutlass::gemm::GemmShape<48, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp192x96x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<192, 96, 32>,
    cutlass::gemm::GemmShape<48, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp224x128x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<224, 128, 32>,
    cutlass::gemm::GemmShape<56, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp224x128x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<224, 128, 32>,
    cutlass::gemm::GemmShape<56, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp224x128x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<224, 128, 32>,
    cutlass::gemm::GemmShape<56, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp224x128x32S4W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<224, 128, 32>,
    cutlass::gemm::GemmShape<56, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp224x128x32S4W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<224, 128, 32>,
    cutlass::gemm::GemmShape<56, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp224x128x32S4W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<224, 128, 32>,
    cutlass::gemm::GemmShape<56, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x64S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x64S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x64S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S2W4x1x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S2W4x1x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S2W4x1x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S3W4x1x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S3W4x1x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S3W4x1x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S4W4x1x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S4W4x1x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S4W4x1x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x64S3W4x1x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x64S3W4x1x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x64S3W4x1x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x64S4W4x1x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x64S4W4x1x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x64S4W4x1x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x96x32S2W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 96, 32>,
    cutlass::gemm::GemmShape<64, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x96x32S2W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 96, 32>,
    cutlass::gemm::GemmShape<64, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x96x32S2W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 96, 32>,
    cutlass::gemm::GemmShape<64, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x96x32S3W4x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 96, 32>,
    cutlass::gemm::GemmShape<64, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x96x32S3W4x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 96, 32>,
    cutlass::gemm::GemmShape<64, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x96x32S3W4x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 96, 32>,
    cutlass::gemm::GemmShape<64, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x32S6W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    6,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x32S6W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    6,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x32S6W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    6,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x64S3W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x64S3W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x64S3W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 64>,
    cutlass::gemm::GemmShape<32, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S2W1x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S2W1x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S2W1x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S4W1x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S4W1x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S4W1x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x64S3W1x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x64S3W1x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x64S3W1x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x64S4W1x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x64S4W1x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x64S4W1x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    4,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x32S10W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    10,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x32S10W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    10,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x32S10W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    10,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x64S5W2x2x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    5,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x64S5W2x2x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    5,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x64S5W2x2x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<32, 32, 64>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    5,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x192x32S3W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 192, 32>,
    cutlass::gemm::GemmShape<48, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x192x32S3W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 192, 32>,
    cutlass::gemm::GemmShape<48, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x192x32S3W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 192, 32>,
    cutlass::gemm::GemmShape<48, 48, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x256x32S2W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 256, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x256x32S2W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 256, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x256x32S2W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 256, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    2,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x256x32S3W2x4x1F16F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 256, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x256x32S3W2x4x1F16F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 256, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp96x256x32S3W2x4x1F16F32Align8GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<96, 256, 32>,
    cutlass::gemm::GemmShape<48, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>,
    float,
    3,
    8,
    8,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt128x128x8S4W4x2x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt128x128x8S5W4x2x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt128x256x8S4W2x4x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt128x256x8S5W2x4x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt128x32x8S5W2x1x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 32, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt128x64x8S5W2x2x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 8>,
    cutlass::gemm::GemmShape<64, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt256x128x8S4W4x2x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt256x128x8S5W4x2x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 8>,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt32x128x8S5W1x2x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<32, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt64x128x8S5W2x2x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80Simt64x64x8S5W2x1x1F32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassSimt,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 64, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S3W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S3W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S3W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S4W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S4W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S4W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S5W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S5W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    5,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x16S5W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    5,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S3W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S3W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S3W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S4W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S4W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x128x32S4W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x16S3W2x4x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x16S3W2x4x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x16S3W2x4x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x32S3W2x4x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x32S3W2x4x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x256x32S3W2x4x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x16S6W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    6,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x16S6W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    6,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x16S6W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 16>,
    cutlass::gemm::GemmShape<64, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    6,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x32S3W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x32S3W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp128x64x32S3W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 64, 32>,
    cutlass::gemm::GemmShape<64, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x16S3W4x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x16S3W4x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x16S3W4x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x32S3W4x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x32S3W4x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x128x32S3W4x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x16S4W4x1x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x16S4W4x1x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x16S4W4x1x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S4W4x1x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S4W4x1x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp256x64x32S4W4x1x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<256, 64, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x16S6W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    6,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x16S6W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    6,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x16S6W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 16>,
    cutlass::gemm::GemmShape<32, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    6,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x32S3W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x32S3W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x128x32S3W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 128, 32>,
    cutlass::gemm::GemmShape<32, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    3,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x16S4W1x4x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x16S4W1x4x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x16S4W1x4x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S4W1x4x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S4W1x4x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x256x32S4W1x4x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 256, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    4,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x16S10W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    10,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x16S10W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    10,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x16S10W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<32, 32, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    10,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x32S5W2x2x1TF32F32Align1GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    5,
    1,
    1,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x32S5W2x2x1TF32F32Align2GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    5,
    2,
    2,
    cutlass::arch::OpMultiplyAdd>;
using Sm80TensorOp64x64x32S5W2x2x1TF32F32Align4GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<32, 32, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    float,
    5,
    4,
    4,
    cutlass::arch::OpMultiplyAdd>;
