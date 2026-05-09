#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>

namespace {

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

template <typename Storage, typename Element, typename LayoutB>
int launch_gemm(
    Storage const* a,
    Storage const* b,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      LayoutB,
      Element,
      cutlass::layout::RowMajor,
      float>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {cutlass_ptr<Storage, Element>(a), k},
      {cutlass_ptr<Storage, Element>(b), ldb},
      {cutlass_ptr<Storage, Element>(c), n},
      {cutlass_ptr<Storage, Element>(c), n},
      {1.0f, 0.0f});
  cutlass::Status status = gemm(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

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

template <
    typename OperatorClass_,
    typename ArchTag_,
    typename ThreadblockShape_,
    typename WarpShape_,
    typename InstructionShape_>
struct GemmPolicy {
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
};

using Sm80TensorOp128x128x32F32GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>>;

using Sm80TensorOp128x128x32F16GemmPolicy = GemmPolicy<
    cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 16>>;

template <typename Element>
struct BiasGemmPolicySelector {
  using Policy = Sm80TensorOp128x128x32F32GemmPolicy;
};

template <>
struct BiasGemmPolicySelector<cutlass::half_t> {
  using Policy = Sm80TensorOp128x128x32F16GemmPolicy;
};

template <>
struct BiasGemmPolicySelector<cutlass::bfloat16_t> {
  using Policy = Sm80TensorOp128x128x32F16GemmPolicy;
};

template <typename Element>
using BiasGemmPolicy = typename BiasGemmPolicySelector<Element>::Policy;

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp>
int launch_gemm_bias(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || bias == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  using Policy = BiasGemmPolicy<Element>;
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      LayoutB,
      Element,
      cutlass::layout::RowMajor,
      float,
      typename Policy::OperatorClass,
      typename Policy::ArchTag,
      typename Policy::ThreadblockShape,
      typename Policy::WarpShape,
      typename Policy::InstructionShape,
      EpilogueOp>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {cutlass_ptr<Storage, Element>(a), k},
      {cutlass_ptr<Storage, Element>(b), ldb},
      {cutlass_ptr<Storage, Element>(bias), 0},
      {cutlass_ptr<Storage, Element>(c), n},
      typename EpilogueOp::Params(1.0f));
  cutlass::Status status = gemm(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename Storage, typename Element, typename LayoutB>
float profile_gemm(
    Storage const* a,
    Storage const* b,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  launch_gemm<Storage, Element, LayoutB>(a, b, c, m, n, k, ldb, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    launch_gemm<Storage, Element, LayoutB>(a, b, c, m, n, k, ldb, stream);
  }
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp>
float profile_gemm_bias(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  launch_gemm_bias<Storage, Element, LayoutB, EpilogueOp>(a, b, bias, c, m, n, k, ldb, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    launch_gemm_bias<Storage, Element, LayoutB, EpilogueOp>(a, b, bias, c, m, n, k, ldb, stream);
  }
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

}  // namespace

static int dinoml_cutlass_legacy_gemm_rrr_f32(
    float const* a,
    float const* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<float, float, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_f32(
    float const* a,
    float const* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<float, float, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_f16(
    half const* a,
    half const* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<half, cutlass::half_t, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_f16(
    half const* a,
    half const* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<half, cutlass::half_t, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<__nv_bfloat16, cutlass::bfloat16_t, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<__nv_bfloat16, cutlass::bfloat16_t, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_bias_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<float, float, cutlass::layout::RowMajor, BiasEpilogue<float>>(
      a, b, bias, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_bias_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<float, float, cutlass::layout::ColumnMajor, BiasEpilogue<float>>(
      a, b, bias, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_bias_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<half, cutlass::half_t, cutlass::layout::RowMajor, BiasEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_bias_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<half, cutlass::half_t, cutlass::layout::ColumnMajor, BiasEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_bias_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::RowMajor,
      BiasEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_bias_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::ColumnMajor,
      BiasEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_bias_relu_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<float, float, cutlass::layout::RowMajor, BiasReluEpilogue<float>>(
      a, b, bias, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_bias_relu_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<float, float, cutlass::layout::ColumnMajor, BiasReluEpilogue<float>>(
      a, b, bias, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_bias_relu_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<half, cutlass::half_t, cutlass::layout::RowMajor, BiasReluEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_bias_relu_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<half, cutlass::half_t, cutlass::layout::ColumnMajor, BiasReluEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, k, stream);
}

static int dinoml_cutlass_legacy_gemm_rrr_bias_relu_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::RowMajor,
      BiasReluEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, n, stream);
}

static int dinoml_cutlass_legacy_gemm_rcr_bias_relu_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::ColumnMajor,
      BiasReluEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, k, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_f32(
    float const* a,
    float const* b,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<float, float, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_f32(
    float const* a,
    float const* b,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<float, float, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_f16(
    half const* a,
    half const* b,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<half, cutlass::half_t, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_f16(
    half const* a,
    half const* b,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<half, cutlass::half_t, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<__nv_bfloat16, cutlass::bfloat16_t, cutlass::layout::RowMajor>(
      a, b, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm<__nv_bfloat16, cutlass::bfloat16_t, cutlass::layout::ColumnMajor>(
      a, b, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_bias_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<float, float, cutlass::layout::RowMajor, BiasEpilogue<float>>(
      a, b, bias, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_bias_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<float, float, cutlass::layout::ColumnMajor, BiasEpilogue<float>>(
      a, b, bias, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_bias_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<half, cutlass::half_t, cutlass::layout::RowMajor, BiasEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_bias_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<half, cutlass::half_t, cutlass::layout::ColumnMajor, BiasEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_bias_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::RowMajor,
      BiasEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_bias_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::ColumnMajor,
      BiasEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_bias_relu_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<float, float, cutlass::layout::RowMajor, BiasReluEpilogue<float>>(
      a, b, bias, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_bias_relu_f32(
    float const* a,
    float const* b,
    float const* bias,
    float* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<float, float, cutlass::layout::ColumnMajor, BiasReluEpilogue<float>>(
      a, b, bias, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_bias_relu_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<half, cutlass::half_t, cutlass::layout::RowMajor, BiasReluEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_bias_relu_f16(
    half const* a,
    half const* b,
    half const* bias,
    half* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<half, cutlass::half_t, cutlass::layout::ColumnMajor, BiasReluEpilogue<cutlass::half_t>>(
      a, b, bias, c, m, n, k, k, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rrr_bias_relu_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::RowMajor,
      BiasReluEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, n, iterations, stream);
}

static float dinoml_profile_cutlass_legacy_gemm_rcr_bias_relu_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16 const* bias,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    int iterations,
    cudaStream_t stream) {
  return profile_gemm_bias<
      __nv_bfloat16,
      cutlass::bfloat16_t,
      cutlass::layout::ColumnMajor,
      BiasReluEpilogue<cutlass::bfloat16_t>>(a, b, bias, c, m, n, k, k, iterations, stream);
}

#define DINOML_CUTLASS_GENERATED_EXPORTS 1

#define DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, SYMBOL_ID) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return dinoml_cutlass_legacy_##OP##_##OLD_SUFFIX(a, b, c, m, n, k, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int iterations, \
    cudaStream_t stream) { \
  return dinoml_profile_cutlass_legacy_##OP##_##OLD_SUFFIX(a, b, c, m, n, k, iterations, stream); \
}

#define DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, SYMBOL_ID) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return dinoml_cutlass_legacy_##OP##_##OLD_SUFFIX(a, b, bias, c, m, n, k, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int iterations, \
    cudaStream_t stream) { \
  return dinoml_profile_cutlass_legacy_##OP##_##OLD_SUFFIX(a, b, bias, c, m, n, k, iterations, stream); \
}

#define DINOML_FORWARD_GEMM_CANDIDATES(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x128x32_align8) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_64x128x32_align8)

#define DINOML_FORWARD_GEMM_BIAS_CANDIDATES(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_64x128x32_align8)

DINOML_FORWARD_GEMM_CANDIDATES(gemm_rrr, float32, float, f32)
DINOML_FORWARD_GEMM_CANDIDATES(gemm_rcr, float32, float, f32)
DINOML_FORWARD_GEMM_CANDIDATES(gemm_rrr, float16, half, f16)
DINOML_FORWARD_GEMM_CANDIDATES(gemm_rcr, float16, half, f16)
DINOML_FORWARD_GEMM_CANDIDATES(gemm_rrr, bfloat16, __nv_bfloat16, bf16)
DINOML_FORWARD_GEMM_CANDIDATES(gemm_rcr, bfloat16, __nv_bfloat16, bf16)

DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rrr_bias, float32, float, f32)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rcr_bias, float32, float, f32)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rrr_bias, float16, half, f16)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rcr_bias, float16, half, f16)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rrr_bias, bfloat16, __nv_bfloat16, bf16)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rcr_bias, bfloat16, __nv_bfloat16, bf16)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rrr_bias_relu, float32, float, f32)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rcr_bias_relu, float32, float, f32)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rrr_bias_relu, float16, half, f16)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rcr_bias_relu, float16, half, f16)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rrr_bias_relu, bfloat16, __nv_bfloat16, bf16)
DINOML_FORWARD_GEMM_BIAS_CANDIDATES(gemm_rcr_bias_relu, bfloat16, __nv_bfloat16, bf16)
