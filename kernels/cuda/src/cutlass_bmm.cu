#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include <cutlass/arch/mma.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm_batched.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
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

template <typename Element, typename LayoutA, typename LayoutB, typename LayoutC, typename EpilogueOp, typename Policy>
using PolicyDeviceBmm = cutlass::gemm::device::GemmBatched<
    Element,
    LayoutA,
    Element,
    LayoutB,
    Element,
    LayoutC,
    typename Policy::ElementAccumulator,
    typename Policy::OperatorClass,
    typename Policy::ArchTag,
    typename Policy::ThreadblockShape,
    typename Policy::WarpShape,
    typename Policy::InstructionShape,
    EpilogueOp,
    cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
    Policy::kStages,
    Policy::kAlignmentA,
    Policy::kAlignmentB,
    typename Policy::Operator>;

template <typename Storage, typename Element, typename LayoutA, typename LayoutB, typename LayoutC, typename Policy>
int launch_bmm_policy(
    Storage const* a,
    Storage const* b,
    Storage* c,
    int batch_count,
    int m,
    int n,
    int k,
    int64_t batch_stride_a,
    int64_t batch_stride_b,
    int64_t batch_stride_c,
    int lda,
    int ldb,
    int ldc,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return 1;
  }
  if (batch_count <= 0 || m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (batch_stride_a < 0 || batch_stride_b < 0 || batch_stride_c <= 0 || lda <= 0 || ldb <= 0 || ldc <= 0) {
    return 2;
  }
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element,
      1,
      typename Policy::ElementAccumulator,
      float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Gemm = PolicyDeviceBmm<Element, LayoutA, LayoutB, LayoutC, EpilogueOp, Policy>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {cutlass_ptr<Storage, Element>(a), lda},
      batch_stride_a,
      {cutlass_ptr<Storage, Element>(b), ldb},
      batch_stride_b,
      {cutlass_ptr<Storage, Element>(c), ldc},
      batch_stride_c,
      {cutlass_ptr<Storage, Element>(c), ldc},
      batch_stride_c,
      {1.0f, 0.0f},
      batch_count);
  cutlass::Status implement_status = Gemm::can_implement(args);
  if (implement_status != cutlass::Status::kSuccess) {
    return 4;
  }
  cutlass::Status status = gemm(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename Storage, typename Element, typename LayoutA, typename LayoutB, typename LayoutC, typename Policy>
int launch_bmm_add_policy(
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
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || d0 == nullptr || c == nullptr) {
    return 1;
  }
  if (batch_count <= 0 || m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  if (batch_stride_a < 0 || batch_stride_b < 0 || batch_stride_d0 <= 0 || batch_stride_c <= 0 || lda <= 0 ||
      ldb <= 0 || ldd0 <= 0 || ldc <= 0) {
    return 2;
  }
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element,
      1,
      typename Policy::ElementAccumulator,
      float>;
  using Gemm = PolicyDeviceBmm<Element, LayoutA, LayoutB, LayoutC, EpilogueOp, Policy>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {cutlass_ptr<Storage, Element>(a), lda},
      batch_stride_a,
      {cutlass_ptr<Storage, Element>(b), ldb},
      batch_stride_b,
      {cutlass_ptr<Storage, Element>(d0), ldd0},
      batch_stride_d0,
      {cutlass_ptr<Storage, Element>(c), ldc},
      batch_stride_c,
      {1.0f, 1.0f},
      batch_count);
  cutlass::Status implement_status = Gemm::can_implement(args);
  if (implement_status != cutlass::Status::kSuccess) {
    return 4;
  }
  cutlass::Status status = gemm(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename Storage, typename Element, typename LayoutA, typename LayoutB, typename LayoutC, typename Policy>
float profile_bmm_policy(
    Storage const* a,
    Storage const* b,
    Storage* c,
    int batch_count,
    int m,
    int n,
    int k,
    int64_t batch_stride_a,
    int64_t batch_stride_b,
    int64_t batch_stride_c,
    int lda,
    int ldb,
    int ldc,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  if (launch_bmm_policy<Storage, Element, LayoutA, LayoutB, LayoutC, Policy>(
          a, b, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_c, lda, ldb, ldc, stream)) {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return -1.0f;
  }
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    if (launch_bmm_policy<Storage, Element, LayoutA, LayoutB, LayoutC, Policy>(
            a, b, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_c, lda, ldb, ldc, stream)) {
      cudaEventDestroy(start);
      cudaEventDestroy(end);
      return -1.0f;
    }
  }
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, typename Element, typename LayoutA, typename LayoutB, typename LayoutC, typename Policy>
float profile_bmm_add_policy(
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
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  if (launch_bmm_add_policy<Storage, Element, LayoutA, LayoutB, LayoutC, Policy>(
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
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return -1.0f;
  }
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    if (launch_bmm_add_policy<Storage, Element, LayoutA, LayoutB, LayoutC, Policy>(
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
      cudaEventDestroy(start);
      cudaEventDestroy(end);
      return -1.0f;
    }
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

#define DINOML_CUTLASS_BMM_GENERATED_EXPORTS 1

#define DINOML_FORWARD_BMM_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, LAYOUT_A, LAYOUT_B, LAYOUT_C, SYMBOL_ID, POLICY) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
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
    cudaStream_t stream) { \
  return launch_bmm_policy<CTYPE, ELEMENT, LAYOUT_A, LAYOUT_B, LAYOUT_C, POLICY>( \
      a, b, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_c, lda, ldb, ldc, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
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
    cudaStream_t stream) { \
  return profile_bmm_policy<CTYPE, ELEMENT, LAYOUT_A, LAYOUT_B, LAYOUT_C, POLICY>( \
      a, b, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_c, lda, ldb, ldc, iterations, stream); \
}

#define DINOML_FORWARD_BMM_ADD_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, LAYOUT_A, LAYOUT_B, LAYOUT_C, SYMBOL_ID, POLICY) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
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
    cudaStream_t stream) { \
  return launch_bmm_add_policy<CTYPE, ELEMENT, LAYOUT_A, LAYOUT_B, LAYOUT_C, POLICY>( \
      a, b, d0, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_d0, batch_stride_c, lda, ldb, ldd0, ldc, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
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
    cudaStream_t stream) { \
  return profile_bmm_add_policy<CTYPE, ELEMENT, LAYOUT_A, LAYOUT_B, LAYOUT_C, POLICY>( \
      a, b, d0, c, batch_count, m, n, k, batch_stride_a, batch_stride_b, batch_stride_d0, batch_stride_c, lda, ldb, ldd0, ldc, iterations, stream); \
}
