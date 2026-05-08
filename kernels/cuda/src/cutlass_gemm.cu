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
  using Gemm = cutlass::gemm::device::Gemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      LayoutB,
      Element,
      cutlass::layout::RowMajor,
      float,
      cutlass::arch::OpClassSimt,
      cutlass::arch::Sm70,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          cutlass::arch::OpClassSimt,
          cutlass::arch::Sm70,
          Element,
          Element,
          Element,
          float>::ThreadblockShape,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          cutlass::arch::OpClassSimt,
          cutlass::arch::Sm70,
          Element,
          Element,
          Element,
          float>::WarpShape,
      typename cutlass::gemm::device::DefaultGemmConfiguration<
          cutlass::arch::OpClassSimt,
          cutlass::arch::Sm70,
          Element,
          Element,
          Element,
          float>::InstructionShape,
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

extern "C" int dinoml_cutlass_gemm_rrr_f32(
    float const* a,
    float const* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<float, float, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, stream);
}

extern "C" int dinoml_cutlass_gemm_rcr_f32(
    float const* a,
    float const* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<float, float, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, stream);
}

extern "C" int dinoml_cutlass_gemm_rrr_f16(
    half const* a,
    half const* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<half, cutlass::half_t, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, stream);
}

extern "C" int dinoml_cutlass_gemm_rcr_f16(
    half const* a,
    half const* b,
    half* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<half, cutlass::half_t, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, stream);
}

extern "C" int dinoml_cutlass_gemm_rrr_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<__nv_bfloat16, cutlass::bfloat16_t, cutlass::layout::RowMajor>(a, b, c, m, n, k, n, stream);
}

extern "C" int dinoml_cutlass_gemm_rcr_bf16(
    __nv_bfloat16 const* a,
    __nv_bfloat16 const* b,
    __nv_bfloat16* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  return launch_gemm<__nv_bfloat16, cutlass::bfloat16_t, cutlass::layout::ColumnMajor>(a, b, c, m, n, k, k, stream);
}

extern "C" int dinoml_cutlass_gemm_rrr_bias_f32(
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

extern "C" int dinoml_cutlass_gemm_rcr_bias_f32(
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

extern "C" int dinoml_cutlass_gemm_rrr_bias_f16(
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

extern "C" int dinoml_cutlass_gemm_rcr_bias_f16(
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

extern "C" int dinoml_cutlass_gemm_rrr_bias_bf16(
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

extern "C" int dinoml_cutlass_gemm_rcr_bias_bf16(
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

extern "C" int dinoml_cutlass_gemm_rrr_bias_relu_f32(
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

extern "C" int dinoml_cutlass_gemm_rcr_bias_relu_f32(
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

extern "C" int dinoml_cutlass_gemm_rrr_bias_relu_f16(
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

extern "C" int dinoml_cutlass_gemm_rcr_bias_relu_f16(
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

extern "C" int dinoml_cutlass_gemm_rrr_bias_relu_bf16(
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

extern "C" int dinoml_cutlass_gemm_rcr_bias_relu_bf16(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_f32(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_f32(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_f16(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_f16(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_bf16(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_bf16(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_f32(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_f32(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_f16(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_f16(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_bf16(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_bf16(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_relu_f32(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_relu_f32(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_relu_f16(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_relu_f16(
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

extern "C" float dinoml_profile_cutlass_gemm_rrr_bias_relu_bf16(
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

extern "C" float dinoml_profile_cutlass_gemm_rcr_bias_relu_bf16(
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
