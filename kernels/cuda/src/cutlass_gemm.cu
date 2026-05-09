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

#include <cmath>

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

enum class BiasActivation {
  kGelu,
  kFastGelu,
  kSigmoid,
  kTanh,
  kSwish,
  kHardSwish,
};

template <typename Storage>
__device__ float activation_load(Storage value) {
  return static_cast<float>(value);
}

template <>
__device__ float activation_load(half value) {
  return __half2float(value);
}

template <>
__device__ float activation_load(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <typename Storage>
__device__ Storage activation_store(float value) {
  return static_cast<Storage>(value);
}

template <>
__device__ half activation_store(float value) {
  return __float2half(value);
}

template <>
__device__ __nv_bfloat16 activation_store(float value) {
  return __float2bfloat16(value);
}

__device__ float apply_activation(BiasActivation activation, float value) {
  switch (activation) {
    case BiasActivation::kGelu: {
      constexpr float kSqrtTwoOverPi = 0.7978845608028654f;
      return 0.5f * value * (1.0f + tanhf(kSqrtTwoOverPi * (value + 0.044715f * value * value * value)));
    }
    case BiasActivation::kFastGelu:
      return value / (1.0f + expf(-1.702f * value));
    case BiasActivation::kSigmoid:
      return 1.0f / (1.0f + expf(-value));
    case BiasActivation::kTanh:
      return tanhf(value);
    case BiasActivation::kSwish:
      return value / (1.0f + expf(-value));
    case BiasActivation::kHardSwish:
      return value * fminf(fmaxf(value + 3.0f, 0.0f), 6.0f) / 6.0f;
  }
  return value;
}

template <typename Storage>
__global__ void activation_kernel(Storage* c, int elements, BiasActivation activation) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  c[index] = activation_store<Storage>(apply_activation(activation, activation_load(c[index])));
}

template <typename Storage, typename Element, typename LayoutB>
int launch_gemm_bias_activation(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    BiasActivation activation,
    cudaStream_t stream) {
  int status = launch_gemm_bias<Storage, Element, LayoutB, BiasEpilogue<Element>>(a, b, bias, c, m, n, k, ldb, stream);
  if (status != 0) {
    return status;
  }
  int elements = m * n;
  int threads = 256;
  int blocks = (elements + threads - 1) / threads;
  activation_kernel<<<blocks, threads, 0, stream>>>(c, elements, activation);
  return cudaGetLastError() == cudaSuccess ? 0 : 4;
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

template <typename Storage, typename Element, typename LayoutB>
float profile_gemm_bias_activation(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    BiasActivation activation,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  launch_gemm_bias_activation<Storage, Element, LayoutB>(a, b, bias, c, m, n, k, ldb, activation, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    launch_gemm_bias_activation<Storage, Element, LayoutB>(a, b, bias, c, m, n, k, ldb, activation, stream);
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

#define DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, LAYOUT_B, LDB, ACTIVATION, SYMBOL_ID) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return launch_gemm_bias_activation<CTYPE, ELEMENT, LAYOUT_B>(a, b, bias, c, m, n, k, LDB, ACTIVATION, stream); \
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
  return profile_gemm_bias_activation<CTYPE, ELEMENT, LAYOUT_B>(a, b, bias, c, m, n, k, LDB, ACTIVATION, iterations, stream); \
}

#define DINOML_FORWARD_GEMM_CANDIDATES(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x128x32_align8) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_64x128x32_align8) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x64x32_align8) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_64x64x32_align8) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_256x128x32_align8) \
DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x128x32_align4)

#define DINOML_FORWARD_GEMM_BIAS_CANDIDATES(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_64x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x64x32_align8) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_64x64x32_align8) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_256x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, OLD_SUFFIX, tensorop_sm80_128x128x32_align4)

#define DINOML_FORWARD_GEMM_RRR_BIAS_ACTIVATION_CANDIDATES(OP, DTYPE_NAME, CTYPE, ELEMENT, ACTIVATION) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::RowMajor, n, ACTIVATION, tensorop_sm80_128x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::RowMajor, n, ACTIVATION, tensorop_sm80_64x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::RowMajor, n, ACTIVATION, tensorop_sm80_128x64x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::RowMajor, n, ACTIVATION, tensorop_sm80_64x64x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::RowMajor, n, ACTIVATION, tensorop_sm80_256x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::RowMajor, n, ACTIVATION, tensorop_sm80_128x128x32_align4)

#define DINOML_FORWARD_GEMM_RCR_BIAS_ACTIVATION_CANDIDATES(OP, DTYPE_NAME, CTYPE, ELEMENT, ACTIVATION) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::ColumnMajor, k, ACTIVATION, tensorop_sm80_128x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::ColumnMajor, k, ACTIVATION, tensorop_sm80_64x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::ColumnMajor, k, ACTIVATION, tensorop_sm80_128x64x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::ColumnMajor, k, ACTIVATION, tensorop_sm80_64x64x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::ColumnMajor, k, ACTIVATION, tensorop_sm80_256x128x32_align8) \
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, cutlass::layout::ColumnMajor, k, ACTIVATION, tensorop_sm80_128x128x32_align4)

#define DINOML_FORWARD_GEMM_BIAS_ACTIVATION_DTYPES(OP, ACTIVATION) \
DINOML_FORWARD_GEMM_RRR_BIAS_ACTIVATION_CANDIDATES(gemm_rrr_bias_##OP, float32, float, float, ACTIVATION) \
DINOML_FORWARD_GEMM_RCR_BIAS_ACTIVATION_CANDIDATES(gemm_rcr_bias_##OP, float32, float, float, ACTIVATION) \
DINOML_FORWARD_GEMM_RRR_BIAS_ACTIVATION_CANDIDATES(gemm_rrr_bias_##OP, float16, half, cutlass::half_t, ACTIVATION) \
DINOML_FORWARD_GEMM_RCR_BIAS_ACTIVATION_CANDIDATES(gemm_rcr_bias_##OP, float16, half, cutlass::half_t, ACTIVATION) \
DINOML_FORWARD_GEMM_RRR_BIAS_ACTIVATION_CANDIDATES(gemm_rrr_bias_##OP, bfloat16, __nv_bfloat16, cutlass::bfloat16_t, ACTIVATION) \
DINOML_FORWARD_GEMM_RCR_BIAS_ACTIVATION_CANDIDATES(gemm_rcr_bias_##OP, bfloat16, __nv_bfloat16, cutlass::bfloat16_t, ACTIVATION)

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

DINOML_FORWARD_GEMM_BIAS_ACTIVATION_DTYPES(gelu, BiasActivation::kGelu)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_DTYPES(fast_gelu, BiasActivation::kFastGelu)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_DTYPES(sigmoid, BiasActivation::kSigmoid)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_DTYPES(tanh, BiasActivation::kTanh)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_DTYPES(swish, BiasActivation::kSwish)
DINOML_FORWARD_GEMM_BIAS_ACTIVATION_DTYPES(hardswish, BiasActivation::kHardSwish)
