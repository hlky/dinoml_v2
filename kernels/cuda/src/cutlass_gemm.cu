#include "cutlass_common.cuh"
#include "../../../third_party/cutlass/examples/45_dual_gemm/device/dual_gemm.h"

namespace {

int validate_split_k_workspace(size_t required_nbytes, void* workspace, size_t workspace_nbytes) {
  if (required_nbytes == 0) {
    return 0;
  }
  if (workspace == nullptr) {
    return 4;
  }
  if (workspace_nbytes < required_nbytes) {
    return 5;
  }
  return 0;
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

template <typename Storage, typename Element, typename LayoutB, typename Policy>
int launch_gemm_policy(
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
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element,
      1,
      typename Policy::ElementAccumulator,
      float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Gemm = PolicyDeviceGemm<Element, LayoutB, EpilogueOp, Policy, false>;
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

template <typename Storage, typename Element, typename LayoutB, typename Policy>
size_t gemm_policy_workspace_size(
    int m,
    int n,
    int k,
    int ldb,
    int split_k) {
  if (m <= 0 || n <= 0 || k <= 0 || split_k <= 1) {
    return 0;
  }
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element,
      1,
      typename Policy::ElementAccumulator,
      float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Gemm = PolicyDeviceGemm<Element, LayoutB, EpilogueOp, Policy, true>;
  typename Gemm::Arguments args(
      {m, n, k},
      {static_cast<Element const*>(nullptr), k},
      {static_cast<Element const*>(nullptr), ldb},
      {static_cast<Element const*>(nullptr), n},
      {static_cast<Element*>(nullptr), n},
      {1.0f, 0.0f},
      split_k);
  return Gemm::get_workspace_size(args);
}

template <typename Storage, typename Element, typename LayoutB, typename Policy>
int launch_gemm_policy_splitk(
    Storage const* a,
    Storage const* b,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int split_k,
    void* workspace,
    size_t workspace_nbytes,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0 || split_k <= 0) {
    return 2;
  }
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      Element,
      1,
      typename Policy::ElementAccumulator,
      float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using Gemm = PolicyDeviceGemm<Element, LayoutB, EpilogueOp, Policy, true>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {cutlass_ptr<Storage, Element>(a), k},
      {cutlass_ptr<Storage, Element>(b), ldb},
      {cutlass_ptr<Storage, Element>(c), n},
      {cutlass_ptr<Storage, Element>(c), n},
      {1.0f, 0.0f},
      split_k);
  int workspace_err = validate_split_k_workspace(Gemm::get_workspace_size(args), workspace, workspace_nbytes);
  if (workspace_err) {
    return workspace_err;
  }
  cutlass::Status status = gemm(args, workspace, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
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
  using Gemm = PolicyDeviceGemm<Element, LayoutB, EpilogueOp, Policy, false>;
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

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
size_t gemm_bias_policy_workspace_size(
    int m,
    int n,
    int k,
    int ldb,
    int split_k) {
  if (m <= 0 || n <= 0 || k <= 0 || split_k <= 1) {
    return 0;
  }
  using Gemm = PolicyDeviceGemm<Element, LayoutB, EpilogueOp, Policy, true>;
  typename Gemm::Arguments args(
      {m, n, k},
      {static_cast<Element const*>(nullptr), k},
      {static_cast<Element const*>(nullptr), ldb},
      {static_cast<Element const*>(nullptr), 0},
      {static_cast<Element*>(nullptr), n},
      typename EpilogueOp::Params(1.0f),
      split_k);
  return Gemm::get_workspace_size(args);
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
int launch_gemm_bias_splitk(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int split_k,
    void* workspace,
    size_t workspace_nbytes,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || bias == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0 || split_k <= 0) {
    return 2;
  }
  using Gemm = PolicyDeviceGemm<Element, LayoutB, EpilogueOp, Policy, true>;
  Gemm gemm;
  typename Gemm::Arguments args(
      {m, n, k},
      {cutlass_ptr<Storage, Element>(a), k},
      {cutlass_ptr<Storage, Element>(b), ldb},
      {cutlass_ptr<Storage, Element>(bias), 0},
      {cutlass_ptr<Storage, Element>(c), n},
      typename EpilogueOp::Params(1.0f),
      split_k);
  int workspace_err = validate_split_k_workspace(Gemm::get_workspace_size(args), workspace, workspace_nbytes);
  if (workspace_err) {
    return workspace_err;
  }
  cutlass::Status status = gemm(args, workspace, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
int launch_gemm_bias_residual(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage const* d0,
    Storage const* d1,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    cudaStream_t stream) {
  if (a == nullptr || b == nullptr || bias == nullptr || d0 == nullptr || c == nullptr) {
    return 1;
  }
  if constexpr (!EpilogueOp::kIsSingleSource) {
    if (d1 == nullptr) {
      return 1;
    }
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  using Gemm = BroadcastDeviceGemm<Element, LayoutB, EpilogueOp, Policy>;
  Gemm gemm;
  if constexpr (EpilogueOp::kIsSingleSource) {
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename EpilogueOp::Params(1.0f),
        cutlass_ptr<Storage, Element>(a),
        cutlass_ptr<Storage, Element>(b),
        cutlass_ptr<Storage, Element>(d0),
        cutlass_ptr<Storage, Element>(c),
        const_cast<Element*>(cutlass_ptr<Storage, Element>(bias)),
        nullptr,
        static_cast<int64_t>(m) * k,
        static_cast<int64_t>(n) * k,
        static_cast<int64_t>(m) * n,
        static_cast<int64_t>(m) * n,
        0,
        static_cast<int64_t>(m) * n,
        k,
        ldb,
        n,
        n,
        0,
        n);
    cutlass::Status status = gemm(args, nullptr, stream);
    return status == cutlass::Status::kSuccess ? 0 : 3;
  } else {
    typename Gemm::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k},
        1,
        typename EpilogueOp::Params(1.0f),
        cutlass_ptr<Storage, Element>(a),
        cutlass_ptr<Storage, Element>(b),
        cutlass_ptr<Storage, Element>(d0),
        cutlass_ptr<Storage, Element>(d1),
        cutlass_ptr<Storage, Element>(c),
        const_cast<Element*>(cutlass_ptr<Storage, Element>(bias)),
        nullptr,
        static_cast<int64_t>(m) * k,
        static_cast<int64_t>(n) * k,
        static_cast<int64_t>(m) * n,
        static_cast<int64_t>(m) * n,
        static_cast<int64_t>(m) * n,
        0,
        static_cast<int64_t>(m) * n,
        k,
        ldb,
        n,
        n,
        n,
        0,
        n);
    cutlass::Status status = gemm(args, nullptr, stream);
    return status == cutlass::Status::kSuccess ? 0 : 3;
  }
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
size_t gemm_bias_residual_policy_workspace_size(
    int m,
    int n,
    int k,
    int ldb,
    int split_k) {
  if constexpr (!EpilogueOp::kSupportsSerialSplitK) {
    return 0;
  } else {
    if (m <= 0 || n <= 0 || k <= 0 || split_k <= 1) {
      return 0;
    }
    using Gemm = BroadcastDeviceGemm<Element, LayoutB, EpilogueOp, Policy>;
    if constexpr (EpilogueOp::kIsSingleSource) {
      typename Gemm::Arguments args(
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k},
          split_k,
          typename EpilogueOp::Params(1.0f),
          static_cast<Element const*>(nullptr),
          static_cast<Element const*>(nullptr),
          static_cast<Element const*>(nullptr),
          static_cast<Element*>(nullptr),
          static_cast<Element*>(nullptr),
          nullptr,
          static_cast<int64_t>(m) * k,
          static_cast<int64_t>(n) * k,
          static_cast<int64_t>(m) * n,
          static_cast<int64_t>(m) * n,
          0,
          static_cast<int64_t>(m) * n,
          k,
          ldb,
          n,
          n,
          0,
          n);
      return Gemm::get_workspace_size(args);
    } else {
      typename Gemm::Arguments args(
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k},
          split_k,
          typename EpilogueOp::Params(1.0f),
          static_cast<Element const*>(nullptr),
          static_cast<Element const*>(nullptr),
          static_cast<Element const*>(nullptr),
          static_cast<Element const*>(nullptr),
          static_cast<Element*>(nullptr),
          static_cast<Element*>(nullptr),
          nullptr,
          static_cast<int64_t>(m) * k,
          static_cast<int64_t>(n) * k,
          static_cast<int64_t>(m) * n,
          static_cast<int64_t>(m) * n,
          static_cast<int64_t>(m) * n,
          0,
          static_cast<int64_t>(m) * n,
          k,
          ldb,
          n,
          n,
          n,
          0,
          n);
      return Gemm::get_workspace_size(args);
    }
  }
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
int launch_gemm_bias_residual_splitk(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage const* d0,
    Storage const* d1,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int split_k,
    void* workspace,
    size_t workspace_nbytes,
    cudaStream_t stream) {
  if constexpr (!EpilogueOp::kSupportsSerialSplitK) {
    return 6;
  } else {
    if (a == nullptr || b == nullptr || bias == nullptr || d0 == nullptr || c == nullptr) {
      return 1;
    }
    if constexpr (!EpilogueOp::kIsSingleSource) {
      if (d1 == nullptr) {
        return 1;
      }
    }
    if (m <= 0 || n <= 0 || k <= 0 || split_k <= 0) {
      return 2;
    }
    using Gemm = BroadcastDeviceGemm<Element, LayoutB, EpilogueOp, Policy>;
    Gemm gemm;
    if constexpr (EpilogueOp::kIsSingleSource) {
      typename Gemm::Arguments args(
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k},
          split_k,
          typename EpilogueOp::Params(1.0f),
          cutlass_ptr<Storage, Element>(a),
          cutlass_ptr<Storage, Element>(b),
          cutlass_ptr<Storage, Element>(d0),
          cutlass_ptr<Storage, Element>(c),
          const_cast<Element*>(cutlass_ptr<Storage, Element>(bias)),
          nullptr,
          static_cast<int64_t>(m) * k,
          static_cast<int64_t>(n) * k,
          static_cast<int64_t>(m) * n,
          static_cast<int64_t>(m) * n,
          0,
          static_cast<int64_t>(m) * n,
          k,
          ldb,
          n,
          n,
          0,
          n);
      int workspace_err = validate_split_k_workspace(Gemm::get_workspace_size(args), workspace, workspace_nbytes);
      if (workspace_err) {
        return workspace_err;
      }
      cutlass::Status status = gemm(args, workspace, stream);
      return status == cutlass::Status::kSuccess ? 0 : 3;
    } else {
      typename Gemm::Arguments args(
          cutlass::gemm::GemmUniversalMode::kGemm,
          {m, n, k},
          split_k,
          typename EpilogueOp::Params(1.0f),
          cutlass_ptr<Storage, Element>(a),
          cutlass_ptr<Storage, Element>(b),
          cutlass_ptr<Storage, Element>(d0),
          cutlass_ptr<Storage, Element>(d1),
          cutlass_ptr<Storage, Element>(c),
          const_cast<Element*>(cutlass_ptr<Storage, Element>(bias)),
          nullptr,
          static_cast<int64_t>(m) * k,
          static_cast<int64_t>(n) * k,
          static_cast<int64_t>(m) * n,
          static_cast<int64_t>(m) * n,
          static_cast<int64_t>(m) * n,
          0,
          static_cast<int64_t>(m) * n,
          k,
          ldb,
          n,
          n,
          n,
          0,
          n);
      int workspace_err = validate_split_k_workspace(Gemm::get_workspace_size(args), workspace, workspace_nbytes);
      if (workspace_err) {
        return workspace_err;
      }
      cutlass::Status status = gemm(args, workspace, stream);
      return status == cutlass::Status::kSuccess ? 0 : 3;
    }
  }
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

template <typename Storage, typename Element, typename LayoutB, typename Policy>
float profile_gemm_policy(
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
  launch_gemm_policy<Storage, Element, LayoutB, Policy>(a, b, c, m, n, k, ldb, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    launch_gemm_policy<Storage, Element, LayoutB, Policy>(a, b, c, m, n, k, ldb, stream);
  }
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, typename Element, typename LayoutB, typename Policy>
float profile_gemm_policy_splitk(
    Storage const* a,
    Storage const* b,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int split_k,
    void* workspace,
    size_t workspace_nbytes,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  if (launch_gemm_policy_splitk<Storage, Element, LayoutB, Policy>(
          a, b, c, m, n, k, ldb, split_k, workspace, workspace_nbytes, stream)) {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return -1.0f;
  }
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    if (launch_gemm_policy_splitk<Storage, Element, LayoutB, Policy>(
            a, b, c, m, n, k, ldb, split_k, workspace, workspace_nbytes, stream)) {
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

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
float profile_gemm_bias_policy(
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
  launch_gemm_bias<Storage, Element, LayoutB, EpilogueOp, Policy>(a, b, bias, c, m, n, k, ldb, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    launch_gemm_bias<Storage, Element, LayoutB, EpilogueOp, Policy>(a, b, bias, c, m, n, k, ldb, stream);
  }
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
float profile_gemm_bias_policy_splitk(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int split_k,
    void* workspace,
    size_t workspace_nbytes,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  if (launch_gemm_bias_splitk<Storage, Element, LayoutB, EpilogueOp, Policy>(
          a, b, bias, c, m, n, k, ldb, split_k, workspace, workspace_nbytes, stream)) {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return -1.0f;
  }
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    if (launch_gemm_bias_splitk<Storage, Element, LayoutB, EpilogueOp, Policy>(
            a, b, bias, c, m, n, k, ldb, split_k, workspace, workspace_nbytes, stream)) {
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

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
float profile_gemm_bias_residual_policy(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage const* d0,
    Storage const* d1,
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
  launch_gemm_bias_residual<Storage, Element, LayoutB, EpilogueOp, Policy>(a, b, bias, d0, d1, c, m, n, k, ldb, stream);
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    launch_gemm_bias_residual<Storage, Element, LayoutB, EpilogueOp, Policy>(a, b, bias, d0, d1, c, m, n, k, ldb, stream);
  }
  cudaEventRecord(end, stream);
  cudaEventSynchronize(end);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, end);
  cudaEventDestroy(start);
  cudaEventDestroy(end);
  return ms / static_cast<float>(iterations);
}

template <typename Storage, typename Element, typename LayoutB, typename EpilogueOp, typename Policy>
float profile_gemm_bias_residual_policy_splitk(
    Storage const* a,
    Storage const* b,
    Storage const* bias,
    Storage const* d0,
    Storage const* d1,
    Storage* c,
    int m,
    int n,
    int k,
    int ldb,
    int split_k,
    void* workspace,
    size_t workspace_nbytes,
    int iterations,
    cudaStream_t stream) {
  if constexpr (!EpilogueOp::kSupportsSerialSplitK) {
    return -1.0f;
  } else {
    if (iterations <= 0) {
      iterations = 20;
    }
    cudaEvent_t start;
    cudaEvent_t end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    if (launch_gemm_bias_residual_splitk<Storage, Element, LayoutB, EpilogueOp, Policy>(
            a, b, bias, d0, d1, c, m, n, k, ldb, split_k, workspace, workspace_nbytes, stream)) {
      cudaEventDestroy(start);
      cudaEventDestroy(end);
      return -1.0f;
    }
    cudaEventRecord(start, stream);
    for (int i = 0; i < iterations; ++i) {
      if (launch_gemm_bias_residual_splitk<Storage, Element, LayoutB, EpilogueOp, Policy>(
              a, b, bias, d0, d1, c, m, n, k, ldb, split_k, workspace, workspace_nbytes, stream)) {
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
}

template <typename Storage, typename Element, typename Epilogue2, typename Policy, bool UseBias, bool BroadcastB1>
int launch_dual_gemm_impl(
    Storage const* a,
    Storage const* b0,
    Storage const* b1,
    Storage const* bias0,
    Storage const* bias1,
    Storage* c,
    int m,
    int n,
    int k,
    cudaStream_t stream) {
  if (a == nullptr || b0 == nullptr || b1 == nullptr || c == nullptr) {
    return 1;
  }
  if (m <= 0 || n <= 0 || k <= 0) {
    return 2;
  }
  using NoBiasEpilogue = cutlass::epilogue::thread::LinearCombination<
      Element,
      1,
      typename Policy::ElementAccumulator,
      float,
      cutlass::epilogue::thread::ScaleType::OnlyAlphaScaling>;
  using BiasAddEpilogue = cutlass::epilogue::thread::LinearCombination<
      Element,
      1,
      typename Policy::ElementAccumulator,
      float,
      cutlass::epilogue::thread::ScaleType::NoBetaScaling>;
  using OutputOp = std::conditional_t<UseBias, BiasAddEpilogue, NoBiasEpilogue>;
  using DualGemm = cutlass::gemm::device::DualGemm<
      Element,
      cutlass::layout::RowMajor,
      Element,
      cutlass::layout::ColumnMajor,
      cutlass::layout::ColumnMajor,
      Element,
      cutlass::layout::RowMajor,
      typename Policy::ElementAccumulator,
      typename Policy::OperatorClass,
      typename Policy::ArchTag,
      typename Policy::ThreadblockShape,
      typename Policy::WarpShape,
      typename Policy::InstructionShape,
      OutputOp,
      OutputOp,
      Epilogue2,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      Policy::kStages,
      false,
      false,
      false,
      Policy::kAlignmentA,
      Policy::kAlignmentB,
      typename Policy::Operator>;

  DualGemm dual_gemm;
  int const b1_ldb = BroadcastB1 ? 0 : k;
  typename cutlass::TensorRef<Element, cutlass::layout::RowMajor> nullptr_ref{};
  typename cutlass::TensorRef<Element const, cutlass::layout::RowMajor> bias0_ref{};
  typename cutlass::TensorRef<Element const, cutlass::layout::RowMajor> bias1_ref{};
  if constexpr (UseBias) {
    if (bias0 == nullptr || bias1 == nullptr) {
      return 1;
    }
    bias0_ref = {cutlass_ptr<Storage, Element>(bias0), 0};
    bias1_ref = {cutlass_ptr<Storage, Element>(bias1), 0};
  }
  typename DualGemm::Arguments args(
      cutlass::gemm::DualGemmMode::kGemm,
      {m, n, k},
      {cutlass_ptr<Storage, Element>(a), k},
      {cutlass_ptr<Storage, Element>(b0), k},
      bias0_ref,
      nullptr_ref,
      {cutlass_ptr<Storage, Element>(b1), b1_ldb},
      bias1_ref,
      nullptr_ref,
      {cutlass_ptr<Storage, Element>(c), n},
      {1.0f, UseBias ? 1.0f : 0.0f},
      {1.0f, UseBias ? 1.0f : 0.0f},
      {},
      1,
      1,
      0,
      0,
      0,
      0,
      0);
  cutlass::Status status = dual_gemm(args, nullptr, stream);
  return status == cutlass::Status::kSuccess ? 0 : 3;
}

template <typename Storage, typename Element, typename Epilogue2, typename Policy, bool UseBias>
int launch_dual_gemm_policy(
    Storage const* a,
    Storage const* b0,
    Storage const* b1,
    Storage const* bias0,
    Storage const* bias1,
    Storage* c,
    int m,
    int n,
    int k,
    int b1_n,
    cudaStream_t stream) {
  if (b1_n == n) {
    return launch_dual_gemm_impl<Storage, Element, Epilogue2, Policy, UseBias, false>(
        a, b0, b1, bias0, bias1, c, m, n, k, stream);
  }
  if (b1_n == 1) {
    return launch_dual_gemm_impl<Storage, Element, Epilogue2, Policy, UseBias, true>(
        a, b0, b1, bias0, bias1, c, m, n, k, stream);
  }
  return 2;
}

template <typename Storage, typename Element, typename Epilogue2, typename Policy, bool UseBias>
float profile_dual_gemm_policy(
    Storage const* a,
    Storage const* b0,
    Storage const* b1,
    Storage const* bias0,
    Storage const* bias1,
    Storage* c,
    int m,
    int n,
    int k,
    int b1_n,
    int iterations,
    cudaStream_t stream) {
  if (iterations <= 0) {
    iterations = 20;
  }
  cudaEvent_t start;
  cudaEvent_t end;
  cudaEventCreate(&start);
  cudaEventCreate(&end);
  if (launch_dual_gemm_policy<Storage, Element, Epilogue2, Policy, UseBias>(
          a, b0, b1, bias0, bias1, c, m, n, k, b1_n, stream)) {
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    return -1.0f;
  }
  cudaEventRecord(start, stream);
  for (int i = 0; i < iterations; ++i) {
    if (launch_dual_gemm_policy<Storage, Element, Epilogue2, Policy, UseBias>(
            a, b0, b1, bias0, bias1, c, m, n, k, b1_n, stream)) {
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


#define DINOML_CUTLASS_GENERATED_EXPORTS 1

#define DINOML_FORWARD_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, OLD_SUFFIX, SYMBOL_ID, POLICY, ALIGN) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return launch_gemm_policy<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, c, m, n, k, DINOML_LDB_##OP, stream); \
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
  return profile_gemm_policy<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, c, m, n, k, DINOML_LDB_##OP, iterations, stream); \
} \
extern "C" size_t dinoml_cutlass_workspace_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    int m, \
    int n, \
    int k, \
    int split_k) { \
  return gemm_policy_workspace_size<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, AlignedGemmPolicy<POLICY, ALIGN>>(m, n, k, DINOML_LDB_##OP, split_k); \
} \
extern "C" int dinoml_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    cudaStream_t stream) { \
  return launch_gemm_policy_splitk<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, c, m, n, k, DINOML_LDB_##OP, split_k, workspace, workspace_nbytes, stream); \
} \
extern "C" float dinoml_profile_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_gemm_policy_splitk<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, c, m, n, k, DINOML_LDB_##OP, split_k, workspace, workspace_nbytes, iterations, stream); \
}

#define DINOML_FORWARD_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, OLD_SUFFIX, SYMBOL_ID, POLICY, ALIGN) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return launch_gemm_bias<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, DINOML_BIAS_EPILOGUE_##OP<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, DINOML_LDB_##OP, stream); \
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
  return profile_gemm_bias_policy<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, DINOML_BIAS_EPILOGUE_##OP<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, DINOML_LDB_##OP, iterations, stream); \
} \
extern "C" size_t dinoml_cutlass_workspace_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    int m, \
    int n, \
    int k, \
    int split_k) { \
  return gemm_bias_policy_workspace_size<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, DINOML_BIAS_EPILOGUE_##OP<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(m, n, k, DINOML_LDB_##OP, split_k); \
} \
extern "C" int dinoml_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    cudaStream_t stream) { \
  return launch_gemm_bias_splitk<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, DINOML_BIAS_EPILOGUE_##OP<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, DINOML_LDB_##OP, split_k, workspace, workspace_nbytes, stream); \
} \
extern "C" float dinoml_profile_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_gemm_bias_policy_splitk<CTYPE, ELEMENT, DINOML_LAYOUT_B_##OP, DINOML_BIAS_EPILOGUE_##OP<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, DINOML_LDB_##OP, split_k, workspace, workspace_nbytes, iterations, stream); \
}

#define DINOML_FORWARD_GEMM_BIAS_ACTIVATION_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, LAYOUT_B, LDB, EPILOGUE, SYMBOL_ID, POLICY, ALIGN) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return launch_gemm_bias<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, LDB, stream); \
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
  return profile_gemm_bias_policy<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, LDB, iterations, stream); \
} \
extern "C" size_t dinoml_cutlass_workspace_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    int m, \
    int n, \
    int k, \
    int split_k) { \
  return gemm_bias_policy_workspace_size<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(m, n, k, LDB, split_k); \
} \
extern "C" int dinoml_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    cudaStream_t stream) { \
  return launch_gemm_bias_splitk<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, LDB, split_k, workspace, workspace_nbytes, stream); \
} \
extern "C" float dinoml_profile_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_gemm_bias_policy_splitk<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, c, m, n, k, LDB, split_k, workspace, workspace_nbytes, iterations, stream); \
}

#define DINOML_FORWARD_GEMM_BIAS_RESIDUAL_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, LAYOUT_B, LDB, EPILOGUE, SYMBOL_ID, POLICY, ALIGN) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return launch_gemm_bias_residual<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, nullptr, c, m, n, k, LDB, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_gemm_bias_residual_policy<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, nullptr, c, m, n, k, LDB, iterations, stream); \
} \
extern "C" size_t dinoml_cutlass_workspace_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    int m, \
    int n, \
    int k, \
    int split_k) { \
  return gemm_bias_residual_policy_workspace_size<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(m, n, k, LDB, split_k); \
} \
extern "C" int dinoml_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    cudaStream_t stream) { \
  return launch_gemm_bias_residual_splitk<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, nullptr, c, m, n, k, LDB, split_k, workspace, workspace_nbytes, stream); \
} \
extern "C" float dinoml_profile_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_gemm_bias_residual_policy_splitk<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, nullptr, c, m, n, k, LDB, split_k, workspace, workspace_nbytes, iterations, stream); \
}

#define DINOML_FORWARD_GEMM_BIAS_RESIDUAL2_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, LAYOUT_B, LDB, EPILOGUE, SYMBOL_ID, POLICY, ALIGN) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE const* d1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    cudaStream_t stream) { \
  return launch_gemm_bias_residual<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, d1, c, m, n, k, LDB, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
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
    cudaStream_t stream) { \
  return profile_gemm_bias_residual_policy<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, d1, c, m, n, k, LDB, iterations, stream); \
} \
extern "C" size_t dinoml_cutlass_workspace_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    int m, \
    int n, \
    int k, \
    int split_k) { \
  return gemm_bias_residual_policy_workspace_size<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(m, n, k, LDB, split_k); \
} \
extern "C" int dinoml_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE const* d1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    cudaStream_t stream) { \
  return launch_gemm_bias_residual_splitk<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, d1, c, m, n, k, LDB, split_k, workspace, workspace_nbytes, stream); \
} \
extern "C" float dinoml_profile_cutlass_splitk_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b, \
    CTYPE const* bias, \
    CTYPE const* d0, \
    CTYPE const* d1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int split_k, \
    void* workspace, \
    size_t workspace_nbytes, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_gemm_bias_residual_policy_splitk<CTYPE, ELEMENT, LAYOUT_B, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>>(a, b, bias, d0, d1, c, m, n, k, LDB, split_k, workspace, workspace_nbytes, iterations, stream); \
}

#define DINOML_FORWARD_DUAL_GEMM_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, EPILOGUE, SYMBOL_ID, POLICY, ALIGN) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b0, \
    CTYPE const* b1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int b1_n, \
    cudaStream_t stream) { \
  return launch_dual_gemm_policy<CTYPE, ELEMENT, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>, false>(a, b0, b1, nullptr, nullptr, c, m, n, k, b1_n, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b0, \
    CTYPE const* b1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int b1_n, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_dual_gemm_policy<CTYPE, ELEMENT, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>, false>(a, b0, b1, nullptr, nullptr, c, m, n, k, b1_n, iterations, stream); \
}

#define DINOML_FORWARD_DUAL_GEMM_BIAS_EXPORT(OP, DTYPE_NAME, CTYPE, ELEMENT, EPILOGUE, SYMBOL_ID, POLICY, ALIGN) \
extern "C" int dinoml_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b0, \
    CTYPE const* b1, \
    CTYPE const* bias0, \
    CTYPE const* bias1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int b1_n, \
    cudaStream_t stream) { \
  return launch_dual_gemm_policy<CTYPE, ELEMENT, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>, true>(a, b0, b1, bias0, bias1, c, m, n, k, b1_n, stream); \
} \
extern "C" float dinoml_profile_cutlass_##OP##_##DTYPE_NAME##_##SYMBOL_ID( \
    CTYPE const* a, \
    CTYPE const* b0, \
    CTYPE const* b1, \
    CTYPE const* bias0, \
    CTYPE const* bias1, \
    CTYPE* c, \
    int m, \
    int n, \
    int k, \
    int b1_n, \
    int iterations, \
    cudaStream_t stream) { \
  return profile_dual_gemm_policy<CTYPE, ELEMENT, EPILOGUE<ELEMENT, AlignedGemmPolicy<POLICY, ALIGN>::ElementAccumulator>, AlignedGemmPolicy<POLICY, ALIGN>, true>(a, b0, b1, bias0, bias1, c, m, n, k, b1_n, iterations, stream); \
}
