#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace dinoml::cutlass_bmm_profiler {

struct BmmRequest {
  std::string dtype;
  int batch_count = 0;
  int m = 0;
  int n = 0;
  int k = 0;
  int64_t batch_stride_a = 0;
  int64_t batch_stride_b = 0;
  int64_t batch_stride_d0 = 0;
  int64_t batch_stride_c = 0;
  int lda = 0;
  int ldb = 0;
  int ldd0 = 0;
  int ldc = 0;
  int iterations = 1;
  int repeats = 1;
  int max_operand_alignment = 0;
  int residual_count = 0;
  std::size_t a_elements = 0;
  std::size_t b_elements = 0;
  std::size_t d0_elements = 0;
  std::size_t c_elements = 0;
};

struct BmmResult {
  std::string profiler_symbol;
  float elapsed_ms = 0.0f;
  std::vector<float> samples_ms;
  std::size_t workspace_nbytes = 0;
};

struct BmmCandidate {
  const char* profiler_symbol;
  int align;
  const char* policy_key;
};

inline void check_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

inline std::size_t dtype_size(const std::string& dtype) {
  if (dtype == "float32") {
    return sizeof(float);
  }
  if (dtype == "float16") {
    return sizeof(__half);
  }
  if (dtype == "bfloat16") {
    return sizeof(__nv_bfloat16);
  }
  throw std::runtime_error("Unsupported CUTLASS BMM profiler dtype: " + dtype);
}

inline std::vector<std::uint8_t> random_storage(std::size_t count, const std::string& dtype, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 0.125f);
  std::vector<std::uint8_t> storage(count * dtype_size(dtype));
  if (dtype == "float32") {
    auto* out = reinterpret_cast<float*>(storage.data());
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = dist(rng);
    }
  } else if (dtype == "float16") {
    auto* out = reinterpret_cast<__half*>(storage.data());
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = __float2half(dist(rng));
    }
  } else {
    auto* out = reinterpret_cast<__nv_bfloat16*>(storage.data());
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = __float2bfloat16(dist(rng));
    }
  }
  return storage;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t nbytes) {
    if (nbytes != 0) {
      check_cuda(cudaMalloc(&ptr_, nbytes), "cudaMalloc");
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      reset();
      ptr_ = other.ptr_;
      other.ptr_ = nullptr;
    }
    return *this;
  }
  ~DeviceBuffer() { reset(); }

  void* get() const { return ptr_; }

  void copy_from(const std::vector<std::uint8_t>& host) {
    if (!host.empty()) {
      check_cuda(cudaMemcpy(ptr_, host.data(), host.size(), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    }
  }

 private:
  void reset() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
      ptr_ = nullptr;
    }
  }

  void* ptr_ = nullptr;
};

void* resolve_profile_symbol(const std::string& symbol);
const std::vector<BmmCandidate>& profiler_candidates();

inline std::vector<const BmmCandidate*> selected_profiler_candidates(const BmmRequest& request) {
  std::unordered_map<std::string, const BmmCandidate*> best_by_policy;
  std::vector<std::string> policy_order;
  for (const auto& candidate : profiler_candidates()) {
    if (request.max_operand_alignment > 0 && candidate.align > request.max_operand_alignment) {
      continue;
    }
    std::string key(candidate.policy_key);
    auto it = best_by_policy.find(key);
    if (it == best_by_policy.end()) {
      policy_order.push_back(key);
      best_by_policy.emplace(key, &candidate);
    } else if (candidate.align > it->second->align) {
      it->second = &candidate;
    }
  }
  std::vector<const BmmCandidate*> selected;
  selected.reserve(policy_order.size());
  for (const auto& key : policy_order) {
    selected.push_back(best_by_policy.at(key));
  }
  return selected;
}

inline float run_candidate(
    const BmmRequest& request,
    const BmmCandidate& candidate,
    void* a,
    void* b,
    void* d0,
    void* c) {
  if (request.residual_count == 0) {
    using Fn = float (*)(
        const void*, const void*, void*, int, int, int, int, int64_t, int64_t, int64_t, int, int, int, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
        a,
        b,
        c,
        request.batch_count,
        request.m,
        request.n,
        request.k,
        request.batch_stride_a,
        request.batch_stride_b,
        request.batch_stride_c,
        request.lda,
        request.ldb,
        request.ldc,
        request.iterations,
        nullptr);
  }
  if (request.residual_count == 1) {
    using Fn = float (*)(
        const void*,
        const void*,
        const void*,
        void*,
        int,
        int,
        int,
        int,
        int64_t,
        int64_t,
        int64_t,
        int64_t,
        int,
        int,
        int,
        int,
        int,
        cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
        a,
        b,
        d0,
        c,
        request.batch_count,
        request.m,
        request.n,
        request.k,
        request.batch_stride_a,
        request.batch_stride_b,
        request.batch_stride_d0,
        request.batch_stride_c,
        request.lda,
        request.ldb,
        request.ldd0,
        request.ldc,
        request.iterations,
        nullptr);
  }
  throw std::runtime_error("CUTLASS BMM profiler supports at most one residual input");
}

inline std::vector<BmmResult> profile_bmm(const BmmRequest& request, std::uint32_t seed) {
  if (request.batch_count <= 0 || request.m <= 0 || request.n <= 0 || request.k <= 0 || request.iterations <= 0 ||
      request.repeats <= 0) {
    throw std::runtime_error("CUTLASS BMM profiler dimensions, iterations, and repeats must be positive");
  }
  auto candidates = selected_profiler_candidates(request);
  if (candidates.empty()) {
    throw std::runtime_error("CUTLASS BMM profiler found no candidate for this problem");
  }

  std::mt19937 rng(seed);
  DeviceBuffer a(request.a_elements * dtype_size(request.dtype));
  DeviceBuffer b(request.b_elements * dtype_size(request.dtype));
  DeviceBuffer c(request.c_elements * dtype_size(request.dtype));
  a.copy_from(random_storage(request.a_elements, request.dtype, rng));
  b.copy_from(random_storage(request.b_elements, request.dtype, rng));
  check_cuda(cudaMemset(c.get(), 0, request.c_elements * dtype_size(request.dtype)), "cudaMemset output");

  DeviceBuffer d0;
  if (request.residual_count == 1) {
    d0 = DeviceBuffer(request.d0_elements * dtype_size(request.dtype));
    d0.copy_from(random_storage(request.d0_elements, request.dtype, rng));
  }

  std::vector<BmmResult> results;
  results.reserve(candidates.size());
  for (const auto* candidate : candidates) {
    BmmResult result;
    result.profiler_symbol = candidate->profiler_symbol;
    result.samples_ms.reserve(static_cast<std::size_t>(request.repeats));
    bool failed = false;
    for (int repeat = 0; repeat < request.repeats; ++repeat) {
      float elapsed_ms = run_candidate(request, *candidate, a.get(), b.get(), d0.get(), c.get());
      if (elapsed_ms < 0.0f) {
        failed = true;
        break;
      }
      result.samples_ms.push_back(elapsed_ms);
    }
    if (failed) {
      continue;
    }
    result.elapsed_ms = result.samples_ms.empty() ? 0.0f : result.samples_ms[0];
    results.push_back(std::move(result));
  }
  if (results.empty()) {
    throw std::runtime_error("CUTLASS BMM profiler failed every candidate for this problem");
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return results;
}

}  // namespace dinoml::cutlass_bmm_profiler
