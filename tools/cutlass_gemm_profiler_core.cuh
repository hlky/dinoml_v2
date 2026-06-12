#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdint>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dinoml::cutlass_gemm_profiler {

struct GemmRequest {
  std::string dtype;
  int m = 0;
  int n = 0;
  int k = 0;
  bool is_dual = false;
  int b1_n = 0;
  int split_k = 1;
  int iterations = 1;
  int repeats = 1;
  int max_operand_alignment = 0;
  bool has_bias = false;
  int residual_count = 0;
};

struct GemmResult {
  std::string profiler_symbol;
  float elapsed_ms = 0.0f;
  std::vector<float> samples_ms;
  std::size_t workspace_nbytes = 0;
};

struct GemmCandidate {
  const char* profiler_symbol;
  const char* workspace_symbol;
  int align;
  bool supports_split_k;
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
  throw std::runtime_error("Unsupported CUTLASS GEMM profiler dtype: " + dtype);
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
void* resolve_workspace_symbol(const std::string& symbol);
const std::vector<GemmCandidate>& profiler_candidates();

inline std::string splitk_symbol(const std::string& symbol) {
  const std::string prefix = "dinoml_profile_cutlass_";
  if (symbol.rfind(prefix, 0) != 0) {
    throw std::runtime_error("Unsupported CUTLASS GEMM profiler symbol for split-K: " + symbol);
  }
  return "dinoml_profile_cutlass_splitk_" + symbol.substr(prefix.size());
}

inline std::vector<const GemmCandidate*> selected_profiler_candidates(const GemmRequest& request) {
  std::unordered_map<std::string, const GemmCandidate*> best_by_policy;
  std::vector<std::string> policy_order;
  for (const auto& candidate : profiler_candidates()) {
    if (request.max_operand_alignment > 0 && candidate.align > request.max_operand_alignment) {
      continue;
    }
    if (request.split_k > 1 && !candidate.supports_split_k) {
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
  std::vector<const GemmCandidate*> selected;
  selected.reserve(policy_order.size());
  for (const auto& key : policy_order) {
    selected.push_back(best_by_policy.at(key));
  }
  return selected;
}

inline float run_candidate(
    const GemmRequest& request,
    const GemmCandidate& candidate,
    void* a,
    void* b0,
    void* b1,
    void* bias,
    void* bias1,
    const std::vector<DeviceBuffer>& residuals,
    void* c,
    void* workspace,
    std::size_t workspace_nbytes) {
  std::string symbol = candidate.profiler_symbol;
  if (request.split_k > 1) {
    symbol = splitk_symbol(symbol);
  }
  if (request.is_dual) {
    if (request.residual_count != 0) {
      throw std::runtime_error("CUTLASS dual GEMM profiler does not support residual inputs");
    }
    if (request.split_k != 1) {
      throw std::runtime_error("CUTLASS dual GEMM profiler does not support split-K");
    }
    if (!request.has_bias) {
      using Fn = float (*)(const void*, const void*, const void*, void*, int, int, int, int, int, cudaStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
          a, b0, b1, c, request.m, request.n, request.k, request.b1_n, request.iterations, nullptr);
    }
    using Fn = float (*)(
        const void*,
        const void*,
        const void*,
        const void*,
        const void*,
        void*,
        int,
        int,
        int,
        int,
        int,
        cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, b1, bias, bias1, c, request.m, request.n, request.k, request.b1_n, request.iterations, nullptr);
  }
  if (!request.has_bias && request.residual_count == 0 && request.split_k == 1) {
    using Fn = float (*)(const void*, const void*, void*, int, int, int, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, c, request.m, request.n, request.k, request.iterations, nullptr);
  }
  if (!request.has_bias && request.residual_count == 0) {
    using Fn = float (*)(const void*, const void*, void*, int, int, int, int, void*, std::size_t, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, c, request.m, request.n, request.k, request.split_k, workspace, workspace_nbytes, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 0 && request.split_k == 1) {
    using Fn = float (*)(const void*, const void*, const void*, void*, int, int, int, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, bias, c, request.m, request.n, request.k, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 0) {
    using Fn = float (*)(const void*, const void*, const void*, void*, int, int, int, int, void*, std::size_t, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, bias, c, request.m, request.n, request.k, request.split_k, workspace, workspace_nbytes, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 1 && request.split_k == 1) {
    using Fn = float (*)(const void*, const void*, const void*, const void*, void*, int, int, int, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, bias, residuals[0].get(), c, request.m, request.n, request.k, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 1) {
    using Fn = float (*)(const void*, const void*, const void*, const void*, void*, int, int, int, int, void*, std::size_t, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, bias, residuals[0].get(), c, request.m, request.n, request.k, request.split_k, workspace, workspace_nbytes, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 2 && request.split_k == 1) {
    using Fn = float (*)(const void*, const void*, const void*, const void*, const void*, void*, int, int, int, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, bias, residuals[0].get(), residuals[1].get(), c, request.m, request.n, request.k, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 2) {
    using Fn = float (*)(const void*, const void*, const void*, const void*, const void*, void*, int, int, int, int, void*, std::size_t, int, cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(symbol))(
        a, b0, bias, residuals[0].get(), residuals[1].get(), c, request.m, request.n, request.k, request.split_k, workspace, workspace_nbytes, request.iterations, nullptr);
  }
  throw std::runtime_error("Unsupported CUTLASS GEMM profiler epilogue input combination");
}

inline std::vector<GemmResult> profile_gemm(const GemmRequest& request, std::uint32_t seed) {
  if (request.m <= 0 || request.n <= 0 || request.k <= 0 || request.iterations <= 0 || request.split_k <= 0 || request.repeats <= 0) {
    throw std::runtime_error("CUTLASS GEMM profiler dimensions, iterations, split_k, and repeats must be positive");
  }
  if (request.residual_count < 0 || request.residual_count > 2) {
    throw std::runtime_error("CUTLASS GEMM profiler supports at most two residual inputs");
  }
  if (request.is_dual && request.b1_n <= 0) {
    throw std::runtime_error("CUTLASS dual GEMM profiler requires positive b1_n");
  }
  auto candidates = selected_profiler_candidates(request);
  if (candidates.empty()) {
    throw std::runtime_error("CUTLASS GEMM profiler found no candidate for this problem");
  }

  std::mt19937 rng(seed);
  const std::size_t a_count = static_cast<std::size_t>(request.m) * static_cast<std::size_t>(request.k);
  const std::size_t b0_count = static_cast<std::size_t>(request.n) * static_cast<std::size_t>(request.k);
  const std::size_t b1_count = static_cast<std::size_t>(request.is_dual ? request.b1_n : request.n) * static_cast<std::size_t>(request.k);
  const std::size_t c_count = static_cast<std::size_t>(request.m) * static_cast<std::size_t>(request.n);
  DeviceBuffer a(a_count * dtype_size(request.dtype));
  DeviceBuffer b0(b0_count * dtype_size(request.dtype));
  DeviceBuffer b1(request.is_dual ? (b1_count * dtype_size(request.dtype)) : 0);
  DeviceBuffer c(c_count * dtype_size(request.dtype));
  a.copy_from(random_storage(a_count, request.dtype, rng));
  b0.copy_from(random_storage(b0_count, request.dtype, rng));
  if (request.is_dual) {
    b1.copy_from(random_storage(b1_count, request.dtype, rng));
  }
  check_cuda(cudaMemset(c.get(), 0, c_count * dtype_size(request.dtype)), "cudaMemset output");

  DeviceBuffer bias;
  DeviceBuffer bias1;
  if (request.has_bias) {
    bias = DeviceBuffer(static_cast<std::size_t>(request.n) * dtype_size(request.dtype));
    bias.copy_from(random_storage(static_cast<std::size_t>(request.n), request.dtype, rng));
    if (request.is_dual) {
      bias1 = DeviceBuffer(static_cast<std::size_t>(request.b1_n) * dtype_size(request.dtype));
      bias1.copy_from(random_storage(static_cast<std::size_t>(request.b1_n), request.dtype, rng));
    }
  }
  std::vector<DeviceBuffer> residuals;
  for (int i = 0; i < request.residual_count; ++i) {
    residuals.emplace_back(c_count * dtype_size(request.dtype));
    residuals.back().copy_from(random_storage(c_count, request.dtype, rng));
  }

  std::size_t workspace_nbytes = 0;
  if (request.split_k > 1) {
    for (const auto* candidate : candidates) {
      using WorkspaceFn = std::size_t (*)(int, int, int, int);
      auto workspace_fn = reinterpret_cast<WorkspaceFn>(resolve_workspace_symbol(candidate->workspace_symbol));
      workspace_nbytes = std::max(workspace_nbytes, workspace_fn(request.m, request.n, request.k, request.split_k));
    }
  }
  DeviceBuffer workspace;
  if (workspace_nbytes > 0) {
    workspace = DeviceBuffer(workspace_nbytes);
  }

  std::vector<GemmResult> results;
  results.reserve(candidates.size());
  for (const auto* candidate : candidates) {
    GemmResult result;
    result.profiler_symbol = candidate->profiler_symbol;
    result.workspace_nbytes = workspace_nbytes;
    result.samples_ms.reserve(static_cast<std::size_t>(request.repeats));
    for (int repeat = 0; repeat < request.repeats; ++repeat) {
      float elapsed_ms = run_candidate(
          request,
          *candidate,
          a.get(),
          b0.get(),
          b1.get(),
          bias.get(),
          bias1.get(),
          residuals,
          c.get(),
          workspace.get(),
          workspace_nbytes);
      if (elapsed_ms < 0.0f) {
        throw std::runtime_error("CUTLASS GEMM profiler symbol failed: " + result.profiler_symbol);
      }
      result.samples_ms.push_back(elapsed_ms);
    }
    result.elapsed_ms = result.samples_ms.empty() ? 0.0f : result.samples_ms[0];
    results.push_back(std::move(result));
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return results;
}

}  // namespace dinoml::cutlass_gemm_profiler
