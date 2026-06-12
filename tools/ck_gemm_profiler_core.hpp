#pragma once

#include <dinoml/math.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dinoml::ck_gemm_profiler {

struct GemmRequest {
  std::string profiler_symbol;
  std::vector<std::string> profiler_symbols;
  std::string dtype;
  int m = 0;
  int n = 0;
  int k = 0;
  bool is_dual = false;
  int b1_n = 0;
  int iterations = 1;
  int repeats = 1;
  bool has_bias = false;
  int residual_count = 0;
};

struct GemmResult {
  std::string profiler_symbol;
  float elapsed_ms = 0.0f;
  std::vector<float> samples_ms;
  std::size_t workspace_nbytes = 0;
  bool ok = true;
  std::string error;
};

inline void check_hip(hipError_t err, const char* what) {
  if (err != hipSuccess) {
    throw std::runtime_error(std::string(what) + ": " + hipGetErrorString(err));
  }
}

inline std::size_t dtype_size(const std::string& dtype) {
  if (dtype == "float32") {
    return sizeof(float);
  }
  if (dtype == "float16") {
    return sizeof(dinoml::math::float16);
  }
  if (dtype == "bfloat16") {
    return sizeof(dinoml::math::bfloat16);
  }
  throw std::runtime_error("Unsupported CK GEMM profiler dtype: " + dtype);
}

inline std::vector<std::uint8_t> random_storage(std::size_t count, const std::string& dtype, std::mt19937& rng) {
  std::normal_distribution<float> dist(0.0f, 0.125f);
  std::vector<std::uint8_t> storage(count * dtype_size(dtype));
  if (dtype == "float32") {
    auto* out = reinterpret_cast<float*>(storage.data());
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = dist(rng);
    }
    return storage;
  }
  if (dtype == "float16") {
    auto* out = reinterpret_cast<dinoml::math::float16*>(storage.data());
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = dinoml::math::cast<dinoml::math::float16>(dist(rng));
    }
    return storage;
  }
  auto* out = reinterpret_cast<dinoml::math::bfloat16*>(storage.data());
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = dinoml::math::cast<dinoml::math::bfloat16>(dist(rng));
  }
  return storage;
}

class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(std::size_t nbytes) {
    if (nbytes != 0) {
      check_hip(hipMalloc(&ptr_, nbytes), "hipMalloc");
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
      check_hip(hipMemcpy(ptr_, host.data(), host.size(), hipMemcpyHostToDevice), "hipMemcpy H2D");
    }
  }

 private:
  void reset() {
    if (ptr_ != nullptr) {
      hipFree(ptr_);
      ptr_ = nullptr;
    }
  }

  void* ptr_ = nullptr;
};

void* resolve_profile_symbol(const std::string& symbol);

inline std::vector<std::string> requested_profiler_symbols(const GemmRequest& request) {
  if (!request.profiler_symbols.empty()) {
    return request.profiler_symbols;
  }
  if (!request.profiler_symbol.empty()) {
    return {request.profiler_symbol};
  }
  return {};
}

inline float run_candidate(
    const GemmRequest& request,
    const std::string& profiler_symbol,
    void* a,
    void* b0,
    void* b1,
    void* bias,
    void* bias1,
    const std::vector<DeviceBuffer>& residuals,
    void* c) {
  if (request.is_dual) {
    if (request.residual_count != 0) {
      throw std::runtime_error("CK dual GEMM profiler does not support residual inputs");
    }
    if (!request.has_bias) {
      using Fn = float (*)(const void*, const void*, const void*, void*, int, int, int, int, int, hipStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
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
        hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        a, b0, b1, bias, bias1, c, request.m, request.n, request.k, request.b1_n, request.iterations, nullptr);
  }
  if (!request.has_bias && request.residual_count == 0) {
    using Fn = float (*)(const void*, const void*, void*, int, int, int, int, hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        a, b0, c, request.m, request.n, request.k, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 0) {
    using Fn = float (*)(const void*, const void*, const void*, void*, int, int, int, int, hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        a, b0, bias, c, request.m, request.n, request.k, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 1) {
    using Fn = float (*)(const void*, const void*, const void*, const void*, void*, int, int, int, int, hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        a, b0, bias, residuals[0].get(), c, request.m, request.n, request.k, request.iterations, nullptr);
  }
  if (request.has_bias && request.residual_count == 2) {
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
        hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        a,
        b0,
        bias,
        residuals[0].get(),
        residuals[1].get(),
        c,
        request.m,
        request.n,
        request.k,
        request.iterations,
        nullptr);
  }
  throw std::runtime_error("Unsupported CK GEMM profiler epilogue input combination");
}

inline std::vector<GemmResult> profile_gemm(const GemmRequest& request, std::uint32_t seed) {
  const auto profiler_symbols = requested_profiler_symbols(request);
  if (profiler_symbols.empty()) {
    throw std::runtime_error("CK GEMM profiler symbol is required");
  }
  if (request.m <= 0 || request.n <= 0 || request.k <= 0 || request.iterations <= 0 || request.repeats <= 0) {
    throw std::runtime_error("CK GEMM profiler dimensions, iterations, and repeats must be positive");
  }
  if (request.residual_count < 0 || request.residual_count > 2) {
    throw std::runtime_error("CK GEMM profiler supports at most two residual inputs");
  }
  if (request.is_dual && request.b1_n <= 0) {
    throw std::runtime_error("CK dual GEMM profiler requires positive b1_n");
  }
  if (!request.has_bias && request.residual_count != 0) {
    throw std::runtime_error("CK GEMM residual profiling requires bias");
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
  check_hip(hipMemset(c.get(), 0, c_count * dtype_size(request.dtype)), "hipMemset output");

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
  residuals.reserve(static_cast<std::size_t>(request.residual_count));
  for (int i = 0; i < request.residual_count; ++i) {
    residuals.emplace_back(c_count * dtype_size(request.dtype));
    residuals.back().copy_from(random_storage(c_count, request.dtype, rng));
  }

  std::vector<GemmResult> results;
  results.reserve(profiler_symbols.size());
  bool all_ok = true;
  for (const auto& profiler_symbol : profiler_symbols) {
    GemmResult result;
    result.profiler_symbol = profiler_symbol;
    result.samples_ms.reserve(static_cast<std::size_t>(request.repeats));
    try {
      for (int repeat = 0; repeat < request.repeats; ++repeat) {
        const float elapsed_ms =
            run_candidate(request, profiler_symbol, a.get(), b0.get(), b1.get(), bias.get(), bias1.get(), residuals, c.get());
        if (!(elapsed_ms >= 0.0f)) {
          throw std::runtime_error("CK GEMM profiler candidate failed: " + profiler_symbol);
        }
        result.samples_ms.push_back(elapsed_ms);
      }
      result.elapsed_ms =
          result.samples_ms.empty() ? 0.0f : *std::min_element(result.samples_ms.begin(), result.samples_ms.end());
    } catch (const std::exception& exc) {
      result.ok = false;
      result.error = exc.what();
      result.elapsed_ms = -1.0f;
      result.samples_ms.clear();
      all_ok = false;
    }
    results.push_back(std::move(result));
  }
  if (all_ok) {
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");
  }
  return results;
}

}  // namespace dinoml::ck_gemm_profiler
