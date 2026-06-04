#pragma once

#include <dinoml/math.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dinoml::ck_bmm_profiler {

struct BmmRequest {
  std::string profiler_symbol;
  std::vector<std::string> profiler_symbols;
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
  throw std::runtime_error("Unsupported CK BMM profiler dtype: " + dtype);
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

inline std::vector<std::string> requested_profiler_symbols(const BmmRequest& request) {
  if (!request.profiler_symbols.empty()) {
    return request.profiler_symbols;
  }
  if (!request.profiler_symbol.empty()) {
    return {request.profiler_symbol};
  }
  return {};
}

inline float run_candidate(
    const BmmRequest& request,
    const std::string& profiler_symbol,
    void* a,
    void* b,
    void* d0,
    void* c) {
  if (request.residual_count == 0) {
    using Fn = float (*)(
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
        int,
        int,
        int,
        int,
        hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
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
        hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
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
  throw std::runtime_error("CK BMM profiler supports at most one residual input");
}

inline std::vector<BmmResult> profile_bmm(const BmmRequest& request, std::uint32_t seed) {
  const auto profiler_symbols = requested_profiler_symbols(request);
  if (profiler_symbols.empty()) {
    throw std::runtime_error("CK BMM profiler symbol is required");
  }
  if (request.batch_count <= 0 || request.m <= 0 || request.n <= 0 || request.k <= 0 || request.iterations <= 0 ||
      request.repeats <= 0) {
    throw std::runtime_error("CK BMM profiler dimensions, iterations, and repeats must be positive");
  }
  if (request.residual_count < 0 || request.residual_count > 1) {
    throw std::runtime_error("CK BMM profiler supports at most one residual input");
  }

  std::mt19937 rng(seed);
  DeviceBuffer a(request.a_elements * dtype_size(request.dtype));
  DeviceBuffer b(request.b_elements * dtype_size(request.dtype));
  DeviceBuffer c(request.c_elements * dtype_size(request.dtype));
  a.copy_from(random_storage(request.a_elements, request.dtype, rng));
  b.copy_from(random_storage(request.b_elements, request.dtype, rng));
  check_hip(hipMemset(c.get(), 0, request.c_elements * dtype_size(request.dtype)), "hipMemset output");

  DeviceBuffer d0;
  if (request.residual_count == 1) {
    d0 = DeviceBuffer(request.d0_elements * dtype_size(request.dtype));
    d0.copy_from(random_storage(request.d0_elements, request.dtype, rng));
  }

  std::vector<BmmResult> results;
  results.reserve(profiler_symbols.size());
  bool all_ok = true;
  for (const auto& profiler_symbol : profiler_symbols) {
    BmmResult result;
    result.profiler_symbol = profiler_symbol;
    result.samples_ms.reserve(static_cast<std::size_t>(request.repeats));
    try {
      for (int repeat = 0; repeat < request.repeats; ++repeat) {
        const float elapsed_ms = run_candidate(request, profiler_symbol, a.get(), b.get(), d0.get(), c.get());
        if (!(elapsed_ms >= 0.0f)) {
          throw std::runtime_error("CK BMM profiler candidate failed: " + profiler_symbol);
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

}  // namespace dinoml::ck_bmm_profiler
