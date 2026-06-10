#pragma once

#include <dinoml/math.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dinoml::ck_conv_profiler {

struct ConvRequest {
  std::string profiler_symbol;
  std::vector<std::string> profiler_symbols;
  std::string dtype;
  int spatial_rank = 2;
  int batch = 0;
  int in_channels = 0;
  int in_height = 0;
  int in_width = 0;
  int out_channels = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int out_height = 0;
  int out_width = 0;
  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 0;
  int pad_w = 0;
  int output_pad_h = 0;
  int output_pad_w = 0;
  int dilation_h = 1;
  int dilation_w = 1;
  int iterations = 1;
  int repeats = 1;
  bool transposed = false;
  bool has_bias = true;
  bool has_residual = false;
  std::size_t x_elements = 0;
  std::size_t weight_elements = 0;
  std::size_t bias_elements = 0;
  std::size_t residual_elements = 0;
  std::size_t output_elements = 0;
};

struct ConvResult {
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
  throw std::runtime_error("Unsupported CK Conv profiler dtype: " + dtype);
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

inline std::vector<std::string> requested_profiler_symbols(const ConvRequest& request) {
  if (!request.profiler_symbols.empty()) {
    return request.profiler_symbols;
  }
  if (!request.profiler_symbol.empty()) {
    return {request.profiler_symbol};
  }
  return {};
}

inline float run_candidate(
    const ConvRequest& request,
    const std::string& profiler_symbol,
    void* x,
    void* weight,
    void* bias,
    void* residual,
    void* output) {
  if (request.spatial_rank == 1) {
    if (!request.has_bias) {
      throw std::runtime_error("CK Conv1d profiler requires bias-enabled kernels");
    }
    if (!request.has_residual) {
      using Fn = float (*)(const void*, const void*, const void*, void*, int, int, int, int, int, int, int, int, int, int, int, hipStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
          x,
          weight,
          bias,
          output,
          request.batch,
          request.in_channels,
          request.in_width,
          request.out_channels,
          request.kernel_w,
          request.out_width,
          request.stride_w,
          request.pad_w,
          request.dilation_w,
          0,
          request.iterations,
          nullptr);
    }
    using Fn = float (*)(
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
        int,
        int,
        int,
        int,
        int,
        int,
        hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        x,
        weight,
        bias,
        residual,
        output,
        request.batch,
        request.in_channels,
        request.in_width,
        request.out_channels,
        request.kernel_w,
        request.out_width,
        request.stride_w,
        request.pad_w,
        request.dilation_w,
        0,
        request.iterations,
        nullptr);
  }
  if (request.transposed) {
    if (!request.has_bias) {
      using Fn = float (*)(
          const void*,
          const void*,
          void*,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          hipStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
          x,
          weight,
          output,
          request.batch,
          request.in_channels,
          request.in_height,
          request.in_width,
          request.out_channels,
          request.kernel_h,
          request.kernel_w,
          request.out_height,
          request.out_width,
          request.stride_h,
          request.stride_w,
          request.pad_h,
          request.pad_w,
          request.output_pad_h,
          request.output_pad_w,
          request.dilation_h,
          request.dilation_w,
          request.iterations,
          nullptr);
    }
    if (!request.has_residual) {
      using Fn = float (*)(
          const void*,
          const void*,
          const void*,
          void*,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          int,
          hipStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
          x,
          weight,
          bias,
          output,
          request.batch,
          request.in_channels,
          request.in_height,
          request.in_width,
          request.out_channels,
          request.kernel_h,
          request.kernel_w,
          request.out_height,
          request.out_width,
          request.stride_h,
          request.stride_w,
          request.pad_h,
          request.pad_w,
          request.output_pad_h,
          request.output_pad_w,
          request.dilation_h,
          request.dilation_w,
          request.iterations,
          nullptr);
    }
    using Fn = float (*)(
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
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        x,
        weight,
        bias,
        residual,
        output,
        request.batch,
        request.in_channels,
        request.in_height,
        request.in_width,
        request.out_channels,
        request.kernel_h,
        request.kernel_w,
        request.out_height,
        request.out_width,
        request.stride_h,
        request.stride_w,
        request.pad_h,
        request.pad_w,
        request.output_pad_h,
        request.output_pad_w,
        request.dilation_h,
        request.dilation_w,
        request.iterations,
        nullptr);
  }
  if (!request.has_residual) {
    using Fn = float (*)(
        const void*,
        const void*,
        const void*,
        void*,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        int,
        hipStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
        x,
        weight,
        bias,
        output,
        request.batch,
        request.in_channels,
        request.in_height,
        request.in_width,
        request.out_channels,
        request.kernel_h,
        request.kernel_w,
        request.out_height,
        request.out_width,
        request.stride_h,
        request.stride_w,
        request.pad_h,
        request.pad_w,
        request.dilation_h,
        request.dilation_w,
        request.iterations,
        nullptr);
  }
  using Fn = float (*)(
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
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      int,
      hipStream_t);
  return reinterpret_cast<Fn>(resolve_profile_symbol(profiler_symbol))(
      x,
      weight,
      bias,
      residual,
      output,
      request.batch,
      request.in_channels,
      request.in_height,
      request.in_width,
      request.out_channels,
      request.kernel_h,
      request.kernel_w,
      request.out_height,
      request.out_width,
      request.stride_h,
      request.stride_w,
      request.pad_h,
      request.pad_w,
      request.dilation_h,
      request.dilation_w,
      request.iterations,
      nullptr);
}

inline std::vector<ConvResult> profile_conv(const ConvRequest& request, std::uint32_t seed) {
  const auto profiler_symbols = requested_profiler_symbols(request);
  if (profiler_symbols.empty()) {
    throw std::runtime_error("CK Conv profiler symbol is required");
  }
  if (request.spatial_rank <= 0 || request.spatial_rank > 2) {
    throw std::runtime_error("CK Conv profiler spatial_rank must be 1 or 2");
  }
  if (request.batch <= 0 || request.in_channels <= 0 || request.in_height <= 0 || request.in_width <= 0 ||
      request.out_channels <= 0 || request.kernel_h <= 0 || request.kernel_w <= 0 || request.out_height <= 0 ||
      request.out_width <= 0 || request.iterations <= 0 || request.repeats <= 0) {
    throw std::runtime_error("CK Conv profiler dimensions, iterations, and repeats must be positive");
  }

  std::mt19937 rng(seed);
  DeviceBuffer x(request.x_elements * dtype_size(request.dtype));
  DeviceBuffer weight(request.weight_elements * dtype_size(request.dtype));
  DeviceBuffer bias;
  DeviceBuffer output(request.output_elements * dtype_size(request.dtype));
  x.copy_from(random_storage(request.x_elements, request.dtype, rng));
  weight.copy_from(random_storage(request.weight_elements, request.dtype, rng));
  if (request.has_bias) {
    bias = DeviceBuffer(request.bias_elements * dtype_size(request.dtype));
    bias.copy_from(random_storage(request.bias_elements, request.dtype, rng));
  }
  check_hip(hipMemset(output.get(), 0, request.output_elements * dtype_size(request.dtype)), "hipMemset output");

  DeviceBuffer residual;
  if (request.has_residual) {
    residual = DeviceBuffer(request.residual_elements * dtype_size(request.dtype));
    residual.copy_from(random_storage(request.residual_elements, request.dtype, rng));
  }

  std::vector<ConvResult> results;
  results.reserve(profiler_symbols.size());
  bool all_ok = true;
  for (const auto& profiler_symbol : profiler_symbols) {
    ConvResult result;
    result.profiler_symbol = profiler_symbol;
    result.samples_ms.reserve(static_cast<std::size_t>(request.repeats));
    try {
      for (int repeat = 0; repeat < request.repeats; ++repeat) {
        const float elapsed_ms =
            run_candidate(request, profiler_symbol, x.get(), weight.get(), bias.get(), residual.get(), output.get());
        if (!(elapsed_ms >= 0.0f)) {
          throw std::runtime_error("CK Conv profiler candidate failed: " + profiler_symbol);
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

}  // namespace dinoml::ck_conv_profiler
