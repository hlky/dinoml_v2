#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace dinoml::cutlass_conv_profiler {

struct ConvRequest {
  std::string dtype;
  int spatial_rank = 2;
  int n = 0;
  int h = 0;
  int w = 0;
  int c = 0;
  int out_h = 0;
  int out_w = 0;
  int out_c = 0;
  int kernel_h = 0;
  int kernel_w = 0;
  int stride_h = 1;
  int stride_w = 1;
  int pad_h = 0;
  int pad_w = 0;
  int dilation_h = 1;
  int dilation_w = 1;
  int iterations = 1;
  int repeats = 1;
  bool has_bias = true;
  int residual_count = 0;
  std::string validation_mode = "fast";
};

struct ConvResult {
  std::string profiler_symbol;
  std::vector<float> samples_ms;
  std::size_t workspace_nbytes = 0;
};

struct ConvCandidate {
  const char* profiler_symbol;
  const char* kernel_symbol;
  const char* predicate_kind;
  int input_channels;
  int min_input_channels;
  int input_channels_multiple;
  int output_channels_multiple;
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
  throw std::runtime_error("Unsupported CUTLASS Conv profiler dtype: " + dtype);
}

inline std::string normalize_validation_mode(const std::string& validation_mode) {
  if (validation_mode.empty() || validation_mode == "fast") {
    return "fast";
  }
  if (validation_mode == "strict" || validation_mode == "debug") {
    return "strict";
  }
  throw std::runtime_error("Unsupported CUTLASS Conv profiler validation_mode: " + validation_mode);
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
    auto* out = reinterpret_cast<__half*>(storage.data());
    for (std::size_t i = 0; i < count; ++i) {
      out[i] = __float2half(dist(rng));
    }
    return storage;
  }
  auto* out = reinterpret_cast<__nv_bfloat16*>(storage.data());
  for (std::size_t i = 0; i < count; ++i) {
    out[i] = __float2bfloat16(dist(rng));
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
  void fill(int value, std::size_t nbytes) {
    if (nbytes != 0) {
      check_cuda(cudaMemset(ptr_, value, nbytes), "cudaMemset");
    }
  }
  void copy_to(std::vector<std::uint8_t>& host) const {
    if (!host.empty()) {
      check_cuda(cudaMemcpy(host.data(), ptr_, host.size(), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
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
const std::vector<ConvCandidate>& profiler_candidates();

inline bool candidate_matches(const ConvCandidate& candidate, const ConvRequest& request) {
  std::string kind(candidate.predicate_kind);
  if (kind == "fallback") {
    return true;
  }
  if (kind == "semantic_input_channels") {
    return candidate.input_channels == request.c;
  }
  if (kind == "natural_alignment") {
    return request.c >= candidate.min_input_channels &&
        candidate.input_channels_multiple > 0 &&
        candidate.output_channels_multiple > 0 &&
        request.c % candidate.input_channels_multiple == 0 &&
        request.out_c % candidate.output_channels_multiple == 0;
  }
  return false;
}

inline float run_candidate(
    const ConvRequest& request,
    const ConvCandidate& candidate,
    void* activation,
    void* weight,
    void* bias,
    void* residual,
    void* output) {
  if (request.spatial_rank == 1) {
    if (!request.has_bias && request.residual_count == 0) {
      using Fn = float (*)(
          const void*, const void*, void*,
          int, int, int, int, int, int,
          int, int, int, int,
          cudaStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
          activation,
          weight,
          output,
          request.n,
          request.w,
          request.c,
          request.out_w,
          request.out_c,
          request.kernel_w,
          request.stride_w,
          request.pad_w,
          request.dilation_w,
          request.iterations,
          nullptr);
    }
    if (request.residual_count == 0) {
      using Fn = float (*)(
          const void*, const void*, const void*, void*,
          int, int, int, int, int, int,
          int, int, int, int,
          cudaStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
          activation,
          weight,
          bias,
          output,
          request.n,
          request.w,
          request.c,
          request.out_w,
          request.out_c,
          request.kernel_w,
          request.stride_w,
          request.pad_w,
          request.dilation_w,
          request.iterations,
          nullptr);
    }
    if (request.residual_count == 1) {
      using Fn = float (*)(
          const void*, const void*, const void*, const void*, void*,
          int, int, int, int, int, int,
          int, int, int, int,
          cudaStream_t);
      return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
          activation,
          weight,
          bias,
          residual,
          output,
          request.n,
          request.w,
          request.c,
          request.out_w,
          request.out_c,
          request.kernel_w,
          request.stride_w,
          request.pad_w,
          request.dilation_w,
          request.iterations,
          nullptr);
    }
    throw std::runtime_error("CUTLASS Conv profiler supports at most one residual input");
  }
  if (!request.has_bias && request.residual_count == 0) {
    using Fn = float (*)(
        const void*, const void*, void*,
        int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int,
        cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
        activation,
        weight,
        output,
        request.n,
        request.h,
        request.w,
        request.c,
        request.out_h,
        request.out_w,
        request.out_c,
        request.kernel_h,
        request.kernel_w,
        request.stride_h,
        request.stride_w,
        request.pad_h,
        request.pad_w,
        request.dilation_h,
        request.dilation_w,
        request.iterations,
        nullptr);
  }
  if (request.residual_count == 0) {
    using Fn = float (*)(
        const void*, const void*, const void*, void*,
        int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int,
        cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
        activation,
        weight,
        bias,
        output,
        request.n,
        request.h,
        request.w,
        request.c,
        request.out_h,
        request.out_w,
        request.out_c,
        request.kernel_h,
        request.kernel_w,
        request.stride_h,
        request.stride_w,
        request.pad_h,
        request.pad_w,
        request.dilation_h,
        request.dilation_w,
        request.iterations,
        nullptr);
  }
  if (request.residual_count == 1) {
    using Fn = float (*)(
        const void*, const void*, const void*, const void*, void*,
        int, int, int, int, int, int, int, int,
        int, int, int, int, int, int, int, int,
        cudaStream_t);
    return reinterpret_cast<Fn>(resolve_profile_symbol(candidate.profiler_symbol))(
        activation,
        weight,
        bias,
        residual,
        output,
        request.n,
        request.h,
        request.w,
        request.c,
        request.out_h,
        request.out_w,
        request.out_c,
        request.kernel_h,
        request.kernel_w,
        request.stride_h,
        request.stride_w,
        request.pad_h,
        request.pad_w,
        request.dilation_h,
        request.dilation_w,
        request.iterations,
        nullptr);
  }
  throw std::runtime_error("CUTLASS Conv profiler supports at most one residual input");
}

inline std::vector<ConvResult> profile_conv(const ConvRequest& request, std::uint32_t seed) {
  if (request.spatial_rank != 1 && request.spatial_rank != 2) {
    throw std::runtime_error("CUTLASS Conv profiler spatial_rank must be 1 or 2");
  }
  const bool is_conv1d = request.spatial_rank == 1;
  if (request.n <= 0 || request.w <= 0 || request.c <= 0 || request.out_w <= 0 || request.out_c <= 0 ||
      request.kernel_w <= 0 || request.iterations <= 0 || request.repeats <= 0 ||
      (!is_conv1d && (request.h <= 0 || request.out_h <= 0 || request.kernel_h <= 0))) {
    throw std::runtime_error("CUTLASS Conv profiler dimensions, iterations, and repeats must be positive");
  }
  const bool strict_validation = (normalize_validation_mode(request.validation_mode) == "strict");
  std::mt19937 rng(seed);
  const std::size_t element_size = dtype_size(request.dtype);
  const std::size_t activation_elements = is_conv1d
      ? static_cast<std::size_t>(request.n) * request.w * request.c
      : static_cast<std::size_t>(request.n) * request.h * request.w * request.c;
  const std::size_t weight_elements = is_conv1d
      ? static_cast<std::size_t>(request.out_c) * request.kernel_w * request.c
      : static_cast<std::size_t>(request.out_c) * request.kernel_h * request.kernel_w * request.c;
  const std::size_t output_elements = is_conv1d
      ? static_cast<std::size_t>(request.n) * request.out_w * request.out_c
      : static_cast<std::size_t>(request.n) * request.out_h * request.out_w * request.out_c;
  DeviceBuffer activation(activation_elements * element_size);
  DeviceBuffer weight(weight_elements * element_size);
  DeviceBuffer bias;
  const std::size_t output_nbytes = output_elements * element_size;
  DeviceBuffer output(output_nbytes);
  activation.copy_from(random_storage(activation_elements, request.dtype, rng));
  weight.copy_from(random_storage(weight_elements, request.dtype, rng));
  if (request.has_bias) {
    bias = DeviceBuffer(static_cast<std::size_t>(request.out_c) * element_size);
    bias.copy_from(random_storage(static_cast<std::size_t>(request.out_c), request.dtype, rng));
  }

  DeviceBuffer residual;
  if (request.residual_count == 1) {
    residual = DeviceBuffer(output_elements * element_size);
    residual.copy_from(random_storage(output_elements, request.dtype, rng));
  }

  std::vector<ConvResult> results;
  for (const auto& candidate : profiler_candidates()) {
    if (!candidate_matches(candidate, request)) {
      continue;
    }
    ConvResult result;
    result.profiler_symbol = candidate.profiler_symbol;
    result.samples_ms.reserve(static_cast<std::size_t>(request.repeats));
    bool failed = false;
    for (int repeat = 0; repeat < request.repeats; ++repeat) {
      if (strict_validation) {
        output.fill(0xA5, output_nbytes);
      }
      float elapsed_ms = run_candidate(request, candidate, activation.get(), weight.get(), bias.get(), residual.get(), output.get());
      if (elapsed_ms < 0.0f) {
        failed = true;
        break;
      }
      if (strict_validation) {
        check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize candidate");
      }
      if (strict_validation && repeat == 0) {
        std::vector<std::uint8_t> output_host(output_nbytes);
        output.copy_to(output_host);
        if (std::all_of(output_host.begin(), output_host.end(), [](std::uint8_t value) { return value == 0xA5; })) {
          failed = true;
          break;
        }
      }
      result.samples_ms.push_back(elapsed_ms);
    }
    if (!failed) {
      results.push_back(std::move(result));
    }
  }
  if (results.empty()) {
    throw std::runtime_error("CUTLASS Conv profiler failed every candidate for this problem");
  }
  check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  return results;
}

}  // namespace dinoml::cutlass_conv_profiler
