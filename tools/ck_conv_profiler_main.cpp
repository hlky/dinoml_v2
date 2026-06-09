#include "ck_conv_profiler_core.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string required_arg(char** begin, char** end, const char* flag) {
  for (char** it = begin; it != end; ++it) {
    if (std::string(*it) == flag) {
      ++it;
      if (it == end) {
        throw std::runtime_error(std::string("Missing value for ") + flag);
      }
      return *it;
    }
  }
  throw std::runtime_error(std::string("Missing required flag ") + flag);
}

int int_arg(char** begin, char** end, const char* flag) {
  return std::stoi(required_arg(begin, end, flag));
}

bool bool_arg(char** begin, char** end, const char* flag) {
  return int_arg(begin, end, flag) != 0;
}

std::size_t size_arg(char** begin, char** end, const char* flag) {
  return static_cast<std::size_t>(std::stoull(required_arg(begin, end, flag)));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto begin = argv + 1;
    auto end = argv + argc;
    dinoml::ck_conv_profiler::ConvRequest request;
    request.profiler_symbol = required_arg(begin, end, "--profiler_symbol");
    request.dtype = required_arg(begin, end, "--dtype");
    request.batch = int_arg(begin, end, "--batch");
    request.in_channels = int_arg(begin, end, "--in_channels");
    request.in_height = int_arg(begin, end, "--in_height");
    request.in_width = int_arg(begin, end, "--in_width");
    request.out_channels = int_arg(begin, end, "--out_channels");
    request.kernel_h = int_arg(begin, end, "--kernel_h");
    request.kernel_w = int_arg(begin, end, "--kernel_w");
    request.out_height = int_arg(begin, end, "--out_height");
    request.out_width = int_arg(begin, end, "--out_width");
    request.stride_h = int_arg(begin, end, "--stride_h");
    request.stride_w = int_arg(begin, end, "--stride_w");
    request.pad_h = int_arg(begin, end, "--pad_h");
    request.pad_w = int_arg(begin, end, "--pad_w");
    request.output_pad_h = int_arg(begin, end, "--output_pad_h");
    request.output_pad_w = int_arg(begin, end, "--output_pad_w");
    request.dilation_h = int_arg(begin, end, "--dilation_h");
    request.dilation_w = int_arg(begin, end, "--dilation_w");
    request.iterations = int_arg(begin, end, "--iterations");
    request.repeats = int_arg(begin, end, "--repeats");
    request.transposed = bool_arg(begin, end, "--transposed");
    request.has_bias = bool_arg(begin, end, "--has_bias");
    request.has_residual = bool_arg(begin, end, "--has_residual");
    request.x_elements = size_arg(begin, end, "--x_elements");
    request.weight_elements = size_arg(begin, end, "--weight_elements");
    request.bias_elements = size_arg(begin, end, "--bias_elements");
    request.residual_elements = size_arg(begin, end, "--residual_elements");
    request.output_elements = size_arg(begin, end, "--output_elements");
    const auto seed = static_cast<std::uint32_t>(std::stoul(required_arg(begin, end, "--seed")));
    const auto results = dinoml::ck_conv_profiler::profile_conv(request, seed);
    for (const auto& result : results) {
      std::cout << result.profiler_symbol << " " << result.elapsed_ms << "\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
