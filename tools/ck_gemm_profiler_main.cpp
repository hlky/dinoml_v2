#include "ck_gemm_profiler_core.hpp"

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

}  // namespace

int main(int argc, char** argv) {
  try {
    auto begin = argv + 1;
    auto end = argv + argc;
    dinoml::ck_gemm_profiler::GemmRequest request;
    request.profiler_symbol = required_arg(begin, end, "--profiler_symbol");
    request.dtype = required_arg(begin, end, "--dtype");
    request.m = int_arg(begin, end, "--m");
    request.n = int_arg(begin, end, "--n");
    request.k = int_arg(begin, end, "--k");
    request.iterations = int_arg(begin, end, "--iterations");
    request.repeats = int_arg(begin, end, "--repeats");
    request.has_bias = bool_arg(begin, end, "--has_bias");
    request.residual_count = int_arg(begin, end, "--residual_count");
    const auto seed = static_cast<std::uint32_t>(std::stoul(required_arg(begin, end, "--seed")));
    const auto results = dinoml::ck_gemm_profiler::profile_gemm(request, seed);
    for (const auto& result : results) {
      std::cout << result.profiler_symbol << " " << result.elapsed_ms << "\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
