#include "ck_bmm_profiler_core.hpp"

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

std::int64_t i64_arg(char** begin, char** end, const char* flag) {
  return std::stoll(required_arg(begin, end, flag));
}

std::size_t size_arg(char** begin, char** end, const char* flag) {
  return static_cast<std::size_t>(std::stoull(required_arg(begin, end, flag)));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto begin = argv + 1;
    auto end = argv + argc;
    dinoml::ck_bmm_profiler::BmmRequest request;
    request.profiler_symbol = required_arg(begin, end, "--profiler_symbol");
    request.dtype = required_arg(begin, end, "--dtype");
    request.batch_count = int_arg(begin, end, "--batch_count");
    request.m = int_arg(begin, end, "--m");
    request.n = int_arg(begin, end, "--n");
    request.k = int_arg(begin, end, "--k");
    request.batch_stride_a = i64_arg(begin, end, "--batch_stride_a");
    request.batch_stride_b = i64_arg(begin, end, "--batch_stride_b");
    request.batch_stride_d0 = i64_arg(begin, end, "--batch_stride_d0");
    request.batch_stride_c = i64_arg(begin, end, "--batch_stride_c");
    request.lda = int_arg(begin, end, "--lda");
    request.ldb = int_arg(begin, end, "--ldb");
    request.ldd0 = int_arg(begin, end, "--ldd0");
    request.ldc = int_arg(begin, end, "--ldc");
    request.iterations = int_arg(begin, end, "--iterations");
    request.repeats = int_arg(begin, end, "--repeats");
    request.residual_count = int_arg(begin, end, "--residual_count");
    request.a_elements = size_arg(begin, end, "--a_elements");
    request.b_elements = size_arg(begin, end, "--b_elements");
    request.d0_elements = size_arg(begin, end, "--d0_elements");
    request.c_elements = size_arg(begin, end, "--c_elements");
    const auto seed = static_cast<std::uint32_t>(std::stoul(required_arg(begin, end, "--seed")));
    const auto results = dinoml::ck_bmm_profiler::profile_bmm(request, seed);
    for (const auto& result : results) {
      std::cout << result.profiler_symbol << " " << result.elapsed_ms << "\n";
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
