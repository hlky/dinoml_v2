#include "cutlass_bmm_profiler_core.cuh"

#include <cstdint>
#include <iostream>
#include <string>
#include <unordered_map>

namespace {

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv) {
  std::unordered_map<std::string, std::string> args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (key.rfind("--", 0) != 0 || i + 1 >= argc) {
      throw std::runtime_error("Expected --key value profiler arguments");
    }
    args[key.substr(2)] = argv[++i];
  }
  return args;
}

std::string get(const std::unordered_map<std::string, std::string>& args, const std::string& key) {
  auto it = args.find(key);
  if (it == args.end()) {
    throw std::runtime_error("Missing required profiler argument --" + key);
  }
  return it->second;
}

int64_t get_i64(const std::unordered_map<std::string, std::string>& args, const std::string& key, int64_t default_value = 0) {
  return std::stoll(args.count(key) ? args.at(key) : std::to_string(default_value));
}

}  // namespace

int main(int argc, char** argv) {
  try {
    auto args = parse_args(argc, argv);
    dinoml::cutlass_bmm_profiler::BmmRequest request;
    request.dtype = get(args, "dtype");
    request.batch_count = static_cast<int>(get_i64(args, "batch-count"));
    request.m = static_cast<int>(get_i64(args, "m"));
    request.n = static_cast<int>(get_i64(args, "n"));
    request.k = static_cast<int>(get_i64(args, "k"));
    request.batch_stride_a = get_i64(args, "batch-stride-a");
    request.batch_stride_b = get_i64(args, "batch-stride-b");
    request.batch_stride_d0 = get_i64(args, "batch-stride-d0", 0);
    request.batch_stride_c = get_i64(args, "batch-stride-c");
    request.lda = static_cast<int>(get_i64(args, "lda"));
    request.ldb = static_cast<int>(get_i64(args, "ldb"));
    request.ldd0 = static_cast<int>(get_i64(args, "ldd0", 0));
    request.ldc = static_cast<int>(get_i64(args, "ldc"));
    request.iterations = static_cast<int>(get_i64(args, "iterations", 20));
    request.repeats = static_cast<int>(get_i64(args, "repeats", 1));
    request.max_operand_alignment = static_cast<int>(get_i64(args, "max-operand-alignment", 0));
    request.residual_count = static_cast<int>(get_i64(args, "residual-count", 0));
    request.a_elements = static_cast<std::size_t>(get_i64(args, "a-elements"));
    request.b_elements = static_cast<std::size_t>(get_i64(args, "b-elements"));
    request.d0_elements = static_cast<std::size_t>(get_i64(args, "d0-elements", 0));
    request.c_elements = static_cast<std::size_t>(get_i64(args, "c-elements"));
    auto seed = static_cast<std::uint32_t>(std::stoul(args.count("seed") ? args["seed"] : "2027"));
    auto results = dinoml::cutlass_bmm_profiler::profile_bmm(request, seed);
    std::cout << "{\"candidates\":[";
    for (std::size_t i = 0; i < results.size(); ++i) {
      const auto& result = results[i];
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << "{\"profiler_symbol\":\"" << result.profiler_symbol
                << "\",\"elapsed_ms\":" << result.elapsed_ms
                << ",\"workspace_nbytes\":" << result.workspace_nbytes
                << ",\"samples_ms\":[";
      for (std::size_t j = 0; j < result.samples_ms.size(); ++j) {
        if (j != 0) {
          std::cout << ",";
        }
        std::cout << result.samples_ms[j];
      }
      std::cout << "]}";
    }
    std::cout << "]}\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
