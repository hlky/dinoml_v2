#include "cutlass_gemm_profiler_core.cuh"

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

}  // namespace

int main(int argc, char** argv) {
  try {
    auto args = parse_args(argc, argv);
    dinoml::cutlass_gemm_profiler::GemmRequest request;
    request.dtype = get(args, "dtype");
    request.m = std::stoi(get(args, "m"));
    request.n = std::stoi(get(args, "n"));
    request.k = std::stoi(get(args, "k"));
    request.split_k = std::stoi(args.count("split-k") ? args["split-k"] : "1");
    request.iterations = std::stoi(args.count("iterations") ? args["iterations"] : "20");
    request.repeats = std::stoi(args.count("repeats") ? args["repeats"] : "1");
    request.max_operand_alignment = std::stoi(args.count("max-operand-alignment") ? args["max-operand-alignment"] : "0");
    request.has_bias = std::stoi(args.count("has-bias") ? args["has-bias"] : "0") != 0;
    request.residual_count = std::stoi(args.count("residual-count") ? args["residual-count"] : "0");
    auto seed = static_cast<std::uint32_t>(std::stoul(args.count("seed") ? args["seed"] : "2027"));
    auto results = dinoml::cutlass_gemm_profiler::profile_gemm(request, seed);
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
