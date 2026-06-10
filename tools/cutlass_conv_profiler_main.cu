#include "cutlass_conv_profiler_core.cuh"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int arg_int(char** argv, int index) {
  return std::atoi(argv[index]);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 15 && argc != 16 && argc != 21 && argc != 22) {
    std::cerr
        << "usage: " << argv[0]
        << " dtype n w c out_w out_c kernel_w stride_w pad_w dilation_w iterations repeats has_bias residual_count [validation_mode]\n"
        << "   or: " << argv[0]
        << " dtype n h w c out_h out_w out_c kernel_h kernel_w stride_h stride_w pad_h pad_w dilation_h dilation_w iterations repeats has_bias residual_count [validation_mode]\n";
    return 2;
  }
  try {
    dinoml::cutlass_conv_profiler::ConvRequest request;
    request.dtype = argv[1];
    request.n = arg_int(argv, 2);
    if (argc == 15 || argc == 16) {
      request.spatial_rank = 1;
      request.w = arg_int(argv, 3);
      request.c = arg_int(argv, 4);
      request.out_w = arg_int(argv, 5);
      request.out_c = arg_int(argv, 6);
      request.kernel_w = arg_int(argv, 7);
      request.stride_w = arg_int(argv, 8);
      request.pad_w = arg_int(argv, 9);
      request.dilation_w = arg_int(argv, 10);
      request.iterations = arg_int(argv, 11);
      request.repeats = arg_int(argv, 12);
      request.has_bias = arg_int(argv, 13) != 0;
      request.residual_count = arg_int(argv, 14);
      if (argc == 16) {
        request.validation_mode = argv[15];
      }
    } else {
      request.spatial_rank = 2;
      request.h = arg_int(argv, 3);
      request.w = arg_int(argv, 4);
      request.c = arg_int(argv, 5);
      request.out_h = arg_int(argv, 6);
      request.out_w = arg_int(argv, 7);
      request.out_c = arg_int(argv, 8);
      request.kernel_h = arg_int(argv, 9);
      request.kernel_w = arg_int(argv, 10);
      request.stride_h = arg_int(argv, 11);
      request.stride_w = arg_int(argv, 12);
      request.pad_h = arg_int(argv, 13);
      request.pad_w = arg_int(argv, 14);
      request.dilation_h = arg_int(argv, 15);
      request.dilation_w = arg_int(argv, 16);
      request.iterations = arg_int(argv, 17);
      request.repeats = arg_int(argv, 18);
      request.has_bias = arg_int(argv, 19) != 0;
      request.residual_count = arg_int(argv, 20);
      if (argc == 22) {
        request.validation_mode = argv[21];
      }
    }
    auto results = dinoml::cutlass_conv_profiler::profile_conv(request, 0xC011A55u);
    std::cout << "[";
    for (std::size_t i = 0; i < results.size(); ++i) {
      const auto& result = results[i];
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << "{\"profiler_symbol\":\"" << result.profiler_symbol << "\",\"workspace_nbytes\":"
                << result.workspace_nbytes << ",\"samples_ms\":[";
      for (std::size_t j = 0; j < result.samples_ms.size(); ++j) {
        if (j != 0) {
          std::cout << ",";
        }
        std::cout << result.samples_ms[j];
      }
      std::cout << "]}";
    }
    std::cout << "]\n";
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << exc.what() << "\n";
    return 1;
  }
}
