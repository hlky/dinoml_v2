#include "cutlass_conv_profiler_core.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

#ifndef DINOML_CUTLASS_CONV_PROFILER_PY_MODULE
#define DINOML_CUTLASS_CONV_PROFILER_PY_MODULE dinoml_cutlass_conv_profiler_bind
#endif

namespace {

std::vector<py::dict> profile_conv_py(py::kwargs kwargs) {
  dinoml::cutlass_conv_profiler::ConvRequest request;
  request.dtype = kwargs["dtype"].cast<std::string>();
  request.spatial_rank = kwargs.contains("spatial_rank") ? kwargs["spatial_rank"].cast<int>() : 2;
  request.n = kwargs["n"].cast<int>();
  request.d = kwargs.contains("d") ? kwargs["d"].cast<int>() : 1;
  request.w = kwargs["w"].cast<int>();
  request.c = kwargs["c"].cast<int>();
  request.out_d = kwargs.contains("out_d") ? kwargs["out_d"].cast<int>() : 1;
  request.out_w = kwargs["out_w"].cast<int>();
  request.out_c = kwargs["out_c"].cast<int>();
  request.kernel_d = kwargs.contains("kernel_d") ? kwargs["kernel_d"].cast<int>() : 1;
  request.kernel_w = kwargs["kernel_w"].cast<int>();
  request.stride_d = kwargs.contains("stride_d") ? kwargs["stride_d"].cast<int>() : 1;
  request.stride_w = kwargs["stride_w"].cast<int>();
  request.pad_d = kwargs.contains("pad_d") ? kwargs["pad_d"].cast<int>() : 0;
  request.pad_w = kwargs["pad_w"].cast<int>();
  request.dilation_d = kwargs.contains("dilation_d") ? kwargs["dilation_d"].cast<int>() : 1;
  request.dilation_w = kwargs["dilation_w"].cast<int>();
  request.h = kwargs.contains("h") ? kwargs["h"].cast<int>() : 1;
  request.out_h = kwargs.contains("out_h") ? kwargs["out_h"].cast<int>() : 1;
  request.kernel_h = kwargs.contains("kernel_h") ? kwargs["kernel_h"].cast<int>() : 1;
  request.stride_h = kwargs.contains("stride_h") ? kwargs["stride_h"].cast<int>() : 1;
  request.pad_h = kwargs.contains("pad_h") ? kwargs["pad_h"].cast<int>() : 0;
  request.dilation_h = kwargs.contains("dilation_h") ? kwargs["dilation_h"].cast<int>() : 1;
  request.groups = kwargs.contains("groups") ? kwargs["groups"].cast<int>() : 1;
  request.iterations = kwargs["iterations"].cast<int>();
  request.repeats = kwargs["repeats"].cast<int>();
  request.has_bias = kwargs.contains("has_bias") ? kwargs["has_bias"].cast<bool>() : true;
  request.residual_count = kwargs["residual_count"].cast<int>();
  if (kwargs.contains("validation_mode")) {
    request.validation_mode = kwargs["validation_mode"].cast<std::string>();
  }
  std::uint32_t seed = kwargs["seed"].cast<std::uint32_t>();
  std::vector<py::dict> rows;
  for (const auto& result : dinoml::cutlass_conv_profiler::profile_conv(request, seed)) {
    py::dict row;
    row["profiler_symbol"] = result.profiler_symbol;
    row["samples_ms"] = result.samples_ms;
    row["workspace_nbytes"] = result.workspace_nbytes;
    rows.push_back(std::move(row));
  }
  return rows;
}

}  // namespace

PYBIND11_MODULE(DINOML_CUTLASS_CONV_PROFILER_PY_MODULE, m) {
  m.def("profile_conv", &profile_conv_py);
}
