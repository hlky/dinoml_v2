#include "ck_conv_profiler_core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

#ifndef DINOML_CK_CONV_PROFILER_PY_MODULE
#define DINOML_CK_CONV_PROFILER_PY_MODULE dinoml_ck_conv_profiler_bind
#endif

namespace {

dinoml::ck_conv_profiler::ConvRequest request_from_kwargs(const py::kwargs& kwargs) {
  dinoml::ck_conv_profiler::ConvRequest request;
  request.profiler_symbol = py::cast<std::string>(kwargs["profiler_symbol"]);
  request.dtype = py::cast<std::string>(kwargs["dtype"]);
  request.batch = py::cast<int>(kwargs["batch"]);
  request.in_channels = py::cast<int>(kwargs["in_channels"]);
  request.in_height = py::cast<int>(kwargs["in_height"]);
  request.in_width = py::cast<int>(kwargs["in_width"]);
  request.out_channels = py::cast<int>(kwargs["out_channels"]);
  request.kernel_h = py::cast<int>(kwargs["kernel_h"]);
  request.kernel_w = py::cast<int>(kwargs["kernel_w"]);
  request.out_height = py::cast<int>(kwargs["out_height"]);
  request.out_width = py::cast<int>(kwargs["out_width"]);
  request.stride_h = py::cast<int>(kwargs["stride_h"]);
  request.stride_w = py::cast<int>(kwargs["stride_w"]);
  request.pad_h = py::cast<int>(kwargs["pad_h"]);
  request.pad_w = py::cast<int>(kwargs["pad_w"]);
  request.dilation_h = py::cast<int>(kwargs["dilation_h"]);
  request.dilation_w = py::cast<int>(kwargs["dilation_w"]);
  request.iterations = py::cast<int>(kwargs["iterations"]);
  request.repeats = py::cast<int>(kwargs["repeats"]);
  request.has_residual = py::cast<bool>(kwargs["has_residual"]);
  request.x_elements = py::cast<std::size_t>(kwargs["x_elements"]);
  request.weight_elements = py::cast<std::size_t>(kwargs["weight_elements"]);
  request.bias_elements = py::cast<std::size_t>(kwargs["bias_elements"]);
  request.residual_elements = py::cast<std::size_t>(kwargs["residual_elements"]);
  request.output_elements = py::cast<std::size_t>(kwargs["output_elements"]);
  return request;
}

}  // namespace

PYBIND11_MODULE(DINOML_CK_CONV_PROFILER_PY_MODULE, m) {
  m.def("profile_conv", [](py::kwargs kwargs) {
    auto request = request_from_kwargs(kwargs);
    auto seed = py::cast<std::uint32_t>(kwargs["seed"]);
    auto results = dinoml::ck_conv_profiler::profile_conv(request, seed);
    py::list out;
    for (const auto& result : results) {
      py::dict item;
      item["profiler_symbol"] = result.profiler_symbol;
      item["elapsed_ms"] = result.elapsed_ms;
      item["samples_ms"] = result.samples_ms;
      item["workspace_nbytes"] = result.workspace_nbytes;
      out.append(item);
    }
    return out;
  });
}
