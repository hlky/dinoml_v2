#include "ck_gemm_profiler_core.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

#ifndef DINOML_CK_GEMM_PROFILER_PY_MODULE
#define DINOML_CK_GEMM_PROFILER_PY_MODULE dinoml_ck_gemm_profiler_bind
#endif

namespace {

dinoml::ck_gemm_profiler::GemmRequest request_from_kwargs(const py::kwargs& kwargs) {
  dinoml::ck_gemm_profiler::GemmRequest request;
  request.profiler_symbol = py::cast<std::string>(kwargs["profiler_symbol"]);
  if (kwargs.contains("profiler_symbols")) {
    request.profiler_symbols = py::cast<std::vector<std::string>>(kwargs["profiler_symbols"]);
  }
  request.dtype = py::cast<std::string>(kwargs["dtype"]);
  request.m = py::cast<int>(kwargs["m"]);
  request.n = py::cast<int>(kwargs["n"]);
  request.k = py::cast<int>(kwargs["k"]);
  request.iterations = py::cast<int>(kwargs["iterations"]);
  request.repeats = py::cast<int>(kwargs["repeats"]);
  request.has_bias = py::cast<bool>(kwargs["has_bias"]);
  request.residual_count = py::cast<int>(kwargs["residual_count"]);
  return request;
}

}  // namespace

PYBIND11_MODULE(DINOML_CK_GEMM_PROFILER_PY_MODULE, m) {
  m.def("profile_gemm", [](py::kwargs kwargs) {
    auto request = request_from_kwargs(kwargs);
    auto seed = py::cast<std::uint32_t>(kwargs["seed"]);
    auto results = dinoml::ck_gemm_profiler::profile_gemm(request, seed);
    py::list out;
    for (const auto& result : results) {
      py::dict item;
      item["profiler_symbol"] = result.profiler_symbol;
      item["elapsed_ms"] = result.elapsed_ms;
      item["samples_ms"] = result.samples_ms;
      item["workspace_nbytes"] = result.workspace_nbytes;
      item["ok"] = result.ok;
      if (!result.error.empty()) {
        item["error"] = result.error;
      }
      out.append(item);
    }
    return out;
  });
}
