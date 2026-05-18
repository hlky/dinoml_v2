#include "cutlass_bmm_profiler_core.cuh"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <string>

namespace py = pybind11;

#ifndef DINOML_CUTLASS_BMM_PROFILER_PY_MODULE
#define DINOML_CUTLASS_BMM_PROFILER_PY_MODULE dinoml_cutlass_bmm_profiler_bind
#endif

namespace {

dinoml::cutlass_bmm_profiler::BmmRequest request_from_kwargs(const py::kwargs& kwargs) {
  dinoml::cutlass_bmm_profiler::BmmRequest request;
  request.dtype = py::cast<std::string>(kwargs["dtype"]);
  request.batch_count = py::cast<int>(kwargs["batch_count"]);
  request.m = py::cast<int>(kwargs["m"]);
  request.n = py::cast<int>(kwargs["n"]);
  request.k = py::cast<int>(kwargs["k"]);
  request.batch_stride_a = py::cast<int64_t>(kwargs["batch_stride_a"]);
  request.batch_stride_b = py::cast<int64_t>(kwargs["batch_stride_b"]);
  request.batch_stride_d0 = py::cast<int64_t>(kwargs["batch_stride_d0"]);
  request.batch_stride_c = py::cast<int64_t>(kwargs["batch_stride_c"]);
  request.lda = py::cast<int>(kwargs["lda"]);
  request.ldb = py::cast<int>(kwargs["ldb"]);
  request.ldd0 = py::cast<int>(kwargs["ldd0"]);
  request.ldc = py::cast<int>(kwargs["ldc"]);
  request.iterations = py::cast<int>(kwargs["iterations"]);
  request.repeats = py::cast<int>(kwargs["repeats"]);
  request.max_operand_alignment = py::cast<int>(kwargs["max_operand_alignment"]);
  request.residual_count = py::cast<int>(kwargs["residual_count"]);
  request.a_elements = py::cast<std::size_t>(kwargs["a_elements"]);
  request.b_elements = py::cast<std::size_t>(kwargs["b_elements"]);
  request.d0_elements = py::cast<std::size_t>(kwargs["d0_elements"]);
  request.c_elements = py::cast<std::size_t>(kwargs["c_elements"]);
  return request;
}

}  // namespace

PYBIND11_MODULE(DINOML_CUTLASS_BMM_PROFILER_PY_MODULE, m) {
  m.def("profile_bmm", [](py::kwargs kwargs) {
    auto request = request_from_kwargs(kwargs);
    auto seed = py::cast<std::uint32_t>(kwargs["seed"]);
    auto results = dinoml::cutlass_bmm_profiler::profile_bmm(request, seed);
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
