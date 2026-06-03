from pathlib import Path


def test_ck_profiler_bind_targets_define_unique_pybind_module_exports():
    cmake = Path("CMakeLists.txt").read_text(encoding="utf-8")

    assert "DINOML_CK_GEMM_PROFILER_PY_MODULE=${_ck_gemm_profiler_output}_bind" in cmake
    assert "DINOML_CK_BMM_PROFILER_PY_MODULE=${_ck_bmm_profiler_output}_bind" in cmake
    assert "DINOML_CK_CONV_PROFILER_PY_MODULE=${_ck_conv_profiler_output}_bind" in cmake
