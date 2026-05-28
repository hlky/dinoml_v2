from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import dinoml as dml
from dinoml import compiler as dml_compiler
from dinoml.backends import get_backend_spec, registered_backend_names
from dinoml.backends import rocm as rocm_backend
from dinoml.backends.rocm import ensure_rocm_support_libs
from dinoml.ir import read_json
from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import apply_execution_plan, build_kernel_manifest
from dinoml.kernels.profiling import (
    _blocked_profile_items,
    _cache_entry,
    _profile_failure_result,
    _profile_report,
    _profile_result,
    _profile_result_from_cache,
    build_execution_plan,
    build_profile_workloads,
)
from dinoml.kernels.providers.ck.bmm import ck_bmm_static_library_name, render_ck_bmm_source
from dinoml.kernels.providers.ck.conv import ck_conv_static_library_name, render_ck_conv_source
from dinoml.kernels.providers.ck.gemm import ck_gemm_static_library_name, render_ck_gemm_source
from dinoml.lowering.rocm import render_rocm_module
from dinoml.lowering.target_specs import lowering_target_spec, storage_type
from dinoml.ops.elementwise import FusedElementwise
from dinoml.runtime import load
from tests.cases import elementwise_case


def test_rocm_target_is_registered_as_distinct_backend():
    assert "rocm" in registered_backend_names()
    target = dml.Target("rocm")
    assert target.to_json() == {
        "name": "rocm",
        "arch": "gfx1201",
        "no_tf32": False,
        "use_fp16_acc": False,
    }

    spec = get_backend_spec("rocm")
    assert spec.default_arch == "gfx1201"
    assert spec.resolve_build_function().__name__ == "build_rocm_module"
    assert spec.cmake.support_build_targets == (
        "dinoml_runtime",
        "dinoml_rocm_runtime",
        "dinoml_rocm_kernels",
    )
    suffix = ".dll" if os.name == "nt" else ".dylib" if sys.platform == "darwin" else ".so"
    assert spec.support_libraries["rocm_runtime_library"].endswith(f"dinoml_rocm_runtime{suffix}")
    assert spec.support_libraries["kernel_library"].endswith(f"dinoml_rocm_kernels{suffix}")


def test_cli_profile_help_mentions_rocm_ck(capsys):
    from dinoml import cli as dinoml_cli

    with pytest.raises(SystemExit) as compile_exit:
        dinoml_cli.main(["compile", "--help"])
    compile_help = capsys.readouterr().out

    with pytest.raises(SystemExit) as profile_exit:
        dinoml_cli.main(["profile", "--help"])
    profile_help = capsys.readouterr().out

    assert compile_exit.value.code == 0
    assert profile_exit.value.code == 0
    assert "ROCm CK" in compile_help
    assert "ROCm CK" in profile_help
    assert "GEMM/BMM/Conv" in profile_help
    assert "blocked profile items" in " ".join(profile_help.lower().split())
    assert "input=1,128,768" in profile_help


def test_cli_profile_summary_handles_rocm_ck_conv_shapes(monkeypatch, capsys):
    from dinoml import cli as dinoml_cli

    def fake_profile_artifact(*_args, **_kwargs):
        return {
            "artifact": "conv_artifact.dinoml",
            "target": {"name": "rocm", "arch": "gfx1201"},
            "iterations": 3,
            "repeats": 2,
            "problems": [
                {
                    "node_id": "n0",
                    "op": "conv2d_bias",
                    "dtype": "float16",
                    "kernel_library": "ck_conv",
                    "profiler_symbol": "dinoml_profile_ck_conv2d_bias_float16_xdl_wide_m_v1",
                    "shape": {
                        "n": 1,
                        "c": 8,
                        "h": 9,
                        "w": 11,
                        "out_n": 1,
                        "out_c": 16,
                        "out_h": 4,
                        "out_w": 6,
                        "kernel_h": 3,
                        "kernel_w": 3,
                    },
                    "conv": {"stride": [2, 2], "padding": [0, 1], "dilation": [1, 1], "groups": 1},
                    "elapsed_ms": 0.25,
                    "workspace_nbytes": 0,
                    "timing": {"sample_count": 2},
                    "tflops": 0.123,
                    "selected": {"candidate_id": "ck_conv2d_bias_float16_xdl_wide_m_v1"},
                }
            ],
            "execution_plan": {"path": "execution_plan.json"},
            "summary": {"profiled": 1, "failed": 0},
        }

    monkeypatch.setattr(dinoml_cli, "profile_artifact", fake_profile_artifact)

    assert dinoml_cli.main(["profile", "conv_artifact.dinoml"]) == 0
    payload = json.loads(capsys.readouterr().out)
    problem = payload["problems"][0]

    assert problem["kernel_library"] == "ck_conv"
    assert problem["candidate_id"] == "ck_conv2d_bias_float16_xdl_wide_m_v1"
    assert problem["shape"]["out_h"] == 4
    assert problem["shape"]["out_w"] == 6
    assert problem["conv"]["stride"] == [2, 2]
    assert "split_k" not in problem


def test_cli_profile_summary_includes_blocked_rocm_ck_items(monkeypatch, capsys):
    from dinoml import cli as dinoml_cli

    def fake_profile_artifact(*_args, **_kwargs):
        return {
            "artifact": "conv_artifact.dinoml",
            "target": {"name": "rocm", "arch": "gfx1201"},
            "iterations": 3,
            "repeats": 2,
            "problems": [],
            "blocked_profile_items": [
                {
                    "op": "conv2d_bias",
                    "dtype": "float16",
                    "kernel_library": "ck_conv",
                    "kernel_symbol": "dinoml_ck_conv2d_bias_float16_xdl_custom_v1",
                    "profiler_symbol": "dinoml_profile_ck_conv2d_bias_float16_xdl_custom_v1",
                    "candidate_set_id": "ck_conv2d_bias_float16_bias_v3",
                    "selected_candidate_id": "ck_conv2d_bias_float16_xdl_custom_v1",
                    "reason": "ck_conv_groups_unsupported_for_profile",
                    "details": {"groups": 2, "supported_groups": [1]},
                }
            ],
            "execution_plan": {"path": "execution_plan.json"},
            "summary": {"profiled": 0, "cached": 0, "failed": 0, "blocked": 1},
        }

    monkeypatch.setattr(dinoml_cli, "profile_artifact", fake_profile_artifact)

    assert dinoml_cli.main(["profile", "conv_artifact.dinoml"]) == 0
    payload = json.loads(capsys.readouterr().out)
    blocked = payload["blocked_profile_items"][0]

    assert payload["summary"]["blocked"] == 1
    assert blocked["kernel_library"] == "ck_conv"
    assert blocked["candidate_id"] == "ck_conv2d_bias_float16_xdl_custom_v1"
    assert blocked["reason"] == "ck_conv_groups_unsupported_for_profile"
    assert blocked["details"] == {"groups": 2, "supported_groups": [1]}


def test_rocm_codegen_plan_uses_arch_specific_support_cache(tmp_path):
    manifest = {
        "target": {"name": "rocm", "arch": "gfx1201"},
        "cache_key": "abcdef0123456789",
        "required_kernels": [],
    }

    plan = create_codegen_plan(manifest, tmp_path)

    assert plan.target == {"name": "rocm", "arch": "gfx1201"}
    assert plan.support_cache_dir == tmp_path / "support" / "rocm-gfx1201" / "abcdef0123456789"


def test_rocm_codegen_plan_sanitizes_feature_suffixed_arch_cache_dir(tmp_path):
    manifest = {
        "target": {"name": "rocm", "arch": "gfx90a:xnack-"},
        "cache_key": "abcdef0123456789",
        "required_kernels": [],
    }

    plan = create_codegen_plan(manifest, tmp_path)

    assert plan.target == {"name": "rocm", "arch": "gfx90a:xnack-"}
    assert plan.support_cache_dir == tmp_path / "support" / "rocm-gfx90a_xnack-" / "abcdef0123456789"


def test_rocm_compile_profile_path_is_admitted_without_cuda_guard(tmp_path, monkeypatch):
    calls = []

    def fake_compile_with_profile(spec, target, output, **kwargs):
        calls.append((spec, target, output, kwargs))
        return dml_compiler.Artifact(Path(output))

    monkeypatch.setattr(dml_compiler, "_compile_with_profile", fake_compile_with_profile)
    spec = elementwise_case().build_spec()

    artifact = dml_compiler.compile(spec, dml.Target("rocm"), tmp_path / "profiled_rocm.dinoml", profile=True)

    assert artifact.path == tmp_path / "profiled_rocm.dinoml"
    assert calls
    assert calls[0][1].name == "rocm"


def test_rocm_cmake_arch_normalizes_whitespace_and_rejects_invalid_values():
    assert rocm_backend._cmake_arch(" gfx1201 ") == "gfx1201"

    with pytest.raises(ValueError, match="Expected ROCm arch"):
        rocm_backend._cmake_arch("")
    with pytest.raises(ValueError, match="Expected ROCm arch"):
        rocm_backend._cmake_arch("   ")
    with pytest.raises(ValueError, match="Expected ROCm arch"):
        rocm_backend._cmake_arch("sm_90")


def test_rocm_support_cache_dir_sanitizes_feature_suffixed_arch():
    assert rocm_backend._rocm_support_cache_dir_name("gfx1201") == "rocm-gfx1201"
    assert rocm_backend._rocm_support_cache_dir_name(" gfx90a:xnack- ") == "rocm-gfx90a_xnack-"


def test_rocm_lowering_target_spec_uses_hip_runtime_contract():
    spec = lowering_target_spec("rocm")
    context = spec.gpu_template_context()

    assert spec.source_extension == "hip"
    assert spec.stream_type == "hipStream_t"
    assert spec.stream_expr == "session->stream"
    assert context["gpu_check_macro"] == "DINO_ROCM_CHECK"
    assert context["gpu_last_error_call"] == "hipGetLastError()"
    assert context["gpu_memset_async"] == "hipMemsetAsync"
    assert context["gpu_warp_full_mask"] == "0xffffffffffffffffull"
    assert storage_type("float16", "rocm") == "half"
    assert storage_type("bfloat16", "rocm") == "dinoml::bfloat16"


def test_rocm_runtime_header_and_source_define_error_check_contract():
    header = Path("runtime/include/dinoml/runtime_rocm.h").read_text(encoding="utf-8")
    source = Path("runtime/src/runtime_rocm.hip").read_text(encoding="utf-8")

    assert "DINO_EXPORT int dino_runtime_rocm_check" in header
    assert "#define DINO_ROCM_CHECK(expr)" in header
    assert "dino_runtime_rocm_check((expr), #expr, __FILE__, __LINE__)" in header
    assert "return _dino_err;" in header
    assert "hipGetErrorString(err)" in source
    assert "hipMemcpyHostToDevice" in source
    assert "hipMemcpyDeviceToHost" in source
    assert "hipMemcpyDeviceToDevice" in source


def test_rocm_runtime_paths_resolve_from_active_rocm_sdk_command(tmp_path, monkeypatch):
    sdk_root = tmp_path / "sdk"
    sdk_bin = sdk_root / "bin"
    llvm_bin = sdk_root / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)

    monkeypatch.setattr(
        rocm_backend.shutil,
        "which",
        lambda name: "rocm-sdk.exe" if name == "rocm-sdk" else None,
    )

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        assert cmd[:2] == ["rocm-sdk.exe", "path"]
        if cmd[-1] == "--root":
            value = sdk_root
        elif cmd[-1] == "--bin":
            value = sdk_bin
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=str(value))

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    assert rocm_backend._rocm_runtime_paths() == [str(sdk_bin), str(llvm_bin)]


def test_rocm_runtime_paths_can_use_active_python_rocm_sdk_module(tmp_path, monkeypatch):
    sdk_root = tmp_path / "sdk"
    sdk_bin = sdk_root / "bin"
    llvm_bin = sdk_root / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)

    def fake_which(name):
        if name == "python":
            return "python.exe"
        return None

    monkeypatch.setattr(rocm_backend.shutil, "which", fake_which)
    monkeypatch.setattr(rocm_backend.sys, "executable", "current-python-without-rocm-sdk.exe")

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        if cmd[:2] == ["current-python-without-rocm-sdk.exe", "-c"]:
            return subprocess.CompletedProcess(cmd, 1)
        if cmd[:2] == ["python.exe", "-c"]:
            return subprocess.CompletedProcess(cmd, 0)
        assert cmd[:4] == ["python.exe", "-m", "rocm_sdk", "path"]
        if cmd[-1] == "--root":
            value = sdk_root
        elif cmd[-1] == "--bin":
            value = sdk_bin
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=str(value))

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    assert rocm_backend._rocm_runtime_paths() == [str(sdk_bin), str(llvm_bin)]


def test_rocm_runtime_paths_prefer_current_python_rocm_sdk_module(tmp_path, monkeypatch):
    sdk_root = tmp_path / "sdk"
    sdk_bin = sdk_root / "bin"
    llvm_bin = sdk_root / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)

    monkeypatch.setattr(rocm_backend.shutil, "which", lambda name: None)
    monkeypatch.setattr(rocm_backend.sys, "executable", "active-python.exe")

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        if cmd[:2] == ["active-python.exe", "-c"]:
            return subprocess.CompletedProcess(cmd, 0)
        assert cmd[:4] == ["active-python.exe", "-m", "rocm_sdk", "path"]
        if cmd[-1] == "--root":
            value = sdk_root
        elif cmd[-1] == "--bin":
            value = sdk_bin
        else:
            raise AssertionError(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout=str(value))

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    assert rocm_backend._rocm_runtime_paths() == [str(sdk_bin), str(llvm_bin)]


def test_rocm_runtime_paths_fallback_to_root_bin_when_sdk_bin_query_missing(tmp_path, monkeypatch):
    sdk_root = tmp_path / "sdk"
    sdk_bin = sdk_root / "bin"
    llvm_bin = sdk_root / "lib" / "llvm" / "bin"
    sdk_bin.mkdir(parents=True)
    llvm_bin.mkdir(parents=True)

    def fake_run_rocm_sdk_path(arg: str) -> str | None:
        if arg == "--root":
            return str(sdk_root)
        if arg == "--bin":
            return None
        raise AssertionError(arg)

    monkeypatch.setattr(rocm_backend, "_run_rocm_sdk_path", fake_run_rocm_sdk_path)

    assert rocm_backend._rocm_runtime_paths() == [str(sdk_bin), str(llvm_bin)]


def test_rocm_sdk_python_probe_treats_launch_failure_as_unavailable(monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise OSError("missing python")

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    assert rocm_backend._python_has_rocm_sdk("missing-python.exe") is False


def test_rocm_sdk_path_query_treats_launch_failure_as_unavailable(monkeypatch):
    monkeypatch.setattr(rocm_backend, "_rocm_sdk_command", lambda: ["rocm-sdk.exe"])

    def fake_run(*_args, **_kwargs):
        raise OSError("broken rocm-sdk")

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    assert rocm_backend._run_rocm_sdk_path("--root") is None


def test_rocm_runtime_paths_allow_regular_hip_sdk_without_rocm_sdk(monkeypatch):
    monkeypatch.setattr(rocm_backend.shutil, "which", lambda name: None)
    monkeypatch.setattr(rocm_backend, "_python_has_rocm_sdk", lambda python: False)

    assert rocm_backend._rocm_runtime_paths() == []


def test_rocm_support_configure_requires_ninja(monkeypatch):
    monkeypatch.setattr(rocm_backend.shutil, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="require Ninja"):
        rocm_backend._with_default_cmake_generator(["cmake", "-S", ".", "-B", "build"])


def test_rocm_module_cmake_imports_and_links_ck_archives():
    cmake = rocm_backend.render_template(
        "rocm_module_cmake.txt.j2",
        {
            "rocm_sdk_cmake": "H:/dinoml_v2/cmake/DinoMLROCmSdk.cmake",
            "runtime_lib": "H:/cache/lib/dinoml_runtime.dll",
            "rocm_runtime_lib": "H:/cache/lib/dinoml_rocm_runtime.dll",
            "kernels_lib": "H:/cache/lib/dinoml_rocm_kernels.dll",
            "ck_gemm_archives": ["H:/cache/lib/dinoml_ck_gemm.a"],
            "ck_bmm_archives": ["H:/cache/lib/dinoml_ck_bmm.a"],
            "ck_conv_archives": ["H:/cache/lib/dinoml_ck_conv.a"],
            "runtime_implib": "",
            "rocm_runtime_implib": "",
            "kernels_implib": "",
            "runtime_include": "H:/dinoml_v2/runtime/include",
            "common_include": "H:/dinoml_v2/kernels/common/include",
            "kernels_include": "H:/dinoml_v2/kernels/rocm/include",
        },
    )

    assert "add_library(dinoml_ck_gemm_0 STATIC IMPORTED GLOBAL)" in cmake
    assert "add_library(dinoml_ck_bmm_0 STATIC IMPORTED GLOBAL)" in cmake
    assert "add_library(dinoml_ck_conv_0 STATIC IMPORTED GLOBAL)" in cmake
    assert 'IMPORTED_LOCATION "H:/cache/lib/dinoml_ck_gemm.a"' in cmake
    assert 'IMPORTED_LOCATION "H:/cache/lib/dinoml_ck_bmm.a"' in cmake
    assert 'IMPORTED_LOCATION "H:/cache/lib/dinoml_ck_conv.a"' in cmake
    assert "dinoml_ck_gemm_0" in cmake[cmake.index("target_link_libraries(module PRIVATE") :]
    assert "dinoml_ck_bmm_0" in cmake[cmake.index("target_link_libraries(module PRIVATE") :]
    assert "dinoml_ck_conv_0" in cmake[cmake.index("target_link_libraries(module PRIVATE") :]


def test_visual_studio_environment_filters_vcvars_payload(monkeypatch):
    vcvars = Path("C:/VS/VC/Auxiliary/Build/vcvars64.bat")
    monkeypatch.setattr(rocm_backend, "_find_vcvars64", lambda: vcvars)

    def fake_run(cmd, *, text, stdout, stderr):
        del text, stdout, stderr
        assert cmd == ["cmd", "/s", "/c", f'"{vcvars}" >nul && set']
        output = "\n".join(
            [
                r"INCLUDE=C:\VS\include",
                r"LIB=C:\VS\lib",
                r"LIBPATH=C:\VS\libpath",
                r"PATH=C:\VS\bin;C:\Windows\System32",
                r"VCToolsVersion=14.43.34808",
                "UNRELATED_SECRET=do-not-copy",
            ]
        )
        return subprocess.CompletedProcess(cmd, 0, stdout=output)

    monkeypatch.setattr(rocm_backend.subprocess, "run", fake_run)

    env = rocm_backend._visual_studio_environment()

    assert env == {
        "INCLUDE": r"C:\VS\include",
        "LIB": r"C:\VS\lib",
        "LIBPATH": r"C:\VS\libpath",
        "PATH": r"C:\VS\bin;C:\Windows\System32",
        "VCToolsVersion": "14.43.34808",
    }


def test_rocm_fused_elementwise_is_publicly_admitted():
    binding = FusedElementwise.backend_kernels["rocm"]

    assert binding.library == "model"
    assert binding.source_template == "fused_elementwise_gpu"


def test_rocm_gemm_manifest_selects_ck_custom_xdl_archive(tmp_path):
    ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16")

    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]
    plan = create_codegen_plan(manifest, tmp_path)

    assert item["kernel_library"] == "ck_gemm"
    assert item["kernel_symbol"] == "dinoml_ck_gemm_rcr_bias_add_relu_float16_xdl_custom_v1"
    assert item["support_archive"] == ck_gemm_static_library_name("gemm_rcr_bias_add_relu", "float16")
    assert len(item["candidates"]) == 27
    assert item["candidates"][0]["ck"]["api"] == "device_gemm_multiple_d_xdl_cshuffle"
    assert item["candidates"][0]["ck"]["mode"] == "custom_ck_xdl_instances"
    assert item["candidates"][1]["ck"]["config"]["name"] == "wide_m"
    candidate_names = {candidate["ck"]["config"]["name"] for candidate in item["candidates"]}
    assert {
        "baseline",
        "wide_m",
        "wide_n",
        "square",
        "skinny_m",
        "skinny_n",
        "wide_m_interwave_v1",
        "wide_m_default_v2",
    } <= candidate_names
    scheduler_pipeline_pairs = [
        (candidate["ck"]["config"]["scheduler"], candidate["ck"]["config"]["pipeline"])
        for candidate in item["candidates"]
    ]
    assert scheduler_pipeline_pairs.count(("default", "v1")) == 9
    assert scheduler_pipeline_pairs.count(("interwave", "v1")) == 9
    assert scheduler_pipeline_pairs.count(("default", "v2")) == 9
    assert plan.external_support_libraries[0]["name"] == "ck_gemm"
    assert plan.external_support_libraries[0]["modules"] == [
        {
            "op": "gemm_rcr_bias_add_relu",
            "dtype": "float16",
            "archive": f"lib/{ck_gemm_static_library_name('gemm_rcr_bias_add_relu', 'float16')}",
            "target": "dinoml_ck_gemm_gemm_rcr_bias_add_relu_float16",
        }
    ]
    _assert_ck_support_plan_unpruned(plan, item)


def test_rocm_gemm_manifest_selects_tuned_ck_candidate_for_aligned_static_shape():
    ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=128, n=128, k=64)

    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    assert item["selected_candidate_id"] == "ck_gemm_rcr_bias_add_relu_float16_xdl_wide_m_v1"
    assert item["kernel_symbol"] == "dinoml_ck_gemm_rcr_bias_add_relu_float16_xdl_wide_m_v1"


@pytest.mark.parametrize(
    ("case", "selected_candidate_id", "config_name", "k_per_block", "vector_width", "cde_vector_width"),
    [
        (
            "gemm",
            "ck_gemm_rcr_bias_add_relu_float32_xdl_wide_m_v1",
            "wide_m",
            32,
            8,
            4,
        ),
        (
            "bmm",
            "ck_bmm_rcr_add_float32_xdl_wide_m_v1",
            "wide_m",
            32,
            4,
            2,
        ),
        (
            "conv",
            "ck_conv2d_bias_float32_xdl_wide_n_v1",
            "wide_n",
            16,
            4,
            4,
        ),
    ],
)
def test_rocm_ck_manifest_selects_float32_candidate_shapes(
    case: str,
    selected_candidate_id: str,
    config_name: str,
    k_per_block: int,
    vector_width: int,
    cde_vector_width: int,
):
    if case == "gemm":
        ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float32", m=128, n=128, k=64)
    elif case == "bmm":
        ir = _rocm_bmm_ir("bmm_rcr_add", "float32", batch=2, m=128, n=128, k=96)
    elif case == "conv":
        ir = _rocm_conv2d_bias_ir("float32", batch=2, in_channels=8, out_channels=64, height=16, width=16)
    else:
        raise AssertionError(f"unhandled CK float32 case: {case}")
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]
    selected = next(candidate for candidate in item["candidates"] if candidate["candidate_id"] == selected_candidate_id)

    assert item["selected_candidate_id"] == selected_candidate_id
    assert item["kernel_library"] == f"ck_{case}"
    assert selected["dtype"] == "float32"
    assert selected["accumulator_dtype"] == "float32"
    assert selected["ck"]["config"]["name"] == config_name
    assert selected["ck"]["config"]["tile"]["k_per_block"] == k_per_block
    assert selected["ck"]["config"]["vector_width"] == vector_width
    assert selected["ck"]["config"]["cde_vector_width"] == cde_vector_width


def test_rocm_gemm_profile_workloads_cover_ck_candidate_set():
    ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=128, n=128, k=64)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    workloads = build_profile_workloads(ir, manifest)

    assert len(workloads) == 27
    assert {workload.kernel_library for workload in workloads} == {"ck_gemm"}
    assert {workload.candidate_id for workload in workloads} >= {
        "ck_gemm_rcr_bias_add_relu_float16_xdl_wide_m_v1",
        "ck_gemm_rcr_bias_add_relu_float16_xdl_codegen_t00_interwave_v1",
        "ck_gemm_rcr_bias_add_relu_float16_xdl_codegen_t00_default_v2",
    }
    assert workloads[0].alignment_context["kind"] == "ck_gemm_profile_alignment_context"


def test_rocm_gemm_profile_workloads_skip_v2_when_k_block_loop_is_odd():
    ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=32, n=32, k=32)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    workloads = build_profile_workloads(ir, manifest)

    candidate_ids = {workload.candidate_id for workload in workloads}
    assert candidate_ids == {
        "ck_gemm_rcr_bias_add_relu_float16_xdl_custom_v1",
        "ck_gemm_rcr_bias_add_relu_float16_xdl_codegen_t08_interwave_v1",
    }
    assert "ck_gemm_rcr_bias_add_relu_float16_xdl_codegen_t08_default_v2" not in candidate_ids


def test_rocm_ck_profile_failure_result_includes_diagnostics():
    ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=128, n=128, k=96)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    workload = build_profile_workloads(ir, manifest)[0]

    result = _profile_failure_result(workload, 1, profile_key="profile-key", error="profile failed")

    diagnostics = result["diagnostics"]
    assert diagnostics["kernel_library"] == "ck_gemm"
    assert diagnostics["candidate_id"] == workload.candidate_id
    assert diagnostics["profiler_symbol"] == workload.profiler_symbol
    assert diagnostics["problem"] == {"m": 128, "n": 128, "k": 96, "split_k": 1}
    assert diagnostics["ck"]["config"]["tile"]["k_per_block"] > 0
    assert result["candidates"][0]["diagnostics"] == diagnostics


def test_rocm_ck_profile_cache_entry_records_launch_metadata():
    ir = _rocm_bmm_ir("bmm_rcr_add", "float16", batch=2, m=64, n=256, k=96)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    workload = next(
        workload
        for workload in build_profile_workloads(ir, manifest)
        if workload.candidate_id == "ck_bmm_rcr_add_float16_xdl_wide_n_v1"
    )
    result = _profile_result(workload, 0.42, 7, profile_key="profile-key", status="ok")

    entry = _cache_entry(workload, result, {"profile_key_payload": "test"})

    assert entry["kernel_library"] == "ck_bmm"
    assert entry["best_candidate_id"] == workload.candidate_id
    assert entry["candidate_set_id"] == workload.candidate_set_id
    assert entry["candidate_set_key"] == workload.candidate_set_key
    assert entry["candidate_config_key"] == workload.candidate_config_key
    assert entry["launch_abi"] == "dinoml_ck_bmm_add_v1"
    assert entry["symbol_id"] == "xdl_wide_n_v1"
    assert entry["kernel_symbol"] == workload.kernel_symbol
    assert entry["profiler_symbol"] == workload.profiler_symbol
    assert entry["split_k"] == 1


def test_rocm_ck_profile_cache_round_trip_preserves_candidate_metadata():
    ir = _rocm_bmm_ir("bmm_rrc_add", "float16", batch=2, m=64, n=128, k=96)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    workload = next(
        workload
        for workload in build_profile_workloads(ir, manifest)
        if workload.candidate_id == "ck_bmm_rrc_add_float16_xdl_wide_n_v1"
    )
    result = _profile_result(workload, 0.37, 7, profile_key="profile-key", status="ok")
    entry = _cache_entry(workload, result, {"profile_key_payload": "test"})

    cached = _profile_result_from_cache(workload, entry)
    execution_plan = build_execution_plan(
        {
            "schema_version": 1,
            "profile_cache_schema_version": 7,
            "target": dict(manifest["target"]),
            "kernel_manifest_cache_key": manifest.get("cache_key"),
            "codegen_plan_cache_key": "test-codegen-plan",
            "problems": [cached],
        }
    )

    assert cached["status"] == "cached"
    assert cached["selected"] == {"candidate_id": workload.candidate_id, "split_k": 1, "reason": "cache_hit"}
    assert cached["kernel_library"] == "ck_bmm"
    assert cached["kernel_symbol"] == workload.kernel_symbol
    assert cached["profiler_symbol"] == workload.profiler_symbol
    assert cached["candidate_set_id"] == workload.candidate_set_id
    assert cached["candidate_set_key"] == workload.candidate_set_key
    assert cached["candidate_config_key"] == workload.candidate_config_key
    assert cached["candidates"][0]["candidate_config_key"] == workload.candidate_config_key
    assert cached["alignment_context"]["problem"]["base_layout"] == "rrc"
    assert cached["leading_dimensions"]["c"] == 64
    assert execution_plan["summary"]["static_selection_count"] == 1
    assert execution_plan["static_selections"][0]["selected_candidate_id"] == workload.candidate_id


@pytest.mark.parametrize(
    ("case", "target_candidate_id", "expected_library"),
    [
        (
            "gemm",
            "ck_gemm_rcr_bias_add_relu_float16_xdl_wide_n_v1",
            "ck_gemm",
        ),
        (
            "bmm",
            "ck_bmm_rcr_add_float16_xdl_wide_n_v1",
            "ck_bmm",
        ),
        (
            "conv",
            "ck_conv2d_bias_float16_xdl_wide_m_v1",
            "ck_conv",
        ),
    ],
)
def test_rocm_ck_execution_plan_static_overlay_selects_profiled_candidate(
    case: str,
    target_candidate_id: str,
    expected_library: str,
):
    if case == "gemm":
        ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=128, n=256, k=64)
    elif case == "bmm":
        ir = _rocm_bmm_ir("bmm_rcr_add", "float16", batch=2, m=128, n=128, k=192)
    elif case == "conv":
        ir = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16)
    else:
        raise AssertionError(f"unhandled CK overlay case: {case}")
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    original_item = manifest["required_kernels"][0]

    execution_plan = _ck_execution_plan_for_candidate(ir, manifest, target_candidate_id)
    overlaid = apply_execution_plan(manifest, execution_plan, strict=True)
    item = overlaid["required_kernels"][0]
    selection = item["execution_plan_selection"]

    assert original_item["kernel_library"] == expected_library
    assert original_item["selected_candidate_id"] != target_candidate_id
    assert execution_plan["summary"]["static_selection_count"] == 1
    assert execution_plan["summary"]["conflict_count"] == 0
    assert item["kernel_library"] == expected_library
    assert item["selected_candidate_id"] == target_candidate_id
    assert item["kernel_symbol"] == selection["kernel_symbol"]
    assert item["profiler_symbol"] == selection["profiler_symbol"]
    assert selection["kernel_library"] == expected_library
    assert selection["selected_candidate_id"] == target_candidate_id
    assert selection["workspace_nbytes"] == 0
    assert selection["split_k"] == 1


def test_rocm_ck_execution_plan_conflict_rejects_static_only_guarded_selection():
    ir_a = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=128, n=256, k=64)
    ir_b = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=128, n=128, k=64)
    manifest = build_kernel_manifest(ir_a, {"name": "rocm", "arch": "gfx1201"})
    original_candidate_id = manifest["required_kernels"][0]["selected_candidate_id"]
    alternate_candidate_id = "ck_gemm_rcr_bias_add_relu_float16_xdl_wide_n_v1"
    assert original_candidate_id != alternate_candidate_id

    result_a = _ck_profile_result_for_candidate(
        ir_a,
        manifest,
        str(original_candidate_id),
        elapsed_ms=0.31,
        profile_key="profile-original-shape",
    )
    result_b = _ck_profile_result_for_candidate(
        ir_b,
        manifest,
        alternate_candidate_id,
        elapsed_ms=0.29,
        profile_key="profile-alternate-shape",
    )
    execution_plan = build_execution_plan(
        {
            "schema_version": 1,
            "profile_cache_schema_version": 7,
            "target": dict(manifest["target"]),
            "kernel_manifest_cache_key": manifest.get("cache_key"),
            "codegen_plan_cache_key": "test-codegen-plan",
            "problems": [result_a, result_b],
        }
    )

    assert execution_plan["summary"]["static_selection_count"] == 0
    assert execution_plan["summary"]["conflict_count"] == 1
    assert execution_plan["conflicts"][0]["reason"] == "profiled_shapes_selected_different_candidate_or_split_k"
    assert execution_plan["conflicts"][0]["selected_candidate_ids"] == sorted(
        {str(original_candidate_id), alternate_candidate_id}
    )
    with pytest.raises(ValueError, match="ck_gemm execution plans only support static selections"):
        apply_execution_plan(manifest, execution_plan, strict=True)


def test_rocm_ck_gemm_execution_plan_prunes_support_exports(tmp_path):
    target_candidate_id = "ck_gemm_rcr_bias_add_relu_float16_xdl_wide_n_v1"
    ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16", m=128, n=256, k=64)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    execution_plan = _ck_execution_plan_for_candidate(ir, manifest, target_candidate_id)
    overlaid = apply_execution_plan(manifest, execution_plan, strict=True)

    plan = create_codegen_plan(overlaid, tmp_path)
    support = plan.external_support_libraries[0]
    entry = support["entries"][0]

    assert support["name"] == "ck_gemm"
    assert support["pruned_by_execution_plan"] is True
    assert support["kernel_symbols"] == ["dinoml_ck_gemm_rcr_bias_add_relu_float16_xdl_wide_n_v1"]
    assert support["profiler_symbols"] == ["dinoml_profile_ck_gemm_rcr_bias_add_relu_float16_xdl_wide_n_v1"]
    assert plan.candidate_profiler_symbols == ("dinoml_profile_ck_gemm_rcr_bias_add_relu_float16_xdl_wide_n_v1",)
    assert support["candidate_config_keys"] == [entry["selected_candidate"]["candidate_config_key"]]
    assert entry["pruned_by_execution_plan"] is True
    assert entry["execution_plan_selection"]["selected_candidate_id"] == target_candidate_id
    assert entry["execution_plan_selection"]["kernel_symbol"] == support["kernel_symbols"][0]
    assert entry["selected_candidate_id"] == target_candidate_id
    assert [candidate["candidate_id"] for candidate in entry["candidates"]] == [target_candidate_id]


@pytest.mark.parametrize(
    ("case", "target_candidate_id", "support_name", "kernel_symbol", "profiler_symbol"),
    [
        (
            "bmm",
            "ck_bmm_rcr_add_float16_xdl_wide_n_v1",
            "ck_bmm",
            "dinoml_ck_bmm_rcr_add_float16_xdl_wide_n_v1",
            "dinoml_profile_ck_bmm_rcr_add_float16_xdl_wide_n_v1",
        ),
        (
            "conv",
            "ck_conv2d_bias_float16_xdl_wide_m_v1",
            "ck_conv",
            "dinoml_ck_conv2d_bias_float16_xdl_wide_m_v1",
            "dinoml_profile_ck_conv2d_bias_float16_xdl_wide_m_v1",
        ),
    ],
)
def test_rocm_ck_bmm_conv_execution_plan_prunes_support_exports(
    tmp_path,
    case: str,
    target_candidate_id: str,
    support_name: str,
    kernel_symbol: str,
    profiler_symbol: str,
):
    if case == "bmm":
        ir = _rocm_bmm_ir("bmm_rcr_add", "float16", batch=2, m=128, n=128, k=192)
    elif case == "conv":
        ir = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16)
    else:
        raise AssertionError(f"unhandled CK pruning case: {case}")
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    execution_plan = _ck_execution_plan_for_candidate(ir, manifest, target_candidate_id)
    overlaid = apply_execution_plan(manifest, execution_plan, strict=True)

    plan = create_codegen_plan(overlaid, tmp_path)
    support = plan.external_support_libraries[0]
    entry = support["entries"][0]

    assert support["name"] == support_name
    assert support["pruned_by_execution_plan"] is True
    assert support["kernel_symbols"] == [kernel_symbol]
    assert support["profiler_symbols"] == [profiler_symbol]
    assert plan.candidate_profiler_symbols == (profiler_symbol,)
    assert support["candidate_config_keys"] == [entry["selected_candidate"]["candidate_config_key"]]
    assert entry["pruned_by_execution_plan"] is True
    assert entry["execution_plan_selection"]["selected_candidate_id"] == target_candidate_id
    assert entry["execution_plan_selection"]["kernel_symbol"] == kernel_symbol
    assert entry["selected_candidate_id"] == target_candidate_id
    assert [candidate["candidate_id"] for candidate in entry["candidates"]] == [target_candidate_id]


def test_ck_profile_wrappers_return_launch_status_for_diagnostics():
    for source_path in (
        Path("kernels/rocm/src/ck_gemm.hip"),
        Path("kernels/rocm/src/ck_bmm.hip"),
        Path("kernels/rocm/src/ck_conv.hip"),
    ):
        source = source_path.read_text(encoding="utf-8")
        assert "const int launch_status = launch_ck_" in source
        assert "return -static_cast<float>(launch_status);" in source


@pytest.mark.parametrize(
    ("source_path", "renderer", "missing_symbol"),
    [
        (
            Path("kernels/rocm/src/ck_gemm.hip"),
            render_ck_gemm_source,
            "dinoml_ck_gemm_rcr_bias_add_relu_float16_xdl_missing_v1",
        ),
        (
            Path("kernels/rocm/src/ck_bmm.hip"),
            render_ck_bmm_source,
            "dinoml_ck_bmm_rcr_add_float16_xdl_missing_v1",
        ),
        (
            Path("kernels/rocm/src/ck_conv.hip"),
            render_ck_conv_source,
            "dinoml_ck_conv2d_bias_float16_xdl_missing_v1",
        ),
    ],
)
def test_ck_source_renderers_reject_missing_plan_symbols(source_path, renderer, missing_symbol):
    source = source_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="missing symbols"):
        renderer(source, {"kernel_symbols": [missing_symbol], "profiler_symbols": []})


def test_rocm_gemm_module_declares_and_calls_ck_symbol():
    ir = _rocm_gemm_ir("gemm_rcr_bias_add_relu", "float16")
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    source = render_rocm_module(ir, kernel_manifest=manifest)

    assert 'extern "C" int dinoml_ck_gemm_rcr_bias_add_relu_float16_xdl_custom_v1' in source
    assert "const half* bias" in source
    assert "const half* d0" in source
    assert (
        "dinoml_ck_gemm_rcr_bias_add_relu_float16_xdl_custom_v1(ptr_a, ptr_b, ptr_bias, ptr_d0, ptr_c"
        in source
    )


def test_ck_gemm_unit_generation_uses_device_op_not_reference_path():
    source = Path("kernels/rocm/src/ck_gemm.hip").read_text(encoding="utf-8")
    candidate = {
        "op": "gemm_rcr_bias_add_relu",
        "dtype": "float16",
        "symbol_id": "xdl_custom_v1",
        "epilogue": "bias_add_relu",
        "launch_abi": "dinoml_ck_gemm_bias_residual_v1",
    }

    rendered = render_ck_gemm_source(
        source,
        {
            "candidates": [candidate],
            "kernel_symbols": ["dinoml_ck_gemm_rcr_bias_add_relu_float16_xdl_custom_v1"],
            "profiler_symbols": ["dinoml_profile_ck_gemm_rcr_bias_add_relu_float16_xdl_custom_v1"],
        },
    )

    assert "ReferenceGemm" not in rendered
    assert "fused_gemm_reference_kernel" not in rendered
    assert "DeviceGemmMultipleD_Xdl_CShuffle" in rendered
    assert (
        "DINOML_CK_GEMM_BIAS_RESIDUAL_EXPORT(gemm_rcr_bias_add_relu, float16, half, "
        "xdl_custom_v1, kRcr, kBiasAddRelu, kBaseline)"
    ) in rendered


def test_rocm_bmm_manifest_selects_ck_custom_xdl_archive(tmp_path):
    ir = _rocm_bmm_ir("bmm_rcr_add", "float16")

    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]
    plan = create_codegen_plan(manifest, tmp_path)

    assert item["kernel_library"] == "ck_bmm"
    assert item["kernel_symbol"] == "dinoml_ck_bmm_rcr_add_float16_xdl_custom_v1"
    assert item["support_archive"] == ck_bmm_static_library_name("bmm_rcr_add", "float16")
    assert len(item["candidates"]) == 7
    assert item["candidates"][0]["ck"]["api"] == "device_batched_gemm_multiple_d_xdl_cshuffle_v3"
    assert item["candidates"][0]["ck"]["mode"] == "custom_ck_xdl_instances"
    assert item["candidates"][1]["ck"]["config"]["name"] == "wide_m"
    assert {candidate["ck"]["config"]["name"] for candidate in item["candidates"]} == {
        "baseline",
        "wide_m",
        "wide_n",
        "square",
        "skinny_m",
        "skinny_n",
        "small",
    }
    assert plan.external_support_libraries[0]["name"] == "ck_bmm"
    assert plan.external_support_libraries[0]["modules"] == [
        {
            "op": "bmm_rcr_add",
            "dtype": "float16",
            "archive": f"lib/{ck_bmm_static_library_name('bmm_rcr_add', 'float16')}",
            "target": "dinoml_ck_bmm_bmm_rcr_add_float16",
        }
    ]
    _assert_ck_support_plan_unpruned(plan, item)


def test_rocm_bmm_manifest_selects_tuned_ck_candidate_for_aligned_static_shape():
    ir = _rocm_bmm_ir("bmm_rcr_add", "float16", batch=2, m=128, n=128, k=96)

    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    assert item["selected_candidate_id"] == "ck_bmm_rcr_add_float16_xdl_wide_m_v1"
    assert item["kernel_symbol"] == "dinoml_ck_bmm_rcr_add_float16_xdl_wide_m_v1"


def test_rocm_bmm_profile_workloads_cover_ck_candidate_set():
    ir = _rocm_bmm_ir("bmm_rcr_add", "float16", batch=2, m=128, n=128, k=192)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    workloads = build_profile_workloads(ir, manifest)

    assert len(workloads) == 7
    assert {workload.kernel_library for workload in workloads} == {"ck_bmm"}
    assert {workload.candidate_id for workload in workloads} >= {
        "ck_bmm_rcr_add_float16_xdl_wide_m_v1",
        "ck_bmm_rcr_add_float16_xdl_small_v1",
    }
    assert workloads[0].batch_count == 2
    assert workloads[0].alignment_context["kind"] == "ck_bmm_profile_alignment_context"


def test_rocm_bmm_profile_workloads_skip_v3_when_k_block_loop_is_too_short():
    ir = _rocm_bmm_ir("bmm_rcr_add", "float16", batch=2, m=128, n=128, k=96)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    workloads = build_profile_workloads(ir, manifest)

    candidate_names = {workload.candidate["ck"]["config"]["name"] for workload in workloads}
    assert candidate_names == {"baseline", "wide_m", "wide_n", "small"}
    assert all(workload.candidate["ck"]["config"]["tile"]["k_per_block"] == 32 for workload in workloads)
    assert "ck_bmm_rcr_add_float16_xdl_square_v1" not in {workload.candidate_id for workload in workloads}


@pytest.mark.parametrize(
    ("m", "n", "k", "selected_candidate_id", "profile_config_names"),
    [
        (
            64,
            256,
            96,
            "ck_bmm_rcr_add_float16_xdl_wide_n_v1",
            {"baseline", "wide_n", "small"},
        ),
        (
            16,
            128,
            192,
            "ck_bmm_rcr_add_float16_xdl_skinny_m_v1",
            {"baseline", "skinny_m", "small"},
        ),
        (
            128,
            16,
            192,
            "ck_bmm_rcr_add_float16_xdl_skinny_n_v1",
            {"baseline", "skinny_n", "small"},
        ),
    ],
)
def test_rocm_bmm_profile_workloads_cover_representative_ck_shape_classes(
    m: int,
    n: int,
    k: int,
    selected_candidate_id: str,
    profile_config_names: set[str],
):
    ir = _rocm_bmm_ir("bmm_rcr_add", "float16", batch=2, m=m, n=n, k=k)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    workloads = build_profile_workloads(ir, manifest)

    assert item["selected_candidate_id"] == selected_candidate_id
    assert {workload.candidate["ck"]["config"]["name"] for workload in workloads} == profile_config_names
    assert all(workload.batch_count == 2 for workload in workloads)
    assert all((workload.m, workload.n, workload.k) == (m, n, k) for workload in workloads)
    assert all(workload.candidate["ck"]["config"]["pipeline"] == "v3" for workload in workloads)


def test_rocm_bmm_profile_workload_preserves_broadcast_batch_and_bias_layout():
    ir = _rocm_bmm_ir(
        "bmm_rcr_add",
        "float16",
        batch=4,
        a_batch=1,
        b_batch=4,
        m=64,
        n=128,
        k=96,
        d0_shape=[128],
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    workloads = build_profile_workloads(ir, manifest)
    workload = workloads[0]

    assert workload.batch_count == 4
    assert workload.a_shape == (1, 64, 96)
    assert workload.b_shape == (4, 128, 96)
    assert workload.output_shape == (4, 64, 128)
    assert workload.residual_shapes == ((128,),)
    assert workload.batch_stride_a == 0
    assert workload.batch_stride_b == 128 * 96
    assert workload.batch_stride_c == 64 * 128
    assert workload.batch_stride_d0 == 0
    assert workload.ldd0 == 0
    assert workload.ldc == 128


def test_rocm_bmm_rrr_profile_workloads_use_layout_specific_alignment():
    ir = _rocm_bmm_ir("bmm_rrr_add", "float16", batch=2, m=64, n=128, k=96)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    workloads = build_profile_workloads(ir, manifest)
    workload = workloads[0]

    assert item["selected_candidate_id"] == "ck_bmm_rrr_add_float16_xdl_wide_n_v1"
    assert item["candidates"][0]["layouts"] == {"a": "row", "b": "row", "c": "row"}
    assert {workload.candidate["ck"]["config"]["name"] for workload in workloads} == {"baseline", "wide_n", "small"}
    assert workload.a_shape == (2, 64, 96)
    assert workload.b_shape == (2, 96, 128)
    assert workload.output_shape == (2, 64, 128)
    assert workload.lda == 96
    assert workload.ldb == 128
    assert workload.ldc == 128
    assert workload.alignment_context["problem"]["base_layout"] == "rrr"
    assert workload.alignment_context["problem"]["b_n"] == 128


def test_rocm_bmm_rrc_profile_workloads_preserve_column_output_layout():
    ir = _rocm_bmm_ir("bmm_rrc_add", "float16", batch=2, m=64, n=128, k=96)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    workloads = build_profile_workloads(ir, manifest)
    workload = workloads[0]

    assert item["selected_candidate_id"] == "ck_bmm_rrc_add_float16_xdl_wide_n_v1"
    assert item["candidates"][0]["layouts"] == {"a": "row", "b": "row", "c": "column"}
    assert {workload.candidate["ck"]["config"]["name"] for workload in workloads} == {"baseline", "wide_n", "small"}
    assert workload.a_shape == (2, 64, 96)
    assert workload.b_shape == (2, 96, 128)
    assert workload.output_shape == (2, 128, 64)
    assert workload.residual_shapes == ((2, 128, 64),)
    assert workload.ldc == 64
    assert workload.ldd0 == 64
    assert workload.batch_stride_c == 64 * 128
    assert workload.batch_stride_d0 == 64 * 128
    assert workload.alignment_context["problem"]["base_layout"] == "rrc"
    assert workload.alignment_context["problem"]["output_layout"] == "c"
    assert workload.alignment_context["problem"]["output_n"] == 128


def test_rocm_bmm_module_declares_and_calls_ck_symbol():
    ir = _rocm_bmm_ir("bmm_rcr_add", "float16")
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    source = render_rocm_module(ir, kernel_manifest=manifest)

    assert 'extern "C" int dinoml_ck_bmm_rcr_add_float16_xdl_custom_v1' in source
    assert "const half* d0" in source
    assert "int64_t batch_stride_d0" in source
    assert "dinoml_ck_bmm_rcr_add_float16_xdl_custom_v1(ptr_a, ptr_b, ptr_d0, ptr_c" in source
    assert "CK BMM launcher failed" in source


def test_ck_bmm_unit_generation_uses_device_op_not_reference_path():
    source = Path("kernels/rocm/src/ck_bmm.hip").read_text(encoding="utf-8")
    candidate = {
        "op": "bmm_rcr_add",
        "dtype": "float16",
        "symbol_id": "xdl_custom_v1",
        "epilogue": "add",
        "launch_abi": "dinoml_ck_bmm_add_v1",
    }

    rendered = render_ck_bmm_source(
        source,
        {
            "candidates": [candidate],
            "kernel_symbols": ["dinoml_ck_bmm_rcr_add_float16_xdl_custom_v1"],
            "profiler_symbols": ["dinoml_profile_ck_bmm_rcr_add_float16_xdl_custom_v1"],
        },
    )

    assert "ReferenceBatchedGemm" not in rendered
    assert "reference" not in rendered.lower()
    assert "DeviceBatchedGemmMultiD_Xdl_CShuffle_V3" in rendered
    assert (
        "DINOML_CK_BMM_ADD_EXPORT(bmm_rcr_add, float16, half, "
        "xdl_custom_v1, kRcr, kAdd, kBaseline)"
    ) in rendered


def test_rocm_conv2d_bias_manifest_selects_ck_custom_xdl_archive(tmp_path):
    ir = _rocm_conv2d_bias_ir("float16")

    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]
    plan = create_codegen_plan(manifest, tmp_path)

    assert item["kernel_library"] == "ck_conv"
    assert item["kernel_symbol"] == "dinoml_ck_conv2d_bias_float16_xdl_custom_v1"
    assert item["support_archive"] == ck_conv_static_library_name("conv2d_bias", "float16")
    assert len(item["candidates"]) == 7
    assert item["candidates"][0]["ck"]["api"] == "device_grouped_conv_fwd_multiple_abd_xdl_cshuffle"
    assert item["candidates"][0]["ck"]["mode"] == "custom_ck_xdl_instances"
    assert item["candidates"][1]["ck"]["config"]["name"] == "wide_n"
    assert {candidate["ck"]["config"]["name"] for candidate in item["candidates"]} == {
        "baseline",
        "wide_n",
        "wide_m",
        "square",
        "skinny_m",
        "skinny_n",
        "small",
    }
    assert plan.external_support_libraries[0]["name"] == "ck_conv"
    assert plan.external_support_libraries[0]["modules"] == [
        {
            "op": "conv2d_bias",
            "dtype": "float16",
            "archive": f"lib/{ck_conv_static_library_name('conv2d_bias', 'float16')}",
            "target": "dinoml_ck_conv_conv2d_bias_float16",
        }
    ]
    _assert_ck_support_plan_unpruned(plan, item)


def test_rocm_conv2d_bias_manifest_selects_tuned_ck_candidate_for_aligned_static_shape():
    ir = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16)

    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    assert item["selected_candidate_id"] == "ck_conv2d_bias_float16_xdl_wide_n_v1"
    assert item["kernel_symbol"] == "dinoml_ck_conv2d_bias_float16_xdl_wide_n_v1"


def test_rocm_conv2d_bias_profile_workloads_cover_ck_candidate_set():
    ir = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    workloads = build_profile_workloads(ir, manifest)

    assert len(workloads) == 7
    assert {workload.kernel_library for workload in workloads} == {"ck_conv"}
    assert {workload.candidate_id for workload in workloads} >= {
        "ck_conv2d_bias_float16_xdl_wide_n_v1",
        "ck_conv2d_bias_float16_xdl_small_v1",
    }
    assert workloads[0].conv_config == {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1}


def test_rocm_conv2d_bias_profile_workloads_skip_unsupported_groups():
    ir = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16, groups=2)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    workloads = build_profile_workloads(ir, manifest)

    assert item["profile_blocked_reason"] == "ck_conv_groups_unsupported_for_profile"
    assert item["profile_blocked_details"] == {"groups": 2, "supported_groups": [1]}
    assert workloads == []


def test_rocm_conv2d_bias_profile_report_summarizes_blocked_groups(tmp_path):
    ir = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16, groups=2)
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    blocked_items = _blocked_profile_items(manifest)

    report = _profile_report(
        tmp_path,
        {"target": {"name": "rocm", "arch": "gfx1201"}},
        manifest,
        {"cache_key": "test-codegen-cache"},
        5,
        3,
        [],
        {"profiled": 0, "cached": 0, "skipped": 0, "failed": 0, "blocked": len(blocked_items)},
        context={
            "fingerprint": {
                "hardware": {"name": "test-gpu"},
                "hardware_key": "test-hardware",
                "support_libraries": [],
                "support_libraries_key": "test-support-libraries",
            }
        },
        blocked_profile_items=blocked_items,
    )

    assert report["summary"]["blocked"] == 1
    assert report["problems"] == []
    assert report["blocked_profile_items"] == [
        {
            "op": "conv2d_bias",
            "dtype": "float16",
            "kernel_library": "ck_conv",
            "kernel_symbol": manifest["required_kernels"][0]["kernel_symbol"],
            "profiler_symbol": manifest["required_kernels"][0]["profiler_symbol"],
            "candidate_set_id": manifest["required_kernels"][0]["candidate_set_id"],
            "candidate_set_key": manifest["required_kernels"][0]["candidate_set_key"],
            "selected_candidate_id": manifest["required_kernels"][0]["selected_candidate_id"],
            "reason": "ck_conv_groups_unsupported_for_profile",
            "details": {"groups": 2, "supported_groups": [1]},
        }
    ]


def test_rocm_conv2d_bias_manifest_reports_malformed_ck_profile_attrs():
    ir = _rocm_conv2d_bias_ir(
        "float16",
        batch=2,
        in_channels=8,
        out_channels=64,
        height=16,
        width=16,
        dilation=[1, 0],
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    workloads = build_profile_workloads(ir, manifest)

    assert item["profile_blocked_reason"] == "ck_conv_attrs_unsupported_for_profile"
    assert "conv2d_bias dilation must contain positive integers" in item["profile_blocked_details"]["error"]
    assert workloads == []
    assert _blocked_profile_items(manifest)[0]["reason"] == "ck_conv_attrs_unsupported_for_profile"


@pytest.mark.parametrize(
    ("batch", "out_channels", "height", "width", "selected_candidate_id", "profile_config_names"),
    [
        (
            1,
            32,
            16,
            16,
            "ck_conv2d_bias_float16_xdl_wide_m_v1",
            {"baseline", "wide_m", "square", "skinny_n", "small"},
        ),
        (
            1,
            64,
            4,
            4,
            "ck_conv2d_bias_float16_xdl_skinny_m_v1",
            {"baseline", "skinny_m", "small"},
        ),
        (
            1,
            16,
            8,
            8,
            "ck_conv2d_bias_float16_xdl_skinny_n_v1",
            {"baseline", "skinny_n", "small"},
        ),
        (
            1,
            16,
            4,
            4,
            "ck_conv2d_bias_float16_xdl_small_v1",
            {"baseline", "small"},
        ),
    ],
)
def test_rocm_conv2d_bias_profile_workloads_cover_representative_ck_shape_classes(
    batch: int,
    out_channels: int,
    height: int,
    width: int,
    selected_candidate_id: str,
    profile_config_names: set[str],
):
    ir = _rocm_conv2d_bias_ir(
        "float16",
        batch=batch,
        in_channels=8,
        out_channels=out_channels,
        height=height,
        width=width,
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    item = manifest["required_kernels"][0]

    workloads = build_profile_workloads(ir, manifest)

    assert item["selected_candidate_id"] == selected_candidate_id
    assert {workload.candidate["ck"]["config"]["name"] for workload in workloads} == profile_config_names
    assert all(workload.x_shape == (batch, 8, height, width) for workload in workloads)
    assert all(workload.weight_shape == (out_channels, 8, 3, 3) for workload in workloads)
    assert all(workload.output_shape == (batch, out_channels, height, width) for workload in workloads)
    assert all(workload.candidate["ck"]["config"]["pipeline"] == "v1" for workload in workloads)
    assert all(workload.semantic_layout["activation"] == "nchw" for workload in workloads)
    assert all(workload.provider_layout["activation"] == "g_nhw_c_strided" for workload in workloads)


def test_rocm_conv2d_bias_profile_workload_preserves_conv_config_and_output_shape():
    ir = _rocm_conv2d_bias_ir(
        "float16",
        batch=1,
        in_channels=8,
        out_channels=16,
        height=9,
        width=11,
        stride=[2, 2],
        padding=[0, 1],
        dilation=[1, 1],
    )
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    workloads = build_profile_workloads(ir, manifest)
    workload = workloads[0]

    assert workload.x_shape == (1, 8, 9, 11)
    assert workload.weight_shape == (16, 8, 3, 3)
    assert workload.bias_shape == (16,)
    assert workload.output_shape == (1, 16, 4, 6)
    assert workload.conv_config == {"stride": [2, 2], "padding": [0, 1], "dilation": [1, 1], "groups": 1}


def test_rocm_conv2d_bias_module_declares_and_calls_ck_symbol():
    ir = _rocm_conv2d_bias_ir("float16")
    manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})

    source = render_rocm_module(ir, kernel_manifest=manifest)

    assert 'extern "C" int dinoml_ck_conv2d_bias_float16_xdl_custom_v1' in source
    assert "const half* weight" in source
    assert "const half* bias" in source
    assert (
        "dinoml_ck_conv2d_bias_float16_xdl_custom_v1(ptr_x, ptr_weight, ptr_bias, ptr_y"
        in source
    )
    assert "CK Conv launcher failed" in source


def test_ck_conv_unit_generation_uses_device_op_not_reference_path():
    source = Path("kernels/rocm/src/ck_conv.hip").read_text(encoding="utf-8")
    candidate = {
        "op": "conv2d_bias",
        "dtype": "float16",
        "symbol_id": "xdl_custom_v1",
        "epilogue": "bias",
        "launch_abi": "dinoml_ck_conv2d_bias_v1",
    }

    rendered = render_ck_conv_source(
        source,
        {
            "candidates": [candidate],
            "kernel_symbols": ["dinoml_ck_conv2d_bias_float16_xdl_custom_v1"],
            "profiler_symbols": ["dinoml_profile_ck_conv2d_bias_float16_xdl_custom_v1"],
        },
    )

    assert "ReferenceConv" not in rendered
    assert "reference" not in rendered.lower()
    assert "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle" in rendered
    assert "DinoConvBiasEpilogue" in rendered
    assert "DINOML_CK_CONV2D_BIAS_EXPORT(conv2d_bias, float16, half, xdl_custom_v1, kBaseline)" in rendered


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE=1 with rocm-sdk on PATH",
)
def test_rocm_support_libraries_build_with_real_toolchain(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    libs = ensure_rocm_support_libs("gfx1201")

    assert libs.runtime_lib.exists()
    assert libs.rocm_runtime_lib.exists()
    assert libs.kernels_lib.exists()
    assert (Path(libs.rocm_runtime_lib).parent / "support_manifest.json").exists()


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_MODULE_COMPILE_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_MODULE_COMPILE_SMOKE=1 with rocm-sdk on the active Python/PATH",
)
def test_rocm_fused_elementwise_module_compiles_with_real_toolchain(tmp_path, monkeypatch):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))
    case = elementwise_case()

    artifact = dml.compile(case.build_spec(), dml.Target("rocm"), tmp_path / "elementwise_rocm.dinoml")

    assert (artifact.path / "module.so").exists()
    if os.name == "nt":
        assert not list(artifact.path.glob("amdhip*.dll"))
        assert not list((artifact.path / "lib").glob("amdhip*.dll"))
    manifest = read_json(artifact.path / "manifest.json")
    assert manifest["target"]["name"] == "rocm"
    assert manifest["files"]["rocm_runtime_library"].endswith("dinoml_rocm_runtime.dll" if os.name == "nt" else "libdinoml_rocm_runtime.so")
    source_manifest = read_json(artifact.path / "debug" / "generated_src" / "source_manifest.json")
    assert source_manifest["target"] == "rocm"
    assert {source["op"] for source in source_manifest["sources"]} == {"fused_elementwise"}
    assert all(str(source["emitted_source_path"]).endswith(".hip") for source in source_manifest["sources"])

    module = load(artifact.path, load_constants=False)
    try:
        assert module.target_name == "rocm"
    finally:
        module.close()


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_PROFILE_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_PROFILE_SMOKE=1 with a working ROCm device/toolchain",
)
def test_rocm_ck_gemm_compile_profile_smoke_with_real_toolchain(tmp_path, monkeypatch):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    class TinyCkGemm(dml.Module):
        def forward(self, a, b, bias, d0):
            return {"c": dml.ops.gemm_rcr_bias_add_relu(a, b, bias, d0)}

    spec = dml.trace(
        TinyCkGemm(),
        {
            "a": dml.TensorSpec([32, 32], "float16"),
            "b": dml.TensorSpec([32, 32], "float16"),
            "bias": dml.TensorSpec([32], "float16"),
            "d0": dml.TensorSpec([32, 32], "float16"),
        },
        name="rocm_ck_gemm_compile_profile_smoke",
    )

    artifact = dml.compile(
        spec,
        dml.Target("rocm"),
        tmp_path / "ck_gemm_profile_rocm.dinoml",
        profile=True,
        profile_iterations=1,
        profile_repeats=1,
        profile_refresh=True,
    )

    manifest = read_json(artifact.path / "manifest.json")
    report = read_json(artifact.path / "debug" / "bootstrap_profile_report.json")
    assert manifest["target"]["name"] == "rocm"
    assert report["target"]["name"] == "rocm"
    assert report["summary"]["profiled"] >= 1
    assert report["summary"]["failed"] == 0
    assert all(problem["kernel_library"] == "ck_gemm" for problem in report["problems"])


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_HEADER_COMPILE_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_HEADER_COMPILE_SMOKE=1 with hipcc or .venv/rocm available",
)
def test_rocm_common_headers_compile_with_real_hipcc(tmp_path):
    hipcc = _find_hipcc()
    if hipcc is None:
        pytest.skip("hipcc not found on PATH, HIPCC, or .venv/rocm/Scripts")
    repo_root = Path(__file__).resolve().parents[2]
    source = tmp_path / "dinoml_rocm_header_smoke.hip"
    obj = tmp_path / ("dinoml_rocm_header_smoke.obj" if os.name == "nt" else "dinoml_rocm_header_smoke.o")
    source.write_text(
        r"""
#include <dinoml/device.h>
#include <dinoml/runtime_rocm.h>
#include <dinoml/math.h>
#include <dinoml/tensor_accessor.h>

extern "C" __global__ void dinoml_rocm_header_smoke_kernel(float* y, half* h, dinoml::bfloat16* b) {
  y[0] = dinoml::math::to_float(h[0]) + dinoml::math::to_float(b[0]);
  h[0] = dinoml::math::from_float<half>(y[0]);
  b[0] = dinoml::math::from_float<dinoml::bfloat16>(y[0]);
  dinoml::access::TensorAccessor accessor;
  y[1] = static_cast<float>(accessor.index(0)) + LDG(y);
}

extern "C" int dinoml_rocm_header_smoke_launch(float* y, half* h, dinoml::bfloat16* b, dinoml::DeviceStream stream) {
  dinoml_rocm_header_smoke_kernel<<<1, 1, 0, stream>>>(y, h, b);
  DINO_ROCM_CHECK(hipGetLastError());
  return 0;
}
""",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            str(hipcc),
            "--offload-arch=" + os.environ.get("DINOML_ROCM_TEST_ARCH", "gfx1201"),
            "-std=c++17",
            "-I",
            str(repo_root / "runtime" / "include"),
            "-I",
            str(repo_root / "kernels" / "common" / "include"),
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    _wait_for_path(obj, timeout_s=10.0)
    assert proc.returncode == 0 and obj.exists(), (
        f"hipcc failed to compile ROCm header smoke\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def _find_hipcc() -> Path | None:
    configured = os.environ.get("HIPCC")
    if configured:
        path = Path(configured)
        if path.exists():
            return path
    found = shutil.which("hipcc")
    if found is not None:
        return Path(found)
    repo_root = Path(__file__).resolve().parents[2]
    suffix = "hipcc.exe" if os.name == "nt" else "hipcc"
    local = repo_root / ".venv" / "rocm" / "Scripts" / suffix
    if local.exists():
        return local
    return None


def _ck_execution_plan_for_candidate(
    ir: dict,
    manifest: dict,
    candidate_id: str,
) -> dict:
    result = _ck_profile_result_for_candidate(
        ir,
        manifest,
        candidate_id,
        0.25,
        profile_key=f"profile-{candidate_id}",
    )
    return build_execution_plan(
        {
            "schema_version": 1,
            "profile_cache_schema_version": 7,
            "target": dict(manifest["target"]),
            "kernel_manifest_cache_key": manifest.get("cache_key"),
            "codegen_plan_cache_key": "test-codegen-plan",
            "problems": [result],
        }
    )


def _ck_profile_result_for_candidate(
    ir: dict,
    manifest: dict,
    candidate_id: str,
    elapsed_ms: float,
    *,
    profile_key: str,
) -> dict:
    workloads = build_profile_workloads(ir, manifest)
    matches = [workload for workload in workloads if workload.candidate_id == candidate_id]
    assert matches, f"candidate {candidate_id!r} was not profileable"
    return _profile_result(
        matches[0],
        elapsed_ms,
        5,
        profile_key=profile_key,
        status="ok",
        reason="test_selected_candidate",
    )


def _assert_ck_support_plan_unpruned(plan, item: dict) -> None:
    support = plan.external_support_libraries[0]
    entry = support["entries"][0]
    expected_config_keys = {str(candidate["candidate_config_key"]) for candidate in item["candidates"]}

    assert support["pruned_by_execution_plan"] is False
    assert entry["pruned_by_execution_plan"] is False
    assert "execution_plan_selection" not in entry
    assert set(support["candidate_config_keys"]) == expected_config_keys
    assert set(entry["candidate_config_keys"]) == expected_config_keys
    assert {str(candidate["candidate_config_key"]) for candidate in entry["candidates"]} == expected_config_keys
    assert len(entry["candidates"]) == len(item["candidates"])


def _rocm_gemm_ir(op_name: str, dtype: str, *, m: int = 2, n: int = 4, k: int = 3) -> dict:
    tensors = [
        _tensor("a", [m, k], dtype, "input"),
        _tensor("b", [n, k], dtype, "input"),
        _tensor("bias", [n], dtype, "input"),
        _tensor("d0", [m, n], dtype, "input"),
        _tensor("c", [m, n], dtype, "output"),
    ]
    return {
        "schema_version": 1,
        "name": "rocm_gemm_smoke",
        "inputs": [
            _io("a", [m, k], dtype),
            _io("b", [n, k], dtype),
            _io("bias", [n], dtype),
            _io("d0", [m, n], dtype),
        ],
        "constants": [],
        "outputs": [_io("c", [m, n], dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": op_name,
                "inputs": ["a", "b", "bias", "d0"],
                "outputs": ["c"],
                "attrs": {},
            }
        ],
        "tensors": tensors,
        "metadata": {},
    }


def _rocm_bmm_ir(
    op_name: str,
    dtype: str,
    *,
    batch: int = 2,
    a_batch: int | None = None,
    b_batch: int | None = None,
    m: int = 3,
    n: int = 4,
    k: int = 5,
    d0_shape: list[int] | None = None,
) -> dict:
    layout = _bmm_layout_from_op(op_name)
    a_batch = batch if a_batch is None else a_batch
    b_batch = batch if b_batch is None else b_batch
    a_shape = [a_batch, k, m] if layout[0] == "c" else [a_batch, m, k]
    b_shape = [b_batch, n, k] if layout[1] == "c" else [b_batch, k, n]
    output_shape = [batch, n, m] if layout[2] == "c" else [batch, m, n]
    d0_shape = output_shape if d0_shape is None else list(d0_shape)
    tensors = [
        _tensor("a", a_shape, dtype, "input"),
        _tensor("b", b_shape, dtype, "input"),
        _tensor("d0", d0_shape, dtype, "input"),
        _tensor("c", output_shape, dtype, "output"),
    ]
    return {
        "schema_version": 1,
        "name": "rocm_bmm_smoke",
        "inputs": [
            _io("a", a_shape, dtype),
            _io("b", b_shape, dtype),
            _io("d0", d0_shape, dtype),
        ],
        "constants": [],
        "outputs": [_io("c", output_shape, dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": op_name,
                "inputs": ["a", "b", "d0"],
                "outputs": ["c"],
                "attrs": {},
            }
        ],
        "tensors": tensors,
        "metadata": {},
    }


def _bmm_layout_from_op(op_name: str) -> str:
    layout = op_name.removeprefix("bmm_").removesuffix("_add")
    assert len(layout) == 3 and set(layout) <= {"c", "r"}
    return layout


def _rocm_conv2d_bias_ir(
    dtype: str,
    *,
    batch: int = 2,
    in_channels: int = 4,
    out_channels: int = 6,
    height: int = 8,
    width: int = 8,
    kernel_h: int = 3,
    kernel_w: int = 3,
    stride: list[int] | tuple[int, int] | None = None,
    padding: list[int] | tuple[int, int] | None = None,
    dilation: list[int] | tuple[int, int] | None = None,
    groups: int = 1,
) -> dict:
    stride = [1, 1] if stride is None else [int(stride[0]), int(stride[1])]
    padding = [1, 1] if padding is None else [int(padding[0]), int(padding[1])]
    dilation = [1, 1] if dilation is None else [int(dilation[0]), int(dilation[1])]
    output_h = (height + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    output_w = (width + 2 * padding[1] - dilation[1] * (kernel_w - 1) - 1) // stride[1] + 1
    tensors = [
        _tensor("x", [batch, in_channels, height, width], dtype, "input"),
        _tensor("weight", [out_channels, in_channels, kernel_h, kernel_w], dtype, "input"),
        _tensor("bias", [out_channels], dtype, "input"),
        _tensor("y", [batch, out_channels, output_h, output_w], dtype, "output"),
    ]
    return {
        "schema_version": 1,
        "name": "rocm_conv2d_bias_smoke",
        "inputs": [
            _io("x", [batch, in_channels, height, width], dtype),
            _io("weight", [out_channels, in_channels, kernel_h, kernel_w], dtype),
            _io("bias", [out_channels], dtype),
        ],
        "constants": [],
        "outputs": [_io("y", [batch, out_channels, output_h, output_w], dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": "conv2d_bias",
                "inputs": ["x", "weight", "bias"],
                "outputs": ["y"],
                "attrs": {"stride": stride, "padding": padding, "dilation": dilation, "groups": groups},
            }
        ],
        "tensors": tensors,
        "metadata": {},
    }


def _io(name: str, shape: list[int], dtype: str) -> dict:
    return {
        "name": name,
        "tensor": name,
        "shape": shape,
        "shape_spec": shape,
        "layout": _dense_layout(shape),
        "dtype": dtype,
    }


def _tensor(name: str, shape: list[int], dtype: str, kind: str) -> dict:
    nbytes = 2 if dtype in {"float16", "bfloat16"} else 4
    for dim in shape:
        nbytes *= dim
    return {
        "name": name,
        "shape": shape,
        "shape_spec": shape,
        "layout": _dense_layout(shape),
        "dtype": dtype,
        "kind": kind,
        "nbytes": nbytes,
    }


def _dense_layout(shape: list[int]) -> dict:
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.insert(0, stride)
        stride *= dim
    return {
        "schema_version": 1,
        "kind": "dense",
        "order": "row_major",
        "strides": strides,
        "storage_offset": 0,
    }


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


def _wait_for_path(path: Path, *, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if path.exists():
            return
        time.sleep(0.1)
