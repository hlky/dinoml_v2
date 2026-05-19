from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

import dinoml as dml
from dinoml.backends import get_backend_spec, registered_backend_names
from dinoml.backends.rocm import ensure_rocm_support_libs
from dinoml.kernels.codegen import create_codegen_plan
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


def test_rocm_codegen_plan_uses_arch_specific_support_cache(tmp_path):
    manifest = {
        "target": {"name": "rocm", "arch": "gfx1201"},
        "cache_key": "abcdef0123456789",
        "required_kernels": [],
    }

    plan = create_codegen_plan(manifest, tmp_path)

    assert plan.target == {"name": "rocm", "arch": "gfx1201"}
    assert plan.support_cache_dir == tmp_path / "support" / "rocm-gfx1201" / "abcdef0123456789"


def test_rocm_compile_fails_before_claiming_any_op_support(tmp_path):
    case = elementwise_case()

    with pytest.raises(NotImplementedError, match="rocm backend does not support op fused_elementwise"):
        dml.compile(case.build_spec(), dml.Target("rocm"), tmp_path / "elementwise_rocm.dinoml")


@pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE") != "1",
    reason="set DINOML_RUN_ROCM_SUPPORT_BUILD_SMOKE=1 in a ROCm SDK environment",
)
def test_rocm_support_libraries_build_with_real_toolchain(tmp_path, monkeypatch):
    monkeypatch.setenv("DINOML_CACHE_DIR", str(tmp_path / "cache"))

    libs = ensure_rocm_support_libs("gfx1201")

    assert libs.runtime_lib.exists()
    assert libs.rocm_runtime_lib.exists()
    assert libs.kernels_lib.exists()
    assert (Path(libs.rocm_runtime_lib).parent / "support_manifest.json").exists()
