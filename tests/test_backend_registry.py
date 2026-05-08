import pytest

import dinoml as dml
from dinoml.backends.registry import BackendSpec, get_backend_spec, registered_backend_names, registered_backend_specs
from dinoml.ir import read_json


class Identity(dml.Module):
    def forward(self, x):
        return dml.ops.output(x, "y")


def test_target_defaults_are_loaded_from_backend_registry():
    assert dml.Target("cuda").arch == "sm_86"
    assert dml.Target("cuda", arch="sm_90").to_json() == {"name": "cuda", "arch": "sm_90"}

    assert dml.Target("cpu").arch == "native"
    assert dml.Target("cpu", arch="sm_86").arch == "native"
    assert dml.Target("cpu", arch="x86_64").to_json() == {"name": "cpu", "arch": "x86_64"}


def test_target_rejects_names_missing_from_backend_registry():
    with pytest.raises(ValueError, match="Unsupported DinoML target 'metal'.*cpu, cuda"):
        dml.Target("metal")


def test_backend_registry_describes_cpu_and_cuda_support():
    assert registered_backend_names() == ("cpu", "cuda")
    assert [spec.name for spec in registered_backend_specs()] == ["cpu", "cuda"]

    cpu = get_backend_spec("cpu")
    assert cpu.default_arch == "native"
    assert cpu.supported_dtypes == frozenset({"float32"})
    assert cpu.build_function == "dinoml.backends.cpu.build_cpu_module"
    assert cpu.cmake.supports_openmp is True
    assert cpu.cmake.requires_cuda is False
    assert cpu.support_libraries == {
        "runtime_library": "lib/libdinoml_runtime.so",
        "kernel_library": "lib/libdinoml_cpu_kernels.so",
    }

    cuda = get_backend_spec("cuda")
    assert cuda.default_arch == "sm_86"
    assert cuda.supported_dtypes == frozenset({"float16", "float32", "bfloat16"})
    assert cuda.build_function == "dinoml.backends.cuda.build_cuda_module"
    assert cuda.cmake.requires_cuda is True
    assert cuda.cmake.supports_cuda_fast_math is True
    assert cuda.support_libraries == {
        "runtime_library": "lib/libdinoml_runtime.so",
        "cuda_runtime_library": "lib/libdinoml_cuda_runtime.so",
        "kernel_library": "lib/libdinoml_cuda_kernels.so",
    }


def test_backend_registry_build_functions_resolve_to_callables():
    assert get_backend_spec("cpu").resolve_build_function().__name__ == "build_cpu_module"
    assert get_backend_spec("cuda").resolve_build_function().__name__ == "build_cuda_module"


def test_compile_uses_backend_registry_for_manifest_and_build_dispatch(tmp_path, monkeypatch):
    calls = []

    def fake_build(ir, *, target, artifact_dir, generated_src_dir, kernel_manifest):
        calls.append(
            {
                "target": target.to_json(),
                "artifact_dir": artifact_dir,
                "generated_src_dir": generated_src_dir,
                "kernel_manifest": kernel_manifest,
            }
        )

    monkeypatch.setattr(BackendSpec, "resolve_build_function", lambda self: fake_build)

    spec = dml.trace(Identity(), inputs={"x": dml.TensorSpec([1, 4], "float32")}, name="registry_dispatch")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "registry_dispatch.dinoml")

    manifest = read_json(artifact.path / "manifest.json")
    assert manifest["files"]["runtime_library"] == get_backend_spec("cpu").support_libraries["runtime_library"]
    assert manifest["files"]["kernel_library"] == get_backend_spec("cpu").support_libraries["kernel_library"]
    assert calls == [
        {
            "target": {"name": "cpu", "arch": "native"},
            "artifact_dir": artifact.path,
            "generated_src_dir": artifact.path / "debug" / "generated_src",
            "kernel_manifest": read_json(artifact.path / "kernel_manifest.json"),
        }
    ]
