from __future__ import annotations

import pytest

from dinoml.lowering.cpp_types import cpu_storage_type, cuda_storage_type, rocm_storage_type
from dinoml.lowering.ops.arange import render_generated_kernel as render_arange_kernel
from dinoml.lowering.ops.full import render_generated_kernel as render_full_kernel
from dinoml.lowering.target_specs import (
    admitted_generated_targets,
    generated_source_extension,
    lowering_target_spec,
    storage_type,
)
from dinoml.ops.creation import Arange, Full


def test_lowering_target_specs_include_rocm_without_generated_module_admission():
    rocm = lowering_target_spec("rocm")

    assert rocm.source_extension == "hip"
    assert rocm.op_template_flavor == "gpu"
    assert rocm.stream_type == "hipStream_t"
    assert rocm.check_macro == "DINO_ROCM_CHECK"
    assert rocm.last_error_call == "hipGetLastError()"
    assert rocm.storage_type("float16") == "half"
    assert rocm.storage_type("bfloat16") == "hip_bfloat16"
    assert not rocm.generated_module_admitted
    assert admitted_generated_targets() == ("cpu", "cuda")


def test_storage_type_helpers_delegate_to_target_specs():
    assert storage_type("float16", "cpu") == "dinoml::math::float16"
    assert storage_type("float16", "cuda") == "half"
    assert storage_type("float16", "rocm") == "half"
    assert cpu_storage_type("bfloat16") == "dinoml::math::bfloat16"
    assert cuda_storage_type("bfloat16") == "__nv_bfloat16"
    assert rocm_storage_type("bfloat16") == "hip_bfloat16"


def test_generated_source_extensions_come_from_target_specs():
    assert generated_source_extension("cpu") == "cpp"
    assert generated_source_extension("cuda") == "cu"
    assert generated_source_extension("rocm") == "hip"

    with pytest.raises(ValueError, match="Unsupported lowering target"):
        generated_source_extension("metal")


def test_creation_gpu_templates_use_backend_spec_names():
    full_tensor_map = {"y": {"shape": [2, 3], "dtype": "float32"}}
    arange_tensor_map = {"y": {"shape": [6], "dtype": "float32"}}
    full_node = {"op": "full", "inputs": [], "outputs": ["y"], "attrs": {"shape": [2, 3], "fill_value": 1.25}}
    arange_node = {
        "op": "arange",
        "inputs": [],
        "outputs": ["y"],
        "attrs": {"start": 1.0, "end": 7.0, "step": 1.0},
    }

    full_source = render_full_kernel("cuda", full_node, full_tensor_map)
    arange_source = render_arange_kernel("cuda", arange_node, arange_tensor_map)

    assert "cudaStream_t stream" in full_source
    assert "DINO_CUDA_CHECK(cudaGetLastError())" in full_source
    assert "cudaStream_t stream" in arange_source
    assert "DINO_CUDA_CHECK(cudaGetLastError())" in arange_source


def test_creation_registry_points_cuda_to_shared_gpu_templates():
    assert Full.backend_kernels["cuda"].source_template == "full_gpu.j2"
    assert Arange.backend_kernels["cuda"].source_template == "arange_gpu.j2"


def test_rocm_target_facts_do_not_admit_rocm_creation_kernels_yet():
    tensor_map = {"y": {"shape": [2], "dtype": "float32"}}
    node = {"op": "full", "inputs": [], "outputs": ["y"], "attrs": {"shape": [2], "fill_value": 1.0}}

    with pytest.raises(ValueError, match="Unsupported full target: rocm"):
        render_full_kernel("rocm", node, tensor_map)
