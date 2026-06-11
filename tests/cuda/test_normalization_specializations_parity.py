from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.normalization_specializations_test_utils import (
    ATOL_BY_DTYPE,
    BATCH_LAYERNORM_SIGMOID_MUL_CASES,
    GROUP_LAYERNORM_CASES,
    LAYER_NORM_CASES,
    LAYERNORM_SIGMOID_MUL_CASES,
    NORMALIZATION_SPECIALIZATION_DTYPES,
    RTOL_BY_DTYPE,
    artifact_stem,
    random_group_inputs,
    random_single_output_inputs,
    torch_group_oracle,
    torch_single_output_oracle,
    trace_group_spec,
    trace_single_output_spec,
)


def _discover_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    for candidate in (
        os.environ.get("CUDACXX"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12.8/bin/nvcc",
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


_NVCC = _discover_nvcc()

pytestmark = pytest.mark.skipif(_NVCC is None, reason="nvcc is required")


def _cuda_target(torch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")
    return dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"), capability


def _skip_if_bfloat16_unsupported(dtype: str, capability: tuple[int, int]) -> None:
    if dtype == "bfloat16" and capability < (8, 0):
        pytest.skip("bfloat16 parity requires CUDA sm_80 or newer")


@pytest.mark.parametrize("case", LAYER_NORM_CASES, ids=lambda case: str(case["tag"]))
@pytest.mark.parametrize("dtype", NORMALIZATION_SPECIALIZATION_DTYPES)
def test_cuda_layer_norm_matches_torch(dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    target, capability = _cuda_target(torch)
    _skip_if_bfloat16_unsupported(dtype, capability)

    spec = trace_single_output_spec("layer_norm", dtype, case, name_prefix="cuda_")
    spec_inputs = random_single_output_inputs("layer_norm", dtype, case)
    artifact = dml.compile(spec, target, tmp_path / f"{artifact_stem('layer_norm', dtype, case)}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_single_output_oracle(torch, "layer_norm", spec_inputs, case, dtype=dtype, device="cuda")
    np.testing.assert_allclose(actual, expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])


@pytest.mark.parametrize("case", LAYERNORM_SIGMOID_MUL_CASES, ids=lambda case: str(case["tag"]))
@pytest.mark.parametrize("dtype", NORMALIZATION_SPECIALIZATION_DTYPES)
def test_cuda_layernorm_sigmoid_mul_matches_torch(dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    target, capability = _cuda_target(torch)
    _skip_if_bfloat16_unsupported(dtype, capability)

    spec = trace_single_output_spec("layernorm_sigmoid_mul", dtype, case, name_prefix="cuda_")
    spec_inputs = random_single_output_inputs("layernorm_sigmoid_mul", dtype, case)
    artifact = dml.compile(spec, target, tmp_path / f"{artifact_stem('layernorm_sigmoid_mul', dtype, case)}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_single_output_oracle(torch, "layernorm_sigmoid_mul", spec_inputs, case, dtype=dtype, device="cuda")
    np.testing.assert_allclose(actual, expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])


@pytest.mark.parametrize("case", BATCH_LAYERNORM_SIGMOID_MUL_CASES, ids=lambda case: str(case["tag"]))
@pytest.mark.parametrize("dtype", NORMALIZATION_SPECIALIZATION_DTYPES)
def test_cuda_batch_layernorm_sigmoid_mul_matches_torch(dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    target, capability = _cuda_target(torch)
    _skip_if_bfloat16_unsupported(dtype, capability)

    spec = trace_single_output_spec("batch_layernorm_sigmoid_mul", dtype, case, name_prefix="cuda_")
    spec_inputs = random_single_output_inputs("batch_layernorm_sigmoid_mul", dtype, case)
    artifact = dml.compile(spec, target, tmp_path / f"{artifact_stem('batch_layernorm_sigmoid_mul', dtype, case)}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_single_output_oracle(torch, "batch_layernorm_sigmoid_mul", spec_inputs, case, dtype=dtype, device="cuda")
    np.testing.assert_allclose(actual, expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])


@pytest.mark.parametrize("op_name", ("group_layernorm", "group_layernorm_sigmoid_mul"))
@pytest.mark.parametrize("case", GROUP_LAYERNORM_CASES, ids=lambda case: str(case["tag"]))
@pytest.mark.parametrize("dtype", NORMALIZATION_SPECIALIZATION_DTYPES)
def test_cuda_group_layernorm_matches_torch(op_name: str, dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    target, capability = _cuda_target(torch)
    _skip_if_bfloat16_unsupported(dtype, capability)

    spec = trace_group_spec(op_name, dtype, case, name_prefix="cuda_")
    spec_inputs = random_group_inputs(dtype, case)
    artifact = dml.compile(spec, target, tmp_path / f"{artifact_stem(op_name, dtype, case)}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)
    finally:
        session.close()
        module.close()

    expected = torch_group_oracle(torch, op_name, spec_inputs, case, dtype=dtype, device="cuda")
    for output_name, expected_value in expected.items():
        np.testing.assert_allclose(
            actual[output_name],
            expected_value,
            atol=ATOL_BY_DTYPE[dtype],
            rtol=RTOL_BY_DTYPE[dtype],
        )
