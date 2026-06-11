from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
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


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


def _skip_if_rocm_unavailable(torch, dtype: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    if dtype == "bfloat16":
        try:
            torch.zeros((1,), device="cuda", dtype=torch.bfloat16)
        except RuntimeError:
            pytest.skip("Torch bfloat16 ROCm device support is unavailable")


@pytest.mark.parametrize("case", LAYER_NORM_CASES, ids=lambda case: str(case["tag"]))
@pytest.mark.parametrize("dtype", NORMALIZATION_SPECIALIZATION_DTYPES)
def test_rocm_layer_norm_matches_torch(dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    _skip_if_rocm_unavailable(torch, dtype)

    spec = trace_single_output_spec("layer_norm", dtype, case, name_prefix="rocm_")
    spec_inputs = random_single_output_inputs("layer_norm", dtype, case)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{artifact_stem('layer_norm', dtype, case)}.dinoml")
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
def test_rocm_layernorm_sigmoid_mul_matches_torch(dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    _skip_if_rocm_unavailable(torch, dtype)

    spec = trace_single_output_spec("layernorm_sigmoid_mul", dtype, case, name_prefix="rocm_")
    spec_inputs = random_single_output_inputs("layernorm_sigmoid_mul", dtype, case)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{artifact_stem('layernorm_sigmoid_mul', dtype, case)}.dinoml")
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
def test_rocm_batch_layernorm_sigmoid_mul_matches_torch(dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    _skip_if_rocm_unavailable(torch, dtype)

    spec = trace_single_output_spec("batch_layernorm_sigmoid_mul", dtype, case, name_prefix="rocm_")
    spec_inputs = random_single_output_inputs("batch_layernorm_sigmoid_mul", dtype, case)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{artifact_stem('batch_layernorm_sigmoid_mul', dtype, case)}.dinoml")
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
def test_rocm_group_layernorm_matches_torch(op_name: str, dtype: str, case: dict[str, object], tmp_path):
    torch = pytest.importorskip("torch")
    _skip_if_rocm_unavailable(torch, dtype)

    spec = trace_group_spec(op_name, dtype, case, name_prefix="rocm_")
    spec_inputs = random_group_inputs(dtype, case)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{artifact_stem(op_name, dtype, case)}.dinoml")
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
