from __future__ import annotations

import os
import shutil
from dataclasses import replace

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load
from tests.upsampling_family_parity import (
    ATOL_BY_DTYPE,
    COMPRESS_TIME_FRAME_SIZES,
    RTOL_BY_DTYPE,
    UPSAMPLING_PARITY_CASES,
    UpsamplingParityCase,
    random_inputs,
    torch_oracle,
    trace_upsampling_parity_spec,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)

_DTYPES = ("float16", "float32", "bfloat16")


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", UPSAMPLING_PARITY_CASES, ids=lambda case: case.name)
def test_rocm_upsampling_family_parity_matches_torch(case: UpsamplingParityCase, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    if dtype == "bfloat16":
        try:
            torch.zeros((1,), device="cuda", dtype=torch.bfloat16)
        except RuntimeError:
            pytest.skip("Torch bfloat16 ROCm device support is unavailable")

    spec = trace_upsampling_parity_spec(case, dtype)
    inputs = random_inputs(case, dtype)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{case.op_name}_{dtype}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(torch, case, inputs, device=torch.device("cuda"), dtype=dtype, native_dtype=True)
    np.testing.assert_allclose(actual, expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("frames", COMPRESS_TIME_FRAME_SIZES)
def test_rocm_upsampling3d_compress_time_parity_matches_torch(frames: int, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")
    if dtype == "bfloat16":
        try:
            torch.zeros((1,), device="cuda", dtype=torch.bfloat16)
        except RuntimeError:
            pytest.skip("Torch bfloat16 ROCm device support is unavailable")

    case = replace(
        UpsamplingParityCase("upsampling3d_compress_time", "upsampling3d_compress_time", (1, frames, 5, 7, 6)),
        input_shape=(1, frames, 5, 7, 6),
    )
    spec = trace_upsampling_parity_spec(case, dtype)
    inputs = random_inputs(case, dtype)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"u3dct_{frames}_{dtype}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(torch, case, inputs, device=torch.device("cuda"), dtype=dtype, native_dtype=True)
    np.testing.assert_allclose(actual, expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])
