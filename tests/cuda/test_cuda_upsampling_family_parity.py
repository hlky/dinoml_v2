from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
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


def _discover_nvcc() -> str | None:
    direct = shutil.which("nvcc")
    if direct:
        return direct
    for candidate in (os.environ.get("CUDACXX"), "/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12.8/bin/nvcc"):
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


_NVCC = _discover_nvcc()
_DTYPES = ("float16", "float32", "bfloat16")

pytestmark = pytest.mark.skipif(_NVCC is None, reason="nvcc is required")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", UPSAMPLING_PARITY_CASES, ids=lambda case: case.name)
def test_cuda_upsampling_family_parity_matches_torch(case: UpsamplingParityCase, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if dtype == "bfloat16" and capability < (8, 0):
        pytest.skip("bfloat16 parity requires CUDA sm_80 or newer")
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")

    spec = trace_upsampling_parity_spec(case, dtype)
    inputs = random_inputs(case, dtype)
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"),
        tmp_path / f"{case.op_name}_{dtype}.dinoml",
    )
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
def test_cuda_upsampling3d_compress_time_parity_matches_torch(frames: int, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if dtype == "bfloat16" and capability < (8, 0):
        pytest.skip("bfloat16 parity requires CUDA sm_80 or newer")
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")

    case = replace(
        UpsamplingParityCase("upsampling3d_compress_time", "upsampling3d_compress_time", (1, frames, 5, 7, 6)),
        input_shape=(1, frames, 5, 7, 6),
    )
    spec = trace_upsampling_parity_spec(case, dtype)
    inputs = random_inputs(case, dtype)
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"),
        tmp_path / f"u3dct_{frames}_{dtype}.dinoml",
    )
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(torch, case, inputs, device=torch.device("cuda"), dtype=dtype, native_dtype=True)
    np.testing.assert_allclose(actual, expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])
