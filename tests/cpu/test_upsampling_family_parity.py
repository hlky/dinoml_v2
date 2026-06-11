from __future__ import annotations

from dataclasses import replace

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


_DTYPES = ("float16", "float32", "bfloat16")


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("case", UPSAMPLING_PARITY_CASES, ids=lambda case: case.name)
def test_cpu_upsampling_family_parity_matches_torch(case: UpsamplingParityCase, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    spec = trace_upsampling_parity_spec(case, dtype)
    inputs = random_inputs(case, dtype)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.op_name}_{dtype}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(torch, case, inputs, device=torch.device("cpu"), dtype=dtype, native_dtype=False)
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("frames", COMPRESS_TIME_FRAME_SIZES)
def test_cpu_upsampling3d_compress_time_parity_matches_torch(frames: int, dtype: str, tmp_path):
    torch = pytest.importorskip("torch")
    case = replace(
        UpsamplingParityCase("upsampling3d_compress_time", "upsampling3d_compress_time", (1, frames, 5, 7, 6)),
        input_shape=(1, frames, 5, 7, 6),
    )
    spec = trace_upsampling_parity_spec(case, dtype)
    inputs = random_inputs(case, dtype)
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"u3dct_{frames}_{dtype}.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(torch, case, inputs, device=torch.device("cpu"), dtype=dtype, native_dtype=False)
    np.testing.assert_allclose(actual.astype(np.float32), expected, atol=ATOL_BY_DTYPE[dtype], rtol=RTOL_BY_DTYPE[dtype])
