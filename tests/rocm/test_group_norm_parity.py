from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)

_GROUP_NORM_DTYPES = ("float16", "float32", "bfloat16")
_GROUP_NORM_OPS = ("group_norm", "group_norm_swish")
_GROUP_NORM_CASES = (
    ([2, 4, 3, 8], 4),
    ([1, 2, 2, 16], 8),
)
_ATOL_BY_DTYPE = {"float16": 0.003, "float32": 1e-5, "bfloat16": 0.02}
_RTOL_BY_DTYPE = {"float16": 0.002, "float32": 1e-5, "bfloat16": 0.02}


class _RocmGroupNormParityModule(dml.Module):
    def __init__(self, op_name: str, num_groups: int):
        self._op_name = op_name
        self._num_groups = num_groups

    def forward(self, x, weight, bias):
        op = getattr(dml.ops, self._op_name)
        y = op(x, self._num_groups, weight, bias, eps=1e-5)
        return dml.ops.output(y, "y")


def _trace_group_norm_parity_spec(op_name: str, dtype: str, shape: list[int], num_groups: int):
    channels = int(shape[-1])
    return dml.trace(
        _RocmGroupNormParityModule(op_name, num_groups),
        inputs={
            "x": dml.TensorSpec(shape, dtype),
            "weight": dml.TensorSpec([channels], dtype),
            "bias": dml.TensorSpec([channels], dtype),
        },
        name=f"rocm_{op_name}_{dtype}_{'x'.join(str(dim) for dim in shape)}_g{num_groups}",
    )


def _random_inputs(dtype: str, shape: list[int]) -> dict[str, np.ndarray]:
    channels = int(shape[-1])
    rng = np.random.default_rng(7)
    inputs = {
        "x": rng.standard_normal(shape, dtype=np.float32).astype(np.float32),
        "weight": rng.standard_normal([channels], dtype=np.float32).astype(np.float32),
        "bias": rng.standard_normal([channels], dtype=np.float32).astype(np.float32),
    }
    if dtype == "float16":
        return {name: value.astype(np.float16) for name, value in inputs.items()}
    if dtype == "bfloat16":
        return inputs
    return inputs


def _torch_oracle(torch, op_name: str, inputs: dict[str, np.ndarray], *, dtype: str, num_groups: int) -> np.ndarray:
    torch_dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    device = torch.device("cuda")
    x = torch.from_numpy(inputs["x"]).to(device=device, dtype=torch_dtype)
    weight = torch.from_numpy(inputs["weight"]).to(device=device, dtype=torch_dtype)
    bias = torch.from_numpy(inputs["bias"]).to(device=device, dtype=torch_dtype)
    x_nchw = x.permute(0, x.ndim - 1, *range(1, x.ndim - 1)).contiguous()
    result = torch.nn.functional.group_norm(x_nchw, num_groups, weight, bias, eps=1e-5)
    result = result.permute(0, *range(2, x.ndim), 1).contiguous()
    if op_name == "group_norm_swish":
        result = torch.nn.functional.silu(result)
    return result.float().cpu().numpy()


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))


@pytest.mark.parametrize("shape,num_groups", _GROUP_NORM_CASES)
@pytest.mark.parametrize("dtype", _GROUP_NORM_DTYPES)
@pytest.mark.parametrize("op_name", _GROUP_NORM_OPS)
def test_rocm_group_norm_parity_matches_torch(op_name: str, dtype: str, shape: list[int], num_groups: int, tmp_path):
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

    spec = _trace_group_norm_parity_spec(op_name, dtype, shape, num_groups)
    spec_inputs = _random_inputs(dtype, shape)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{op_name}_{dtype}_group_norm_parity_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(spec_inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = _torch_oracle(torch, op_name, spec_inputs, dtype=dtype, num_groups=num_groups)
    np.testing.assert_allclose(actual, expected, atol=_ATOL_BY_DTYPE[dtype], rtol=_RTOL_BY_DTYPE[dtype])
