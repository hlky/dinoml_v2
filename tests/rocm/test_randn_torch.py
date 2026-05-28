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


class _TorchRandnRocmModule(dml.Module):
    def forward(self):
        return {
            "float32": dml.ops.output(dml.ops.randn([1024], dtype="float32", seed=7, rng="torch"), "float32"),
            "bfloat16": dml.ops.output(dml.ops.randn([1024], dtype="bfloat16", seed=7, rng="torch"), "bfloat16"),
        }


def test_rocm_randn_torch_matches_torch_hip(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    spec = dml.trace(_TorchRandnRocmModule(), inputs={}, name="torch_randn_rocm")
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "torch_randn_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({})
    finally:
        session.close()
        module.close()

    float_generator = torch.Generator(device="cuda")
    float_generator.manual_seed(7)
    expected_float = torch.randn((1024,), dtype=torch.float32, device="cuda", generator=float_generator).cpu().numpy()
    bfloat_generator = torch.Generator(device="cuda")
    bfloat_generator.manual_seed(7)
    expected_bfloat = torch.randn((1024,), dtype=torch.bfloat16, device="cuda", generator=bfloat_generator).float().cpu().numpy()
    np.testing.assert_allclose(actual["float32"], expected_float, atol=0, rtol=0)
    np.testing.assert_allclose(actual["bfloat16"], expected_bfloat, atol=0, rtol=0)


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))
