from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.runtime import load


class _TorchRandnCpuModule(dml.Module):
    def forward(self):
        return {
            "small": dml.ops.output(dml.ops.randn([12], dtype="float32", seed=7, rng="torch"), "small"),
            "block": dml.ops.output(dml.ops.randn([17], dtype="float32", seed=7, rng="torch"), "block"),
            "float16": dml.ops.output(dml.ops.randn([17], dtype="float16", seed=7, rng="torch"), "float16"),
            "bfloat16": dml.ops.output(dml.ops.randn([17], dtype="bfloat16", seed=7, rng="torch"), "bfloat16"),
            "legacy": dml.ops.output(dml.ops.randn([12], dtype="float32", seed=7), "legacy"),
        }


def test_randn_torch_reference_matches_torch_cpu():
    torch = pytest.importorskip("torch")
    spec = dml.trace(_TorchRandnCpuModule(), inputs={}, name="torch_randn_reference")

    actual = reference_numpy(spec, {})
    small_generator = torch.Generator(device="cpu")
    small_generator.manual_seed(7)
    block_generator = torch.Generator(device="cpu")
    block_generator.manual_seed(7)

    np.testing.assert_allclose(actual["small"], torch.randn(12, generator=small_generator).numpy(), atol=0, rtol=0)
    np.testing.assert_allclose(actual["block"], torch.randn(17, generator=block_generator).numpy(), atol=1e-6, rtol=0)
    half_generator = torch.Generator(device="cpu")
    half_generator.manual_seed(7)
    bfloat_generator = torch.Generator(device="cpu")
    bfloat_generator.manual_seed(7)
    np.testing.assert_allclose(actual["float16"], torch.randn(17, dtype=torch.float16, generator=half_generator).numpy(), atol=0, rtol=0)
    np.testing.assert_allclose(
        actual["bfloat16"],
        torch.randn(17, dtype=torch.bfloat16, generator=bfloat_generator).float().numpy(),
        atol=0,
        rtol=0,
    )
    assert not np.allclose(actual["legacy"], actual["small"])


def test_cpu_randn_torch_matches_torch(tmp_path):
    torch = pytest.importorskip("torch")
    spec = dml.trace(_TorchRandnCpuModule(), inputs={}, name="torch_randn_cpu")
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / "torch_randn_cpu.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy({})
    finally:
        session.close()
        module.close()

    small_generator = torch.Generator(device="cpu")
    small_generator.manual_seed(7)
    block_generator = torch.Generator(device="cpu")
    block_generator.manual_seed(7)
    half_generator = torch.Generator(device="cpu")
    half_generator.manual_seed(7)
    bfloat_generator = torch.Generator(device="cpu")
    bfloat_generator.manual_seed(7)
    np.testing.assert_allclose(actual["small"], torch.randn(12, generator=small_generator).numpy(), atol=0, rtol=0)
    np.testing.assert_allclose(actual["block"], torch.randn(17, generator=block_generator).numpy(), atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        actual["float16"],
        torch.randn(17, dtype=torch.float16, generator=half_generator).numpy(),
        atol=0.002,
        rtol=0,
    )
    np.testing.assert_allclose(
        actual["bfloat16"],
        torch.randn(17, dtype=torch.bfloat16, generator=bfloat_generator).float().numpy(),
        atol=0.008,
        rtol=0,
    )
    assert not np.allclose(actual["legacy"], actual["small"])


def test_randn_rejects_unknown_rng():
    with pytest.raises(ValueError, match="randn rng must be one of"):
        dml.ops.randn([2, 3], rng="unknown")
