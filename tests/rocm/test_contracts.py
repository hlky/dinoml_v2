from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.ir import array_from_storage, array_to_storage
from dinoml.reference import reference_numpy
from dinoml.runtime import load
from tests.cases import GraphCase, standard_cases


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


ROCM_SIMPLE_CASES = [case for case in standard_cases() if case.rocm]


@pytest.mark.parametrize("case", ROCM_SIMPLE_CASES, ids=lambda case: case.name)
def test_rocm_simple_artifact_compiles_and_runs(case: GraphCase, tmp_path):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    spec = case.build_spec()
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{case.name}_rocm.dinoml")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(case.inputs())
    finally:
        session.close()
        module.close()

    expected = reference_numpy(spec, case.inputs())
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=case.atol, rtol=case.rtol)


class _FlashAttentionBiasModule(dml.nn.Module):
    def forward(self, q, k, v, bias):
        return dml.ops.output(dml.ops.flash_attention_bias(q, k, v, bias, causal=True), "out")


def test_rocm_flash_attention_bias_bfloat16_head_dim128_compiles_and_uses_bias(tmp_path):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    spec = dml.trace(
        _FlashAttentionBiasModule(),
        inputs={
            "q": dml.TensorSpec([1, 4, 4, 128], "bfloat16"),
            "k": dml.TensorSpec([1, 4, 2, 128], "bfloat16"),
            "v": dml.TensorSpec([1, 4, 2, 128], "bfloat16"),
            "bias": dml.TensorSpec([4, 4, 4], "bfloat16"),
        },
        name="rocm_flash_attention_bias_bfloat16_head_dim128_contract",
    )
    rng = np.random.default_rng(123)
    inputs = {
        "q": _bf16_storage(rng.normal(size=(1, 4, 4, 128)).astype(np.float32) * np.float32(0.125)),
        "k": _bf16_storage(rng.normal(size=(1, 4, 2, 128)).astype(np.float32) * np.float32(0.125)),
        "v": _bf16_storage(rng.normal(size=(1, 4, 2, 128)).astype(np.float32) * np.float32(0.125)),
        "bias": _bf16_storage(_strong_bias([4, 4, 4], masked_key=2)),
    }

    _assert_bias_changes_reference(spec, inputs)
    _compile_run_and_compare(spec, inputs, tmp_path / "flash_attention_bias_bfloat16_hdim128.dinoml")


class _FlashAttentionStaticKvCacheBiasModule(dml.nn.Module):
    def forward(self, q, past_key, past_value, new_key, new_value, cache_seqlens, bias):
        return dml.ops.output(
            dml.ops.flash_attention_static_kv_cache_bias(
                q,
                past_key,
                past_value,
                new_key,
                new_value,
                cache_seqlens,
                bias,
            ),
            "out",
        )


def test_rocm_flash_attention_static_kv_cache_bias_bfloat16_head_dim128_compiles_and_uses_bias(tmp_path):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    spec = dml.trace(
        _FlashAttentionStaticKvCacheBiasModule(),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 128], "bfloat16"),
            "past_key": dml.TensorSpec([1, 2, 6, 128], "bfloat16"),
            "past_value": dml.TensorSpec([1, 2, 6, 128], "bfloat16"),
            "new_key": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "new_value": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "cache_seqlens": dml.TensorSpec([1], "int32"),
            "bias": dml.TensorSpec([4, 1, 6], "bfloat16"),
        },
        name="rocm_flash_attention_static_kv_cache_bias_bfloat16_head_dim128_contract",
    )
    rng = np.random.default_rng(321)
    inputs = {
        "q": _bf16_storage(rng.normal(size=(1, 1, 4, 128)).astype(np.float32) * np.float32(0.125)),
        "past_key": _bf16_storage(rng.normal(size=(1, 2, 6, 128)).astype(np.float32) * np.float32(0.125)),
        "past_value": _bf16_storage(rng.normal(size=(1, 2, 6, 128)).astype(np.float32) * np.float32(0.125)),
        "new_key": _bf16_storage(rng.normal(size=(1, 2, 1, 128)).astype(np.float32) * np.float32(0.125)),
        "new_value": _bf16_storage(rng.normal(size=(1, 2, 1, 128)).astype(np.float32) * np.float32(0.125)),
        "cache_seqlens": np.asarray([4], dtype=np.int32),
        "bias": _bf16_storage(_strong_bias([4, 1, 6], masked_key=3)),
    }

    _assert_bias_changes_reference(spec, inputs)
    _compile_run_and_compare(spec, inputs, tmp_path / "flash_attention_static_kv_cache_bias_bfloat16_hdim128.dinoml")


def _compile_run_and_compare(spec, inputs, artifact_path):
    artifact = dml.compile(spec, dml.Target("rocm"), artifact_path)
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)
    finally:
        session.close()
        module.close()

    expected = reference_numpy(spec, inputs)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], atol=3e-2, rtol=3e-2)


def _assert_bias_changes_reference(spec, inputs) -> None:
    zero_bias_inputs = dict(inputs)
    zero_bias_inputs["bias"] = np.zeros_like(inputs["bias"])
    expected_masked = reference_numpy(spec, inputs)["out"]
    expected_unmasked = reference_numpy(spec, zero_bias_inputs)["out"]
    max_delta = float(np.max(np.abs(expected_masked.astype(np.float32) - expected_unmasked.astype(np.float32))))
    assert max_delta > 1.0e-3


def _bf16_storage(value: np.ndarray) -> np.ndarray:
    return array_to_storage(np.asarray(value, dtype=np.float32), "bfloat16")


def _bf16_value(value: np.ndarray) -> np.ndarray:
    return array_from_storage(_bf16_storage(value), "bfloat16")


def _strong_bias(shape: list[int], *, masked_key: int) -> np.ndarray:
    bias = np.zeros(shape, dtype=np.float32)
    bias[..., masked_key] = -1.0e4
    return _bf16_value(bias)


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))
