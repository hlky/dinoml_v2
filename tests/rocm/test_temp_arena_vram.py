from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.gpu_memory_validation import hip_device_synchronize, hip_mem_get_info, load_hip_runtime
from dinoml.runtime import load
from dinoml.shapes import Dim


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


class _ThreeLayerNormModule(dml.Module):
    def forward(self, x, weight, bias):
        h0 = dml.ops.layer_norm(x, weight, bias)
        h1 = dml.ops.layer_norm(h0, weight, bias)
        return dml.ops.output(dml.ops.layer_norm(h1, weight, bias), "output")


def test_rocm_temp_arena_policies_track_bucketed_vram_usage(tmp_path):
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    hip_runtime = load_hip_runtime()
    batch = Dim("batch", min=256, max=1024, buckets=(256, 1024))
    hidden = 2048
    spec = dml.trace(
        _ThreeLayerNormModule(),
        inputs={
            "x": dml.TensorSpec([batch, hidden], "float32"),
            "weight": dml.TensorSpec([hidden], "float32"),
            "bias": dml.TensorSpec([hidden], "float32"),
        },
        name="rocm_temp_arena_vram_bucketed_contract",
    )

    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / "temp_arena_vram_rocm.dinoml")
    metadata = json.loads((Path(artifact.path) / "metadata.json").read_text(encoding="utf-8"))
    plans_by_batch = {
        int(plan["dim_values"]["batch"]): int(plan["arena_nbytes"])
        for plan in metadata["memory_plan"]["bucket_plans"]
    }
    expected_small = plans_by_batch[256]
    expected_large = plans_by_batch[1024]
    output_names = ("output",)
    small_inputs = _inputs(batch_size=256, hidden=hidden)
    large_inputs = _inputs(batch_size=1024, hidden=hidden)

    module = load(artifact.path, load_constants=True)
    try:
        eager_create = _measure_create_arena_release(module, hip_runtime, policy="eager_max")
        lazy_grow_create = _measure_create_arena_release(module, hip_runtime, policy="lazy_grow")
        lazy_exact_small = _measure_lazy_exact_run(module, hip_runtime, small_inputs, output_names)
        lazy_exact_large = _measure_lazy_exact_run(module, hip_runtime, large_inputs, output_names)
        lazy_grow_points = _measure_lazy_grow_sequence(
            module,
            hip_runtime,
            [("small", small_inputs), ("large", large_inputs)],
            output_names,
        )
    finally:
        module.close()

    assert eager_create["observed_temp_bytes"] >= expected_large
    assert lazy_grow_create["observed_temp_bytes"] == 0
    assert lazy_exact_small >= expected_small
    assert lazy_exact_large >= expected_large
    assert lazy_exact_small < lazy_exact_large
    assert lazy_grow_points["small"] >= expected_small
    assert lazy_grow_points["large"] >= expected_large
    assert lazy_grow_points["small"] < lazy_grow_points["large"]


def _inputs(*, batch_size: int, hidden: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(batch_size + hidden)
    return {
        "x": rng.normal(0.0, 0.1, (batch_size, hidden)).astype(np.float32),
        "weight": rng.normal(1.0, 0.01, (hidden,)).astype(np.float32),
        "bias": rng.normal(0.0, 0.01, (hidden,)).astype(np.float32),
    }


def _sync_and_free_bytes(hip_runtime) -> int:
    hip_device_synchronize(hip_runtime)
    free, _total = hip_mem_get_info(hip_runtime)
    return free


def _measure_create_arena_release(module, hip_runtime, *, policy: str) -> dict[str, int]:
    free_after_module = _sync_and_free_bytes(hip_runtime)
    session = module.create_session(temp_arena_policy=policy)
    try:
        free_after_session = _sync_and_free_bytes(hip_runtime)
        session.release_temp_arena()
        free_after_release = _sync_and_free_bytes(hip_runtime)
    finally:
        session.close()
        _sync_and_free_bytes(hip_runtime)
    return {
        "create_delta": free_after_module - free_after_session,
        "observed_temp_bytes": free_after_release - free_after_session,
    }


def _measure_lazy_exact_run(module, hip_runtime, inputs, output_names) -> int:
    session = module.create_session(temp_arena_policy="lazy_exact_bucket")
    try:
        session.run_numpy_device_outputs(inputs, device_outputs=output_names)
        _sync_and_free_bytes(hip_runtime)
        session._free_cuda_buffers()
        free_after_buffers = _sync_and_free_bytes(hip_runtime)
        session.release_temp_arena()
        free_after_release = _sync_and_free_bytes(hip_runtime)
    finally:
        session.close()
        _sync_and_free_bytes(hip_runtime)
    return free_after_release - free_after_buffers


def _measure_lazy_grow_sequence(module, hip_runtime, scenarios, output_names) -> dict[str, int]:
    session = module.create_session(temp_arena_policy="lazy_grow")
    try:
        points: dict[str, int] = {}
        for name, inputs in scenarios:
            session.run_numpy_device_outputs(inputs, device_outputs=output_names)
            _sync_and_free_bytes(hip_runtime)
            session._free_cuda_buffers()
            free_after_buffers = _sync_and_free_bytes(hip_runtime)
            points[name] = free_after_buffers
        session.release_temp_arena()
        free_after_release = _sync_and_free_bytes(hip_runtime)
    finally:
        session.close()
        _sync_and_free_bytes(hip_runtime)
    return {name: free_after_release - free_after_buffers for name, free_after_buffers in points.items()}


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))
