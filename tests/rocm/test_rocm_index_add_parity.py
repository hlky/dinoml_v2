from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.runtime import load
from tests.index_add_parity import (
    ATOL_BY_DTYPE,
    INDEX_ADD_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_index_add_spec,
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


@pytest.mark.parametrize("case", INDEX_ADD_CASES, ids=lambda case: case.name)
def test_rocm_index_add_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("Torch HIP/CUDA device is not available")
    if not _rocm_module_compile_toolchain_available():
        pytest.skip("ROCm module compile toolchain not found from active Python/PATH")

    spec = trace_index_add_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(spec, dml.Target("rocm"), tmp_path / f"{case.name}.dinoml")
    _assert_generated_index_add_kernel(artifact.path, backend="rocm", suffix=".hip")
    module = load(artifact.path)
    session = module.create_session()
    try:
        actual = session.run_numpy(inputs)["y"]
    finally:
        session.close()
        module.close()

    expected = torch_oracle(case, inputs)
    np.testing.assert_allclose(
        actual.astype(np.float32),
        expected.astype(np.float32),
        atol=ATOL_BY_DTYPE[case.dtype],
        rtol=RTOL_BY_DTYPE[case.dtype],
    )


def _assert_generated_index_add_kernel(
    artifact_path: str | os.PathLike[str],
    *,
    backend: str,
    suffix: str,
) -> None:
    artifact_dir = Path(artifact_path)
    manifest = json.loads((artifact_dir / "kernel_manifest.json").read_text(encoding="utf-8"))
    entries = [item for item in manifest.get("required_kernels", []) if item.get("op") == "index_add"]
    assert len(entries) == 1
    generated = entries[0].get("generated_source")
    assert isinstance(generated, dict)
    assert str(generated.get("generated_function_name", "")).startswith("index_add_")
    source_hash = str(generated.get("source_hash", ""))
    assert source_hash
    assert str(generated.get("source_key", "")) == f"{backend}:{generated['generated_function_name']}"
    emitted_source_path = Path("debug") / "generated_src" / "ops" / "index_add" / f"{source_hash}{suffix}"
    source_path = artifact_dir / emitted_source_path
    if source_path.is_file():
        assert "serialized placeholder" not in source_path.read_text(encoding="utf-8")
