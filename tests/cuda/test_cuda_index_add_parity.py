from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import dinoml as dml
from dinoml.runtime import load
from tests.index_add_parity import (
    ATOL_BY_DTYPE,
    INDEX_ADD_CASES,
    RTOL_BY_DTYPE,
    random_inputs,
    torch_oracle,
    trace_index_add_spec,
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

pytestmark = pytest.mark.skipif(_NVCC is None, reason="nvcc is required")


@pytest.mark.parametrize("case", INDEX_ADD_CASES, ids=lambda case: case.name)
def test_cuda_index_add_parity(case, tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA runtime is required")
    capability = torch.cuda.get_device_capability()
    if _NVCC is not None:
        os.environ.setdefault("CUDACXX", _NVCC)
        nvcc_parent = str(Path(_NVCC).parent)
        if nvcc_parent not in os.environ.get("PATH", ""):
            os.environ["PATH"] = nvcc_parent + os.pathsep + os.environ.get("PATH", "")

    spec = trace_index_add_spec(case)
    inputs = random_inputs(case)
    artifact = dml.compile(
        spec,
        dml.Target("cuda", arch=f"sm_{capability[0]}{capability[1]}"),
        tmp_path / f"{case.name}.dinoml",
    )
    _assert_generated_index_add_kernel(artifact.path, suffix=".cu")
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


def _assert_generated_index_add_kernel(artifact_path: str | os.PathLike[str], *, suffix: str) -> None:
    artifact_dir = Path(artifact_path)
    manifest = json.loads((artifact_dir / "kernel_manifest.json").read_text(encoding="utf-8"))
    entries = [item for item in manifest.get("required_kernels", []) if item.get("op") == "index_add"]
    assert len(entries) == 1
    generated = entries[0].get("generated_source")
    assert isinstance(generated, dict)
    assert str(generated.get("generated_function_name", "")).startswith("index_add_")
    source_hash = str(generated.get("source_hash", ""))
    assert source_hash
    assert str(generated.get("source_key", "")) == f"cuda:{generated['generated_function_name']}"
    emitted_source_path = Path("debug") / "generated_src" / "ops" / "index_add" / f"{source_hash}{suffix}"
    source_path = artifact_dir / emitted_source_path
    if source_path.is_file():
        assert "serialized placeholder" not in source_path.read_text(encoding="utf-8")
