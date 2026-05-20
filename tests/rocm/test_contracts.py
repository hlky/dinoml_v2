from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.reference import reference_numpy
from dinoml.runtime import load
from tests.cases import GraphCase, standard_cases


pytestmark = pytest.mark.skipif(
    os.environ.get("DINOML_RUN_ROCM_CONTRACTS") != "1",
    reason="set DINOML_RUN_ROCM_CONTRACTS=1 in the ROCm venv to compile/run ROCm artifacts",
)


ROCM_SIMPLE_CASES = [case for case in standard_cases() if case.name != "provider_ops"]


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


def _rocm_module_compile_toolchain_available() -> bool:
    if rocm_backend._rocm_sdk_command() is not None:
        return True
    if shutil.which("hipconfig") is not None:
        return True
    return bool(os.environ.get("ROCM_PATH") or os.environ.get("HIP_PATH"))
