from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.runtime import load
from tests.cases import GraphCase, standard_cases


@pytest.mark.parametrize("case", [case for case in standard_cases() if case.cpu], ids=lambda case: case.name)
def test_cpu_artifact_compiles_and_runs(case: GraphCase, tmp_path):
    spec = case.build_spec()
    artifact = dml.compile(spec, dml.Target("cpu"), tmp_path / f"{case.name}_cpu.dinoml")
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
