from __future__ import annotations

import pytest

from dinoml.reference import reference_numpy
from tests.conv3d_frontend_parity import CONV3D_FRONTEND_CASES, random_inputs, torch_oracle, trace_conv3d_frontend_spec


@pytest.mark.parametrize("case", CONV3D_FRONTEND_CASES, ids=lambda case: case.name)
def test_conv3d_frontend_spellings_trace_to_existing_conv3d_ops(case):
    spec = trace_conv3d_frontend_spec(case)

    assert [node["op"] for node in spec.ir["nodes"]] == ["conv3d_bias"]
    node = spec.ir["nodes"][0]
    if case.use_bias:
        assert node["attrs"].get("source_op") is None
        assert node["attrs"].get("bias_mode") is None
    else:
        assert node["attrs"]["source_op"] == "conv3d"
        assert node["attrs"]["bias_mode"] == "explicit_zero_constant"

    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["y"]
    expected = torch_oracle(case, inputs)
    assert actual.shape == expected.shape


def test_conv3d_module_rejects_non_divisible_groups():
    import dinoml as dml

    with pytest.raises(ValueError, match="Conv3d in_channels must be divisible by groups"):
        dml.nn.Conv3d(3, 4, kernel_size=3, groups=2)
