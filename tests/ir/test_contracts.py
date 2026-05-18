from __future__ import annotations

import pytest

from dinoml.reference import reference_numpy
from dinoml.passes import PassManager, validate_ir
from tests.cases import GraphCase, ir_cases


def _node_ops(spec) -> set[str]:
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)
    ops = {str(node["op"]) for node in lowered["nodes"]}
    for node in lowered["nodes"]:
        if node["op"] == "fused_elementwise":
            ops.update(str(sub_op["op"]) for sub_op in node.get("attrs", {}).get("sub_ops", ()))
    return ops


@pytest.mark.parametrize("case", ir_cases(), ids=lambda case: case.name)
def test_ir_traces_validates_and_reference_executes(case: GraphCase):
    spec = case.build_spec()
    ops = _node_ops(spec)
    materialized_expected_ops = {
        op
        for op in case.expected_ops
        if op
        not in {
            "reshape",
            "flatten",
            "unsqueeze",
            "squeeze",
            "identity",
            "transpose",
            "split",
            "chunk",
            "meshgrid",
            "pixel_shuffle",
            "pixel_unshuffle",
        }
    }
    assert materialized_expected_ops <= ops

    outputs = reference_numpy(spec, case.inputs())

    assert {output["name"] for output in spec.ir["outputs"]} == set(outputs)
    for name, value in outputs.items():
        assert tuple(value.shape) == tuple(next(output["shape"] for output in spec.ir["outputs"] if output["name"] == name))
