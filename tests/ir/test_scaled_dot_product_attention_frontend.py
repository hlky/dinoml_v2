from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy
from tests.scaled_dot_product_attention_frontend_parity import (
    ATOL,
    RTOL,
    SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES,
    random_inputs,
    torch_oracle,
    trace_scaled_dot_product_attention_frontend_spec,
)


@pytest.mark.parametrize("case", SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES, ids=lambda case: case.name)
def test_scaled_dot_product_attention_frontend_routes_to_flash_attention(case):
    spec = trace_scaled_dot_product_attention_frontend_spec(case)
    inputs = random_inputs(case)
    actual = reference_numpy(spec, inputs)["out"].astype(np.float32)
    expected = torch_oracle(case, inputs)

    assert [node["op"] for node in spec.ir["nodes"]] == [
        "permute0213",
        "permute0213",
        "permute0213",
        "flash_attention",
        "permute0213",
    ]
    np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def test_scaled_dot_product_attention_rejects_attn_mask():
    class _MaskModule(dml.Module):
        def forward(self, q, k, v, mask):
            return dml.ops.output(
                dml.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask),
                "out",
            )

    with pytest.raises(NotImplementedError, match="attn_mask=None"):
        dml.trace(
            _MaskModule(),
            inputs={
                "q": dml.TensorSpec([1, 4, 2, 64], "float16"),
                "k": dml.TensorSpec([1, 4, 2, 64], "float16"),
                "v": dml.TensorSpec([1, 4, 2, 64], "float16"),
                "mask": dml.TensorSpec([4, 4], "float16"),
            },
            name="scaled_dot_product_attention_mask_unsupported",
        )


def test_scaled_dot_product_attention_rejects_dropout_scale_and_gqa():
    q = dml.TensorSpec([1, 4, 2, 64], "float16")
    for kwargs, match in (
        ({"dropout_p": 0.1}, "dropout_p=0.0"),
        ({"scale": 0.5}, "scale=None"),
        ({"enable_gqa": True}, "enable_gqa=False"),
    ):
        class _Module(dml.Module):
            def forward(self, q, k, v):
                return dml.ops.output(dml.nn.functional.scaled_dot_product_attention(q, k, v, **kwargs), "out")

        with pytest.raises(NotImplementedError, match=match):
            dml.trace(
                _Module(),
                inputs={"q": q, "k": q, "v": q},
                name=f"scaled_dot_product_attention_{match}",
            )
