from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import dinoml as dml


ATOL = 2.0e-3
RTOL = 2.0e-3


@dataclass(frozen=True)
class ScaledDotProductAttentionFrontendCase:
    name: str
    q_shape: tuple[int, int, int, int]
    kv_shape: tuple[int, int, int, int]
    causal: bool
    dtype: str = "float16"


SCALED_DOT_PRODUCT_ATTENTION_FRONTEND_CASES = (
    ScaledDotProductAttentionFrontendCase(
        name="scaled_dot_product_attention_nocausal_f16",
        q_shape=(1, 2, 4, 64),
        kv_shape=(1, 2, 4, 64),
        causal=False,
    ),
    ScaledDotProductAttentionFrontendCase(
        name="scaled_dot_product_attention_causal_f16",
        q_shape=(1, 2, 4, 64),
        kv_shape=(1, 2, 4, 64),
        causal=True,
    ),
)


class _ScaledDotProductAttentionModule(dml.Module):
    def __init__(self, case: ScaledDotProductAttentionFrontendCase):
        self.case = case

    def forward(self, q, k, v):
        return dml.ops.output(
            dml.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=self.case.causal,
                scale=None,
                enable_gqa=False,
            ),
            "out",
        )


def trace_scaled_dot_product_attention_frontend_spec(
    case: ScaledDotProductAttentionFrontendCase,
    *,
    dtype: str | None = None,
):
    dtype = dtype or case.dtype
    return dml.trace(
        _ScaledDotProductAttentionModule(case),
        inputs={
            "q": dml.TensorSpec(list(case.q_shape), dtype),
            "k": dml.TensorSpec(list(case.kv_shape), dtype),
            "v": dml.TensorSpec(list(case.kv_shape), dtype),
        },
        name=case.name,
    )


def random_inputs(
    case: ScaledDotProductAttentionFrontendCase,
    *,
    dtype: str | None = None,
    seed: int = 123,
) -> dict[str, np.ndarray]:
    dtype = dtype or case.dtype
    rng = np.random.default_rng(seed)
    q = rng.normal(size=case.q_shape).astype(np.float16)
    k = rng.normal(size=case.kv_shape).astype(np.float16)
    v = rng.normal(size=case.kv_shape).astype(np.float16)
    if dtype == "bfloat16":
        q = q.astype(np.float32)
        k = k.astype(np.float32)
        v = v.astype(np.float32)
    return {"q": q, "k": k, "v": v}


def torch_oracle(case: ScaledDotProductAttentionFrontendCase, inputs: dict[str, np.ndarray]) -> np.ndarray:
    torch = __import__("torch")
    q = torch.from_numpy(inputs["q"]).to(dtype=torch.float16)
    k = torch.from_numpy(inputs["k"]).to(dtype=torch.float16)
    v = torch.from_numpy(inputs["v"]).to(dtype=torch.float16)
    result = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=case.causal,
        scale=None,
        enable_gqa=False,
    )
    return result.cpu().numpy().astype(np.float16).astype(np.float32)
