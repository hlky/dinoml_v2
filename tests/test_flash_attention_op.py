from __future__ import annotations

import math

import numpy as np

import dinoml as dml
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.reference import reference_numpy


class _FlashAttentionModule(dml.nn.Module):
    def __init__(self, *, causal: bool):
        self.causal = causal

    def forward(self, q, k, v):
        return dml.ops.output(dml.ops.flash_attention(q, k, v, causal=self.causal), "out")


class _FlashAttentionQKVModule(dml.nn.Module):
    def __init__(self, *, causal: bool):
        self.causal = causal

    def forward(self, qkv):
        return dml.ops.output(dml.ops.flash_attention_qkv(qkv, causal=self.causal), "out")


def test_flash_attention_reference_matches_naive_causal_attention():
    spec = dml.trace(
        _FlashAttentionModule(causal=True),
        inputs={
            "q": dml.TensorSpec([1, 4, 2, 64], "float16"),
            "k": dml.TensorSpec([1, 4, 2, 64], "float16"),
            "v": dml.TensorSpec([1, 4, 2, 64], "float16"),
        },
        name="flash_attention_reference",
    )
    rng = np.random.default_rng(123)
    inputs = {
        name: rng.normal(size=(1, 4, 2, 64)).astype(np.float16)
        for name in ("q", "k", "v")
    }

    actual = reference_numpy(spec, inputs)["out"].astype(np.float32)
    expected = _naive_flash_attention(inputs["q"], inputs["k"], inputs["v"], causal=True)

    np.testing.assert_allclose(actual, expected.astype(np.float16).astype(np.float32), atol=2.0e-3, rtol=2.0e-3)


def test_flash_attention_rocm_manifest_requests_ck_archive():
    spec = dml.trace(
        _FlashAttentionModule(causal=False),
        inputs={
            "q": dml.TensorSpec([1, 5, 2, 64], "float16"),
            "k": dml.TensorSpec([1, 5, 2, 64], "float16"),
            "v": dml.TensorSpec([1, 5, 2, 64], "float16"),
        },
        name="flash_attention_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    required = manifest["required_kernels"]

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention"
    assert required[0]["kernel_library"] == "flash_attn_ck"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_ck_fwd_float16_v1"
    assert required[0]["support_archive"].endswith("dinoml_flash_attn_ck.lib") or required[0][
        "support_archive"
    ].endswith("libdinoml_flash_attn_ck.a")


def test_flash_attention_cuda_manifest_requests_flash_attn_archive():
    spec = dml.trace(
        _FlashAttentionModule(causal=False),
        inputs={
            "q": dml.TensorSpec([1, 5, 2, 64], "float16"),
            "k": dml.TensorSpec([1, 5, 2, 64], "float16"),
            "v": dml.TensorSpec([1, 5, 2, 64], "float16"),
        },
        name="flash_attention_cuda_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("cuda").to_json())
    required = manifest["required_kernels"]

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention"
    assert required[0]["kernel_library"] == "flash_attn_cuda"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_cuda_fwd_float16_v1"
    assert required[0]["support_archive"].endswith("dinoml_flash_attn_cuda.lib") or required[0][
        "support_archive"
    ].endswith("libdinoml_flash_attn_cuda.a")


def test_flash_attention_qkv_cuda_manifest_requests_flash_attn_archive():
    spec = dml.trace(
        _FlashAttentionQKVModule(causal=True),
        inputs={"qkv": dml.TensorSpec([1, 5, 3, 2, 64], "float16")},
        name="flash_attention_qkv_cuda_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("cuda").to_json())
    required = manifest["required_kernels"]

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention_qkv"
    assert required[0]["kernel_library"] == "flash_attn_cuda"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_cuda_qkv_fwd_float16_v1"


def _naive_flash_attention(q, k, v, *, causal: bool) -> np.ndarray:
    q_value = q.astype(np.float32)
    k_value = k.astype(np.float32)
    v_value = v.astype(np.float32)
    _, seqlen_q, _, head_dim = q_value.shape
    seqlen_k = k_value.shape[1]
    scores = np.einsum("bqhd,bkhd->bhqk", q_value, k_value) * np.float32(1.0 / math.sqrt(head_dim))
    if causal:
        q_idx = np.arange(seqlen_q)[:, None]
        k_idx = np.arange(seqlen_k)[None, :]
        scores = np.where(k_idx > q_idx, np.float32(-1.0e30), scores)
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(shifted)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    return np.einsum("bhqk,bkhd->bqhd", probs, v_value)
