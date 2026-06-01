from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.compiler import _validate_mvp_runtime_contract
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.ops import render_launch
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


class _FlashAttentionStaticKvCacheModule(dml.nn.Module):
    def forward(self, q, past_key, past_value, new_key, new_value, cache_seqlens):
        return dml.ops.output(
            dml.ops.flash_attention_static_kv_cache(q, past_key, past_value, new_key, new_value, cache_seqlens),
            "out",
        )


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


def test_flash_attention_static_kv_cache_reference_matches_naive_attention():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheModule(),
        inputs={
            "q": dml.TensorSpec([2, 1, 4, 32], "float16"),
            "past_key": dml.TensorSpec([2, 2, 5, 32], "float16"),
            "past_value": dml.TensorSpec([2, 2, 5, 32], "float16"),
            "new_key": dml.TensorSpec([2, 2, 1, 32], "float16"),
            "new_value": dml.TensorSpec([2, 2, 1, 32], "float16"),
            "cache_seqlens": dml.TensorSpec([2], "int32"),
        },
        name="flash_attention_static_kv_cache_reference",
    )
    rng = np.random.default_rng(123)
    inputs = {
        "q": rng.normal(size=(2, 1, 4, 32)).astype(np.float16),
        "past_key": rng.normal(size=(2, 2, 5, 32)).astype(np.float16),
        "past_value": rng.normal(size=(2, 2, 5, 32)).astype(np.float16),
        "new_key": rng.normal(size=(2, 2, 1, 32)).astype(np.float16),
        "new_value": rng.normal(size=(2, 2, 1, 32)).astype(np.float16),
        "cache_seqlens": np.asarray([3, 5], dtype=np.int32),
    }

    actual = reference_numpy(spec, inputs)["out"].astype(np.float32)
    expected = _naive_static_kv_cache_attention(
        inputs["q"],
        inputs["past_key"],
        inputs["past_value"],
        inputs["new_key"],
        inputs["new_value"],
        inputs["cache_seqlens"],
    )

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


def test_flash_attention_rocm_manifest_requests_ck_bfloat16_head_dim128_archive():
    spec = dml.trace(
        _FlashAttentionModule(causal=True),
        inputs={
            "q": dml.TensorSpec([1, 5, 4, 128], "bfloat16"),
            "k": dml.TensorSpec([1, 5, 2, 128], "bfloat16"),
            "v": dml.TensorSpec([1, 5, 2, 128], "bfloat16"),
        },
        name="flash_attention_bfloat16_head_dim128_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    required = manifest["required_kernels"]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = next(node for node in spec.ir["nodes"] if node["op"] == "flash_attention")
    launch = render_launch("rocm", node, tensors, kernel_manifest=manifest)

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention"
    assert required[0]["kernel_library"] == "flash_attn_ck"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_ck_fwd_bfloat16_v1"
    assert "dinoml_flash_attn_ck_fwd_bfloat16_v1" in launch
    assert "!= 64" not in launch
    assert "> 256" in launch


def test_flash_attention_qkv_rocm_manifest_requests_ck_bfloat16_head_dim128_archive():
    spec = dml.trace(
        _FlashAttentionQKVModule(causal=True),
        inputs={"qkv": dml.TensorSpec([1, 5, 3, 2, 128], "bfloat16")},
        name="flash_attention_qkv_bfloat16_head_dim128_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    required = manifest["required_kernels"]

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention_qkv"
    assert required[0]["kernel_library"] == "flash_attn_ck"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_ck_qkv_fwd_bfloat16_v1"


def test_flash_attention_rocm_support_manifest_records_requested_ck_dtype_modules(tmp_path: Path):
    archive = tmp_path / "libdinoml_flash_attn_ck.a"
    archive.write_bytes(b"archive")
    modules = rocm_backend._required_flash_attn_ck_modules(
        {
            "required_kernels": [
                {
                    "kernel_library": "flash_attn_ck",
                    "op": "flash_attention",
                    "dtype": "float16",
                    "kernel_symbol": "dinoml_flash_attn_ck_fwd_float16_v1",
                },
                {
                    "kernel_library": "flash_attn_ck",
                    "op": "flash_attention",
                    "dtype": "bfloat16",
                    "kernel_symbol": "dinoml_flash_attn_ck_fwd_bfloat16_v1",
                },
                {
                    "kernel_library": "flash_attn_ck",
                    "op": "flash_attention",
                    "dtype": "bfloat16",
                    "kernel_symbol": "dinoml_flash_attn_ck_fwd_bfloat16_v1",
                },
            ]
        },
        archive=archive,
        target="dinoml_flash_attn_ck",
    )

    assert [module["dtype"] for module in modules] == ["bfloat16", "float16"]
    assert [module["kernel_symbol"] for module in modules] == [
        "dinoml_flash_attn_ck_fwd_bfloat16_v1",
        "dinoml_flash_attn_ck_fwd_float16_v1",
    ]
    assert all(module["archive"] == archive.name for module in modules)
    assert all(module["archive_sha256"] for module in modules)


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


def test_flash_attention_static_kv_cache_cuda_manifest_requests_flash_attn_archive():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheModule(),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 64], "bfloat16"),
            "past_key": dml.TensorSpec([1, 2, 8, 64], "bfloat16"),
            "past_value": dml.TensorSpec([1, 2, 8, 64], "bfloat16"),
            "new_key": dml.TensorSpec([1, 2, 1, 64], "bfloat16"),
            "new_value": dml.TensorSpec([1, 2, 1, 64], "bfloat16"),
            "cache_seqlens": dml.TensorSpec([1], "int32"),
        },
        name="flash_attention_static_kv_cache_cuda_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("cuda").to_json())
    required = manifest["required_kernels"]

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention_static_kv_cache"
    assert required[0]["kernel_library"] == "flash_attn_cuda"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_cuda_static_kv_cache_fwd_bfloat16_v1"


def test_flash_attention_static_kv_cache_cuda_lowering_uses_cache_seqlens_symbol():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheModule(),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 64], "float16"),
            "past_key": dml.TensorSpec([1, 2, 8, 64], "float16"),
            "past_value": dml.TensorSpec([1, 2, 8, 64], "float16"),
            "new_key": dml.TensorSpec([1, 2, 1, 64], "float16"),
            "new_value": dml.TensorSpec([1, 2, 1, 64], "float16"),
            "cache_seqlens": dml.TensorSpec([1], "int32"),
        },
        name="flash_attention_static_kv_cache_cuda_lowering",
    )
    manifest = build_kernel_manifest(spec.ir, dml.Target("cuda").to_json())
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = next(node for node in spec.ir["nodes"] if node["op"] == "flash_attention_static_kv_cache")

    launch = render_launch("cuda", node, tensors, kernel_manifest=manifest)

    assert "dinoml_flash_attn_cuda_static_kv_cache_fwd_float16_v1" in launch
    assert "ptr_cache_seqlens" in launch


def test_flash_attention_static_kv_cache_cuda_contract_accepts_int32_cache_seqlens():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheModule(),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 64], "bfloat16"),
            "past_key": dml.TensorSpec([1, 2, 8, 64], "bfloat16"),
            "past_value": dml.TensorSpec([1, 2, 8, 64], "bfloat16"),
            "new_key": dml.TensorSpec([1, 2, 1, 64], "bfloat16"),
            "new_value": dml.TensorSpec([1, 2, 1, 64], "bfloat16"),
            "cache_seqlens": dml.TensorSpec([1], "int32"),
        },
        name="flash_attention_static_kv_cache_cuda_contract",
    )

    _validate_mvp_runtime_contract(spec.ir, dml.Target("cuda"))


def _naive_flash_attention(q, k, v, *, causal: bool) -> np.ndarray:
    q_value = q.astype(np.float32)
    k_value = k.astype(np.float32)
    v_value = v.astype(np.float32)
    _, seqlen_q, heads_q, head_dim = q_value.shape
    heads_k = k_value.shape[2]
    if heads_q != heads_k:
        repeat = heads_q // heads_k
        k_value = np.repeat(k_value, repeat, axis=2)
        v_value = np.repeat(v_value, repeat, axis=2)
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


def _naive_static_kv_cache_attention(q, past_key, past_value, new_key, new_value, cache_seqlens) -> np.ndarray:
    outputs = []
    for batch_idx, valid_past in enumerate(cache_seqlens.tolist()):
        k = np.concatenate(
            [
                np.transpose(past_key[batch_idx : batch_idx + 1, :, :valid_past, :], (0, 2, 1, 3)),
                np.transpose(new_key[batch_idx : batch_idx + 1], (0, 2, 1, 3)),
            ],
            axis=1,
        )
        v = np.concatenate(
            [
                np.transpose(past_value[batch_idx : batch_idx + 1, :, :valid_past, :], (0, 2, 1, 3)),
                np.transpose(new_value[batch_idx : batch_idx + 1], (0, 2, 1, 3)),
            ],
            axis=1,
        )
        outputs.append(_naive_flash_attention(q[batch_idx : batch_idx + 1], k, v, causal=False))
    return np.concatenate(outputs, axis=0)
