from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import dinoml as dml
from dinoml.backends import rocm as rocm_backend
from dinoml.compiler import _validate_mvp_runtime_contract
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.lowering.cuda import render_cuda_module
from dinoml.lowering.rocm import render_rocm_module
from dinoml.lowering.ops import render_launch
from dinoml.reference import reference_numpy


class _FlashAttentionModule(dml.nn.Module):
    def __init__(self, *, causal: bool):
        self.causal = causal

    def forward(self, q, k, v):
        return dml.ops.output(dml.ops.flash_attention(q, k, v, causal=self.causal), "out")


class _FlashAttentionBiasModule(dml.nn.Module):
    def __init__(self, *, causal: bool):
        self.causal = causal

    def forward(self, q, k, v, bias):
        return dml.ops.output(dml.ops.flash_attention_bias(q, k, v, bias, causal=self.causal), "out")


class _FlashAttentionQKVModule(dml.nn.Module):
    def __init__(self, *, causal: bool):
        self.causal = causal

    def forward(self, qkv):
        return dml.ops.output(dml.ops.flash_attention_qkv(qkv, causal=self.causal), "out")


class _FlashAttentionVarlenModule(dml.nn.Module):
    def __init__(self, *, max_seqlen: int, causal: bool = False):
        self.max_seqlen = int(max_seqlen)
        self.causal = bool(causal)

    def forward(self, q, k, v, cu_seqlens):
        return dml.ops.output(
            dml.ops.flash_attention_varlen(q, k, v, cu_seqlens, max_seqlen=self.max_seqlen, causal=self.causal),
            "out",
        )


class _RuntimeIndexSelectModule(dml.nn.Module):
    def forward(self, x, indices):
        return dml.ops.output(dml.ops.runtime_index_select(x, 0, indices), "out")


class _FlashAttentionStaticKvCacheModule(dml.nn.Module):
    def forward(self, q, past_key, past_value, new_key, new_value, cache_seqlens):
        return dml.ops.output(
            dml.ops.flash_attention_static_kv_cache(q, past_key, past_value, new_key, new_value, cache_seqlens),
            "out",
        )


class _FlashAttentionStaticKvCacheAdvanceModule(dml.nn.Module):
    def forward(self, q, past_key, past_value, new_key, new_value, cache_seqlens):
        return dml.ops.output(
            dml.ops.flash_attention_static_kv_cache(
                q,
                past_key,
                past_value,
                new_key,
                new_value,
                cache_seqlens,
                advance_cache_seqlens=True,
            ),
            "out",
        )


class _FlashAttentionStaticKvCacheBiasModule(dml.nn.Module):
    def forward(self, q, past_key, past_value, new_key, new_value, cache_seqlens, bias):
        return dml.ops.output(
            dml.ops.flash_attention_static_kv_cache_bias(
                q,
                past_key,
                past_value,
                new_key,
                new_value,
                cache_seqlens,
                bias,
            ),
            "out",
        )


class _FlashAttentionStaticKvCacheBiasStateModule(dml.nn.Module):
    def __init__(self, *, max_cache_len: int):
        self.max_cache_len = int(max_cache_len)

    def forward(self, q, new_key, new_value, bias):
        past_key = dml.state("past_key", dml.TensorSpec([1, 2, self.max_cache_len, 128], "bfloat16"))
        past_value = dml.state("past_value", dml.TensorSpec([1, 2, self.max_cache_len, 128], "bfloat16"))
        cache_seqlens = dml.state("cache_seqlens", dml.TensorSpec([1], "int32"))
        return dml.ops.output(
            dml.ops.flash_attention_static_kv_cache_bias(
                q,
                past_key,
                past_value,
                new_key,
                new_value,
                cache_seqlens,
                bias,
                advance_cache_seqlens=True,
            ),
            "out",
        )


class _FlashAttentionStaticKvCacheStateModule(dml.nn.Module):
    def __init__(self, *, max_cache_len: int):
        self.max_cache_len = int(max_cache_len)

    def forward(self, q, new_key, new_value):
        past_key = dml.state("past_key", dml.TensorSpec([1, 2, self.max_cache_len, 64], "bfloat16"))
        past_value = dml.state("past_value", dml.TensorSpec([1, 2, self.max_cache_len, 64], "bfloat16"))
        cache_seqlens = dml.state("cache_seqlens", dml.TensorSpec([1], "int32"))
        return dml.ops.output(
            dml.ops.flash_attention_static_kv_cache(
                q,
                past_key,
                past_value,
                new_key,
                new_value,
                cache_seqlens,
                advance_cache_seqlens=True,
            ),
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


def test_flash_attention_bias_reference_matches_naive_per_head_bias():
    spec = dml.trace(
        _FlashAttentionBiasModule(causal=False),
        inputs={
            "q": dml.TensorSpec([1, 4, 2, 64], "float16"),
            "k": dml.TensorSpec([1, 4, 1, 64], "float16"),
            "v": dml.TensorSpec([1, 4, 1, 64], "float16"),
            "bias": dml.TensorSpec([2, 4, 4], "float16"),
        },
        name="flash_attention_bias_reference",
    )
    rng = np.random.default_rng(123)
    inputs = {
        "q": rng.normal(size=(1, 4, 2, 64)).astype(np.float16),
        "k": rng.normal(size=(1, 4, 1, 64)).astype(np.float16),
        "v": rng.normal(size=(1, 4, 1, 64)).astype(np.float16),
        "bias": rng.normal(size=(2, 4, 4)).astype(np.float16),
    }

    actual = reference_numpy(spec, inputs)["out"].astype(np.float32)
    expected = _naive_flash_attention_bias(inputs["q"], inputs["k"], inputs["v"], inputs["bias"], causal=False)

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
        "cache_seqlens": np.asarray([3, 4], dtype=np.int32),
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


def test_flash_attention_static_kv_cache_bias_reference_matches_naive_attention():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheBiasModule(),
        inputs={
            "q": dml.TensorSpec([2, 1, 4, 32], "float16"),
            "past_key": dml.TensorSpec([2, 2, 5, 32], "float16"),
            "past_value": dml.TensorSpec([2, 2, 5, 32], "float16"),
            "new_key": dml.TensorSpec([2, 2, 1, 32], "float16"),
            "new_value": dml.TensorSpec([2, 2, 1, 32], "float16"),
            "cache_seqlens": dml.TensorSpec([2], "int32"),
            "bias": dml.TensorSpec([2, 4, 1, 5], "float16"),
        },
        name="flash_attention_static_kv_cache_bias_reference",
    )
    rng = np.random.default_rng(123)
    inputs = {
        "q": rng.normal(size=(2, 1, 4, 32)).astype(np.float16),
        "past_key": rng.normal(size=(2, 2, 5, 32)).astype(np.float16),
        "past_value": rng.normal(size=(2, 2, 5, 32)).astype(np.float16),
        "new_key": rng.normal(size=(2, 2, 1, 32)).astype(np.float16),
        "new_value": rng.normal(size=(2, 2, 1, 32)).astype(np.float16),
        "cache_seqlens": np.asarray([2, 4], dtype=np.int32),
        "bias": rng.normal(size=(2, 4, 1, 5)).astype(np.float16),
    }

    actual = reference_numpy(spec, inputs)["out"].astype(np.float32)
    expected = _naive_static_kv_cache_attention_bias(
        inputs["q"],
        inputs["past_key"],
        inputs["past_value"],
        inputs["new_key"],
        inputs["new_value"],
        inputs["cache_seqlens"],
        inputs["bias"],
    )

    np.testing.assert_allclose(actual, expected.astype(np.float16).astype(np.float32), atol=2.0e-3, rtol=2.0e-3)


def test_flash_attention_varlen_reference_matches_chunked_attention():
    spec = dml.trace(
        _FlashAttentionVarlenModule(max_seqlen=3),
        inputs={
            "q": dml.TensorSpec([5, 2, 32], "float16"),
            "k": dml.TensorSpec([5, 1, 32], "float16"),
            "v": dml.TensorSpec([5, 1, 32], "float16"),
            "cu_seqlens": dml.TensorSpec([3], "int32"),
        },
        name="flash_attention_varlen_reference",
    )
    rng = np.random.default_rng(123)
    inputs = {
        "q": rng.normal(size=(5, 2, 32)).astype(np.float16),
        "k": rng.normal(size=(5, 1, 32)).astype(np.float16),
        "v": rng.normal(size=(5, 1, 32)).astype(np.float16),
        "cu_seqlens": np.asarray([0, 3, 5], dtype=np.int32),
    }

    actual = reference_numpy(spec, inputs)["out"].astype(np.float32)
    expected = np.concatenate(
        [
            _naive_flash_attention(inputs["q"][None, :3], inputs["k"][None, :3], inputs["v"][None, :3], causal=False)[0],
            _naive_flash_attention(inputs["q"][None, 3:], inputs["k"][None, 3:], inputs["v"][None, 3:], causal=False)[0],
        ],
        axis=0,
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


def test_flash_attention_bias_rocm_manifest_requests_ck_bfloat16_head_dim128_archive():
    spec = dml.trace(
        _FlashAttentionBiasModule(causal=False),
        inputs={
            "q": dml.TensorSpec([1, 5, 4, 128], "bfloat16"),
            "k": dml.TensorSpec([1, 5, 2, 128], "bfloat16"),
            "v": dml.TensorSpec([1, 5, 2, 128], "bfloat16"),
            "bias": dml.TensorSpec([4, 5, 5], "bfloat16"),
        },
        name="flash_attention_bias_bfloat16_head_dim128_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    required = manifest["required_kernels"]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = next(node for node in spec.ir["nodes"] if node["op"] == "flash_attention_bias")
    launch = render_launch("rocm", node, tensors, kernel_manifest=manifest)

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention_bias"
    assert required[0]["kernel_library"] == "flash_attn_ck"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_ck_bias_fwd_bfloat16_v1"
    assert "dinoml_flash_attn_ck_bias_fwd_bfloat16_v1" in launch
    assert "ptr_bias" in launch
    assert "shape_bias_0" in launch


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


def test_flash_attention_varlen_rocm_manifest_requests_ck_bfloat16_archive():
    total_seq = dml.Dim("total_seq", min=1, max=8, typical=5, buckets=(5, 8))
    cu_count = dml.Dim("cu_count", min=2, max=4, typical=3, buckets=(3, 4))
    spec = dml.trace(
        _FlashAttentionVarlenModule(max_seqlen=4),
        inputs={
            "q": dml.TensorSpec([total_seq, 4, 128], "bfloat16"),
            "k": dml.TensorSpec([total_seq, 2, 128], "bfloat16"),
            "v": dml.TensorSpec([total_seq, 2, 128], "bfloat16"),
            "cu_seqlens": dml.TensorSpec([cu_count], "int32"),
        },
        name="flash_attention_varlen_rocm_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    required = manifest["required_kernels"]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = next(node for node in spec.ir["nodes"] if node["op"] == "flash_attention_varlen")
    launch = render_launch("rocm", node, tensors, kernel_manifest=manifest)

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention_varlen"
    assert required[0]["kernel_library"] == "flash_attn_ck"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_ck_varlen_fwd_bfloat16_v1"
    assert "dinoml_flash_attn_ck_varlen_fwd_bfloat16_v1" in launch
    assert "ptr_cu_seqlens" in launch
    assert "shape_cu_seqlens_0 - 1" in launch
    _validate_mvp_runtime_contract(spec.ir, dml.Target("rocm"))


def test_runtime_index_select_rocm_contract_accepts_dynamic_int32_indices():
    rows = dml.Dim("rows", min=1, max=8, typical=5, buckets=(5, 8))
    selected = dml.Dim("selected_rows", min=1, max=8, typical=5, buckets=(5, 8))
    spec = dml.trace(
        _RuntimeIndexSelectModule(),
        inputs={
            "x": dml.TensorSpec([rows, 16], "bfloat16"),
            "indices": dml.TensorSpec([selected], "int32"),
        },
        name="runtime_index_select_rocm",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    source = render_rocm_module(spec.ir, kernel_manifest=manifest)

    assert "runtime_index_select_" in source
    assert "const int32_t* DINO_RESTRICT indices" in source
    _validate_mvp_runtime_contract(spec.ir, dml.Target("rocm"))


def test_flash_attention_static_kv_cache_rocm_manifest_requests_ck_bfloat16_head_dim128_archive():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheModule(),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 128], "bfloat16"),
            "past_key": dml.TensorSpec([1, 2, 8, 128], "bfloat16"),
            "past_value": dml.TensorSpec([1, 2, 8, 128], "bfloat16"),
            "new_key": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "new_value": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "cache_seqlens": dml.TensorSpec([1], "int32"),
        },
        name="flash_attention_static_kv_cache_rocm_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    required = manifest["required_kernels"]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = next(node for node in spec.ir["nodes"] if node["op"] == "flash_attention_static_kv_cache")
    launch = render_launch("rocm", node, tensors, kernel_manifest=manifest)

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention_static_kv_cache"
    assert required[0]["kernel_library"] == "flash_attn_ck"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_ck_static_kv_cache_fwd_bfloat16_v1"
    assert "dinoml_flash_attn_ck_static_kv_cache_fwd_bfloat16_v1" in launch
    assert "ptr_cache_seqlens" in launch
    assert "flash_attention_static_kv_cache_scratch" in launch
    assert ", 0, session->flash_attention_static_kv_cache_scratch" in launch


def test_flash_attention_static_kv_cache_bias_rocm_manifest_requests_ck_bfloat16_head_dim128_archive():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheBiasModule(),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 128], "bfloat16"),
            "past_key": dml.TensorSpec([1, 2, 8, 128], "bfloat16"),
            "past_value": dml.TensorSpec([1, 2, 8, 128], "bfloat16"),
            "new_key": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "new_value": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "cache_seqlens": dml.TensorSpec([1], "int32"),
            "bias": dml.TensorSpec([4, 1, 8], "bfloat16"),
        },
        name="flash_attention_static_kv_cache_bias_rocm_manifest",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    required = manifest["required_kernels"]
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = next(node for node in spec.ir["nodes"] if node["op"] == "flash_attention_static_kv_cache_bias")
    launch = render_launch("rocm", node, tensors, kernel_manifest=manifest)

    assert len(required) == 1
    assert required[0]["op"] == "flash_attention_static_kv_cache_bias"
    assert required[0]["kernel_library"] == "flash_attn_ck"
    assert required[0]["kernel_symbol"] == "dinoml_flash_attn_ck_static_kv_cache_bias_fwd_bfloat16_v1"
    assert "dinoml_flash_attn_ck_static_kv_cache_bias_fwd_bfloat16_v1" in launch
    assert "ptr_bias" in launch
    assert "shape_bias_0" in launch
    assert "flash_attention_static_kv_cache_scratch" in launch
    assert ", 0, session->flash_attention_static_kv_cache_scratch" in launch
    _validate_mvp_runtime_contract(spec.ir, dml.Target("rocm"))


def test_flash_attention_static_kv_cache_bias_can_use_session_state_buffers():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheBiasStateModule(max_cache_len=8),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 128], "bfloat16"),
            "new_key": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "new_value": dml.TensorSpec([1, 2, 1, 128], "bfloat16"),
            "bias": dml.TensorSpec([4, 1, 8], "bfloat16"),
        },
        name="flash_attention_static_kv_cache_bias_state",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("rocm").to_json())
    source = render_rocm_module(spec.ir, kernel_manifest=manifest)
    input_names = {item["name"] for item in spec.ir["inputs"]}
    state_names = {item["name"] for item in spec.ir["states"]}

    assert "past_key" not in input_names
    assert "past_value" not in input_names
    assert "cache_seqlens" not in input_names
    assert state_names == {"past_key", "past_value", "cache_seqlens"}
    assert "void* state_past_key" in source
    assert "void* state_past_value" in source
    assert "void* state_cache_seqlens" in source
    assert "ptr_past_key = static_cast" in source
    assert "ptr_past_value = static_cast" in source
    assert "ptr_cache_seqlens = static_cast" in source
    assert "session->state_past_key" in source
    assert "session->state_past_value" in source
    assert "session->state_cache_seqlens" in source
    assert "dinoml_flash_attn_ck_static_kv_cache_bias_fwd_bfloat16_v1" in source
    assert ", 1, session->flash_attention_static_kv_cache_scratch" in source
    _validate_mvp_runtime_contract(spec.ir, dml.Target("rocm"))


def test_flash_attention_static_kv_cache_cuda_can_use_session_state_buffers_and_advance_cache_seqlens():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheStateModule(max_cache_len=8),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 64], "bfloat16"),
            "new_key": dml.TensorSpec([1, 2, 1, 64], "bfloat16"),
            "new_value": dml.TensorSpec([1, 2, 1, 64], "bfloat16"),
        },
        name="flash_attention_static_kv_cache_cuda_state",
    )

    manifest = build_kernel_manifest(spec.ir, dml.Target("cuda").to_json())
    source = render_cuda_module(spec.ir, kernel_manifest=manifest)
    input_names = {item["name"] for item in spec.ir["inputs"]}
    state_names = {item["name"] for item in spec.ir["states"]}

    assert "past_key" not in input_names
    assert "past_value" not in input_names
    assert "cache_seqlens" not in input_names
    assert state_names == {"past_key", "past_value", "cache_seqlens"}
    assert "void* state_past_key" in source
    assert "void* state_past_value" in source
    assert "void* state_cache_seqlens" in source
    assert "ptr_past_key = static_cast" in source
    assert "ptr_past_value = static_cast" in source
    assert "ptr_cache_seqlens = static_cast" in source
    assert "session->state_past_key" in source
    assert "session->state_past_value" in source
    assert "session->state_cache_seqlens" in source
    assert "dinoml_flash_attn_cuda_static_kv_cache_fwd_bfloat16_v1" in source
    assert ", 1, session->stream))" in source
    _validate_mvp_runtime_contract(spec.ir, dml.Target("cuda"))


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


def test_flash_attention_static_kv_cache_cuda_lowering_can_advance_input_cache_seqlens():
    spec = dml.trace(
        _FlashAttentionStaticKvCacheAdvanceModule(),
        inputs={
            "q": dml.TensorSpec([1, 1, 4, 64], "float16"),
            "past_key": dml.TensorSpec([1, 2, 8, 64], "float16"),
            "past_value": dml.TensorSpec([1, 2, 8, 64], "float16"),
            "new_key": dml.TensorSpec([1, 2, 1, 64], "float16"),
            "new_value": dml.TensorSpec([1, 2, 1, 64], "float16"),
            "cache_seqlens": dml.TensorSpec([1], "int32"),
        },
        name="flash_attention_static_kv_cache_cuda_advance_input",
    )
    manifest = build_kernel_manifest(spec.ir, dml.Target("cuda").to_json())
    tensors = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    node = next(node for node in spec.ir["nodes"] if node["op"] == "flash_attention_static_kv_cache")

    launch = render_launch("cuda", node, tensors, kernel_manifest=manifest)

    assert "dinoml_flash_attn_cuda_static_kv_cache_fwd_float16_v1" in launch
    assert ", 1, session->stream))" in launch


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


def _naive_flash_attention_bias(q, k, v, bias, *, causal: bool) -> np.ndarray:
    q_value = q.astype(np.float32)
    k_value = k.astype(np.float32)
    v_value = v.astype(np.float32)
    bias_value = bias.astype(np.float32)
    batch, seqlen_q, heads_q, head_dim = q_value.shape
    heads_k = k_value.shape[2]
    if heads_q != heads_k:
        repeat = heads_q // heads_k
        k_value = np.repeat(k_value, repeat, axis=2)
        v_value = np.repeat(v_value, repeat, axis=2)
    seqlen_k = k_value.shape[1]
    if bias_value.ndim == 2:
        bias_value = bias_value.reshape(1, 1, seqlen_q, seqlen_k)
    elif bias_value.ndim == 3:
        bias_value = bias_value.reshape(1, bias_value.shape[0], seqlen_q, seqlen_k)
    scores = np.einsum("bqhd,bkhd->bhqk", q_value, k_value) * np.float32(1.0 / math.sqrt(head_dim))
    scores = scores + np.broadcast_to(bias_value, (batch, heads_q, seqlen_q, seqlen_k))
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


def _naive_static_kv_cache_attention_bias(q, past_key, past_value, new_key, new_value, cache_seqlens, bias) -> np.ndarray:
    bias_value = bias.astype(np.float32)
    if bias_value.ndim == 2:
        bias_value = bias_value.reshape(1, 1, bias_value.shape[0], bias_value.shape[1])
    elif bias_value.ndim == 3:
        bias_value = bias_value.reshape(1, bias_value.shape[0], bias_value.shape[1], bias_value.shape[2])
    broadcast_bias = np.broadcast_to(bias_value, (q.shape[0], q.shape[2], q.shape[1], past_key.shape[2]))
    outputs = []
    for batch_idx, valid_past in enumerate(cache_seqlens.tolist()):
        total_len = int(valid_past) + 1
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
        outputs.append(
            _naive_flash_attention_bias(
                q[batch_idx : batch_idx + 1],
                k,
                v,
                broadcast_bias[batch_idx : batch_idx + 1, :, :, :total_len],
                causal=False,
            )
        )
    return np.concatenate(outputs, axis=0)
