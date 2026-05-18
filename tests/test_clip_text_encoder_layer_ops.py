import math

import numpy as np

import dinoml as dml
from dinoml.reference import reference_numpy
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.passes import PassManager, validate_ir


BATCH = 2
SEQ_LEN = 4
HIDDEN = 6
NUM_HEADS = 2
HEAD_DIM = HIDDEN // NUM_HEADS
INTERMEDIATE = 8
EPS = 1.0e-5
MASK_FILL = np.float32(-1.0e4)
ATTN_SCALE = 1.0 / math.sqrt(HEAD_DIM)


def _encoder_layer_constants():
    rng = np.random.default_rng(2027)

    def _weight(shape, scale):
        return (rng.standard_normal(shape).astype(np.float32) / scale).astype(np.float32)

    def _bias(length, scale):
        return (rng.standard_normal((length,)).astype(np.float32) / scale).astype(np.float32)

    causal = np.zeros((1, SEQ_LEN, SEQ_LEN), dtype=np.float32)
    rows, cols = np.triu_indices(SEQ_LEN, k=1)
    causal[:, rows, cols] = MASK_FILL

    return {
        "ln1_weight": _weight((HIDDEN,), 4.0),
        "ln1_bias": _bias(HIDDEN, 6.0),
        "q_weight": _weight((HIDDEN, HIDDEN), 5.0),
        "q_bias": _bias(HIDDEN, 7.0),
        "k_weight": _weight((HIDDEN, HIDDEN), 5.0),
        "k_bias": _bias(HIDDEN, 7.0),
        "v_weight": _weight((HIDDEN, HIDDEN), 5.0),
        "v_bias": _bias(HIDDEN, 7.0),
        "out_weight": _weight((HIDDEN, HIDDEN), 5.0),
        "out_bias": _bias(HIDDEN, 7.0),
        "ln2_weight": _weight((HIDDEN,), 4.0),
        "ln2_bias": _bias(HIDDEN, 6.0),
        "fc1_weight": _weight((INTERMEDIATE, HIDDEN), 4.0),
        "fc1_bias": _bias(INTERMEDIATE, 6.0),
        "fc2_weight": _weight((HIDDEN, INTERMEDIATE), 4.0),
        "fc2_bias": _bias(HIDDEN, 6.0),
        # Keep the causal path honest: this bounded slice uses a static additive
        # causal mask parameter instead of inventing a broader mask-building op.
        "causal_mask": causal,
    }


CONSTANTS = _encoder_layer_constants()


def _hidden_states():
    return np.array(
        [
            [
                [0.25, -0.50, 0.75, 1.00, -0.25, 0.50],
                [1.50, 0.25, -0.75, 0.50, 1.25, -1.00],
                [-0.25, 0.75, 0.50, -1.25, 0.00, 0.25],
                [0.50, -1.50, 1.25, 0.75, -0.50, 1.00],
            ],
            [
                [-1.00, 0.50, 0.25, -0.75, 1.50, 0.00],
                [0.75, -0.25, -1.25, 1.00, 0.50, -0.50],
                [1.25, 1.50, -0.50, 0.25, -1.00, 0.75],
                [-0.50, 0.00, 1.00, -1.50, 0.25, 1.25],
            ],
        ],
        dtype=np.float32,
    )


def _attention_mask():
    return np.array(
        [
            [True, True, True, False],
            [True, False, True, False],
        ],
        dtype=np.bool_,
    )


def _reference_layer_norm(x, weight, bias):
    x = np.asarray(x, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.maximum(np.mean(x * x, axis=-1, keepdims=True) - mean * mean, 0.0)
    return ((x - mean) * (1.0 / np.sqrt(variance + EPS)) * weight + bias).astype(np.float32)


def _project_heads(x):
    x = x.reshape(BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM)
    x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(BATCH * NUM_HEADS, SEQ_LEN, HEAD_DIM)


def _merge_heads(x):
    x = x.reshape(BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM)
    x = np.transpose(x, (0, 2, 1, 3))
    return x.reshape(BATCH, SEQ_LEN, HIDDEN)


def _softmax_last_dim(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _reference_attention(hidden_states, attention_mask=None):
    q = hidden_states @ CONSTANTS["q_weight"].T + CONSTANTS["q_bias"]
    k = hidden_states @ CONSTANTS["k_weight"].T + CONSTANTS["k_bias"]
    v = hidden_states @ CONSTANTS["v_weight"].T + CONSTANTS["v_bias"]

    q = _project_heads(q)
    k = _project_heads(k)
    v = _project_heads(v)

    scores = np.matmul(q, np.swapaxes(k, -1, -2)).astype(np.float32)
    scores = scores * ATTN_SCALE
    scores = scores + CONSTANTS["causal_mask"]

    if attention_mask is not None:
        keep = attention_mask.astype(np.bool_)[:, None, None, :]
        keep = np.broadcast_to(keep, (BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN)).reshape(BATCH * NUM_HEADS, SEQ_LEN, SEQ_LEN)
        scores = scores + np.where(keep, np.float32(0.0), MASK_FILL).astype(np.float32)

    probs = _softmax_last_dim(scores).astype(np.float32)
    context = np.matmul(probs, v).astype(np.float32)
    context = _merge_heads(context)
    return (context @ CONSTANTS["out_weight"].T + CONSTANTS["out_bias"]).astype(np.float32)


def _reference_encoder_layer(hidden_states, attention_mask=None):
    residual = np.asarray(hidden_states, dtype=np.float32)
    attn_input = _reference_layer_norm(residual, CONSTANTS["ln1_weight"], CONSTANTS["ln1_bias"])
    attn_output = _reference_attention(attn_input, attention_mask=attention_mask)
    hidden_states = (residual + attn_output).astype(np.float32)

    residual = hidden_states
    mlp_input = _reference_layer_norm(hidden_states, CONSTANTS["ln2_weight"], CONSTANTS["ln2_bias"])
    intermediate = mlp_input @ CONSTANTS["fc1_weight"].T + CONSTANTS["fc1_bias"]
    intermediate = (intermediate / (1.0 + np.exp(-1.702 * intermediate))).astype(np.float32)
    mlp_output = (intermediate @ CONSTANTS["fc2_weight"].T + CONSTANTS["fc2_bias"]).astype(np.float32)
    return (residual + mlp_output).astype(np.float32)


class _ClipTextEncoderLayerBase(dml.Module):
    def __init__(self):
        self.ln1_weight = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["ln1_weight"])
        self.ln1_bias = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["ln1_bias"])
        self.q_weight = dml.Parameter([HIDDEN, HIDDEN], dtype="float32", value=CONSTANTS["q_weight"])
        self.q_bias = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["q_bias"])
        self.k_weight = dml.Parameter([HIDDEN, HIDDEN], dtype="float32", value=CONSTANTS["k_weight"])
        self.k_bias = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["k_bias"])
        self.v_weight = dml.Parameter([HIDDEN, HIDDEN], dtype="float32", value=CONSTANTS["v_weight"])
        self.v_bias = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["v_bias"])
        self.out_weight = dml.Parameter([HIDDEN, HIDDEN], dtype="float32", value=CONSTANTS["out_weight"])
        self.out_bias = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["out_bias"])
        self.ln2_weight = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["ln2_weight"])
        self.ln2_bias = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["ln2_bias"])
        self.fc1_weight = dml.Parameter([INTERMEDIATE, HIDDEN], dtype="float32", value=CONSTANTS["fc1_weight"])
        self.fc1_bias = dml.Parameter([INTERMEDIATE], dtype="float32", value=CONSTANTS["fc1_bias"])
        self.fc2_weight = dml.Parameter([HIDDEN, INTERMEDIATE], dtype="float32", value=CONSTANTS["fc2_weight"])
        self.fc2_bias = dml.Parameter([HIDDEN], dtype="float32", value=CONSTANTS["fc2_bias"])
        self.causal_mask = dml.Parameter([1, SEQ_LEN, SEQ_LEN], dtype="float32", value=CONSTANTS["causal_mask"])

    def _attention(self, hidden_states, attention_mask=None):
        q = dml.ops.gemm_rcr_bias(hidden_states, self.q_weight, self.q_bias)
        k = dml.ops.gemm_rcr_bias(hidden_states, self.k_weight, self.k_bias)
        v = dml.ops.gemm_rcr_bias(hidden_states, self.v_weight, self.v_bias)

        q = dml.ops.reshape(q, [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM])
        k = dml.ops.reshape(k, [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM])
        v = dml.ops.reshape(v, [BATCH, SEQ_LEN, NUM_HEADS, HEAD_DIM])

        q = dml.ops.permute0213(q)
        k = dml.ops.permute0213(k)
        v = dml.ops.permute0213(v)

        q = dml.ops.flatten(q, start_dim=0, end_dim=1)
        k = dml.ops.flatten(k, start_dim=0, end_dim=1)
        v = dml.ops.flatten(v, start_dim=0, end_dim=1)

        scores = dml.ops.bmm_rcr(q, k)
        scores = scores * ATTN_SCALE
        scores = scores + self.causal_mask
        if attention_mask is not None:
            keep = dml.ops.reshape(attention_mask, [BATCH, 1, 1, SEQ_LEN])
            keep = dml.ops.expand(keep, [BATCH, NUM_HEADS, SEQ_LEN, SEQ_LEN])
            keep = dml.ops.reshape(keep, [BATCH * NUM_HEADS, SEQ_LEN, SEQ_LEN])
            zeros = dml.ops.full([BATCH * NUM_HEADS, SEQ_LEN, SEQ_LEN], 0.0, dtype="float32")
            masked = dml.ops.full([BATCH * NUM_HEADS, SEQ_LEN, SEQ_LEN], float(MASK_FILL), dtype="float32")
            scores = scores + dml.ops.where(keep, zeros, masked)
        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, v)
        context = dml.ops.reshape(context, [BATCH, NUM_HEADS, SEQ_LEN, HEAD_DIM])
        context = dml.ops.permute0213(context)
        context = dml.ops.reshape(context, [BATCH, SEQ_LEN, HIDDEN])
        return dml.ops.gemm_rcr_bias(context, self.out_weight, self.out_bias)

    def _encoder_layer(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = dml.ops.layer_norm(hidden_states, self.ln1_weight, self.ln1_bias, eps=EPS)
        hidden_states = self._attention(hidden_states, attention_mask=attention_mask)
        hidden_states = dml.ops.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = dml.ops.layer_norm(hidden_states, self.ln2_weight, self.ln2_bias, eps=EPS)
        hidden_states = dml.ops.gemm_rcr_bias_quick_gelu(hidden_states, self.fc1_weight, self.fc1_bias)
        hidden_states = dml.ops.gemm_rcr_bias(hidden_states, self.fc2_weight, self.fc2_bias)
        return dml.ops.add(residual, hidden_states)


class ClipTextEncoderLayerStaticCausalMaskModule(_ClipTextEncoderLayerBase):
    def forward(self, hidden_states):
        return dml.ops.output(self._encoder_layer(hidden_states), "out")


class ClipTextEncoderLayerWithPaddingMaskModule(_ClipTextEncoderLayerBase):
    def forward(self, hidden_states, attention_mask):
        return dml.ops.output(self._encoder_layer(hidden_states, attention_mask=attention_mask), "out")


def _trace_static_causal():
    return dml.trace(
        ClipTextEncoderLayerStaticCausalMaskModule(),
        inputs={"hidden_states": dml.TensorSpec([BATCH, SEQ_LEN, HIDDEN], "float32")},
        name="clip_text_encoder_layer_static_causal_mask",
    )


def _trace_with_padding_mask():
    return dml.trace(
        ClipTextEncoderLayerWithPaddingMaskModule(),
        inputs={
            "hidden_states": dml.TensorSpec([BATCH, SEQ_LEN, HIDDEN], "float32"),
            "attention_mask": dml.TensorSpec([BATCH, SEQ_LEN], "bool"),
        },
        name="clip_text_encoder_layer_with_padding_mask",
    )


def test_clip_text_encoder_layer_static_causal_mask_cpu_reference():
    spec = _trace_static_causal()
    hidden_states = _hidden_states()

    node_ops = [node["op"] for node in spec.ir["nodes"]]
    assert node_ops.count("layer_norm") == 2
    assert node_ops.count("gemm_rcr_bias") == 5
    assert node_ops.count("gemm_rcr_bias_quick_gelu") == 1
    assert node_ops.count("permute0213") == 4
    assert node_ops.count("bmm_rcr") == 1
    assert node_ops.count("bmm_rrr") == 1
    assert node_ops.count("softmax") == 1
    assert node_ops.count("add") >= 3
    assert spec.ir["outputs"][0]["shape"] == [BATCH, SEQ_LEN, HIDDEN]
    assert spec.ir["outputs"][0]["dtype"] == "float32"

    actual = reference_numpy(spec, {"hidden_states": hidden_states})["out"]
    expected = _reference_encoder_layer(hidden_states)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clip_text_encoder_layer_bool_padding_mask_cpu_reference():
    spec = _trace_with_padding_mask()
    hidden_states = _hidden_states()
    attention_mask = _attention_mask()

    node_ops = [node["op"] for node in spec.ir["nodes"]]
    assert node_ops.count("layer_norm") == 2
    assert node_ops.count("gemm_rcr_bias") == 5
    assert node_ops.count("gemm_rcr_bias_quick_gelu") == 1
    assert node_ops.count("permute0213") == 4
    assert node_ops.count("bmm_rcr") == 1
    assert node_ops.count("bmm_rrr") == 1
    assert node_ops.count("softmax") == 1
    assert node_ops.count("expand") == 1
    assert node_ops.count("where") == 1
    assert node_ops.count("full") == 2
    assert spec.ir["outputs"][0]["shape"] == [BATCH, SEQ_LEN, HIDDEN]
    assert spec.ir["outputs"][0]["dtype"] == "float32"

    actual = reference_numpy(spec, {"hidden_states": hidden_states, "attention_mask": attention_mask})["out"]
    expected = _reference_encoder_layer(hidden_states, attention_mask=attention_mask)

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_clip_text_encoder_layer_manifest_keeps_provider_and_model_kernels_honest():
    spec = _trace_with_padding_mask()
    lowered, _ = PassManager().run(spec.ir)
    validate_ir(lowered)

    manifest = build_kernel_manifest(lowered, {"name": "cuda", "arch": "sm_86"})
    required = manifest["required_kernels"]
    ops = [entry["op"] for entry in required]

    layer_norm_entries = [entry for entry in required if entry["op"] == "layer_norm"]
    gemm_bias_entries = [entry for entry in required if entry["op"] == "gemm_rcr_bias"]
    quick_gelu_entries = [entry for entry in required if entry["op"] == "gemm_rcr_bias_quick_gelu"]
    bmm_entries = [entry for entry in required if entry["op"] in {"bmm_rcr", "bmm_rrr"}]
    softmax_entries = [entry for entry in required if entry["op"] == "softmax"]

    assert "layer_norm" in ops
    assert "gemm_rcr_bias" in ops
    assert "gemm_rcr_bias_quick_gelu" in ops
    assert "bmm_rcr" in ops
    assert "bmm_rrr" in ops
    assert "softmax" in ops

    assert len(layer_norm_entries) >= 1
    assert len(gemm_bias_entries) >= 1
    assert len(quick_gelu_entries) == 1
    assert len(softmax_entries) == 1
    assert len(bmm_entries) == 2

    assert all(entry["kernel_library"] == "model" for entry in layer_norm_entries)
    assert all(entry["kernel_symbol"] == "generated_layer_norm" for entry in layer_norm_entries)
    assert all(entry["kernel_library"] == "cutlass_gemm" for entry in gemm_bias_entries)
    assert all(entry["kernel_symbol"].startswith("dinoml_cutlass_gemm_rcr_bias_float32_") for entry in gemm_bias_entries)
    [quick_gelu_entry] = quick_gelu_entries
    assert quick_gelu_entry["kernel_library"] == "cutlass_gemm"
    assert quick_gelu_entry["kernel_symbol"].startswith("dinoml_cutlass_gemm_rcr_bias_quick_gelu_float32_")
    assert quick_gelu_entry["candidate_set"]["epilogue_config"]["activation"] == "quick_gelu"
    assert all(entry["kernel_library"] == "cutlass_bmm" for entry in bmm_entries)
    assert any(entry["kernel_symbol"].startswith("dinoml_cutlass_bmm_rcr_float32_") for entry in bmm_entries)
    assert any(entry["kernel_symbol"].startswith("dinoml_cutlass_bmm_rrr_float32_") for entry in bmm_entries)
    assert softmax_entries[0]["kernel_library"] == "model"
    assert softmax_entries[0]["kernel_symbol"] == "generated_softmax"
