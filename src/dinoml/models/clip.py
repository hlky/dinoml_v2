from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import numpy as np

import dinoml as dml


@dataclass(frozen=True)
class LegacyCLIPTextConfig:
    """Bounded legacy OpenAI CLIP text-tower config.

    This helper intentionally models only the bounded text path currently
    landed in DinoML: explicit `position_ids`, static traced sequence length,
    and the two source CLIP EOS pooling branches (`eos_token_id == 2`
    argmax/highest-token-id compatibility, otherwise first EOS equality match).
    """

    vocab_size: int
    max_position_embeddings: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    projection_dim: int
    layer_norm_eps: float = 1.0e-5
    eos_token_id: int = 2
    mask_fill_value: float = -1.0e4

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.max_position_embeddings <= 0:
            raise ValueError("max_position_embeddings must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def build_clip_causal_mask(seq_len: int, mask_fill_value: float = -1.0e4) -> np.ndarray:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    causal = np.zeros((1, seq_len, seq_len), dtype=np.float32)
    rows, cols = np.triu_indices(seq_len, k=1)
    causal[:, rows, cols] = np.float32(mask_fill_value)
    return causal


class _LegacyCLIPTextEncoderLayer(dml.Module):
    def __init__(self, config: LegacyCLIPTextConfig, weights: Mapping[str, np.ndarray], layer_idx: int):
        prefix = f"text_model.encoder.layers.{layer_idx}"
        self.config = config
        self.layer_norm1_weight = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.layer_norm1.weight", (config.hidden_size,)),
        )
        self.layer_norm1_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.layer_norm1.bias", (config.hidden_size,)),
        )
        self.q_proj_weight = dml.Parameter(
            [config.hidden_size, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                f"{prefix}.self_attn.q_proj.weight",
                (config.hidden_size, config.hidden_size),
            ),
        )
        self.q_proj_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.self_attn.q_proj.bias", (config.hidden_size,)),
        )
        self.k_proj_weight = dml.Parameter(
            [config.hidden_size, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                f"{prefix}.self_attn.k_proj.weight",
                (config.hidden_size, config.hidden_size),
            ),
        )
        self.k_proj_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.self_attn.k_proj.bias", (config.hidden_size,)),
        )
        self.v_proj_weight = dml.Parameter(
            [config.hidden_size, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                f"{prefix}.self_attn.v_proj.weight",
                (config.hidden_size, config.hidden_size),
            ),
        )
        self.v_proj_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.self_attn.v_proj.bias", (config.hidden_size,)),
        )
        self.out_proj_weight = dml.Parameter(
            [config.hidden_size, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                f"{prefix}.self_attn.out_proj.weight",
                (config.hidden_size, config.hidden_size),
            ),
        )
        self.out_proj_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.self_attn.out_proj.bias", (config.hidden_size,)),
        )
        self.layer_norm2_weight = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.layer_norm2.weight", (config.hidden_size,)),
        )
        self.layer_norm2_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.layer_norm2.bias", (config.hidden_size,)),
        )
        self.fc1_weight = dml.Parameter(
            [config.intermediate_size, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                f"{prefix}.mlp.fc1.weight",
                (config.intermediate_size, config.hidden_size),
            ),
        )
        self.fc1_bias = dml.Parameter(
            [config.intermediate_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.mlp.fc1.bias", (config.intermediate_size,)),
        )
        self.fc2_weight = dml.Parameter(
            [config.hidden_size, config.intermediate_size],
            dtype="float32",
            value=_weight_value(
                weights,
                f"{prefix}.mlp.fc2.weight",
                (config.hidden_size, config.intermediate_size),
            ),
        )
        self.fc2_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, f"{prefix}.mlp.fc2.bias", (config.hidden_size,)),
        )

    def _attention(self, hidden_states, attention_mask, causal_mask):
        seq_len = _hidden_sequence_length(hidden_states.shape)
        batch = _first_static_dim(hidden_states.shape)
        hidden = self.config.hidden_size
        num_heads = self.config.num_attention_heads
        head_dim = self.config.head_dim

        q = dml.ops.gemm_rcr_bias(hidden_states, self.q_proj_weight, self.q_proj_bias)
        k = dml.ops.gemm_rcr_bias(hidden_states, self.k_proj_weight, self.k_proj_bias)
        v = dml.ops.gemm_rcr_bias(hidden_states, self.v_proj_weight, self.v_proj_bias)

        q = dml.ops.reshape(q, [batch, seq_len, num_heads, head_dim])
        k = dml.ops.reshape(k, [batch, seq_len, num_heads, head_dim])
        v = dml.ops.reshape(v, [batch, seq_len, num_heads, head_dim])

        q = dml.ops.permute0213(q)
        k = dml.ops.permute0213(k)
        v = dml.ops.permute0213(v)

        q = dml.ops.flatten(q, start_dim=0, end_dim=1)
        k = dml.ops.flatten(k, start_dim=0, end_dim=1)
        v = dml.ops.flatten(v, start_dim=0, end_dim=1)

        scores = dml.ops.bmm_rcr(q, k)
        scores = dml.ops.mul(scores, 1.0 / math.sqrt(head_dim))
        scores = dml.ops.add(scores, causal_mask)

        keep = dml.ops.reshape(attention_mask, [batch, 1, 1, seq_len])
        keep = dml.ops.expand(keep, [batch, num_heads, seq_len, seq_len])
        keep = dml.ops.reshape(keep, [batch * num_heads, seq_len, seq_len])
        zeros = dml.ops.full([batch * num_heads, seq_len, seq_len], 0.0, dtype="float32")
        masked = dml.ops.full(
            [batch * num_heads, seq_len, seq_len],
            float(self.config.mask_fill_value),
            dtype="float32",
        )
        scores = dml.ops.add(scores, dml.ops.where(keep, zeros, masked))

        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, v)
        context = dml.ops.reshape(context, [batch, num_heads, seq_len, head_dim])
        context = dml.ops.permute0213(context)
        context = dml.ops.reshape(context, [batch, seq_len, hidden])
        return dml.ops.gemm_rcr_bias(context, self.out_proj_weight, self.out_proj_bias)

    def forward(self, hidden_states, attention_mask, causal_mask):
        residual = hidden_states
        hidden_states = dml.ops.layer_norm(
            hidden_states,
            self.layer_norm1_weight,
            self.layer_norm1_bias,
            eps=self.config.layer_norm_eps,
        )
        hidden_states = self._attention(hidden_states, attention_mask, causal_mask)
        hidden_states = dml.ops.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = dml.ops.layer_norm(
            hidden_states,
            self.layer_norm2_weight,
            self.layer_norm2_bias,
            eps=self.config.layer_norm_eps,
        )
        hidden_states = dml.ops.gemm_rcr_bias_fast_gelu(hidden_states, self.fc1_weight, self.fc1_bias)
        hidden_states = dml.ops.gemm_rcr_bias(hidden_states, self.fc2_weight, self.fc2_bias)
        return dml.ops.add(residual, hidden_states)


class LegacyCLIPTextModelWithProjection(dml.Module):
    def __init__(self, config: LegacyCLIPTextConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.token_embedding_weight = dml.Parameter(
            [config.vocab_size, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                "text_model.embeddings.token_embedding.weight",
                (config.vocab_size, config.hidden_size),
            ),
        )
        self.position_embedding_weight = dml.Parameter(
            [config.max_position_embeddings, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                "text_model.embeddings.position_embedding.weight",
                (config.max_position_embeddings, config.hidden_size),
            ),
        )
        self.layers = [
            _LegacyCLIPTextEncoderLayer(config, weights, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.final_layer_norm_weight = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, "text_model.final_layer_norm.weight", (config.hidden_size,)),
        )
        self.final_layer_norm_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, "text_model.final_layer_norm.bias", (config.hidden_size,)),
        )
        self.text_projection_weight = dml.Parameter(
            [config.projection_dim, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                "text_projection.weight",
                (config.projection_dim, config.hidden_size),
            ),
        )
        self.causal_mask = dml.Parameter(
            [1, config.max_position_embeddings, config.max_position_embeddings],
            dtype="float32",
            value=build_clip_causal_mask(config.max_position_embeddings, config.mask_fill_value),
        )

    def _causal_mask_for_sequence(self, seq_len: int):
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if seq_len > self.config.max_position_embeddings:
            raise ValueError(
                "traced seq_len must be less than or equal to max_position_embeddings"
            )
        return dml.ops.dynamic_slice(
            self.causal_mask,
            start_indices=(0, 0, 0),
            slice_sizes=(1, seq_len, seq_len),
        )

    def _pool_hidden_state(self, input_ids, hidden_states):
        if self.config.eos_token_id == 2:
            indices = dml.ops.argmax(input_ids, dim=-1, keepdim=True)
        else:
            eos_mask = dml.ops.eq(input_ids, self.config.eos_token_id)
            indices = dml.ops.argmax(eos_mask, dim=-1, keepdim=True)
        pooled = dml.ops.batch_gather(hidden_states, indices)
        return dml.ops.squeeze(pooled, 1)

    def encode_text(self, input_ids, attention_mask, position_ids):
        seq_len = _sequence_length(input_ids.shape)
        token_embeddings = dml.ops.embedding(self.token_embedding_weight, input_ids)
        position_embeddings = dml.ops.embedding(self.position_embedding_weight, position_ids)
        hidden_states = dml.ops.add(token_embeddings, position_embeddings)
        causal_mask = self._causal_mask_for_sequence(seq_len)

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, causal_mask)

        last_hidden_state = dml.ops.layer_norm(
            hidden_states,
            self.final_layer_norm_weight,
            self.final_layer_norm_bias,
            eps=self.config.layer_norm_eps,
        )
        pooled_output = self._pool_hidden_state(input_ids, last_hidden_state)
        return dml.ops.gemm_rcr(pooled_output, self.text_projection_weight)

    def forward(self, input_ids, attention_mask, position_ids):
        text_features = self.encode_text(input_ids, attention_mask, position_ids)
        return dml.ops.output(text_features, "text_features")


def _weight_value(weights: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in weights:
        raise KeyError(f"Missing CLIP weight: {name}")
    value = np.asarray(weights[name], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"Weight {name} has shape {value.shape}, expected {shape}")
    return value


def _first_static_dim(shape: list[int]) -> int:
    if not shape:
        raise ValueError("expected rank >= 1 tensor")
    batch = int(shape[0])
    if batch <= 0:
        raise ValueError("expected positive static batch dimension")
    return batch


def _sequence_length(shape: list[int]) -> int:
    if not shape:
        raise ValueError("expected rank >= 1 tensor")
    dim = int(shape[-1])
    if dim <= 0:
        raise ValueError("expected positive static sequence dimension")
    return dim


def _hidden_sequence_length(shape: list[int]) -> int:
    if len(shape) < 2:
        raise ValueError("expected rank >= 2 tensor")
    dim = int(shape[-2])
    if dim <= 0:
        raise ValueError("expected positive static sequence dimension")
    return dim
