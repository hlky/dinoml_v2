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
    landed in DinoML: static traced sequence length, optional default
    `position_ids`, and the two source CLIP EOS pooling branches
    (`eos_token_id == 2` argmax/highest-token-id compatibility, otherwise
    first EOS equality match).
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
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@dataclass(frozen=True)
class LegacyCLIPVisionEmbeddingsConfig:
    """Bounded CLIP vision-embeddings config.

    This helper intentionally models only the admitted CLIP vision embedding
    slice currently landing in DinoML: fixed square NCHW pixel input, patch
    projection, patch flatten/transposition into a sequence, CLS prepend, and
    learned absolute position add. It does not admit interpolation, arbitrary
    image sizes, or the full vision encoder/projection path.
    """

    hidden_size: int
    image_size: int
    patch_size: int
    num_channels: int = 3

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.patch_size > self.image_size:
            raise ValueError("patch_size must be less than or equal to image_size")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_positions(self) -> int:
        return self.num_patches + 1


@dataclass(frozen=True)
class LegacyCLIPVisionConfig:
    """Bounded CLIP vision-wrapper config.

    This helper intentionally models only the admitted CLIP vision-wrapper
    slice currently landed in DinoML: fixed square NCHW pixel input,
    source-faithful embeddings, pre-LayerNorm, either a no-op encoder
    (`num_hidden_layers == 0`) or one-or-more real encoder blocks,
    CLS pool, post-LayerNorm, and bias-free visual projection. It does not
    admit interpolation or arbitrary image sizes.
    """

    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_hidden_layers: int
    projection_dim: int
    image_size: int
    patch_size: int
    num_channels: int = 3
    layer_norm_eps: float = 1.0e-5

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_hidden_layers < 0:
            raise ValueError("num_hidden_layers must be non-negative")
        if self.projection_dim <= 0:
            raise ValueError("projection_dim must be positive")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.num_channels <= 0:
            raise ValueError("num_channels must be positive")
        if self.patch_size > self.image_size:
            raise ValueError("patch_size must be less than or equal to image_size")

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def num_positions(self) -> int:
        return self.num_patches + 1

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
        hidden_states = dml.ops.gemm_rcr_bias_quick_gelu(hidden_states, self.fc1_weight, self.fc1_bias)
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

    def _default_position_ids(self, seq_len: int) -> dml.Parameter:
        return dml.Parameter(np.arange(seq_len, dtype=np.int64), dtype="int64")

    def encode_text(self, input_ids, attention_mask, position_ids=None):
        seq_len = _sequence_length(input_ids.shape)
        if position_ids is None:
            position_ids = self._default_position_ids(seq_len)
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

    def forward(self, input_ids, attention_mask, position_ids=None):
        text_features = self.encode_text(input_ids, attention_mask, position_ids)
        return dml.ops.output(text_features, "text_features")


class LegacyCLIPVisionEmbeddings(dml.Module):
    def __init__(
        self,
        config: LegacyCLIPVisionEmbeddingsConfig | LegacyCLIPVisionConfig,
        weights: Mapping[str, np.ndarray],
    ):
        self.config = config
        self.class_embedding = dml.Parameter(
            [1, 1, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                "vision_model.embeddings.class_embedding",
                (config.hidden_size,),
            ).reshape(1, 1, config.hidden_size),
        )
        self.patch_embedding_weight = dml.Parameter(
            [config.hidden_size, config.num_channels, config.patch_size, config.patch_size],
            dtype="float32",
            value=_weight_value(
                weights,
                "vision_model.embeddings.patch_embedding.weight",
                (config.hidden_size, config.num_channels, config.patch_size, config.patch_size),
            ),
        )
        self.position_embedding_weight = dml.Parameter(
            [config.num_positions, config.hidden_size],
            dtype="float32",
            value=_weight_value(
                weights,
                "vision_model.embeddings.position_embedding.weight",
                (config.num_positions, config.hidden_size),
            ),
        )
        self.position_ids = dml.Parameter(
            np.arange(config.num_positions, dtype=np.int64).reshape(1, config.num_positions),
            dtype="int64",
        )

    def encode_pixels(self, pixel_values):
        batch, channels, height, width = _nchw_image_shape(pixel_values.shape)
        if channels != self.config.num_channels:
            raise ValueError(
                f"expected pixel_values channel dimension {self.config.num_channels}, got {channels}"
            )
        if height != self.config.image_size or width != self.config.image_size:
            raise ValueError(
                f"Input image size ({height}*{width}) doesn't match model "
                f"({self.config.image_size}*{self.config.image_size})."
            )

        patch_embeds = dml.ops.conv2d(
            pixel_values,
            self.patch_embedding_weight,
            stride=(self.config.patch_size, self.config.patch_size),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
        )
        patch_embeds = dml.ops.flatten(patch_embeds, start_dim=2)
        patch_embeds = dml.ops.permute021(patch_embeds)

        class_embeds = dml.ops.expand(self.class_embedding, [batch, 1, self.config.hidden_size])

        embeddings = dml.ops.concatenate([class_embeds, patch_embeds], dim=1)
        position_embeddings = dml.ops.embedding(self.position_embedding_weight, self.position_ids)
        return dml.ops.add(embeddings, position_embeddings)

    def forward(self, pixel_values):
        embeddings = self.encode_pixels(pixel_values)
        return dml.ops.output(embeddings, "embeddings")


class _LegacyCLIPVisionEncoderLayer(dml.Module):
    def __init__(self, config: LegacyCLIPVisionConfig, weights: Mapping[str, np.ndarray], layer_idx: int):
        prefix = f"vision_model.encoder.layers.{layer_idx}"
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

    def _attention(self, hidden_states):
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
        probs = dml.ops.softmax(scores, dim=-1)
        context = dml.ops.bmm_rrr(probs, v)
        context = dml.ops.reshape(context, [batch, num_heads, seq_len, head_dim])
        context = dml.ops.permute0213(context)
        context = dml.ops.reshape(context, [batch, seq_len, hidden])
        return dml.ops.gemm_rcr_bias(context, self.out_proj_weight, self.out_proj_bias)

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = dml.ops.layer_norm(
            hidden_states,
            self.layer_norm1_weight,
            self.layer_norm1_bias,
            eps=self.config.layer_norm_eps,
        )
        hidden_states = self._attention(hidden_states)
        hidden_states = dml.ops.add(residual, hidden_states)

        residual = hidden_states
        hidden_states = dml.ops.layer_norm(
            hidden_states,
            self.layer_norm2_weight,
            self.layer_norm2_bias,
            eps=self.config.layer_norm_eps,
        )
        hidden_states = dml.ops.gemm_rcr_bias_quick_gelu(hidden_states, self.fc1_weight, self.fc1_bias)
        hidden_states = dml.ops.gemm_rcr_bias(hidden_states, self.fc2_weight, self.fc2_bias)
        return dml.ops.add(residual, hidden_states)


class LegacyCLIPVisionModelWithProjection(dml.Module):
    def __init__(self, config: LegacyCLIPVisionConfig, weights: Mapping[str, np.ndarray]):
        self.config = config
        self.embeddings = LegacyCLIPVisionEmbeddings(config, weights)
        self.layers = [
            _LegacyCLIPVisionEncoderLayer(config, weights, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.pre_layrnorm_weight = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, "vision_model.pre_layrnorm.weight", (config.hidden_size,)),
        )
        self.pre_layrnorm_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, "vision_model.pre_layrnorm.bias", (config.hidden_size,)),
        )
        self.post_layernorm_weight = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, "vision_model.post_layernorm.weight", (config.hidden_size,)),
        )
        self.post_layernorm_bias = dml.Parameter(
            [config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, "vision_model.post_layernorm.bias", (config.hidden_size,)),
        )
        self.visual_projection_weight = dml.Parameter(
            [config.projection_dim, config.hidden_size],
            dtype="float32",
            value=_weight_value(weights, "visual_projection.weight", (config.projection_dim, config.hidden_size)),
        )

    def encode_vision(self, pixel_values):
        hidden_states = self.embeddings.encode_pixels(pixel_values)
        hidden_states = dml.ops.layer_norm(
            hidden_states,
            self.pre_layrnorm_weight,
            self.pre_layrnorm_bias,
            eps=self.config.layer_norm_eps,
        )
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        batch = _first_static_dim(hidden_states.shape)
        pooled_output = dml.ops.dynamic_slice(
            hidden_states,
            start_indices=(0, 0, 0),
            slice_sizes=(batch, 1, self.config.hidden_size),
        )
        pooled_output = dml.ops.squeeze(pooled_output, 1)
        pooled_output = dml.ops.layer_norm(
            pooled_output,
            self.post_layernorm_weight,
            self.post_layernorm_bias,
            eps=self.config.layer_norm_eps,
        )
        image_features = dml.ops.gemm_rcr(pooled_output, self.visual_projection_weight)
        return hidden_states, pooled_output, image_features

    def forward(self, pixel_values):
        last_hidden_state, pooler_output, image_features = self.encode_vision(pixel_values)
        return (
            dml.ops.output(last_hidden_state, "last_hidden_state"),
            dml.ops.output(pooler_output, "pooler_output"),
            dml.ops.output(image_features, "image_features"),
        )


class LegacyCLIPModel(dml.Module):
    """Bounded CLIP two-tower wrapper composed from the admitted tower slices."""

    def __init__(
        self,
        text_config: LegacyCLIPTextConfig,
        vision_config: LegacyCLIPVisionConfig,
        weights: Mapping[str, np.ndarray],
    ):
        if text_config.projection_dim != vision_config.projection_dim:
            raise ValueError("text and vision projection_dim must match")
        self.text_model = LegacyCLIPTextModelWithProjection(text_config, weights)
        self.vision_model = LegacyCLIPVisionModelWithProjection(vision_config, weights)
        self.logit_scale = dml.Parameter(
            [1],
            dtype="float32",
            value=np.asarray([_scalar_weight_value(weights, "logit_scale")], dtype=np.float32),
        )

    def get_text_features(self, input_ids, attention_mask, position_ids=None):
        return self.text_model.encode_text(input_ids, attention_mask, position_ids)

    def get_image_features(self, pixel_values):
        _, _, image_features = self.vision_model.encode_vision(pixel_values)
        return image_features

    def _normalize_features(self, features):
        return dml.ops.div(features, dml.ops.vector_norm(features, dim=-1, keepdim=True))

    def forward(self, input_ids, pixel_values, attention_mask, position_ids=None):
        text_features = self.get_text_features(input_ids, attention_mask, position_ids)
        image_features = self.get_image_features(pixel_values)
        text_embeds = self._normalize_features(text_features)
        image_embeds = self._normalize_features(image_features)
        logits_per_text = dml.ops.gemm_rcr(text_embeds, image_embeds)
        logits_per_text = dml.ops.mul(logits_per_text, dml.ops.exp(self.logit_scale))
        logits_per_image = dml.ops.transpose(logits_per_text, 0, 1)
        return (
            dml.ops.output(logits_per_image, "logits_per_image"),
            dml.ops.output(logits_per_text, "logits_per_text"),
            dml.ops.output(text_embeds, "text_embeds"),
            dml.ops.output(image_embeds, "image_embeds"),
        )


def legacy_clip_configs_from_transformers_clip_config(
    clip_config: object,
) -> tuple[LegacyCLIPTextConfig, LegacyCLIPVisionConfig]:
    """Derive bounded LegacyCLIP configs from a Transformers CLIPConfig.

    This adapter intentionally admits only the already-supported inference
    surface: dense CLIP text/vision towers with `hidden_act="quick_gelu"` and
    the existing DinoML `LegacyCLIPModel` weight namespace.
    """

    text_config = getattr(clip_config, "text_config", None)
    vision_config = getattr(clip_config, "vision_config", None)
    projection_dim = getattr(clip_config, "projection_dim", None)
    if text_config is None or vision_config is None or projection_dim is None:
        raise TypeError("expected a Transformers CLIPConfig-like object with text_config, vision_config, and projection_dim")
    _validate_transformers_clip_hidden_act(text_config, tower="text")
    _validate_transformers_clip_hidden_act(vision_config, tower="vision")
    return (
        LegacyCLIPTextConfig(
            vocab_size=int(text_config.vocab_size),
            max_position_embeddings=int(text_config.max_position_embeddings),
            hidden_size=int(text_config.hidden_size),
            intermediate_size=int(text_config.intermediate_size),
            num_attention_heads=int(text_config.num_attention_heads),
            num_hidden_layers=int(text_config.num_hidden_layers),
            projection_dim=int(projection_dim),
            layer_norm_eps=float(text_config.layer_norm_eps),
            eos_token_id=int(text_config.eos_token_id),
        ),
        LegacyCLIPVisionConfig(
            hidden_size=int(vision_config.hidden_size),
            intermediate_size=int(vision_config.intermediate_size),
            num_attention_heads=int(vision_config.num_attention_heads),
            num_hidden_layers=int(vision_config.num_hidden_layers),
            projection_dim=int(projection_dim),
            image_size=int(vision_config.image_size),
            patch_size=int(vision_config.patch_size),
            num_channels=int(vision_config.num_channels),
            layer_norm_eps=float(vision_config.layer_norm_eps),
        ),
    )


def legacy_clip_weights_from_transformers_state_dict(
    state_dict: Mapping[str, object],
    text_config: LegacyCLIPTextConfig,
    vision_config: LegacyCLIPVisionConfig,
) -> dict[str, np.ndarray]:
    """Convert a Transformers CLIPModel state dict into LegacyCLIP weights."""

    required = _legacy_clip_required_weight_names(text_config, vision_config)
    missing = [name for name in required if name not in state_dict]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} missing total)"
        raise KeyError(f"Missing Transformers CLIP state_dict weights: {preview}{suffix}")
    return {name: _transformers_state_value_to_numpy(state_dict[name], name) for name in required}


def legacy_clip_model_from_transformers_clip_model(clip_model: object) -> LegacyCLIPModel:
    """Build a bounded DinoML LegacyCLIPModel from a local Transformers CLIPModel.

    The adapter is intentionally inference-only: it transfers config and
    checkpoint weights for the existing LegacyCLIP text/vision towers and
    contrastive head, but it does not add tokenizer/processor plumbing, loss,
    position interpolation, FlashAttention dispatch, or other broader CLIP
    surfaces beyond the current LegacyCLIPModel contract.
    """

    clip_config = getattr(clip_model, "config", None)
    state_dict_fn = getattr(clip_model, "state_dict", None)
    if clip_config is None or state_dict_fn is None:
        raise TypeError("expected a Transformers CLIPModel-like object with config and state_dict()")
    text_config, vision_config = legacy_clip_configs_from_transformers_clip_config(clip_config)
    weights = legacy_clip_weights_from_transformers_state_dict(
        state_dict_fn(),
        text_config,
        vision_config,
    )
    return LegacyCLIPModel(text_config, vision_config, weights)


def _weight_value(weights: Mapping[str, np.ndarray], name: str, shape: tuple[int, ...]) -> np.ndarray:
    if name not in weights:
        raise KeyError(f"Missing CLIP weight: {name}")
    value = np.asarray(weights[name], dtype=np.float32)
    if value.shape != shape:
        raise ValueError(f"Weight {name} has shape {value.shape}, expected {shape}")
    return value


def _scalar_weight_value(weights: Mapping[str, np.ndarray], name: str) -> np.float32:
    if name not in weights:
        raise KeyError(f"Missing CLIP weight: {name}")
    value = np.asarray(weights[name], dtype=np.float32)
    if value.shape not in {(), (1,)}:
        raise ValueError(f"Weight {name} has shape {value.shape}, expected scalar or [1]")
    return np.float32(value.reshape(-1)[0] if value.shape == (1,) else value)


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


def _nchw_image_shape(shape: list[int]) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError("expected rank-4 NCHW image tensor")
    batch, channels, height, width = (int(dim) for dim in shape)
    if batch <= 0:
        raise ValueError("expected positive static batch dimension")
    if channels <= 0 or height <= 0 or width <= 0:
        raise ValueError("expected positive static NCHW image dimensions")
    return batch, channels, height, width


def _validate_transformers_clip_hidden_act(config: object, *, tower: str) -> None:
    hidden_act = getattr(config, "hidden_act", None)
    if hidden_act != "quick_gelu":
        raise ValueError(
            f"LegacyCLIP {tower} adapter only supports Transformers hidden_act='quick_gelu', got {hidden_act!r}"
        )


def _legacy_clip_required_weight_names(
    text_config: LegacyCLIPTextConfig,
    vision_config: LegacyCLIPVisionConfig,
) -> list[str]:
    names = [
        "text_model.embeddings.token_embedding.weight",
        "text_model.embeddings.position_embedding.weight",
        "text_model.final_layer_norm.weight",
        "text_model.final_layer_norm.bias",
        "text_projection.weight",
        "vision_model.embeddings.class_embedding",
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.position_embedding.weight",
        "vision_model.pre_layrnorm.weight",
        "vision_model.pre_layrnorm.bias",
        "vision_model.post_layernorm.weight",
        "vision_model.post_layernorm.bias",
        "visual_projection.weight",
        "logit_scale",
    ]
    for layer_idx in range(text_config.num_hidden_layers):
        prefix = f"text_model.encoder.layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.layer_norm1.weight",
                f"{prefix}.layer_norm1.bias",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.out_proj.weight",
                f"{prefix}.self_attn.out_proj.bias",
                f"{prefix}.layer_norm2.weight",
                f"{prefix}.layer_norm2.bias",
                f"{prefix}.mlp.fc1.weight",
                f"{prefix}.mlp.fc1.bias",
                f"{prefix}.mlp.fc2.weight",
                f"{prefix}.mlp.fc2.bias",
            ]
        )
    for layer_idx in range(vision_config.num_hidden_layers):
        prefix = f"vision_model.encoder.layers.{layer_idx}"
        names.extend(
            [
                f"{prefix}.layer_norm1.weight",
                f"{prefix}.layer_norm1.bias",
                f"{prefix}.self_attn.q_proj.weight",
                f"{prefix}.self_attn.q_proj.bias",
                f"{prefix}.self_attn.k_proj.weight",
                f"{prefix}.self_attn.k_proj.bias",
                f"{prefix}.self_attn.v_proj.weight",
                f"{prefix}.self_attn.v_proj.bias",
                f"{prefix}.self_attn.out_proj.weight",
                f"{prefix}.self_attn.out_proj.bias",
                f"{prefix}.layer_norm2.weight",
                f"{prefix}.layer_norm2.bias",
                f"{prefix}.mlp.fc1.weight",
                f"{prefix}.mlp.fc1.bias",
                f"{prefix}.mlp.fc2.weight",
                f"{prefix}.mlp.fc2.bias",
            ]
        )
    return names


def _transformers_state_value_to_numpy(value: object, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.size == 0:
        raise ValueError(f"Transformers CLIP weight {name} is empty")
    return array
