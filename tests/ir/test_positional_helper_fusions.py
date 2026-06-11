from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy


def _cropped_pos_embed_oracle(
    *,
    embed_dim: int,
    pos_embed_max_size: int,
    base_size: int,
    interpolation_scale: float,
    patch_size: int,
    height: int,
    width: int,
) -> np.ndarray:
    crop_h = height // patch_size
    crop_w = width // patch_size
    top = (pos_embed_max_size - crop_h) // 2
    left = (pos_embed_max_size - crop_w) // 2
    grid_h = np.arange(pos_embed_max_size, dtype=np.float32) / np.float32(pos_embed_max_size / base_size) / np.float32(
        interpolation_scale
    )
    grid_w = np.arange(pos_embed_max_size, dtype=np.float32) / np.float32(pos_embed_max_size / base_size) / np.float32(
        interpolation_scale
    )
    grid_0, grid_1 = np.meshgrid(grid_w, grid_h, indexing="xy")
    pair_dim = embed_dim // 4
    omega = np.arange(pair_dim, dtype=np.float32) / np.float32(pair_dim)
    omega = np.float32(1.0) / np.power(np.float32(10000.0), omega, dtype=np.float32)
    out_0 = grid_0.reshape(-1, 1) * omega.reshape(1, -1)
    out_1 = grid_1.reshape(-1, 1) * omega.reshape(1, -1)
    pos_embed = np.concatenate(
        [
            np.sin(out_0).astype(np.float32, copy=False),
            np.cos(out_0).astype(np.float32, copy=False),
            np.sin(out_1).astype(np.float32, copy=False),
            np.cos(out_1).astype(np.float32, copy=False),
        ],
        axis=1,
    )
    spatial = pos_embed.reshape(1, pos_embed_max_size, pos_embed_max_size, embed_dim)
    return spatial[:, top : top + crop_h, left : left + crop_w, :].reshape(1, crop_h * crop_w, embed_dim)


def _gaussian_fourier_projection_oracle(
    x: np.ndarray,
    weight: np.ndarray,
    *,
    log: bool,
    flip_sin_to_cos: bool,
) -> np.ndarray:
    x_value = np.asarray(x, dtype=np.float32)
    weight_value = np.asarray(weight, dtype=np.float32)
    if log:
        x_value = np.log(x_value)
    x_proj = x_value[:, None] * weight_value[None, :] * np.float32(2.0 * np.pi)
    if flip_sin_to_cos:
        return np.concatenate([np.cos(x_proj), np.sin(x_proj)], axis=1).astype(np.float32, copy=False)
    return np.concatenate([np.sin(x_proj), np.cos(x_proj)], axis=1).astype(np.float32, copy=False)


def _get_fourier_embeds_from_boundingbox_oracle(embed_dim: int, box: np.ndarray) -> np.ndarray:
    box_value = np.asarray(box, dtype=np.float32)
    emb = np.power(np.float32(100.0), np.arange(embed_dim, dtype=np.float32) / np.float32(embed_dim))
    projected = box_value[..., None] * emb.reshape(1, 1, 1, embed_dim)
    stacked = np.stack((np.sin(projected), np.cos(projected)), axis=-1)
    return np.transpose(stacked, (0, 1, 3, 4, 2)).reshape(box.shape[0], box.shape[1], embed_dim * 8).astype(
        np.float32,
        copy=False,
    )


def _relative_attention_bucket(
    relative_position: int,
    *,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> int:
    bucket = 0
    n = num_buckets
    rel = int(relative_position)
    if bidirectional:
        n //= 2
        if rel > 0:
            bucket += n
        rel = abs(rel)
    else:
        rel = -min(rel, 0)
    max_exact = n // 2
    if rel < max_exact:
        return bucket + rel
    scaled = max_exact + int(
        np.log(np.float32(rel) / np.float32(max_exact))
        / np.log(np.float32(max_distance) / np.float32(max_exact))
        * np.float32(n - max_exact)
    )
    return bucket + min(scaled, n - 1)


def _relative_attention_bias_oracle(
    embedding: np.ndarray,
    *,
    query_length: int,
    key_length: int,
    bidirectional: bool,
    num_buckets: int,
    max_distance: int,
) -> np.ndarray:
    embedding_value = np.asarray(embedding, dtype=np.float32)
    _, heads = embedding_value.shape
    result = np.empty((1, heads, query_length, key_length), dtype=np.float32)
    for query_idx in range(query_length):
        for key_idx in range(key_length):
            bucket = _relative_attention_bucket(
                key_idx - query_idx,
                bidirectional=bidirectional,
                num_buckets=num_buckets,
                max_distance=max_distance,
            )
            result[0, :, query_idx, key_idx] = embedding_value[bucket]
    return result


def _sinusoidal_positional_embedding_oracle(x: np.ndarray, *, max_seq_len: int) -> np.ndarray:
    x_value = np.asarray(x, dtype=np.float32)
    _, seq_len, embed_dim = x_value.shape
    position = np.arange(max_seq_len, dtype=np.float32).reshape(max_seq_len, 1)
    div_term = np.exp(
        np.arange(0, embed_dim, 2, dtype=np.float32) * (-np.log(np.float32(10000.0)) / np.float32(embed_dim))
    ).astype(np.float32, copy=False)
    pe = np.zeros((1, max_seq_len, embed_dim), dtype=np.float32)
    pe[:, :, 0::2] = np.sin(position * div_term).reshape(1, max_seq_len, -1)
    pe[:, :, 1::2] = np.cos(position * div_term[: pe[:, :, 1::2].shape[2]]).reshape(1, max_seq_len, -1)
    return (x_value + pe[:, :seq_len, :]).astype(np.float32, copy=False)


def test_cropped_pos_embed_reference_matches_oracle():
    class CroppedModule(dml.Module):
        def forward(self):
            return dml.ops.output(
                dml.ops.cropped_pos_embed(
                    embed_dim=16,
                    pos_embed_max_size=8,
                    base_size=4,
                    interpolation_scale=1.0,
                    patch_size=2,
                    height=8,
                    width=12,
                ),
                "out",
            )

    spec = dml.trace(CroppedModule(), inputs={}, name="cropped_pos_embed_fusion")
    actual = reference_numpy(spec, {})["out"]
    expected = _cropped_pos_embed_oracle(
        embed_dim=16,
        pos_embed_max_size=8,
        base_size=4,
        interpolation_scale=1.0,
        patch_size=2,
        height=8,
        width=12,
    )
    assert Counter(node["op"] for node in spec.ir["nodes"])["cropped_pos_embed"] == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_gaussian_fourier_projection_reference_matches_oracle():
    class GaussianModule(dml.Module):
        def forward(self, x, weight):
            return dml.ops.output(
                dml.ops.gaussian_fourier_projection(x, weight, log=True, flip_sin_to_cos=False),
                "out",
            )

    x = np.array([0.125, 0.5, 1.25, 3.0], dtype=np.float32)
    weight = np.linspace(-0.75, 0.5, num=6, dtype=np.float32)
    spec = dml.trace(
        GaussianModule(),
        inputs={"x": dml.TensorSpec([4], "float32"), "weight": dml.TensorSpec([6], "float32")},
        name="gaussian_fourier_projection_fusion",
    )
    actual = reference_numpy(spec, {"x": x, "weight": weight})["out"]
    expected = _gaussian_fourier_projection_oracle(x, weight, log=True, flip_sin_to_cos=False)
    assert Counter(node["op"] for node in spec.ir["nodes"])["gaussian_fourier_projection"] == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_get_fourier_embeds_from_boundingbox_reference_matches_oracle():
    class BoxModule(dml.Module):
        def forward(self, box):
            return dml.ops.output(dml.ops.get_fourier_embeds_from_boundingbox(4, box), "out")

    box = np.linspace(-0.5, 0.75, num=24, dtype=np.float32).reshape(2, 3, 4)
    spec = dml.trace(
        BoxModule(),
        inputs={"box": dml.TensorSpec([2, 3, 4], "float32")},
        name="get_fourier_embeds_from_boundingbox_fusion",
    )
    actual = reference_numpy(spec, {"box": box})["out"]
    expected = _get_fourier_embeds_from_boundingbox_oracle(4, box)
    assert Counter(node["op"] for node in spec.ir["nodes"])["get_fourier_embeds_from_boundingbox"] == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_relative_attention_bias_reference_matches_oracle():
    class RelativeBiasModule(dml.Module):
        def forward(self, embedding):
            return dml.ops.output(
                dml.ops.relative_attention_bias(
                    embedding,
                    3,
                    5,
                    bidirectional=True,
                    num_buckets=16,
                    max_distance=32,
                ),
                "out",
            )

    embedding = np.linspace(-1.0, 1.0, num=32, dtype=np.float32).reshape(16, 2)
    spec = dml.trace(
        RelativeBiasModule(),
        inputs={"embedding": dml.TensorSpec([16, 2], "float32")},
        name="relative_attention_bias_fusion",
    )
    actual = reference_numpy(spec, {"embedding": embedding})["out"]
    expected = _relative_attention_bias_oracle(
        embedding,
        query_length=3,
        key_length=5,
        bidirectional=True,
        num_buckets=16,
        max_distance=32,
    )
    assert Counter(node["op"] for node in spec.ir["nodes"])["relative_attention_bias"] == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_sinusoidal_positional_embedding_reference_matches_oracle():
    class SinusoidalModule(dml.Module):
        def forward(self, x):
            return dml.ops.output(dml.ops.sinusoidal_positional_embedding(x, 8, 6), "out")

    x = np.linspace(-1.25, 1.5, num=96, dtype=np.float32).reshape(2, 6, 8)
    spec = dml.trace(
        SinusoidalModule(),
        inputs={"x": dml.TensorSpec([2, 6, 8], "float32")},
        name="sinusoidal_positional_embedding_fusion",
    )
    actual = reference_numpy(spec, {"x": x})["out"]
    expected = _sinusoidal_positional_embedding_oracle(x, max_seq_len=6)
    assert Counter(node["op"] for node in spec.ir["nodes"])["sinusoidal_positional_embedding"] == 1
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("name", "builder", "message"),
    [
        (
            "cropped_bad_embed_dim",
            lambda: dml.ops.cropped_pos_embed(
                embed_dim=10,
                pos_embed_max_size=8,
                base_size=4,
                interpolation_scale=1.0,
                patch_size=2,
                height=8,
                width=12,
            ),
            "divisible by 4",
        ),
        (
            "relative_bad_rank",
            lambda emb: dml.ops.relative_attention_bias(emb, 3, 5),
            "shape \\[num_buckets, heads\\]",
        ),
        (
            "sinusoidal_bad_hidden",
            lambda x: dml.ops.sinusoidal_positional_embedding(x, 6, 6),
            "embed_dim must match x.shape\\[-1\\]",
        ),
    ],
)
def test_positional_helper_fusions_validate_shapes(name, builder, message):
    if name == "cropped_bad_embed_dim":
        class CroppedBad(dml.Module):
            def forward(self):
                return dml.ops.output(builder(), "out")

        with pytest.raises(ValueError, match=message):
            dml.trace(CroppedBad(), inputs={}, name=name)
        return

    if name == "relative_bad_rank":
        class RelativeBad(dml.Module):
            def forward(self, embedding):
                return dml.ops.output(builder(embedding), "out")

        with pytest.raises(ValueError, match=message):
            dml.trace(RelativeBad(), inputs={"embedding": dml.TensorSpec([16], "float32")}, name=name)
        return

    class SinusoidalBad(dml.Module):
        def forward(self, x):
            return dml.ops.output(builder(x), "out")

    with pytest.raises(ValueError, match=message):
        dml.trace(SinusoidalBad(), inputs={"x": dml.TensorSpec([2, 6, 8], "float32")}, name=name)
