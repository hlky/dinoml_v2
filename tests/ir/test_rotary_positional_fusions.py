from __future__ import annotations

from collections import Counter

import numpy as np

import dinoml as dml
from dinoml.reference import reference_numpy


def _get_1d_rotary_oracle(
    dim: int,
    pos: np.ndarray | int,
    *,
    theta: float = 10000.0,
    use_real: bool = True,
    linear_factor: float = 1.0,
    ntk_factor: float = 1.0,
    repeat_interleave_real: bool = True,
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    if isinstance(pos, int):
        positions = np.arange(pos, dtype=np.float32)
    else:
        positions = np.asarray(pos, dtype=np.float32)
    rotary_dim = dim // 2
    exponent = -np.log(np.float32(theta * ntk_factor)) * np.arange(rotary_dim, dtype=np.float32) / np.float32(rotary_dim)
    freqs = positions[:, None] * (np.exp(exponent).astype(np.float32, copy=False) / np.float32(linear_factor))[None, :]
    if not use_real:
        return np.cos(freqs).astype(np.float32, copy=False), np.sin(freqs).astype(np.float32, copy=False)
    base_cos = np.cos(freqs).astype(np.float32, copy=False)
    base_sin = np.sin(freqs).astype(np.float32, copy=False)
    if repeat_interleave_real:
        return np.repeat(base_cos, 2, axis=1), np.repeat(base_sin, 2, axis=1)
    return np.concatenate([base_cos, base_cos], axis=1), np.concatenate([base_sin, base_sin], axis=1)


def _get_2d_rotary_pos_embed_oracle(embed_dim: int, crops_coords, grid_size) -> tuple[np.ndarray, np.ndarray]:
    start, stop = crops_coords
    grid_h = np.linspace(start[0], stop[0] * (grid_size[0] - 1) / grid_size[0], grid_size[0], dtype=np.float32)
    grid_w = np.linspace(start[1], stop[1] * (grid_size[1] - 1) / grid_size[1], grid_size[1], dtype=np.float32)
    grid_0, grid_1 = np.meshgrid(grid_w, grid_h, indexing="xy")
    cos_h, sin_h = _get_1d_rotary_oracle(embed_dim // 2, grid_0.reshape(-1), use_real=True)
    cos_w, sin_w = _get_1d_rotary_oracle(embed_dim // 2, grid_1.reshape(-1), use_real=True)
    return np.concatenate([cos_h, cos_w], axis=1), np.concatenate([sin_h, sin_w], axis=1)


def _get_2d_rotary_pos_embed_lumina_oracle(
    embed_dim: int,
    len_h: int,
    len_w: int,
    *,
    linear_factor: float,
    ntk_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    real_h, imag_h = _get_1d_rotary_oracle(
        embed_dim // 2,
        len_h,
        use_real=False,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
    )
    real_w, imag_w = _get_1d_rotary_oracle(
        embed_dim // 2,
        len_w,
        use_real=False,
        linear_factor=linear_factor,
        ntk_factor=ntk_factor,
    )
    quarter_dim = embed_dim // 4
    real = np.empty((len_h, len_w, embed_dim // 2), dtype=np.float32)
    imag = np.empty_like(real)
    for h_idx in range(len_h):
        for w_idx in range(len_w):
            for dim_idx in range(quarter_dim):
                base = 2 * dim_idx
                real[h_idx, w_idx, base] = real_h[h_idx, dim_idx]
                imag[h_idx, w_idx, base] = imag_h[h_idx, dim_idx]
                real[h_idx, w_idx, base + 1] = real_w[w_idx, dim_idx]
                imag[h_idx, w_idx, base + 1] = imag_w[w_idx, dim_idx]
    return real, imag


def _get_3d_rotary_pos_embed_oracle(
    embed_dim: int,
    crops_coords,
    grid_size,
    temporal_size: int,
    *,
    theta: float = 10000.0,
    grid_type: str = "linspace",
    max_size=None,
) -> tuple[np.ndarray, np.ndarray]:
    if grid_type == "linspace":
        start, stop = crops_coords
        grid_h = np.linspace(start[0], stop[0] * (grid_size[0] - 1) / grid_size[0], grid_size[0], dtype=np.float32)
        grid_w = np.linspace(start[1], stop[1] * (grid_size[1] - 1) / grid_size[1], grid_size[1], dtype=np.float32)
        grid_t = np.linspace(0.0, temporal_size * (temporal_size - 1) / temporal_size, temporal_size, dtype=np.float32)
    else:
        max_h, max_w = max_size
        grid_h = np.arange(max_h, dtype=np.float32)
        grid_w = np.arange(max_w, dtype=np.float32)
        grid_t = np.arange(temporal_size, dtype=np.float32)
    dim_t = embed_dim // 4
    dim_h = (embed_dim // 8) * 3
    dim_w = (embed_dim // 8) * 3
    t_cos, t_sin = _get_1d_rotary_oracle(dim_t, grid_t, theta=theta, use_real=True)
    h_cos, h_sin = _get_1d_rotary_oracle(dim_h, grid_h, theta=theta, use_real=True)
    w_cos, w_sin = _get_1d_rotary_oracle(dim_w, grid_w, theta=theta, use_real=True)
    if grid_type == "slice":
        h_cos = h_cos[: grid_size[0]]
        h_sin = h_sin[: grid_size[0]]
        w_cos = w_cos[: grid_size[1]]
        w_sin = w_sin[: grid_size[1]]

    def combine(freq_t, freq_h, freq_w):
        t = np.broadcast_to(freq_t[:, None, None, :], (temporal_size, grid_size[0], grid_size[1], freq_t.shape[1]))
        h = np.broadcast_to(freq_h[None, :, None, :], (temporal_size, grid_size[0], grid_size[1], freq_h.shape[1]))
        w = np.broadcast_to(freq_w[None, None, :, :], (temporal_size, grid_size[0], grid_size[1], freq_w.shape[1]))
        return np.concatenate([t, h, w], axis=-1).reshape(temporal_size * grid_size[0] * grid_size[1], -1)

    return combine(t_cos, h_cos, w_cos), combine(t_sin, h_sin, w_sin)


def _get_3d_rotary_pos_embed_allegro_oracle(
    height: int,
    width: int,
    num_frames: int,
    *,
    vae_scale_factor_spatial: int,
    patch_size: int,
    interpolation_scale_h: float,
    interpolation_scale_t: float,
    interpolation_scale_w: float,
    attention_head_dim: int,
) -> tuple[np.ndarray, ...]:
    grid_h = height // (vae_scale_factor_spatial * patch_size)
    grid_w = width // (vae_scale_factor_spatial * patch_size)
    dim_axis = attention_head_dim // 3
    t_cos, t_sin = _get_1d_rotary_oracle(
        dim_axis,
        np.linspace(0.0, num_frames * (num_frames - 1) / num_frames, num_frames, dtype=np.float32) / np.float32(interpolation_scale_t),
        use_real=True,
        repeat_interleave_real=False,
    )
    h_cos, h_sin = _get_1d_rotary_oracle(
        dim_axis,
        np.linspace(0.0, grid_h - 1, grid_h, dtype=np.float32) / np.float32(interpolation_scale_h),
        use_real=True,
        repeat_interleave_real=False,
    )
    w_cos, w_sin = _get_1d_rotary_oracle(
        dim_axis,
        np.linspace(0.0, grid_w - 1, grid_w, dtype=np.float32) / np.float32(interpolation_scale_w),
        use_real=True,
        repeat_interleave_real=False,
    )
    grid_t_vals = np.arange(num_frames, dtype=np.int64)
    grid_h_vals = np.arange(grid_h, dtype=np.int64)
    grid_w_vals = np.arange(grid_w, dtype=np.int64)
    grid_t, grid_h_arr, grid_w_arr = np.meshgrid(grid_t_vals, grid_h_vals, grid_w_vals, indexing="ij")
    return (
        t_cos,
        t_sin,
        h_cos,
        h_sin,
        w_cos,
        w_sin,
        grid_t.reshape(1, -1),
        grid_h_arr.reshape(1, -1),
        grid_w_arr.reshape(1, -1),
    )


def test_get_2d_rotary_pos_embed_reference_matches_oracle():
    class Rotary2dModule(dml.Module):
        def forward(self):
            cos, sin = dml.ops.get_2d_rotary_pos_embed(64, ((0.0, 0.0), (1.5, 2.25)), (4, 5))
            return {"cos": dml.ops.output(cos, "cos"), "sin": dml.ops.output(sin, "sin")}

    spec = dml.trace(Rotary2dModule(), inputs={}, name="rotary_2d_fusion")
    actual = reference_numpy(spec, {})
    expected_cos, expected_sin = _get_2d_rotary_pos_embed_oracle(64, ((0.0, 0.0), (1.5, 2.25)), (4, 5))
    assert Counter(node["op"] for node in spec.ir["nodes"])["get_2d_rotary_pos_embed"] == 1
    np.testing.assert_allclose(actual["cos"], expected_cos, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual["sin"], expected_sin, rtol=1e-5, atol=1e-5)


def test_get_2d_rotary_pos_embed_lumina_reference_matches_oracle():
    class Rotary2dLuminaModule(dml.Module):
        def forward(self):
            real, imag = dml.ops.get_2d_rotary_pos_embed_lumina(64, 4, 5, linear_factor=1.25, ntk_factor=1.5)
            return {"real": dml.ops.output(real, "real"), "imag": dml.ops.output(imag, "imag")}

    spec = dml.trace(Rotary2dLuminaModule(), inputs={}, name="rotary_2d_lumina_fusion")
    actual = reference_numpy(spec, {})
    expected_real, expected_imag = _get_2d_rotary_pos_embed_lumina_oracle(64, 4, 5, linear_factor=1.25, ntk_factor=1.5)
    assert Counter(node["op"] for node in spec.ir["nodes"])["get_2d_rotary_pos_embed_lumina"] == 1
    np.testing.assert_allclose(actual["real"], expected_real, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual["imag"], expected_imag, rtol=1e-5, atol=1e-5)


def test_get_3d_rotary_pos_embed_reference_matches_linspace_oracle():
    class Rotary3dLinspaceModule(dml.Module):
        def forward(self):
            cos, sin = dml.ops.get_3d_rotary_pos_embed(64, ((0.0, 0.0), (1.25, 2.0)), (3, 4), 2, grid_type="linspace")
            return {"cos": dml.ops.output(cos, "cos"), "sin": dml.ops.output(sin, "sin")}

    spec = dml.trace(Rotary3dLinspaceModule(), inputs={}, name="rotary_3d_linspace_fusion")
    actual = reference_numpy(spec, {})
    expected_cos, expected_sin = _get_3d_rotary_pos_embed_oracle(64, ((0.0, 0.0), (1.25, 2.0)), (3, 4), 2, grid_type="linspace")
    assert Counter(node["op"] for node in spec.ir["nodes"])["get_3d_rotary_pos_embed"] == 1
    np.testing.assert_allclose(actual["cos"], expected_cos, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual["sin"], expected_sin, rtol=1e-5, atol=1e-5)


def test_get_3d_rotary_pos_embed_reference_matches_slice_oracle():
    class Rotary3dSliceModule(dml.Module):
        def forward(self):
            cos, sin = dml.ops.get_3d_rotary_pos_embed(
                64,
                ((0.0, 0.0), (1.25, 2.0)),
                (3, 4),
                2,
                grid_type="slice",
                max_size=(6, 7),
            )
            return {"cos": dml.ops.output(cos, "cos"), "sin": dml.ops.output(sin, "sin")}

    spec = dml.trace(Rotary3dSliceModule(), inputs={}, name="rotary_3d_slice_fusion")
    actual = reference_numpy(spec, {})
    expected_cos, expected_sin = _get_3d_rotary_pos_embed_oracle(
        64,
        ((0.0, 0.0), (1.25, 2.0)),
        (3, 4),
        2,
        grid_type="slice",
        max_size=(6, 7),
    )
    assert Counter(node["op"] for node in spec.ir["nodes"])["get_3d_rotary_pos_embed"] == 1
    np.testing.assert_allclose(actual["cos"], expected_cos, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual["sin"], expected_sin, rtol=1e-5, atol=1e-5)


def test_get_3d_rotary_pos_embed_allegro_reference_matches_oracle():
    class Rotary3dAllegroModule(dml.Module):
        def forward(self):
            (freqs, grids) = dml.ops.get_3d_rotary_pos_embed_allegro(
                96,
                128,
                3,
                vae_scale_factor_spatial=8,
                patch_size=2,
                interpolation_scale_h=2.0,
                interpolation_scale_t=2.2,
                interpolation_scale_w=2.0,
                attention_head_dim=96,
            )
            (t_freqs, h_freqs, w_freqs) = freqs
            t_cos, t_sin = t_freqs
            h_cos, h_sin = h_freqs
            w_cos, w_sin = w_freqs
            grid_t, grid_h, grid_w = grids
            return {
                "t_cos": dml.ops.output(t_cos, "t_cos"),
                "t_sin": dml.ops.output(t_sin, "t_sin"),
                "h_cos": dml.ops.output(h_cos, "h_cos"),
                "h_sin": dml.ops.output(h_sin, "h_sin"),
                "w_cos": dml.ops.output(w_cos, "w_cos"),
                "w_sin": dml.ops.output(w_sin, "w_sin"),
                "grid_t": dml.ops.output(grid_t, "grid_t"),
                "grid_h": dml.ops.output(grid_h, "grid_h"),
                "grid_w": dml.ops.output(grid_w, "grid_w"),
            }

    spec = dml.trace(Rotary3dAllegroModule(), inputs={}, name="rotary_3d_allegro_fusion")
    actual = reference_numpy(spec, {})
    expected = _get_3d_rotary_pos_embed_allegro_oracle(
        96,
        128,
        3,
        vae_scale_factor_spatial=8,
        patch_size=2,
        interpolation_scale_h=2.0,
        interpolation_scale_t=2.2,
        interpolation_scale_w=2.0,
        attention_head_dim=96,
    )
    assert Counter(node["op"] for node in spec.ir["nodes"])["get_3d_rotary_pos_embed_allegro"] == 1
    for name, expected_value in zip(
        ("t_cos", "t_sin", "h_cos", "h_sin", "w_cos", "w_sin", "grid_t", "grid_h", "grid_w"),
        expected,
    ):
        np.testing.assert_allclose(actual[name], expected_value, rtol=1e-5, atol=1e-5)
