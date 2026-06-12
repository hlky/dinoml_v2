from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.lowering.ops import render_generated_kernels
from dinoml.reference import reference_numpy


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def _multi_level_level_index(roi: np.ndarray, *, im_h: int, im_w: int, spatial_scale: float) -> int:
    x1 = np.clip(float(roi[1]) * spatial_scale, 0.0, float(im_w))
    y1 = np.clip(float(roi[2]) * spatial_scale, 0.0, float(im_h))
    x2 = np.clip(float(roi[3]) * spatial_scale, 0.0, float(im_w))
    y2 = np.clip(float(roi[4]) * spatial_scale, 0.0, float(im_h))
    first_threshold = (224.0 * 224.0) / (float(im_h) * float(im_w) * 4.0)
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0) / float(im_h * im_w)
    if area > first_threshold * 16.0:
        return 3
    if area > first_threshold * 4.0:
        return 2
    if area > first_threshold:
        return 1
    return 0


def test_roi_align_reference_matches_torchvision_oracle():
    class Tiny(dml.nn.Module):
        def forward(self, x, rois):
            return dml.ops.output(
                dml.ops.roi_align(
                    x,
                    rois,
                    pooled_size=(2, 3),
                    sampling_ratio=2,
                    spatial_scale=1.0,
                ),
                "y",
            )

    spec = dml.trace(
        Tiny(),
        inputs={
            "x": dml.TensorSpec([2, 3, 8, 8], "float32"),
            "rois": dml.TensorSpec([4, 5], "float32"),
        },
        name="roi_align_torchvision_oracle",
    )
    x = np.arange(2 * 3 * 8 * 8, dtype=np.float32).reshape(2, 3, 8, 8) / np.float32(10.0)
    rois = np.array(
        [
            [0.0, 0.0, 0.0, 7.0, 7.0],
            [1.0, 1.0, 1.0, 6.0, 6.0],
            [0.0, 2.0, 1.0, 6.5, 7.0],
            [1.0, 0.5, 2.0, 5.0, 7.0],
        ],
        dtype=np.float32,
    )
    actual = reference_numpy(spec, {"x": x, "rois": rois})["y"]
    expected = torchvision.ops.roi_align(
        torch.from_numpy(x),
        torch.from_numpy(rois),
        output_size=(2, 3),
        spatial_scale=1.0,
        sampling_ratio=2,
        aligned=False,
    )
    np.testing.assert_allclose(actual, expected.numpy(), atol=1e-5, rtol=1e-5)


def test_multi_level_roi_align_reference_matches_v1_style_level_oracle():
    class Tiny(dml.nn.Module):
        def forward(self, p2, p3, p4, p5, rois):
            return dml.ops.output(
                dml.ops.multi_level_roi_align(
                    p2,
                    p3,
                    p4,
                    p5,
                    rois,
                    pooled_size=(2, 2),
                    sampling_ratio=2,
                    spatial_scale=1.0,
                    im_shape=(64, 64),
                ),
                "y",
            )

    spec = dml.trace(
        Tiny(),
        inputs={
            "p2": dml.TensorSpec([2, 2, 16, 16], "float32"),
            "p3": dml.TensorSpec([2, 2, 8, 8], "float32"),
            "p4": dml.TensorSpec([2, 2, 4, 4], "float32"),
            "p5": dml.TensorSpec([2, 2, 2, 2], "float32"),
            "rois": dml.TensorSpec([4, 5], "float32"),
        },
        name="multi_level_roi_align_oracle",
    )
    p2 = np.arange(2 * 2 * 16 * 16, dtype=np.float32).reshape(2, 2, 16, 16) / np.float32(50.0)
    p3 = np.arange(2 * 2 * 8 * 8, dtype=np.float32).reshape(2, 2, 8, 8) / np.float32(40.0)
    p4 = np.arange(2 * 2 * 4 * 4, dtype=np.float32).reshape(2, 2, 4, 4) / np.float32(30.0)
    p5 = np.arange(2 * 2 * 2 * 2, dtype=np.float32).reshape(2, 2, 2, 2) / np.float32(20.0)
    rois = np.array(
        [
            [0.0, 2.0, 2.0, 10.0, 10.0],
            [1.0, 4.0, 4.0, 24.0, 24.0],
            [0.0, 8.0, 8.0, 40.0, 40.0],
            [1.0, 0.0, 0.0, 63.0, 63.0],
        ],
        dtype=np.float32,
    )
    actual = reference_numpy(spec, {"p2": p2, "p3": p3, "p4": p4, "p5": p5, "rois": rois})["y"]
    features = [p2, p3, p4, p5]
    expected = []
    for roi in rois:
        level_index = _multi_level_level_index(roi, im_h=64, im_w=64, spatial_scale=1.0)
        feature = features[level_index]
        level_rois = torch.from_numpy(roi[None, :])
        level_expected = torchvision.ops.roi_align(
            torch.from_numpy(feature),
            level_rois,
            output_size=(2, 2),
            spatial_scale=float(feature.shape[2]) / 64.0,
            sampling_ratio=2,
            aligned=False,
        )
        expected.append(level_expected.numpy()[0])
    expected_array = np.stack(expected, axis=0)
    np.testing.assert_allclose(actual, expected_array, atol=1e-5, rtol=1e-5)


def test_roi_align_family_frontend_and_generated_sources():
    class Tiny(dml.nn.Module):
        def forward(self, x, rois, p2, p3, p4, p5):
            single = dml.ops.roi_align(x, rois, pooled_size=(2, 2), sampling_ratio=2, spatial_scale=1.0)
            multi = dml.ops.multi_level_roi_align(
                p2,
                p3,
                p4,
                p5,
                rois,
                pooled_size=(2, 2),
                sampling_ratio=2,
                spatial_scale=1.0,
                im_shape=(64, 64),
            )
            return {
                "single": dml.ops.output(single, "single"),
                "multi": dml.ops.output(multi, "multi"),
            }

    spec = dml.trace(
        Tiny(),
        inputs={
            "x": dml.TensorSpec([2, 3, 8, 8], "float32"),
            "rois": dml.TensorSpec([4, 5], "float32"),
            "p2": dml.TensorSpec([2, 3, 16, 16], "float32"),
            "p3": dml.TensorSpec([2, 3, 8, 8], "float32"),
            "p4": dml.TensorSpec([2, 3, 4, 4], "float32"),
            "p5": dml.TensorSpec([2, 3, 2, 2], "float32"),
        },
        name="roi_align_family_generated_sources",
    )
    assert [node["op"] for node in spec.ir["nodes"]] == ["roi_align", "multi_level_roi_align"]
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}
    for backend in ("cuda", "rocm"):
        sources = render_generated_kernels(backend, spec.ir["nodes"], tensor_map)
        assert len(sources) == 2
        assert "roi_align_" in sources[0]
        assert "multi_level_roi_align_" in sources[1]

    class BadPositionSensitive(dml.nn.Module):
        def forward(self, x, rois):
            return dml.ops.output(
                dml.ops.roi_align(x, rois, pooled_size=2, position_sensitive=True),
                "y",
            )

    with pytest.raises(NotImplementedError, match="position_sensitive=True"):
        dml.trace(
            BadPositionSensitive(),
            inputs={
                "x": dml.TensorSpec([1, 2, 4, 4], "float32"),
                "rois": dml.TensorSpec([1, 5], "float32"),
            },
            name="roi_align_bad_position_sensitive",
        )
