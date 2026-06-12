from __future__ import annotations

import numpy as np
import pytest

import dinoml as dml
from dinoml.reference import reference_numpy


torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")


def _nms_inputs() -> tuple[np.ndarray, np.ndarray]:
    boxes = np.array(
        [
            [
                [0.0, 0.0, 4.0, 4.0],
                [0.5, 0.5, 4.5, 4.5],
                [6.0, 6.0, 9.0, 9.0],
                [0.0, 0.0, 0.2, 0.2],
                [7.0, 7.0, 10.0, 10.0],
            ],
            [
                [1.0, 1.0, 5.0, 5.0],
                [1.5, 1.5, 5.5, 5.5],
                [8.0, 8.0, 10.0, 10.0],
                [9.0, 9.0, 12.0, 12.0],
                [0.0, 0.0, 0.1, 0.1],
            ],
        ],
        dtype=np.float32,
    )
    scores = np.array(
        [
            [0.95, 0.90, 0.80, 0.70, 0.60],
            [0.99, 0.85, 0.75, 0.65, 0.55],
        ],
        dtype=np.float32,
    )
    return boxes, scores


def _batched_nms_boxes() -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 4.0, 4.0],
            [0.2, 0.2, 4.2, 4.2],
            [5.0, 5.0, 8.0, 8.0],
            [6.0, 6.0, 9.0, 9.0],
            [20.0, 20.0, 24.0, 24.0],
        ],
        dtype=np.float32,
    )


def _efficient_nms_inputs() -> tuple[np.ndarray, np.ndarray]:
    boxes = np.array(
        [
            [
                [[0.0, 0.0, 4.0, 4.0], [0.0, 0.0, 4.0, 4.0], [10.0, 10.0, 12.0, 12.0]],
                [[0.4, 0.4, 4.4, 4.4], [5.0, 5.0, 8.0, 8.0], [10.5, 10.5, 12.5, 12.5]],
                [[7.0, 7.0, 9.5, 9.5], [5.2, 5.2, 8.2, 8.2], [30.0, 30.0, 31.0, 31.0]],
                [[15.0, 15.0, 18.0, 18.0], [15.2, 15.2, 18.2, 18.2], [32.0, 32.0, 35.0, 35.0]],
            ],
            [
                [[1.0, 1.0, 4.0, 4.0], [6.0, 6.0, 8.0, 8.0], [12.0, 12.0, 14.0, 14.0]],
                [[1.2, 1.2, 4.2, 4.2], [6.1, 6.1, 8.1, 8.1], [12.1, 12.1, 14.1, 14.1]],
                [[20.0, 20.0, 24.0, 24.0], [25.0, 25.0, 29.0, 29.0], [0.0, 0.0, 0.1, 0.1]],
                [[21.0, 21.0, 25.0, 25.0], [25.2, 25.2, 29.2, 29.2], [40.0, 40.0, 44.0, 44.0]],
            ],
        ],
        dtype=np.float32,
    )
    scores = np.array(
        [
            [
                [0.95, 0.93, 0.20],
                [0.90, 0.89, 0.18],
                [0.70, 0.87, 0.10],
                [0.60, 0.50, 0.05],
            ],
            [
                [0.98, 0.80, 0.60],
                [0.94, 0.79, 0.58],
                [0.78, 0.77, 0.30],
                [0.76, 0.75, 0.20],
            ],
        ],
        dtype=np.float32,
    )
    return boxes, scores


def _torch_nms_boxes_oracle(boxes: np.ndarray, scores: np.ndarray, *, pre_nms_top: int, max_output: int, iou_threshold: float, min_box_size: float) -> np.ndarray:
    output = np.zeros((boxes.shape[0], max_output, 4), dtype=np.float32)
    for batch_idx in range(boxes.shape[0]):
        batch_boxes = torch.from_numpy(boxes[batch_idx])
        batch_scores = torch.from_numpy(scores[batch_idx])
        widths = batch_boxes[:, 2] - batch_boxes[:, 0]
        heights = batch_boxes[:, 3] - batch_boxes[:, 1]
        valid = (widths >= min_box_size) & (heights >= min_box_size)
        top_order = torch.argsort(batch_scores, descending=True, stable=True)[:pre_nms_top]
        top_order = top_order[valid[top_order]]
        keep = torchvision.ops.nms(batch_boxes[top_order], batch_scores[top_order], iou_threshold)
        keep = top_order[keep[:max_output]]
        if keep.numel():
            output[batch_idx, : keep.numel()] = batch_boxes[keep].numpy()
    return output


def _torch_batched_keep_oracle(boxes: np.ndarray, *, iou_threshold: float, keep_n: int) -> np.ndarray:
    scores = torch.arange(boxes.shape[0], 0, -1, dtype=torch.float32)
    keep = torchvision.ops.nms(torch.from_numpy(boxes), scores, iou_threshold)
    if keep_n >= 0:
        keep = keep[:keep_n]
    mask = np.zeros((boxes.shape[0],), dtype=np.int64)
    mask[keep.numpy()] = 1
    return mask


def _torch_efficient_nms_oracle(
    boxes: np.ndarray,
    scores: np.ndarray,
    *,
    pre_nms_top: int,
    max_output: int,
    iou_threshold: float,
    min_box_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = boxes.shape[0]
    num_det = np.zeros((batch, 1), dtype=np.int64)
    det_boxes = np.zeros((batch, max_output, 4), dtype=np.float32)
    det_scores = np.zeros((batch, max_output), dtype=np.float32)
    det_classes = np.zeros((batch, max_output), dtype=np.int64)
    for batch_idx in range(batch):
        flat_boxes = boxes[batch_idx].reshape(-1, 4)
        flat_scores = scores[batch_idx].reshape(-1)
        class_ids = np.tile(np.arange(scores.shape[2], dtype=np.int64), scores.shape[1])
        widths = flat_boxes[:, 2] - flat_boxes[:, 0]
        heights = flat_boxes[:, 3] - flat_boxes[:, 1]
        valid = (widths >= min_box_size) & (heights >= min_box_size)
        valid_indices = np.nonzero(valid)[0]
        valid_scores = torch.from_numpy(flat_scores[valid])
        top_order = torch.argsort(valid_scores, descending=True, stable=True)[:pre_nms_top]
        candidate_indices = valid_indices[top_order.numpy()]
        keep = torchvision.ops.batched_nms(
            torch.from_numpy(flat_boxes[candidate_indices]),
            torch.from_numpy(flat_scores[candidate_indices]),
            torch.from_numpy(class_ids[candidate_indices]),
            iou_threshold,
        )[:max_output]
        chosen = candidate_indices[keep.numpy()]
        count = chosen.shape[0]
        num_det[batch_idx, 0] = count
        if count:
            det_boxes[batch_idx, :count] = flat_boxes[chosen]
            det_scores[batch_idx, :count] = flat_scores[chosen]
            det_classes[batch_idx, :count] = class_ids[chosen]
    return num_det, det_boxes, det_scores, det_classes


def test_nms_reference_matches_torchvision_oracle():
    class Tiny(dml.nn.Module):
        def forward(self, boxes, scores):
            return dml.ops.output(
                dml.ops.nms(
                    boxes,
                    scores,
                    pre_nms_top=4,
                    max_output=3,
                    iou_threshold=0.5,
                    min_box_size=0.5,
                ),
                "y",
            )

    spec = dml.trace(
        Tiny(),
        inputs={"boxes": dml.TensorSpec([2, 5, 4], "float32"), "scores": dml.TensorSpec([2, 5], "float32")},
        name="nms_reference_oracle",
    )
    boxes, scores = _nms_inputs()
    actual = reference_numpy(spec, {"boxes": boxes, "scores": scores})["y"]
    expected = _torch_nms_boxes_oracle(boxes, scores, pre_nms_top=4, max_output=3, iou_threshold=0.5, min_box_size=0.5)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_batched_nms_reference_matches_sorted_torchvision_nms():
    class Tiny(dml.nn.Module):
        def forward(self, boxes):
            return dml.ops.output(dml.ops.batched_nms(boxes, iou_threshold=0.5, keep_n=3), "y")

    spec = dml.trace(
        Tiny(),
        inputs={"boxes": dml.TensorSpec([5, 4], "float32")},
        name="batched_nms_reference_oracle",
    )
    boxes = _batched_nms_boxes()
    actual = reference_numpy(spec, {"boxes": boxes})["y"]
    expected = _torch_batched_keep_oracle(boxes, iou_threshold=0.5, keep_n=3)
    np.testing.assert_array_equal(actual, expected)


def test_efficient_nms_reference_matches_flattened_batched_nms_oracle():
    class Tiny(dml.nn.Module):
        def forward(self, boxes, scores):
            num_det, det_boxes, det_scores, det_classes = dml.ops.efficient_nms(
                boxes,
                scores,
                pre_nms_top=5,
                max_output=3,
                iou_threshold=0.5,
                min_box_size=0.5,
            )
            return {
                "num_detections": dml.ops.output(num_det, "num_detections"),
                "detection_boxes": dml.ops.output(det_boxes, "detection_boxes"),
                "detection_scores": dml.ops.output(det_scores, "detection_scores"),
                "detection_classes": dml.ops.output(det_classes, "detection_classes"),
            }

    spec = dml.trace(
        Tiny(),
        inputs={"boxes": dml.TensorSpec([2, 4, 3, 4], "float32"), "scores": dml.TensorSpec([2, 4, 3], "float32")},
        name="efficient_nms_reference_oracle",
    )
    boxes, scores = _efficient_nms_inputs()
    actual = reference_numpy(spec, {"boxes": boxes, "scores": scores})
    expected = _torch_efficient_nms_oracle(
        boxes,
        scores,
        pre_nms_top=5,
        max_output=3,
        iou_threshold=0.5,
        min_box_size=0.5,
    )
    np.testing.assert_array_equal(actual["num_detections"], expected[0])
    np.testing.assert_allclose(actual["detection_boxes"], expected[1], atol=1e-6, rtol=1e-6)
    np.testing.assert_allclose(actual["detection_scores"], expected[2], atol=1e-6, rtol=1e-6)
    np.testing.assert_array_equal(actual["detection_classes"], expected[3])
