from __future__ import annotations

import numpy as np

import dinoml as dml


SCORE_SHAPE = [2, 4]
FEATURE_SHAPE = [2, 4, 3]
TOP_K = 2


class CandidateSelection(dml.Module):
    def forward(self, scores, features):
        top_scores, top_indices = dml.ops.topk(scores, TOP_K, dim=-1)
        selected_features = dml.ops.batch_gather(features, top_indices)
        return {
            "top_scores": top_scores,
            "selected_features": selected_features,
        }


def build_spec() -> dml.ir.ModelSpec:
    return dml.trace(
        CandidateSelection(),
        inputs={
            "scores": dml.TensorSpec(SCORE_SHAPE, "float32"),
            "features": dml.TensorSpec(FEATURE_SHAPE, "float32"),
        },
        name="candidate_selection",
    )


def build_validation_inputs() -> dict[str, np.ndarray]:
    scores = np.array(
        [
            [0.1, 2.5, 2.5, -1.0],
            [3.0, 1.0, 4.0, 4.0],
        ],
        dtype=np.float32,
    )
    features = (np.arange(int(np.prod(FEATURE_SHAPE)), dtype=np.float32) / 10.0).reshape(
        FEATURE_SHAPE
    )
    return {"scores": scores, "features": features}


def torch_reference(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    import torch

    scores = torch.from_numpy(inputs["scores"])
    features = torch.from_numpy(inputs["features"])
    top_scores, top_indices = torch.topk(
        scores, TOP_K, dim=-1, largest=True, sorted=True
    )
    batch_ids = torch.arange(scores.shape[0]).unsqueeze(1)
    selected_features = features[batch_ids, top_indices]
    return {
        "top_scores": top_scores.numpy().astype(np.float32),
        "selected_features": selected_features.numpy().astype(np.float32),
    }
