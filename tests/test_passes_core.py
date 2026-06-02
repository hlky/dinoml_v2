from __future__ import annotations

from dinoml.passes.core import _dynamic_slice_static_contiguous_view


def test_dynamic_slice_static_contiguous_view_rejects_noncontiguous_outer_slices():
    node = {
        "inputs": ["x"],
        "outputs": ["y"],
        "attrs": {
            "start_indices": [0, 2, 0],
            "slice_sizes": [2, 2, 3],
        },
    }
    input_tensor = {"shape": [2, 4, 3]}
    output_tensor = {"shape": [2, 2, 3]}

    assert _dynamic_slice_static_contiguous_view(node, input_tensor, output_tensor) is None


def test_dynamic_slice_static_contiguous_view_accepts_single_outer_block_slice():
    node = {
        "inputs": ["x"],
        "outputs": ["y"],
        "attrs": {
            "start_indices": [1, 1, 0],
            "slice_sizes": [1, 2, 3],
        },
    }
    input_tensor = {"shape": [2, 4, 3]}
    output_tensor = {"shape": [1, 2, 3]}

    assert _dynamic_slice_static_contiguous_view(node, input_tensor, output_tensor) == ("x", 15)
