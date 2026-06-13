from __future__ import annotations

import dinoml as dml
from dinoml.lowering.ops import render_generated_kernels, render_launch


class _ModeModule(dml.Module):
    def forward(self, x):
        values, indices = x.mode(dim=-1)
        return {
            "values": dml.ops.output(values, "values"),
            "indices": dml.ops.output(indices, "indices"),
        }


def test_mode_gpu_pair_uses_single_fused_launch():
    spec = dml.trace(_ModeModule(), inputs={"x": dml.TensorSpec([4, 8], "float32")}, name="mode_gpu_pair")
    tensor_map = {tensor["name"]: tensor for tensor in spec.ir["tensors"]}

    sources = render_generated_kernels("rocm", spec.ir["nodes"], tensor_map)
    launches = [render_launch("rocm", node, tensor_map) for node in spec.ir["nodes"]]

    assert len(sources) == 1
    assert "ptr_x, ptr_t0, ptr_t1" in launches[0]
    assert "paired mode_indices is produced by the mode_values launch" in launches[1]
    assert "candidate_count" in sources[0]
