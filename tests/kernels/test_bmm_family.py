from dinoml.kernels.families.bmm import bmm_op_spec


def test_bmm_output_shape_spec_treats_symbolic_product_as_non_one_batch():
    batch_heads = {"op": "mul", "args": [{"name": "batch", "min": 1, "max": 1, "typical": 1}, 16]}
    spec = bmm_op_spec("bmm_rcr")

    output = spec.output_shape_spec(
        [
            [batch_heads, 1, 128],
            [batch_heads, 2048, 128],
        ]
    )

    assert output == [batch_heads, 1, 2048]
