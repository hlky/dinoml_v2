from __future__ import annotations

from pathlib import Path

import pytest

from dinoml.kernels.families.dual_gemm import DUAL_GEMM_OPS, dual_gemm_op_spec, dual_gemm_problem
from dinoml.kernels.manifest import build_kernel_manifest
from dinoml.kernels.providers.cutlass.gemm import cutlass_gemm_candidates
from tests.dual_gemm_parity import DUAL_GEMM_CASES, trace_dual_gemm_spec


def test_dual_gemm_family_exposes_local_anchor_surface():
    assert DUAL_GEMM_OPS == (
        "dual_gemm_rcr_relu",
        "dual_gemm_rcr_gelu",
        "dual_gemm_rcr_fast_gelu",
        "dual_gemm_rcr_quick_gelu",
        "dual_gemm_rcr_sigmoid",
        "dual_gemm_rcr_tanh",
        "dual_gemm_rcr_silu",
        "dual_gemm_rcr_hardswish",
        "dual_gemm_rcr_elup1",
        "dual_gemm_rcr_bias_relu",
        "dual_gemm_rcr_bias_gelu",
        "dual_gemm_rcr_bias_fast_gelu",
        "dual_gemm_rcr_bias_quick_gelu",
        "dual_gemm_rcr_bias_sigmoid",
        "dual_gemm_rcr_bias_tanh",
        "dual_gemm_rcr_bias_swish",
        "dual_gemm_rcr_bias_hardswish",
        "dual_gemm_rcr_bias_elup1",
    )


@pytest.mark.parametrize(
    ("op_name", "shapes", "expected_output"),
    [
        ("dual_gemm_rcr_silu", ([5, 7], [11, 7], [11, 7]), (5, 11)),
        ("dual_gemm_rcr_fast_gelu", ([2, 3, 7], [11, 7], [1, 7]), (2, 3, 11)),
        ("dual_gemm_rcr_bias_fast_gelu", ([5, 7], [11, 7], [11, 7], [11], [11]), (5, 11)),
        ("dual_gemm_rcr_bias_fast_gelu", ([2, 3, 7], [11, 7], [1, 7], [11], [1]), (2, 3, 11)),
    ],
)
def test_dual_gemm_family_validates_anchor_shapes(op_name, shapes, expected_output):
    spec = dual_gemm_op_spec(op_name)
    assert tuple(spec.validate_shapes(shapes)) == expected_output


def test_dual_gemm_problem_flattens_leading_dims():
    m, n, k, output = dual_gemm_problem(
        "dual_gemm_rcr_fast_gelu",
        [[2, 3, 7], [11, 7], [11, 7]],
    )
    assert (m, n, k, output) == (6, 11, 7, (2, 3, 11))


@pytest.mark.parametrize("cmake_var", ["DINOML_CK_GEMM_OPS", "DINOML_CUTLASS_GEMM_OPS"])
def test_dual_gemm_family_ops_are_exposed_in_cmake_static_archive_lists(cmake_var: str):
    cmake_text = Path(__file__).resolve().parents[2].joinpath("CMakeLists.txt").read_text(encoding="utf-8")
    marker = f"set({cmake_var}"
    start = cmake_text.find(marker)
    assert start >= 0
    tail = cmake_text[start + len(marker) :]
    block = tail.split("\n    )", 1)[0]
    listed_ops = {token for token in block.split() if token and token != "CACHE" and token != "STRING"}
    assert set(DUAL_GEMM_OPS).issubset(listed_ops)


@pytest.mark.parametrize(
    ("case_name", "dtype"),
    [
        ("dual_gemm_fast_gelu_f16_broadcast_dynamic", "float16"),
        ("dual_gemm_bias_fast_gelu_bf16_dynamic", "bfloat16"),
    ],
)
def test_dual_gemm_cuda_manifest_avoids_invalid_align1_candidates(case_name: str, dtype: str):
    case = next(case for case in DUAL_GEMM_CASES if case.name == case_name)
    manifest = build_kernel_manifest(trace_dual_gemm_spec(case).ir, {"name": "cuda", "arch": "sm_89"})
    item = manifest["required_kernels"][0]

    candidate_alignments = {int(candidate["cutlass"]["align"]) for candidate in item["candidates"]}
    selected = next(candidate for candidate in item["candidates"] if candidate["candidate_id"] == item["selected_candidate_id"])

    assert item["kernel_library"] == "cutlass_gemm"
    assert int(selected["cutlass"]["align"]) >= 2
    assert 1 not in candidate_alignments


@pytest.mark.parametrize(
    ("op_name", "dtype", "instruction"),
    [
        ("dual_gemm_rcr_fast_gelu", "float16", [16, 8, 16]),
        ("dual_gemm_rcr_bias_fast_gelu", "bfloat16", [16, 8, 16]),
        ("dual_gemm_rcr_silu", "float32", [16, 8, 8]),
    ],
)
def test_dual_gemm_cutlass_candidates_stay_on_dual_supported_policy_subset(
    op_name: str,
    dtype: str,
    instruction: list[int],
):
    candidates = cutlass_gemm_candidates(op_name, dtype)

    assert candidates
    assert {
        (
            tuple(int(dim) for dim in candidate["cutlass"]["threadblock"]),
            int(candidate["cutlass"]["stages"]),
            tuple(int(dim) for dim in candidate["cutlass"]["warp_count"]),
            tuple(int(dim) for dim in candidate["cutlass"]["instruction"]),
            str(candidate["cutlass"]["opclass"]),
        )
        for candidate in candidates
    } == {
        ((128, 64, 32), 3, (2, 2, 1), tuple(instruction), "tensorop"),
    }
    assert {str(candidate["accumulator_dtype"]) for candidate in candidates} == {"float32"}


def test_dual_gemm_cutlass_runtime_wrapper_keeps_b1_column_major_with_zero_stride_broadcast():
    source = Path(__file__).resolve().parents[2].joinpath("kernels", "cuda", "src", "cutlass_gemm.cu").read_text(
        encoding="utf-8"
    )

    assert "cutlass::layout::ColumnMajor,\n      cutlass::layout::ColumnMajor," in source
    assert "int const b1_ldb = BroadcastB1 ? 0 : k;" in source


@pytest.mark.parametrize(
    ("op_name", "shapes", "message"),
    [
        ("dual_gemm_rcr_silu", ([5, 7], [11, 8], [11, 7]), "B0"),
        ("dual_gemm_rcr_fast_gelu", ([5, 7], [11, 7], [3, 7]), "B1"),
        ("dual_gemm_rcr_bias_fast_gelu", ([5, 7], [11, 7], [11, 7], [10], [11]), "bias0"),
        ("dual_gemm_rcr_bias_fast_gelu", ([5, 7], [11, 7], [1, 7], [11], [11]), "bias1"),
    ],
)
def test_dual_gemm_family_rejects_invalid_shapes(op_name, shapes, message):
    spec = dual_gemm_op_spec(op_name)
    with pytest.raises(ValueError, match=message):
        spec.validate_shapes(shapes)
