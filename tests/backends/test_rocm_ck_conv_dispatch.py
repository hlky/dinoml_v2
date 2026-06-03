from __future__ import annotations

from dinoml.kernels.codegen import create_codegen_plan
from dinoml.kernels.manifest import apply_execution_plan, build_kernel_manifest
from dinoml.lowering.rocm import render_rocm_module
from tests.backends.test_rocm_scaffold import (
    _ck_execution_plan_from_results,
    _ck_profile_result_for_candidate,
    _rocm_conv2d_bias_ir,
)


def test_rocm_ck_conv_execution_plan_conflict_applies_guarded_dispatch():
    ir_a = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16)
    ir_b = _rocm_conv2d_bias_ir("float16", batch=1, in_channels=8, out_channels=32, height=16, width=16)
    manifest = build_kernel_manifest(ir_a, {"name": "rocm", "arch": "gfx1201"})
    default_candidate_id = str(manifest["required_kernels"][0]["selected_candidate_id"])
    alternate_candidate_id = "ck_conv2d_bias_float16_xdl_wide_m_v1"
    assert default_candidate_id != alternate_candidate_id

    execution_plan = _ck_execution_plan_from_results(
        manifest,
        _ck_profile_result_for_candidate(ir_a, manifest, default_candidate_id, 0.31, profile_key="profile-default"),
        _ck_profile_result_for_candidate(ir_b, manifest, alternate_candidate_id, 0.29, profile_key="profile-alt"),
    )
    overlaid = apply_execution_plan(manifest, execution_plan, strict=True)
    item = overlaid["required_kernels"][0]
    dispatch = item["execution_plan_dispatch"]

    assert execution_plan["summary"]["static_selection_count"] == 0
    assert execution_plan["summary"]["conflict_count"] == 1
    assert item["selected_candidate_id"] == default_candidate_id
    assert "execution_plan_selection" not in item
    assert len(dispatch) == 2
    assert {entry["selected_candidate_id"] for entry in dispatch} == {default_candidate_id, alternate_candidate_id}
    assert {
        tuple((key, entry["shape"][key]) for key in ("n", "c", "h", "w", "out_n", "out_c", "out_h", "out_w"))
        for entry in dispatch
    } == {
        (("n", 2), ("c", 8), ("h", 16), ("w", 16), ("out_n", 2), ("out_c", 64), ("out_h", 16), ("out_w", 16)),
        (("n", 1), ("c", 8), ("h", 16), ("w", 16), ("out_n", 1), ("out_c", 32), ("out_h", 16), ("out_w", 16)),
    }


def test_rocm_ck_conv_execution_plan_guarded_dispatch_exports_all_referenced_symbols(tmp_path):
    ir_a = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16)
    ir_b = _rocm_conv2d_bias_ir("float16", batch=1, in_channels=8, out_channels=32, height=16, width=16)
    manifest = build_kernel_manifest(ir_a, {"name": "rocm", "arch": "gfx1201"})
    default_candidate_id = str(manifest["required_kernels"][0]["selected_candidate_id"])
    alternate_candidate_id = "ck_conv2d_bias_float16_xdl_wide_m_v1"
    execution_plan = _ck_execution_plan_from_results(
        manifest,
        _ck_profile_result_for_candidate(ir_a, manifest, default_candidate_id, 0.31, profile_key="profile-default"),
        _ck_profile_result_for_candidate(ir_b, manifest, alternate_candidate_id, 0.29, profile_key="profile-alt"),
    )
    overlaid = apply_execution_plan(manifest, execution_plan, strict=True)

    plan = create_codegen_plan(overlaid, tmp_path)
    support = plan.external_support_libraries[0]
    entry = support["entries"][0]

    assert support["name"] == "ck_conv"
    assert support["pruned_by_execution_plan"] is True
    assert len(entry["execution_plan_dispatch"]) == 2
    assert {candidate["candidate_id"] for candidate in entry["candidates"]} == {
        default_candidate_id,
        alternate_candidate_id,
    }
    assert set(support["kernel_symbols"]) == {
        str(selection["kernel_symbol"]) for selection in entry["execution_plan_dispatch"]
    }
    assert set(support["profiler_symbols"]) == {
        str(selection["profiler_symbol"]) for selection in entry["execution_plan_dispatch"]
    }
    assert set(plan.candidate_profiler_symbols) == set(support["profiler_symbols"])


def test_rocm_conv_module_dispatches_ck_execution_plan_by_shape():
    ir_a = _rocm_conv2d_bias_ir("float16", batch=2, in_channels=8, out_channels=64, height=16, width=16)
    ir_b = _rocm_conv2d_bias_ir("float16", batch=1, in_channels=8, out_channels=32, height=16, width=16)
    manifest = build_kernel_manifest(ir_a, {"name": "rocm", "arch": "gfx1201"})
    default_candidate_id = str(manifest["required_kernels"][0]["selected_candidate_id"])
    execution_plan = _ck_execution_plan_from_results(
        manifest,
        _ck_profile_result_for_candidate(ir_a, manifest, default_candidate_id, 0.31, profile_key="profile-default"),
        _ck_profile_result_for_candidate(
            ir_b,
            manifest,
            "ck_conv2d_bias_float16_xdl_wide_m_v1",
            0.29,
            profile_key="profile-alt",
        ),
    )
    overlaid = apply_execution_plan(manifest, execution_plan, strict=True)

    source = render_rocm_module(ir_a, kernel_manifest=overlaid)

    assert source.count('extern "C" int dinoml_ck_conv2d_bias_float16_') >= 2
    assert "(shape_x_0) == 2" in source
    assert "(shape_x_0) == 1" in source
    assert "(shape_weight_2) == 3" in source
    assert "(shape_y_1) == 64" in source
    assert "(shape_y_1) == 32" in source
    assert "else {" in source
    assert "CK Conv launcher failed" in source
