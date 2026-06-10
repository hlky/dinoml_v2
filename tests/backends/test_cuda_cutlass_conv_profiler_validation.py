from pathlib import Path

import pytest

from dinoml.kernels.profiling import (
    CUTLASS_CONV_VALIDATION_FAST,
    CUTLASS_CONV_VALIDATION_STRICT,
    ConvProfileWorkload,
    _CutlassConvProfiler,
    _profile_cache_lookup,
    _profile_report,
)


def _cutlass_conv_workload() -> ConvProfileWorkload:
    candidate = {
        "candidate_id": "cand0",
        "candidate_config_key": "cfg0",
        "profiler_symbol": "profile_conv_symbol",
        "status": "runtime",
        "profiler_status": "runtime_profiler",
    }
    return ConvProfileWorkload(
        node_id="n0",
        op="conv2d_bias",
        dtype="float16",
        kernel_symbol="kernel_conv_symbol",
        profiler_symbol="profile_conv_symbol",
        candidate_set_id="cutlass_conv_set",
        candidate_set_key="cutlass_conv_set_key",
        candidate_id="cand0",
        candidate_config_key="cfg0",
        candidate=candidate,
        x_tensor="x",
        weight_tensor="w",
        bias_tensor="b",
        residual_tensor=None,
        output_tensor="y",
        x_shape=(2, 8, 16, 16),
        weight_shape=(64, 8, 3, 3),
        bias_shape=(64,),
        residual_shape=None,
        output_shape=(2, 64, 14, 14),
        conv_config={"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1]},
        semantic_layout={},
        provider_layout={},
        layout_translation={},
        weight_transform={},
        temporary_buffers=(),
        workspace_nbytes=0,
        source_op=None,
        bias_mode=None,
        shape_source="static",
        shape_case_id="case0",
        dim_values={},
        dim_sources={},
    )


def _cutlass_conv1d_workload() -> ConvProfileWorkload:
    candidate = {
        "candidate_id": "cand1",
        "candidate_config_key": "cfg1",
        "profiler_symbol": "profile_conv1d_symbol",
        "status": "runtime",
        "profiler_status": "runtime_profiler",
    }
    return ConvProfileWorkload(
        node_id="n1",
        op="conv1d_bias_add_relu",
        dtype="float16",
        kernel_symbol="kernel_conv1d_symbol",
        profiler_symbol="profile_conv1d_symbol",
        candidate_set_id="cutlass_conv1d_set",
        candidate_set_key="cutlass_conv1d_set_key",
        candidate_id="cand1",
        candidate_config_key="cfg1",
        candidate=candidate,
        x_tensor="x",
        weight_tensor="w",
        bias_tensor="b",
        residual_tensor="r",
        output_tensor="y",
        x_shape=(2, 8, 16),
        weight_shape=(64, 8, 3),
        bias_shape=(64,),
        residual_shape=(2, 64, 8),
        output_shape=(2, 64, 8),
        conv_config={"stride": [2], "padding": [1], "dilation": [1]},
        semantic_layout={},
        provider_layout={},
        layout_translation={},
        weight_transform={},
        temporary_buffers=(),
        workspace_nbytes=0,
        source_op=None,
        bias_mode=None,
        shape_source="static",
        shape_case_id="case1",
        dim_values={},
        dim_sources={},
    )


def _cutlass_transposed_conv2d_workload() -> ConvProfileWorkload:
    candidate = {
        "candidate_id": "cand0",
        "candidate_config_key": "cfg0",
        "profiler_symbol": "profile_transposed_conv_symbol",
        "status": "runtime",
        "profiler_status": "runtime_profiler",
    }
    return ConvProfileWorkload(
        node_id="n0",
        op="transposed_conv2d",
        dtype="float16",
        kernel_symbol="kernel_transposed_conv_symbol",
        profiler_symbol="profile_transposed_conv_symbol",
        candidate_set_id="cutlass_transposed_conv_set",
        candidate_set_key="cutlass_transposed_conv_set_key",
        candidate_id="cand0",
        candidate_config_key="cfg0",
        candidate=candidate,
        x_tensor="x",
        weight_tensor="w",
        bias_tensor=None,
        residual_tensor=None,
        output_tensor="y",
        x_shape=(2, 8, 16, 16),
        weight_shape=(8, 64, 3, 3),
        bias_shape=None,
        residual_shape=None,
        output_shape=(2, 64, 32, 32),
        conv_config={"stride": [2, 2], "padding": [1, 1], "output_padding": [1, 1], "dilation": [1, 1]},
        semantic_layout={},
        provider_layout={},
        layout_translation={},
        weight_transform={},
        temporary_buffers=(),
        workspace_nbytes=0,
        source_op=None,
        bias_mode=None,
        shape_source="static",
        shape_case_id="case0",
        dim_values={},
        dim_sources={},
    )


def _profile_context() -> dict[str, object]:
    return {
        "fingerprint": {
            "hardware": {"name": "fake-gpu"},
            "hardware_key": "hw-key",
            "support_libraries": [],
            "support_libraries_key": "libs-key",
        },
        "support_libraries_by_name": {},
    }


def test_cutlass_conv_profiler_defaults_to_fast_validation_mode():
    workload = _cutlass_conv_workload()
    calls = []

    class FakeModule:
        def profile_conv(self, **kwargs):
            calls.append(kwargs)
            return [{"profiler_symbol": workload.profiler_symbol, "samples_ms": [0.21, 0.20], "workspace_nbytes": 0}]

    profiler = _CutlassConvProfiler(
        {(workload.op, workload.dtype): FakeModule()},
        {(workload.op, workload.dtype): [workload.candidate]},
    )
    rows = profiler.profile(workload, iterations=7, repeats=2)

    assert calls[0]["validation_mode"] == CUTLASS_CONV_VALIDATION_FAST
    assert rows[0]["candidate"]["candidate_id"] == workload.candidate_id
    assert rows[0]["samples_ms"] == [0.21, 0.20]


def test_cutlass_conv_profiler_forwards_strict_validation_mode():
    workload = _cutlass_conv_workload()
    calls = []

    class FakeModule:
        def profile_conv(self, **kwargs):
            calls.append(kwargs)
            return [{"profiler_symbol": workload.profiler_symbol, "samples_ms": [0.19], "workspace_nbytes": 0}]

    profiler = _CutlassConvProfiler(
        {(workload.op, workload.dtype): FakeModule()},
        {(workload.op, workload.dtype): [workload.candidate]},
        validation_mode="strict",
    )
    profiler.profile(workload, iterations=5, repeats=1)

    assert calls[0]["validation_mode"] == CUTLASS_CONV_VALIDATION_STRICT


def test_cutlass_transposed_conv2d_profiler_marks_bias_absent():
    workload = _cutlass_transposed_conv2d_workload()
    calls = []

    class FakeModule:
        def profile_conv(self, **kwargs):
            calls.append(kwargs)
            return [{"profiler_symbol": workload.profiler_symbol, "samples_ms": [0.17], "workspace_nbytes": 0}]

    profiler = _CutlassConvProfiler(
        {(workload.op, workload.dtype): FakeModule()},
        {(workload.op, workload.dtype): [workload.candidate]},
    )
    rows = profiler.profile(workload, iterations=5, repeats=1)

    assert calls[0]["has_bias"] is False
    assert calls[0]["c"] == workload.x_shape[1]
    assert calls[0]["out_c"] == workload.output_shape[1]
    assert calls[0]["residual_count"] == 0
    assert rows[0]["candidate"]["candidate_id"] == workload.candidate_id


def test_cutlass_conv1d_profiler_forwards_1d_shape_contract():
    workload = _cutlass_conv1d_workload()
    calls = []

    class FakeModule:
        def profile_conv(self, **kwargs):
            calls.append(kwargs)
            return [{"profiler_symbol": workload.profiler_symbol, "samples_ms": [0.23], "workspace_nbytes": 0}]

    profiler = _CutlassConvProfiler(
        {(workload.op, workload.dtype): FakeModule()},
        {(workload.op, workload.dtype): [workload.candidate]},
    )
    rows = profiler.profile(workload, iterations=9, repeats=1)

    assert calls[0]["spatial_rank"] == 1
    assert calls[0]["w"] == workload.x_shape[2]
    assert calls[0]["out_w"] == workload.output_shape[2]
    assert calls[0]["kernel_w"] == workload.weight_shape[2]
    assert calls[0]["residual_count"] == 1
    assert rows[0]["candidate"]["candidate_id"] == workload.candidate_id


def test_cutlass_conv_profiler_surfaces_strict_validation_failures():
    workload = _cutlass_conv_workload()

    class FakeModule:
        def profile_conv(self, **kwargs):
            if kwargs["validation_mode"] == CUTLASS_CONV_VALIDATION_STRICT:
                return []
            return [{"profiler_symbol": workload.profiler_symbol, "samples_ms": [0.18], "workspace_nbytes": 0}]

    profiler = _CutlassConvProfiler(
        {(workload.op, workload.dtype): FakeModule()},
        {(workload.op, workload.dtype): [workload.candidate]},
        validation_mode="strict",
    )

    with pytest.raises(RuntimeError, match="returned no usable candidate timings"):
        profiler.profile(workload, iterations=5, repeats=1)


def test_cutlass_conv_profile_cache_key_tracks_validation_mode():
    workload = _cutlass_conv_workload()
    manifest = {"target": {"name": "cuda", "arch": "sm_89"}}

    fast_lookup = _profile_cache_lookup(
        workload,
        manifest,
        {"cache_key": "kernel-cache"},
        {"cache_key": "codegen-cache"},
        context=_profile_context(),
        cutlass_conv_validation_mode="fast",
    )
    strict_lookup = _profile_cache_lookup(
        workload,
        manifest,
        {"cache_key": "kernel-cache"},
        {"cache_key": "codegen-cache"},
        context=_profile_context(),
        cutlass_conv_validation_mode="strict",
    )

    assert fast_lookup.profile_key != strict_lookup.profile_key
    assert fast_lookup.key_payload["profile_variant"]["validation_mode"] == CUTLASS_CONV_VALIDATION_FAST
    assert strict_lookup.key_payload["profile_variant"]["validation_mode"] == CUTLASS_CONV_VALIDATION_STRICT


def test_profile_report_records_cutlass_conv_validation_policy():
    report = _profile_report(
        Path("."),
        {"target": {"name": "cuda", "arch": "sm_89"}},
        {"cache_key": "kernel-cache"},
        {"cache_key": "codegen-cache"},
        5,
        2,
        [],
        {"profiled": 0, "cached": 0, "skipped": 0, "failed": 0, "blocked": 0},
        context=_profile_context(),
        cutlass_conv_validation_mode="strict",
    )

    assert report["schema_version"] >= 8
    assert report["validation_policy"] == {"cutlass_conv": CUTLASS_CONV_VALIDATION_STRICT}
