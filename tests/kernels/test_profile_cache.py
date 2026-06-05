from __future__ import annotations

from dataclasses import replace

import dinoml.kernels.profiling as profiling_mod
from dinoml.kernels.gemm import gemm_op_spec
from dinoml.kernels.manifest import PROFILE_CACHE_SCHEMA_VERSION, build_kernel_manifest
from dinoml.kernels.profile_cache import ProfileCacheWrite, SQLiteProfileCacheBackend, default_profile_cache_path
from dinoml.kernels.profiling import (
    _cache_entry,
    _hardware_cache_payload,
    _parse_hipinfo_devices,
    _prepare_profile_workloads,
    _profile_cache_lookup,
    _profile_result,
    build_profile_workloads,
)


def test_default_profile_cache_path_is_shared_per_target_directory(tmp_path):
    first = default_profile_cache_path({"support_cache_dir": str(tmp_path / "support" / "rocm-gfx1201" / "aaaa")})
    second = default_profile_cache_path({"support_cache_dir": str(tmp_path / "support" / "rocm-gfx1201" / "bbbb")})

    assert first == second
    assert first.name == f"profile_cache.v{PROFILE_CACHE_SCHEMA_VERSION}.sqlite3"


def test_prepare_profile_workloads_uses_backend_lookup_and_dedupes():
    target = {"name": "rocm", "arch": "gfx1201"}
    manifest = build_kernel_manifest(_bmm_ir("bmm_rcr_add", "float16"), target)
    workload = build_profile_workloads(_bmm_ir("bmm_rcr_add", "float16"), manifest)[0]
    context = _fake_profile_context(workload.kernel_library)
    lookup = _profile_cache_lookup(workload, {"target": target}, manifest, {"cache_key": "plan-a"}, context=context)
    entry = _cache_entry(
        workload,
        _profile_result(workload, 0.21, 5, profile_key=lookup.profile_key, status="ok"),
        lookup.key_payload,
    )

    class FakeBackend:
        def __init__(self) -> None:
            self.calls = []

        def lookup_many(self, lookups):
            self.calls.append([item.profile_key for item in lookups])
            return {lookup.profile_key: dict(entry)}

    backend = FakeBackend()
    prepared, unique = _prepare_profile_workloads(
        [workload, replace(workload, node_id="n1")],
        {"target": target},
        manifest,
        {"cache_key": "plan-b"},
        backend,
        iterations=5,
        repeats=1,
        cutlass_conv_validation_mode="fast",
        refresh=False,
        context=context,
    )

    assert len(backend.calls) == 1
    assert backend.calls[0] == [lookup.profile_key]
    assert len(unique) == 1
    assert unique[0].resolution == "cache"
    assert prepared[0].representative is True
    assert prepared[1].representative is False
    assert prepared[0].profile_key == prepared[1].profile_key


def test_prepare_profile_workloads_pre_dedupes_before_profile_cache_lookup(monkeypatch):
    target = {"name": "rocm", "arch": "gfx1201"}
    manifest = build_kernel_manifest(_bmm_ir("bmm_rcr_add", "float16"), target)
    workload = build_profile_workloads(_bmm_ir("bmm_rcr_add", "float16"), manifest)[0]
    duplicate = replace(
        workload,
        shape_source="workflow_bucket",
        shape_case_id="shape_m=64",
        dim_values={"m": workload.m},
        dim_sources={"m": "metadata"},
    )
    context = _fake_profile_context(workload.kernel_library)
    lookup = _profile_cache_lookup(workload, {"target": target}, manifest, {"cache_key": "plan-a"}, context=context)
    entry = _cache_entry(
        workload,
        _profile_result(workload, 0.21, 5, profile_key=lookup.profile_key, status="ok"),
        lookup.key_payload,
    )
    calls = 0
    original = profiling_mod._profile_cache_lookup

    def counting_lookup(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(profiling_mod, "_profile_cache_lookup", counting_lookup)

    class FakeBackend:
        def lookup_many(self, lookups):
            return {lookup.profile_key: dict(entry) for lookup in lookups}

    prepared, unique = _prepare_profile_workloads(
        [workload, duplicate],
        {"target": target},
        manifest,
        {"cache_key": "plan-b"},
        FakeBackend(),
        iterations=5,
        repeats=1,
        cutlass_conv_validation_mode="fast",
        refresh=False,
        context=context,
    )

    assert calls == 1
    assert len(unique) == 1
    assert unique[0].resolution == "cache"
    assert prepared[0].profile_key == prepared[1].profile_key


def test_sqlite_profile_cache_round_trip(tmp_path):
    target = {"name": "rocm", "arch": "gfx1201"}
    ir = _bmm_ir("bmm_rcr_add", "float16")
    manifest = build_kernel_manifest(ir, target)
    workload = build_profile_workloads(ir, manifest)[0]
    context = _fake_profile_context(workload.kernel_library, support_fields={"source_sha256": "support-a"})
    lookup = _profile_cache_lookup(workload, {"target": target}, manifest, {"cache_key": "plan-a"}, context=context)
    entry = _cache_entry(
        workload,
        _profile_result(workload, 0.33, 7, profile_key=lookup.profile_key, status="ok"),
        lookup.key_payload,
    )

    backend = SQLiteProfileCacheBackend(tmp_path / "profile_cache.sqlite3")
    backend.upsert_many([ProfileCacheWrite(lookup, entry)])
    rows = backend.lookup_many([lookup])
    backend.close()

    assert rows[lookup.profile_key]["profile_key"] == lookup.profile_key
    assert rows[lookup.profile_key]["key"] == lookup.key_payload
    assert rows[lookup.profile_key]["best_candidate_id"] == workload.candidate_id


def test_rocm_profile_key_reuses_across_artifact_identity_changes():
    target = {"name": "rocm", "arch": "gfx1201"}
    ir = _bmm_ir("bmm_rcr_add", "float16")
    manifest = build_kernel_manifest(ir, target)
    workload = build_profile_workloads(ir, manifest)[0]
    context = _fake_profile_context(workload.kernel_library)

    first = _profile_cache_lookup(
        workload,
        {"target": target},
        {"support_cache_key": "support-a", "cache_key": "manifest-a"},
        {"cache_key": "plan-a"},
        context=context,
    )
    second = _profile_cache_lookup(
        workload,
        {"target": target},
        {"support_cache_key": "support-b", "cache_key": "manifest-b"},
        {"cache_key": "plan-b"},
        context=context,
    )

    assert first.profile_key == second.profile_key


def test_rocm_profile_key_ignores_shape_case_metadata_when_problem_matches():
    target = {"name": "rocm", "arch": "gfx1201"}
    ir = _bmm_ir("bmm_rcr_add", "float16")
    manifest = build_kernel_manifest(ir, target)
    workload = build_profile_workloads(ir, manifest)[0]
    context = _fake_profile_context(workload.kernel_library)
    alternate = replace(
        workload,
        shape_source="workflow_bucket",
        shape_case_id="shape_m=64",
        dim_values={"m": workload.m},
        dim_sources={"m": "metadata"},
    )

    first = _profile_cache_lookup(workload, {"target": target}, manifest, {"cache_key": "plan-a"}, context=context)
    second = _profile_cache_lookup(alternate, {"target": target}, manifest, {"cache_key": "plan-b"}, context=context)

    assert first.profile_key == second.profile_key


def test_profile_key_differs_when_support_fingerprint_changes():
    target = {"name": "rocm", "arch": "gfx1201"}
    ir = _bmm_ir("bmm_rcr_add", "float16")
    manifest = build_kernel_manifest(ir, target)
    workload = build_profile_workloads(ir, manifest)[0]

    first = _profile_cache_lookup(
        workload,
        {"target": target},
        manifest,
        {"cache_key": "plan-a"},
        context=_fake_profile_context(workload.kernel_library, support_fields={"source_sha256": "support-a"}),
    )
    second = _profile_cache_lookup(
        workload,
        {"target": target},
        manifest,
        {"cache_key": "plan-b"},
        context=_fake_profile_context(workload.kernel_library, support_fields={"source_sha256": "support-b"}),
    )

    assert first.profile_key != second.profile_key


def test_parse_hipinfo_devices_extracts_stable_identity_fields():
    devices = _parse_hipinfo_devices(
        """
--------------------------------------------------------------------------------
device#                           0
Name:                             AMD Radeon RX 9070 XT
pciBusID:                         4
pciDeviceID:                      0
pciDomainID:                      0
multiProcessorCount:              32
totalGlobalMem:                   15.92 GB
major:                            12
minor:                            0
asicRevision:                     0
gcnArchName:                      gfx1201
"""
    )

    assert devices == [
        {
            "index": 0,
            "name": "AMD Radeon RX 9070 XT",
            "pci_bus_id": 4,
            "pci_device_id": 0,
            "pci_domain_id": 0,
            "multi_processor_count": 32,
            "total_global_mem_bytes": 17093969838,
            "major": 12,
            "minor": 0,
            "asic_revision": 0,
            "gcn_arch_name": "gfx1201",
        }
    ]


def test_rocm_hardware_cache_payload_uses_device_identity_not_tool_versions():
    base = {
        "backend": "rocm",
        "target_arch": "gfx1201",
        "hip_visible_devices": "",
        "rocr_visible_devices": "",
        "gpu_device_ordinal": "",
        "devices": [
            {
                "name": "AMD Radeon RX 9070 XT",
                "gcn_arch_name": "gfx1201",
                "major": 12,
                "minor": 0,
                "multi_processor_count": 32,
                "total_global_mem_bytes": 17093969838,
                "pci_bus_id": 4,
                "pci_device_id": 0,
                "pci_domain_id": 0,
                "asic_revision": 0,
            }
        ],
    }

    first = _hardware_cache_payload(
        {
            **base,
            "hipconfig": {"available": "false"},
            "rocminfo": {"available": "false"},
            "hipinfo": {"available": "true"},
        }
    )
    second = _hardware_cache_payload(
        {
            **base,
            "hipconfig": {"available": "true", "version": "7.2.53211-158bd99533"},
            "rocminfo": {"available": "true", "version": "rocminfo 7.2"},
            "hipinfo": {"available": "true"},
        }
    )

    assert first == second


def test_profile_key_differs_when_target_changes():
    ir = _bmm_ir("bmm_rcr_add", "float16")
    rocm_manifest = build_kernel_manifest(ir, {"name": "rocm", "arch": "gfx1201"})
    rocm_workload = build_profile_workloads(ir, rocm_manifest)[0]

    first = _profile_cache_lookup(
        rocm_workload,
        {"target": {"name": "rocm", "arch": "gfx1201"}},
        rocm_manifest,
        {"cache_key": "plan-a"},
        context=_fake_profile_context(rocm_workload.kernel_library),
    )
    second = _profile_cache_lookup(
        rocm_workload,
        {"target": {"name": "rocm", "arch": "gfx1203"}},
        rocm_manifest,
        {"cache_key": "plan-b"},
        context=_fake_profile_context(rocm_workload.kernel_library),
    )

    assert first.profile_key != second.profile_key


def test_profile_key_differs_when_candidate_identity_changes():
    target = {"name": "rocm", "arch": "gfx1201"}
    ir = _bmm_ir("bmm_rcr_add", "float16")
    manifest = build_kernel_manifest(ir, target)
    workloads = build_profile_workloads(ir, manifest)
    assert len(workloads) >= 2
    context = _fake_profile_context(workloads[0].kernel_library)

    first = _profile_cache_lookup(workloads[0], {"target": target}, manifest, {"cache_key": "plan-a"}, context=context)
    second = _profile_cache_lookup(workloads[1], {"target": target}, manifest, {"cache_key": "plan-b"}, context=context)

    assert workloads[0].candidate_id != workloads[1].candidate_id
    assert first.profile_key != second.profile_key


def test_cuda_profile_key_reuses_across_artifact_identity_changes():
    target = {"name": "cuda", "arch": "sm_80"}
    ir = _gemm_ir("gemm_rcr_bias_add_relu", "float16")
    manifest = build_kernel_manifest(ir, target)
    workload = build_profile_workloads(ir, manifest)[0]
    context = _fake_profile_context(workload.kernel_library, support_fields={"source_sha256": "cutlass-a"})

    first = _profile_cache_lookup(
        workload,
        {"target": target},
        {"support_cache_key": "support-a", "cache_key": "manifest-a"},
        {"cache_key": "plan-a"},
        context=context,
    )
    second = _profile_cache_lookup(
        workload,
        {"target": target},
        {"support_cache_key": "support-b", "cache_key": "manifest-b"},
        {"cache_key": "plan-b"},
        context=context,
    )

    assert workload.kernel_library.startswith("cutlass_")
    assert first.profile_key == second.profile_key


def test_conv_profile_key_differs_when_conv_semantics_change():
    target = {"name": "rocm", "arch": "gfx1201"}
    ir = _conv_ir("conv2d_bias", "float16")
    manifest = build_kernel_manifest(ir, target)
    workload = build_profile_workloads(ir, manifest)[0]
    context = _fake_profile_context(workload.kernel_library)
    alternate = replace(workload, conv_config={"stride": [2, 2], "padding": [1, 1], "dilation": [1, 1], "groups": 1})

    first = _profile_cache_lookup(workload, {"target": target}, manifest, {"cache_key": "plan-a"}, context=context)
    second = _profile_cache_lookup(alternate, {"target": target}, manifest, {"cache_key": "plan-b"}, context=context)

    assert first.profile_key != second.profile_key


def _fake_profile_context(
    kernel_library: str,
    *,
    hardware_key: str = "hardware-a",
    support_fields: dict | None = None,
) -> dict:
    library = {"name": kernel_library, **(support_fields or {})}
    return {
        "fingerprint": {
            "hardware_key": hardware_key,
            "support_libraries_key": "support-all-a",
            "hardware": {},
            "support_libraries": [library],
        },
        "support_libraries_by_name": {kernel_library: library},
    }


def _gemm_ir(op_name: str, dtype: str, *, m: int = 64, n: int = 128, k: int = 96) -> dict:
    spec = gemm_op_spec(op_name)
    layout = op_name.removeprefix("gemm_").split("_", 1)[0]
    a_shape = [k, m] if layout[0] == "c" else [m, k]
    b_shape = [n, k] if layout[1] == "c" else [k, n]
    output_shape = [n, m] if layout[2] == "c" else [m, n]
    extra_shapes = {"bias": [n], "d0": output_shape, "d1": output_shape}
    epilogue_inputs = list(spec.epilogue.inputs)
    tensors = [
        _tensor("a", a_shape, dtype, "input"),
        _tensor("b", b_shape, dtype, "input"),
        *[_tensor(name, extra_shapes[name], dtype, "input") for name in epilogue_inputs],
        _tensor("c", output_shape, dtype, "output"),
    ]
    return {
        "schema_version": 1,
        "name": "profile_cache_gemm",
        "inputs": [
            _io("a", a_shape, dtype),
            _io("b", b_shape, dtype),
            *[_io(name, extra_shapes[name], dtype) for name in epilogue_inputs],
        ],
        "constants": [],
        "outputs": [_io("c", output_shape, dtype)],
        "nodes": [{"id": "n0", "op": op_name, "inputs": ["a", "b", *epilogue_inputs], "outputs": ["c"], "attrs": {}}],
        "tensors": tensors,
        "metadata": {},
    }


def _bmm_ir(op_name: str, dtype: str, *, batch: int = 2, m: int = 64, n: int = 128, k: int = 96) -> dict:
    layout = op_name.removeprefix("bmm_").removesuffix("_add")
    a_shape = [batch, k, m] if layout[0] == "c" else [batch, m, k]
    b_shape = [batch, n, k] if layout[1] == "c" else [batch, k, n]
    output_shape = [batch, n, m] if layout[2] == "c" else [batch, m, n]
    has_d0 = op_name.endswith("_add")
    tensors = [
        _tensor("a", a_shape, dtype, "input"),
        _tensor("b", b_shape, dtype, "input"),
        *([_tensor("d0", output_shape, dtype, "input")] if has_d0 else []),
        _tensor("c", output_shape, dtype, "output"),
    ]
    return {
        "schema_version": 1,
        "name": "profile_cache_bmm",
        "inputs": [
            _io("a", a_shape, dtype),
            _io("b", b_shape, dtype),
            *([_io("d0", output_shape, dtype)] if has_d0 else []),
        ],
        "constants": [],
        "outputs": [_io("c", output_shape, dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": op_name,
                "inputs": ["a", "b", *(["d0"] if has_d0 else [])],
                "outputs": ["c"],
                "attrs": {},
            }
        ],
        "tensors": tensors,
        "metadata": {},
    }


def _conv_ir(op_name: str, dtype: str) -> dict:
    tensors = [
        _tensor("x", [2, 8, 16, 16], dtype, "input"),
        _tensor("weight", [16, 8, 3, 3], dtype, "input"),
        _tensor("bias", [16], dtype, "input"),
        _tensor("y", [2, 16, 16, 16], dtype, "output"),
    ]
    return {
        "schema_version": 1,
        "name": "profile_cache_conv",
        "inputs": [
            _io("x", [2, 8, 16, 16], dtype),
            _io("weight", [16, 8, 3, 3], dtype),
            _io("bias", [16], dtype),
        ],
        "constants": [],
        "outputs": [_io("y", [2, 16, 16, 16], dtype)],
        "nodes": [
            {
                "id": "n0",
                "op": op_name,
                "inputs": ["x", "weight", "bias"],
                "outputs": ["y"],
                "attrs": {"stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 1},
            }
        ],
        "tensors": tensors,
        "metadata": {},
    }


def _io(name: str, shape: list[int], dtype: str) -> dict:
    return {
        "name": name,
        "tensor": name,
        "shape": shape,
        "shape_spec": shape,
        "layout": _dense_layout(shape),
        "dtype": dtype,
    }


def _tensor(name: str, shape: list[int], dtype: str, kind: str) -> dict:
    nbytes = 2 if dtype in {"float16", "bfloat16"} else 4
    for dim in shape:
        nbytes *= dim
    return {
        "name": name,
        "shape": shape,
        "shape_spec": shape,
        "layout": _dense_layout(shape),
        "dtype": dtype,
        "kind": kind,
        "nbytes": nbytes,
    }


def _dense_layout(shape: list[int]) -> dict:
    stride = 1
    strides = []
    for dim in reversed(shape):
        strides.insert(0, stride)
        stride *= dim
    return {
        "schema_version": 1,
        "kind": "dense",
        "order": "row_major",
        "strides": strides,
        "storage_offset": 0,
    }
