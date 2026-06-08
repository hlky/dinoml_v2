from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from tools import benchmark_qwen2_5_vl_static_cache_pipeline as pipeline_tool
from tools import qwen2_5_vl_benchmark_common as qwen_common


def _config(*, dtype: str = "float16", num_layers: int = 1):
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_layers,
            num_key_value_heads=2,
            num_attention_heads=8,
            head_dim=4,
            dtype=dtype,
            mask_fill_value=-1.0e4,
        ),
        vision_config=SimpleNamespace(dtype=dtype),
    )


def _write_artifact_metadata(path, *, inputs: list[str], outputs: list[str], target: str = "rocm") -> None:
    path.mkdir()
    (path / "manifest.json").write_text(
        json.dumps(
            {
                "runtime_abi_version": 1,
                "target": {"name": target},
                "files": {
                    "module": "module.so",
                    "metadata": "metadata.json",
                    "runtime_library": "lib/libdinoml_runtime.so",
                    "kernel_library": "lib/libdinoml_cuda_kernels.so",
                },
            }
        ),
        encoding="utf-8",
    )
    (path / "metadata.json").write_text(
        json.dumps(
            {
                "inputs": [{"name": name} for name in inputs],
                "outputs": [{"name": name} for name in outputs],
            }
        ),
        encoding="utf-8",
    )
    (path / "module.so").write_bytes(b"module")
    (path / "lib").mkdir()
    (path / "lib" / "libdinoml_runtime.so").write_bytes(b"runtime")
    (path / "lib" / "libdinoml_cuda_kernels.so").write_bytes(b"kernel")


def test_static_cache_pipeline_defaults_use_qwen_real_fixture_paths():
    assert pipeline_tool.DEFAULT_IMAGE == qwen_common.DEFAULT_IMAGE
    assert pipeline_tool.DEFAULT_PROMPT == qwen_common.DEFAULT_PROMPT
    assert pipeline_tool.DEFAULT_SNAPSHOT == qwen_common.DEFAULT_SNAPSHOT
    assert str(pipeline_tool.DEFAULT_SNAPSHOT).endswith(r"Qwen2.5-VL-3B-Instruct")
    assert str(pipeline_tool.DEFAULT_IMAGE).endswith("0000017352_0001.jpg")


def test_transformers_reference_forwards_mm_token_type_ids():
    assert "mm_token_type_ids" in pipeline_tool.TRANSFORMERS_REFERENCE_INPUTS


def test_transformers_reference_reports_generation_time(monkeypatch, tmp_path):
    class FakeTensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        @property
        def shape(self):
            return self.value.shape

        @property
        def dtype(self):
            return self.value.dtype

        def __getitem__(self, key):
            return FakeTensor(self.value[key])

        def to(self, _device):
            return self

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

        def tolist(self):
            return self.value.tolist()

    class FakeInferenceMode:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

    class FakeTorch:
        bfloat16 = "bfloat16"
        float16 = "float16"
        float32 = "float32"
        cuda = FakeCuda

        @staticmethod
        def device(name):
            return name

        @staticmethod
        def inference_mode():
            return FakeInferenceMode()

        @staticmethod
        def argmax(value):
            return SimpleNamespace(item=lambda: int(np.argmax(value.value)))

    class FakeModel:
        def to(self, _device):
            return self

        def eval(self):
            return None

        def __call__(self, *, input_ids, attention_mask=None, past_key_values=None, position_ids=None, **_kwargs):
            del attention_mask, past_key_values, position_ids
            return SimpleNamespace(
                logits=FakeTensor([[[0.0, 0.0, 1.0, 3.0]]]),
                past_key_values="past",
            )

        def generate(self, *, input_ids, **_kwargs):
            prompt = np.asarray(input_ids.value)
            sequences = np.concatenate([prompt, np.asarray([[3, 1]], dtype=prompt.dtype)], axis=1)
            return SimpleNamespace(
                sequences=FakeTensor(sequences),
                logits=(
                    FakeTensor([[0.0, 0.0, 1.0, 3.0]]),
                    FakeTensor([[0.0, 5.0, 0.0, 0.0]]),
                ),
            )

    class FakeQwen:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return FakeModel()

    fake_processor = SimpleNamespace(
        tokenizer=SimpleNamespace(pad_token_id=0),
        batch_decode=lambda ids, skip_special_tokens, clean_up_tokenization_spaces: ["decoded"],
    )
    fake_transformers = SimpleNamespace(Qwen2_5_VLForConditionalGeneration=FakeQwen)
    ticks = iter([10.0, 10.125])

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(pipeline_tool.time, "perf_counter", lambda: next(ticks))

    result = pipeline_tool.run_transformers_reference(
        snapshot=tmp_path,
        processor=fake_processor,
        processed_torch={
            "input_ids": FakeTensor([[10, 11]]),
            "attention_mask": FakeTensor([[1, 1]]),
            "pixel_values": FakeTensor([[1.0]]),
            "mm_token_type_ids": FakeTensor([[0, 1]]),
        },
        dtype="bfloat16",
        device_name="cuda",
        max_new_tokens=2,
        stop_token_ids=(),
    )

    assert result["generated_token_count"] == 2
    assert result["generated_ids"] == [3, 1]
    assert result["generation_time_ms"] == pytest.approx(125.0)
    assert result["prefill_output_preview"]["argmax_token_id"] == 3
    assert result["first_decode_output_preview"]["argmax_token_id"] == 1


def test_release_gpu_memory_before_transformers_reference_tolerates_ipc_cleanup_failure(monkeypatch):
    class FakeCuda:
        empty_cache_called = False

        @staticmethod
        def is_available():
            return True

        @classmethod
        def empty_cache(cls):
            cls.empty_cache_called = True

        @staticmethod
        def ipc_collect():
            raise RuntimeError("ipc cleanup unavailable")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=FakeCuda))

    pipeline_tool.release_gpu_memory_before_transformers_reference()

    assert FakeCuda.empty_cache_called is True


def test_qwen_real_image_benchmark_longest_side_resize_preserves_aspect_ratio():
    image = Image.new("RGB", (2496, 3150))

    resized = qwen_common.resize_image_longest_side(image, 1024)

    assert resized.size == (811, 1024)


def test_qwen_real_image_benchmark_longest_side_resize_rejects_non_positive_value():
    image = Image.new("RGB", (10, 20))

    with pytest.raises(ValueError, match="--longest-side must be positive"):
        qwen_common.resize_image_longest_side(image, 0)


@pytest.mark.parametrize(
    ("decode_outputs", "expected_mode"),
    (
        (["logits", "new_key_0", "new_value_0"], "static"),
        (["logits", "present_key_0", "present_value_0"], "dynamic"),
    ),
)
def test_validate_artifacts_accepts_qwen_prefill_grid_contract(tmp_path, decode_outputs, expected_mode):
    prefill = tmp_path / "prefill.dinoml"
    decode = tmp_path / "decode.dinoml"
    _write_artifact_metadata(
        prefill,
        inputs=[
            "input_ids",
            "pixel_values",
            "image_grid_thw",
            "vision_cos",
            "vision_sin",
            "vision_full_cu_seqlens",
            "vision_window_cu_seqlens",
            "vision_reverse_window_index",
            "text_cos",
            "text_sin",
            "attention_mask",
        ],
        outputs=["logits", "present_key_0", "present_value_0"],
    )
    _write_artifact_metadata(
        decode,
        inputs=(
            ["input_ids", "cos", "sin", "attention_mask", "cache_seqlens", "past_key_0", "past_value_0"]
            if expected_mode == "static"
            else ["input_ids", "cos", "sin", "attention_mask", "past_key_0", "past_value_0"]
        ),
        outputs=decode_outputs,
    )

    result = pipeline_tool.validate_artifacts(prefill_artifact=prefill, decode_artifact=decode)

    assert result["status"] == "validated"
    assert result["compiled"] is True
    assert result["decode_mode"] == expected_mode
    assert result["use_decode_attention_mask"] is True
    assert result["prefill"]["target"] == "rocm"


def test_validate_artifacts_rejects_prefill_without_image_grid(tmp_path):
    prefill = tmp_path / "prefill.dinoml"
    decode = tmp_path / "decode.dinoml"
    _write_artifact_metadata(
        prefill,
        inputs=["input_ids", "pixel_values", "vision_cos", "vision_sin", "text_cos", "text_sin", "attention_mask"],
        outputs=["logits", "present_key_0", "present_value_0"],
    )
    _write_artifact_metadata(
        decode,
        inputs=["input_ids", "cos", "sin", "attention_mask", "cache_seqlens", "past_key_0", "past_value_0"],
        outputs=["logits", "new_key_0", "new_value_0"],
    )

    with pytest.raises(ValueError, match="image_grid_thw"):
        pipeline_tool.validate_artifacts(prefill_artifact=prefill, decode_artifact=decode)


def test_validate_artifacts_rejects_missing_manifest_files(tmp_path):
    prefill = tmp_path / "prefill.dinoml"
    decode = tmp_path / "decode.dinoml"
    _write_artifact_metadata(
        prefill,
        inputs=[
            "input_ids",
            "pixel_values",
            "image_grid_thw",
            "vision_cos",
            "vision_sin",
            "vision_full_cu_seqlens",
            "vision_window_cu_seqlens",
            "vision_reverse_window_index",
            "text_cos",
            "text_sin",
            "attention_mask",
        ],
        outputs=["logits", "present_key_0", "present_value_0"],
    )
    _write_artifact_metadata(
        decode,
        inputs=["input_ids", "cos", "sin", "attention_mask", "cache_seqlens", "past_key_0", "past_value_0"],
        outputs=["logits", "new_key_0", "new_value_0"],
    )
    (prefill / "module.so").unlink()

    with pytest.raises(FileNotFoundError, match="manifest-declared files"):
        pipeline_tool.validate_artifacts(prefill_artifact=prefill, decode_artifact=decode)


def test_validate_artifacts_rejects_non_gpu_targets(tmp_path):
    prefill = tmp_path / "prefill.dinoml"
    decode = tmp_path / "decode.dinoml"
    _write_artifact_metadata(
        prefill,
        inputs=[
            "input_ids",
            "pixel_values",
            "image_grid_thw",
            "vision_cos",
            "vision_sin",
            "vision_full_cu_seqlens",
            "vision_window_cu_seqlens",
            "vision_reverse_window_index",
            "text_cos",
            "text_sin",
            "attention_mask",
        ],
        outputs=["logits", "present_key_0", "present_value_0"],
        target="cpu",
    )
    _write_artifact_metadata(
        decode,
        inputs=["input_ids", "cos", "sin", "attention_mask", "cache_seqlens", "past_key_0", "past_value_0"],
        outputs=["logits", "new_key_0", "new_value_0"],
        target="cpu",
    )

    with pytest.raises(ValueError, match="GPU backend"):
        pipeline_tool.validate_artifacts(prefill_artifact=prefill, decode_artifact=decode)


def test_compare_generated_outputs_reports_first_qwen_mismatch():
    result = pipeline_tool.compare_generated_outputs(
        {"status": "ok", "generated_ids": [1, 2, 3], "text": "abc"},
        {"status": "ok", "generated_ids": [1, 4, 3], "text": "axc"},
    )

    assert result == {
        "status": "mismatch",
        "generated_ids_match": False,
        "text_matches": False,
        "dinoml_generated_token_count": 3,
        "transformers_generated_token_count": 3,
        "first_mismatch_index": 1,
        "dinoml_token": 2,
        "transformers_token": 4,
    }


def test_build_verification_summary_requires_prefill_and_first_decode_parity():
    verification = pipeline_tool.build_verification_summary(
        artifacts={"status": "validated", "compiled": True, "prefill": {"target": "rocm"}},
        dinoml={
            "status": "ok",
            "execution_mode": "device_pointers",
            "generated_ids": [1, 2],
            "text": "same",
            "_prefill_logits": np.asarray([[[1.0, 2.0]]], dtype=np.float32),
            "_first_decode_logits": np.asarray([[[3.0, 4.0]]], dtype=np.float32),
        },
        transformers={
            "status": "ok",
            "generated_ids": [1, 2],
            "text": "same",
            "_prefill_logits": np.asarray([[[1.5, 2.5]]], dtype=np.float32),
            "_first_decode_logits": np.asarray([[[3.0, 4.0]]], dtype=np.float32),
        },
    )

    assert verification["artifact_compilation"]["status"] == "ok"
    assert verification["artifact_execution"]["status"] == "ok"
    assert verification["prefill_parity"]["status"] == "mismatch"
    assert verification["first_decode_step_parity"]["status"] == "ok"
    assert verification["final_generated_output_parity"]["status"] == "ok"
    assert verification["acceptance_met"] is False


def test_build_verification_summary_distinguishes_not_requested_and_invalid_artifacts():
    not_requested = pipeline_tool.build_verification_summary(
        artifacts={"status": "not_requested", "requested": False, "compiled": False, "executed": False},
        dinoml={"status": "not_requested"},
        transformers={"status": "not_requested"},
    )
    invalid = pipeline_tool.build_verification_summary(
        artifacts={
            "status": "invalid",
            "requested": True,
            "compiled": False,
            "executed": False,
            "error_type": "ValueError",
            "error": "bad artifact",
        },
        dinoml={"status": "not_run", "reason": "artifacts_invalid"},
        transformers={"status": "not_requested"},
    )

    assert not_requested["artifact_compilation"]["status"] == "not_requested"
    assert not_requested["artifact_execution"]["status"] == "not_requested"
    assert invalid["artifact_compilation"] == {
        "status": "invalid",
        "error_type": "ValueError",
        "error": "bad artifact",
    }
    assert invalid["artifact_execution"]["status"] == "not_run"
    assert invalid["acceptance_met"] is False


def test_compare_saved_runs_reloads_probe_arrays_and_checks_acceptance_surface(tmp_path):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    dinoml_json = build_dir / "dinoml.json"
    transformers_json = build_dir / "transformers.json"
    dinoml_probe_path = dinoml_json.with_suffix(".probes.npz")
    transformers_probe_path = transformers_json.with_suffix(".probes.npz")
    dinoml_json.write_text(
        json.dumps(
            {
                "artifacts": {
                    "status": "validated",
                    "requested": True,
                    "compiled": True,
                    "executed": True,
                    "prefill": {"target": "rocm"},
                },
                "dinoml": {
                    "status": "ok",
                    "execution_mode": "device_pointers_sequential_modules",
                    "generated_ids": [7, 8],
                    "text": "same",
                },
                "probe_arrays": {"path": str(Path("build") / dinoml_probe_path.name)},
            }
        ),
        encoding="utf-8",
    )
    transformers_json.write_text(
        json.dumps(
            {
                "transformers": {
                    "status": "ok",
                    "generated_ids": [7, 8],
                    "text": "same",
                },
                "probe_arrays": {"path": str(Path("build") / transformers_probe_path.name)},
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(
        dinoml_probe_path,
        dinoml_prefill_logits=np.asarray([[[1.0, 2.0]]], dtype=np.float32),
        dinoml_first_decode_logits=np.asarray([[[3.0, 4.0]]], dtype=np.float32),
    )
    np.savez_compressed(
        transformers_probe_path,
        transformers_prefill_logits=np.asarray([[[1.0, 2.0]]], dtype=np.float32),
        transformers_first_decode_logits=np.asarray([[[3.5, 4.5]]], dtype=np.float32),
    )

    result = pipeline_tool.compare_saved_runs(dinoml_json, transformers_json)

    assert result["status"] == "ok"
    assert result["verification"]["artifact_compilation"]["status"] == "ok"
    assert result["verification"]["artifact_execution"]["status"] == "ok"
    assert result["verification"]["prefill_parity"]["status"] == "ok"
    assert result["verification"]["first_decode_step_parity"]["status"] == "mismatch"
    assert result["verification"]["final_generated_output_parity"]["status"] == "ok"
    assert result["verification"]["acceptance_met"] is False


def test_run_artifacts_unloads_prefill_before_decode_device_stage(monkeypatch, tmp_path):
    class FakeSession:
        def __init__(self, stage):
            self.stage = stage

        def close(self):
            events.append(f"{self.stage}:session_close")
            return None

    class FakeModule:
        target_name = "rocm"

        def __init__(self, stage):
            self.stage = stage

        def create_session(self):
            events.append(f"{self.stage}:session_create")
            return FakeSession(self.stage)

        def close(self):
            events.append(f"{self.stage}:module_close")
            return None

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def empty_cache():
            return None

        @staticmethod
        def ipc_collect():
            return None

    class FakeTorch:
        cuda = FakeCuda

        @staticmethod
        def device(name):
            return name

    events: list[str] = []
    calls = {"prefill": 0, "decode": 0}

    def fake_load(path, *, load_constants):
        assert load_constants is True
        stage = "prefill" if "prefill" in path.name else "decode"
        events.append(f"{stage}:load")
        return FakeModule(stage)

    def fake_prepare_device_pipeline(**_kwargs):
        events.append("prepare")
        return {"prepared": True}

    def fake_run_prefill_device(session, **_kwargs):
        assert session.stage == "prefill"
        calls["prefill"] += 1
        events.append("prefill:run")
        return 10, {
            "logits_shape": [1, 1, 16],
            "argmax_token_id": 10,
            "logits": np.asarray([[[1.0, 2.0]]], dtype=np.float32),
        }

    def fake_run_decode_device(session, **_kwargs):
        assert session.stage == "decode"
        calls["decode"] += 1
        events.append("decode:run")
        return [10, 11], [1.0], {
            "status": "ok",
            "input_token_id": 10,
            "argmax_token_id": 11,
            "logits_shape": [1, 1, 16],
            "logits": np.asarray([[[3.0, 4.0]]], dtype=np.float32),
        }

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(pipeline_tool.runtime, "load", fake_load)
    monkeypatch.setattr(pipeline_tool, "prepare_device_pipeline", fake_prepare_device_pipeline)
    monkeypatch.setattr(pipeline_tool, "run_prefill_device", fake_run_prefill_device)
    monkeypatch.setattr(pipeline_tool, "run_decode_device", fake_run_decode_device)

    processor = SimpleNamespace(
        batch_decode=lambda ids, skip_special_tokens, clean_up_tokenization_spaces: ["decoded"]
    )
    result = pipeline_tool.run_artifacts(
        prefill_artifact=tmp_path / "prefill.dinoml",
        decode_artifact=tmp_path / "decode.dinoml",
        artifact_modes={"decode_mode": "static", "use_decode_attention_mask": True},
        processor=processor,
        prefill_inputs={},
        full_inputs={},
        config=_config(),
        prefill_len=2,
        max_cache_len=4,
        max_new_tokens=2,
        stop_token_ids=(),
        warmup=1,
        iterations=1,
    )

    assert result["execution_mode"] == "device_pointers_sequential_modules"
    assert result["module_residency"] == "prefill_unloaded_before_decode"
    assert result["generated_ids"] == [10, 11]
    assert result["decode_step_runs_ms"] == [[1.0]]
    assert result["prefill_output_preview"] == {"logits_shape": [1, 1, 16], "argmax_token_id": 10}
    assert result["first_decode_output_preview"] == {
        "status": "ok",
        "input_token_id": 10,
        "argmax_token_id": 11,
        "logits_shape": [1, 1, 16],
    }
    assert calls == {"prefill": 2, "decode": 2}
    first_decode_load = events.index("decode:load")
    assert events.index("prefill:module_close") < first_decode_load
