from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

import dinoml as dml
from dinoml import runtime
from dinoml.backends.target import Target
from dinoml.compiler import compile as compile_model
from dinoml.models.glm_ocr import GlmOcrTextModel
from dinoml.models.glm_ocr.workflow_common import load_glm_ocr_config, load_glm_ocr_weights

try:
    from tools.benchmark_glm_ocr_static_cache_pipeline import (
        DEFAULT_PROMPT,
        build_pipeline_inputs,
        decode_step_inputs,
        load_processor_and_inputs,
        load_stop_token_ids,
        prefill_output_shapes,
        run_pipeline_once,
        seed_dynamic_decode_cache,
        update_dynamic_decode_cache,
        validate_artifacts,
    )
except ModuleNotFoundError:
    from benchmark_glm_ocr_static_cache_pipeline import (
        DEFAULT_PROMPT,
        build_pipeline_inputs,
        decode_step_inputs,
        load_processor_and_inputs,
        load_stop_token_ids,
        prefill_output_shapes,
        run_pipeline_once,
        seed_dynamic_decode_cache,
        update_dynamic_decode_cache,
        validate_artifacts,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Localize GLM OCR decode parity drift against Transformers.")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prefill-artifact", type=Path, required=True)
    parser.add_argument("--decode-artifact", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--transformers-device", default="cuda")
    parser.add_argument("--longest-side", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--detail-layer", type=int, default=None)
    parser.add_argument("--attention-detail-layer", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args()


def artifact_target(path: Path) -> Target:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    target = manifest["target"]
    return Target(
        name=str(target["name"]),
        arch=str(target.get("arch") or ""),
        no_tf32=bool(target.get("no_tf32", False)),
        use_fp16_acc=bool(target.get("use_fp16_acc", False)),
    )


def dtype_to_float32(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.uint16:
        import torch

        return torch.from_numpy(array).view(torch.bfloat16).float().cpu().numpy()
    return array.astype(np.float32, copy=False)


def topk_tokens(logits: np.ndarray, *, k: int) -> list[dict[str, float]]:
    flat = np.asarray(logits).reshape(-1)
    count = min(k, int(flat.shape[0]))
    if count <= 0:
        return []
    indices = np.argpartition(-flat, count - 1)[:count]
    indices = indices[np.argsort(-flat[indices])]
    return [{"token_id": int(index), "logit": float(flat[index])} for index in indices]


def first_difference(a: list[int], b: list[int]) -> int | None:
    limit = min(len(a), len(b))
    for index in range(limit):
        if int(a[index]) != int(b[index]):
            return index
    if len(a) != len(b):
        return limit
    return None


def detokenize_window(processor, token_ids: list[int]) -> str:
    if not token_ids:
        return ""
    return processor.post_process_image_text_to_text(
        np.asarray([token_ids], dtype=np.int64),
        skip_special_tokens=True,
    )[0]


def run_transformers_greedy(
    *,
    snapshot: Path,
    processed_torch: dict[str, Any],
    stop_token_ids: tuple[int, ...],
    max_new_tokens: int,
    device_name: str,
    dtype_name: str,
) -> dict[str, Any]:
    import torch
    from transformers import GlmOcrForConditionalGeneration

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    device = torch.device(device_name)
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
    except TypeError:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, torch_dtype=dtype)
    model = model.to(device).eval()
    inputs = {name: value.to(device) for name, value in processed_torch.items()}

    with torch.inference_mode():
        prefill = model(
            **inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        prefill_logits = prefill.logits[:, -1:, :].detach().to(torch.float32).cpu().numpy()
        prefill_past_key_values = [
            (
                layer.keys.detach().to(torch.float32).cpu().numpy().copy(),
                layer.values.detach().to(torch.float32).cpu().numpy().copy(),
            )
            for layer in prefill.past_key_values.layers
        ]
        generated_ids: list[int] = []
        next_id = int(torch.argmax(prefill.logits[0, -1, :]).item())
        past_key_values = prefill.past_key_values
        for step in range(max_new_tokens):
            generated_ids.append(next_id)
            if next_id in stop_token_ids or step == max_new_tokens - 1:
                break
            token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            outputs = model(
                input_ids=token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())
    return {
        "prefill_logits": prefill_logits,
        "prefill_past_key_values": prefill_past_key_values,
        "generated_ids": generated_ids,
    }


class GlmOcrDecodeLayerDump(dml.nn.Module):
    def __init__(self, config, weights):
        self.config = config
        self.language_model = GlmOcrTextModel(config.text_config, weights)

    def forward(self, input_ids, cos, sin, attention_mask=None, **past_key_values):
        hidden_states = self.language_model.embed_tokens(input_ids)
        outputs: dict[str, Any] = {
            "embed_hidden": dml.ops.output(hidden_states, "embed_hidden"),
        }
        for layer_idx, layer in enumerate(self.language_model.layers):
            hidden_states, _, _ = layer.forward_with_cache(
                hidden_states,
                cos,
                sin,
                past_key_values[f"past_key_{layer_idx}"],
                past_key_values[f"past_value_{layer_idx}"],
                attention_mask,
            )
            outputs[f"layer_hidden_{layer_idx}"] = dml.ops.output(hidden_states, f"layer_hidden_{layer_idx}")
        outputs["final_hidden"] = dml.ops.output(self.language_model.norm(hidden_states), "final_hidden")
        return outputs


class GlmOcrDecodeLayerStageDump(dml.nn.Module):
    def __init__(self, config, weights, *, target_layer: int):
        self.config = config
        self.target_layer = int(target_layer)
        self.language_model = GlmOcrTextModel(config.text_config, weights)

    def forward(self, input_ids, cos, sin, attention_mask=None, **past_key_values):
        hidden_states = self.language_model.embed_tokens(input_ids)
        for layer_idx in range(self.target_layer):
            hidden_states, _, _ = self.language_model.layers[layer_idx].forward_with_cache(
                hidden_states,
                cos,
                sin,
                past_key_values[f"past_key_{layer_idx}"],
                past_key_values[f"past_value_{layer_idx}"],
                attention_mask,
            )
        layer = self.language_model.layers[self.target_layer]
        outputs: dict[str, Any] = {"layer_input": dml.ops.output(hidden_states, "layer_input")}
        residual = hidden_states
        hidden_states = layer.input_layernorm(hidden_states)
        outputs["input_layernorm"] = dml.ops.output(hidden_states, "input_layernorm")
        hidden_states, _, _ = layer.self_attn.forward_with_cache(
            hidden_states,
            cos,
            sin,
            past_key_values[f"past_key_{self.target_layer}"],
            past_key_values[f"past_value_{self.target_layer}"],
            attention_mask,
        )
        outputs["self_attn"] = dml.ops.output(hidden_states, "self_attn")
        hidden_states = layer.post_self_attn_layernorm(hidden_states)
        outputs["post_self_attn_layernorm"] = dml.ops.output(hidden_states, "post_self_attn_layernorm")
        hidden_states = dml.ops.add(residual, hidden_states)
        outputs["after_attn_residual"] = dml.ops.output(hidden_states, "after_attn_residual")
        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        outputs["post_attention_layernorm"] = dml.ops.output(hidden_states, "post_attention_layernorm")
        hidden_states = layer.mlp(hidden_states)
        outputs["mlp"] = dml.ops.output(hidden_states, "mlp")
        hidden_states = layer.post_mlp_layernorm(hidden_states)
        outputs["post_mlp_layernorm"] = dml.ops.output(hidden_states, "post_mlp_layernorm")
        hidden_states = dml.ops.add(residual, hidden_states)
        outputs["layer_output"] = dml.ops.output(hidden_states, "layer_output")
        return outputs


class GlmOcrDecodeAttentionStageDump(dml.nn.Module):
    def __init__(self, config, weights, *, target_layer: int):
        self.config = config
        self.target_layer = int(target_layer)
        self.language_model = GlmOcrTextModel(config.text_config, weights)

    def forward(self, input_ids, cos, sin, attention_mask=None, **past_key_values):
        hidden_states = self.language_model.embed_tokens(input_ids)
        for layer_idx in range(self.target_layer):
            hidden_states, _, _ = self.language_model.layers[layer_idx].forward_with_cache(
                hidden_states,
                cos,
                sin,
                past_key_values[f"past_key_{layer_idx}"],
                past_key_values[f"past_value_{layer_idx}"],
                attention_mask,
            )
        layer = self.language_model.layers[self.target_layer]
        attn = layer.self_attn
        hidden_states = layer.input_layernorm(hidden_states)
        outputs: dict[str, Any] = {"attn_input": dml.ops.output(hidden_states, "attn_input")}
        q = attn.q_proj(hidden_states)
        k = attn.k_proj(hidden_states)
        v = attn.v_proj(hidden_states)
        outputs["q_proj"] = dml.ops.output(q, "q_proj")
        outputs["k_proj"] = dml.ops.output(k, "k_proj")
        outputs["v_proj"] = dml.ops.output(v, "v_proj")
        batch, seq_len, _ = q.shape_spec
        q = dml.ops.reshape(q, [batch, seq_len, attn.config.num_attention_heads, attn.config.head_dim])
        k = dml.ops.reshape(k, [batch, seq_len, attn.config.num_key_value_heads, attn.config.head_dim])
        v = dml.ops.reshape(v, [batch, seq_len, attn.config.num_key_value_heads, attn.config.head_dim])
        q, k = dml.ops.glm_ocr_text_rope(q, k, cos, sin, attn.config.rotary_dim)
        outputs["q_rope"] = dml.ops.output(q, "q_rope")
        outputs["k_rope"] = dml.ops.output(k, "k_rope")
        outputs["v_reshape"] = dml.ops.output(v, "v_reshape")
        q = dml.ops.permute(q, (0, 2, 1, 3))
        k = dml.ops.permute(k, (0, 2, 1, 3))
        v = dml.ops.permute(v, (0, 2, 1, 3))
        present_key = dml.ops.concatenate([past_key_values[f"past_key_{self.target_layer}"], k], dim=2)
        present_value = dml.ops.concatenate([past_key_values[f"past_value_{self.target_layer}"], v], dim=2)
        attn_key = dml.ops.permute(present_key, (0, 2, 1, 3))
        attn_value = dml.ops.permute(present_value, (0, 2, 1, 3))
        q_flash = dml.ops.permute(q, (0, 2, 1, 3))
        outputs["q_flash"] = dml.ops.output(q_flash, "q_flash")
        outputs["attn_key"] = dml.ops.output(attn_key, "attn_key")
        outputs["attn_value"] = dml.ops.output(attn_value, "attn_value")
        if attention_mask is None:
            context = dml.ops.flash_attention(q_flash, attn_key, attn_value, causal=False)
        else:
            context = dml.ops.flash_attention_bias(q_flash, attn_key, attn_value, attention_mask, causal=False)
        outputs["flash_context"] = dml.ops.output(context, "flash_context")
        context = dml.ops.reshape(context, [batch, seq_len, attn.config.q_proj_size])
        attn_output = attn.o_proj(context)
        outputs["o_proj"] = dml.ops.output(attn_output, "o_proj")
        return outputs


def build_decode_debug_artifact(
    *,
    snapshot: Path,
    target: Target,
    past_len: int,
    dtype: str,
    use_attention_mask: bool,
) -> Path:
    config = load_glm_ocr_config(snapshot=snapshot, dtype=dtype)
    weights = load_glm_ocr_weights(config=config, snapshot=snapshot)
    batch = 1
    inputs: dict[str, Any] = {
        "input_ids": dml.TensorSpec([batch, 1], "int64"),
        "cos": dml.TensorSpec([batch, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": dml.TensorSpec([batch, 1, config.text_config.head_dim], config.text_config.dtype),
    }
    if use_attention_mask:
        inputs["attention_mask"] = dml.TensorSpec(
            [config.text_config.num_attention_heads, 1, past_len + 1],
            config.text_config.dtype,
        )
    for layer_idx in range(int(config.text_config.num_hidden_layers)):
        shape = [batch, config.text_config.num_key_value_heads, past_len, config.text_config.head_dim]
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec(shape, config.text_config.dtype)
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec(shape, config.text_config.dtype)
    spec = dml.trace(
        GlmOcrDecodeLayerDump(config, weights),
        inputs=inputs,
        name=f"glm_ocr_decode_layer_dump_past{past_len}",
    )
    artifact_dir = Path(tempfile.mkdtemp(prefix="glm_ocr_decode_parity_", dir=str(Path.cwd() / "build")))
    compile_model(spec, target=target, output=artifact_dir, clean=True, profile=False)
    return artifact_dir


def build_layer_stage_artifact(
    *,
    snapshot: Path,
    target: Target,
    past_len: int,
    dtype: str,
    use_attention_mask: bool,
    target_layer: int,
) -> Path:
    config = load_glm_ocr_config(snapshot=snapshot, dtype=dtype)
    weights = load_glm_ocr_weights(config=config, snapshot=snapshot)
    batch = 1
    inputs: dict[str, Any] = {
        "input_ids": dml.TensorSpec([batch, 1], "int64"),
        "cos": dml.TensorSpec([batch, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": dml.TensorSpec([batch, 1, config.text_config.head_dim], config.text_config.dtype),
    }
    if use_attention_mask:
        inputs["attention_mask"] = dml.TensorSpec(
            [config.text_config.num_attention_heads, 1, past_len + 1],
            config.text_config.dtype,
        )
    for layer_idx in range(int(config.text_config.num_hidden_layers)):
        shape = [batch, config.text_config.num_key_value_heads, past_len, config.text_config.head_dim]
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec(shape, config.text_config.dtype)
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec(shape, config.text_config.dtype)
    spec = dml.trace(
        GlmOcrDecodeLayerStageDump(config, weights, target_layer=target_layer),
        inputs=inputs,
        name=f"glm_ocr_decode_layer_{target_layer}_stage_dump_past{past_len}",
    )
    artifact_dir = Path(tempfile.mkdtemp(prefix="glm_ocr_decode_stage_parity_", dir=str(Path.cwd() / "build")))
    compile_model(spec, target=target, output=artifact_dir, clean=True, profile=False)
    return artifact_dir


def build_attention_stage_artifact(
    *,
    snapshot: Path,
    target: Target,
    past_len: int,
    dtype: str,
    use_attention_mask: bool,
    target_layer: int,
) -> Path:
    config = load_glm_ocr_config(snapshot=snapshot, dtype=dtype)
    weights = load_glm_ocr_weights(config=config, snapshot=snapshot)
    batch = 1
    inputs: dict[str, Any] = {
        "input_ids": dml.TensorSpec([batch, 1], "int64"),
        "cos": dml.TensorSpec([batch, 1, config.text_config.head_dim], config.text_config.dtype),
        "sin": dml.TensorSpec([batch, 1, config.text_config.head_dim], config.text_config.dtype),
    }
    if use_attention_mask:
        inputs["attention_mask"] = dml.TensorSpec(
            [config.text_config.num_attention_heads, 1, past_len + 1],
            config.text_config.dtype,
        )
    for layer_idx in range(int(config.text_config.num_hidden_layers)):
        shape = [batch, config.text_config.num_key_value_heads, past_len, config.text_config.head_dim]
        inputs[f"past_key_{layer_idx}"] = dml.TensorSpec(shape, config.text_config.dtype)
        inputs[f"past_value_{layer_idx}"] = dml.TensorSpec(shape, config.text_config.dtype)
    spec = dml.trace(
        GlmOcrDecodeAttentionStageDump(config, weights, target_layer=target_layer),
        inputs=inputs,
        name=f"glm_ocr_decode_attention_layer_{target_layer}_stage_dump_past{past_len}",
    )
    artifact_dir = Path(tempfile.mkdtemp(prefix="glm_ocr_decode_attn_parity_", dir=str(Path.cwd() / "build")))
    compile_model(spec, target=target, output=artifact_dir, clean=True, profile=False)
    return artifact_dir


def dinoml_state_before_step(
    *,
    prefill_session,
    decode_session,
    prefill_inputs: dict[str, np.ndarray],
    full_inputs: dict[str, np.ndarray],
    config,
    prefill_len: int,
    shared_prefix: list[int],
    use_attention_mask: bool,
) -> dict[str, Any]:
    prefill_outputs = prefill_session.run_numpy(prefill_inputs)
    cache = seed_dynamic_decode_cache(prefill_outputs)
    next_id = int(np.argmax(prefill_outputs["logits"][0, 0, :]))
    if shared_prefix and next_id != int(shared_prefix[0]):
        raise ValueError(f"Prefill next token mismatch: DinoML produced {next_id}, expected shared prefix {shared_prefix[0]}")
    for index, token in enumerate(shared_prefix[:-1]):
        if int(token) != next_id:
            raise ValueError(f"DinoML shared-prefix mismatch at token {index}: expected {token}, saw {next_id}")
        position = prefill_len + index
        step_inputs = decode_step_inputs(
            full_inputs=full_inputs,
            config=config,
            cache=cache,
            next_id=int(token),
            position=position,
            decode_mode="dynamic",
            use_attention_mask=use_attention_mask,
        )
        step_outputs = decode_session.run_numpy(step_inputs)
        cache = update_dynamic_decode_cache(step_outputs)
        next_id = int(np.argmax(step_outputs["logits"][0, 0, :]))
    if not shared_prefix:
        raise ValueError("No shared prefix exists; use prefill parity instead.")
    past_len = prefill_len + len(shared_prefix) - 1
    final_step_inputs = decode_step_inputs(
        full_inputs=full_inputs,
        config=config,
        cache=cache,
        next_id=int(shared_prefix[-1]),
        position=past_len,
        decode_mode="dynamic",
        use_attention_mask=use_attention_mask,
    )
    final_step_outputs = decode_session.run_numpy(final_step_inputs)
    return {
        "prefill_outputs": prefill_outputs,
        "past_len": past_len,
        "step_inputs": final_step_inputs,
        "step_outputs": final_step_outputs,
    }


def capture_transformers_step(
    *,
    snapshot: Path,
    processed_torch: dict[str, Any],
    shared_prefix: list[int],
    dtype_name: str,
    device_name: str,
) -> dict[str, Any]:
    import torch
    from transformers import GlmOcrForConditionalGeneration

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    device = torch.device(device_name)
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
    except TypeError:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, torch_dtype=dtype)
    model = model.to(device).eval()
    inputs = {name: value.to(device) for name, value in processed_torch.items()}

    with torch.inference_mode():
        prefill = model(
            **inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        prefill_logits = prefill.logits[:, -1:, :].detach().to(torch.float32).cpu().numpy()
        next_id = int(torch.argmax(prefill.logits[0, -1, :]).item())
        if shared_prefix and next_id != int(shared_prefix[0]):
            raise ValueError(
                f"Transformers prefill next token mismatch: produced {next_id}, expected shared prefix {shared_prefix[0]}"
            )
        past_key_values = prefill.past_key_values
        for index, token in enumerate(shared_prefix[:-1]):
            if int(token) != next_id:
                raise ValueError(f"Transformers shared-prefix mismatch at token {index}: expected {token}, saw {next_id}")
            outputs = model(
                input_ids=torch.tensor([[int(token)]], dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())

        captures: dict[str, np.ndarray] = {}
        handles = []

        def store_tensor(name: str):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captures[name] = tensor.detach().to(torch.float32).cpu().numpy()

            return hook

        handles.append(model.model.language_model.embed_tokens.register_forward_hook(store_tensor("embed_hidden")))
        for layer_idx, layer in enumerate(model.model.language_model.layers):
            handles.append(layer.register_forward_hook(store_tensor(f"layer_hidden_{layer_idx}")))
        handles.append(model.model.language_model.norm.register_forward_hook(store_tensor("final_hidden")))
        try:
            final_outputs = model(
                input_ids=torch.tensor([[int(shared_prefix[-1])]], dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        finally:
            for handle in handles:
                handle.remove()

    return {
        "prefill_logits": prefill_logits,
        "prefill_past_key_values": [
            (
                key.detach().to(torch.float32).cpu().numpy(),
                value.detach().to(torch.float32).cpu().numpy(),
            )
            for key, value in prefill.past_key_values
        ],
        "step_logits": final_outputs.logits[:, -1:, :].detach().to(torch.float32).cpu().numpy(),
        "captures": captures,
    }


def capture_transformers_layer_detail(
    *,
    snapshot: Path,
    processed_torch: dict[str, Any],
    shared_prefix: list[int],
    dtype_name: str,
    device_name: str,
    target_layer: int,
) -> dict[str, np.ndarray]:
    import torch
    from transformers import GlmOcrForConditionalGeneration

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    device = torch.device(device_name)
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
    except TypeError:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, torch_dtype=dtype)
    model = model.to(device).eval()
    inputs = {name: value.to(device) for name, value in processed_torch.items()}

    with torch.inference_mode():
        prefill = model(
            **inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        next_id = int(torch.argmax(prefill.logits[0, -1, :]).item())
        past_key_values = prefill.past_key_values
        for index, token in enumerate(shared_prefix[:-1]):
            if int(token) != next_id:
                raise ValueError(f"Transformers shared-prefix mismatch at token {index}: expected {token}, saw {next_id}")
            outputs = model(
                input_ids=torch.tensor([[int(token)]], dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())

        layer = model.model.language_model.layers[int(target_layer)]
        captures: dict[str, np.ndarray] = {}
        handles = []

        def save_output(name: str):
            def hook(_module, _inputs, output):
                tensor = output[0] if isinstance(output, tuple) else output
                captures[name] = tensor.detach().to(torch.float32).cpu().numpy()

            return hook

        def save_input(name: str):
            def hook(_module, inputs):
                tensor = inputs[0]
                captures[name] = tensor.detach().to(torch.float32).cpu().numpy()

            return hook

        handles.append(layer.input_layernorm.register_forward_pre_hook(save_input("layer_input")))
        handles.append(layer.input_layernorm.register_forward_hook(save_output("input_layernorm")))
        handles.append(layer.self_attn.register_forward_hook(save_output("self_attn")))
        handles.append(layer.post_self_attn_layernorm.register_forward_hook(save_output("post_self_attn_layernorm")))
        handles.append(layer.post_attention_layernorm.register_forward_hook(save_output("post_attention_layernorm")))
        handles.append(layer.mlp.register_forward_hook(save_output("mlp")))
        handles.append(layer.post_mlp_layernorm.register_forward_hook(save_output("post_mlp_layernorm")))
        handles.append(layer.register_forward_hook(save_output("layer_output")))
        try:
            model(
                input_ids=torch.tensor([[int(shared_prefix[-1])]], dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        finally:
            for handle in handles:
                handle.remove()

    captures["after_attn_residual"] = captures["layer_input"] + captures["post_self_attn_layernorm"]
    return captures


def capture_transformers_attention_detail(
    *,
    snapshot: Path,
    processed_torch: dict[str, Any],
    shared_prefix: list[int],
    dtype_name: str,
    device_name: str,
    target_layer: int,
) -> dict[str, np.ndarray]:
    import torch
    from transformers import GlmOcrForConditionalGeneration
    from transformers.models.glm_ocr.modeling_glm_ocr import (
        ALL_ATTENTION_FUNCTIONS,
        apply_rotary_pos_emb,
        eager_attention_forward,
    )

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype_name]
    device = torch.device(device_name)
    try:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, dtype=dtype)
    except TypeError:
        model = GlmOcrForConditionalGeneration.from_pretrained(snapshot, torch_dtype=dtype)
    model = model.to(device).eval()
    inputs = {name: value.to(device) for name, value in processed_torch.items()}

    with torch.inference_mode():
        prefill = model(
            **inputs,
            use_cache=True,
            return_dict=True,
            logits_to_keep=1,
        )
        next_id = int(torch.argmax(prefill.logits[0, -1, :]).item())
        past_key_values = prefill.past_key_values
        for index, token in enumerate(shared_prefix[:-1]):
            if int(token) != next_id:
                raise ValueError(f"Transformers shared-prefix mismatch at token {index}: expected {token}, saw {next_id}")
            outputs = model(
                input_ids=torch.tensor([[int(token)]], dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            past_key_values = outputs.past_key_values
            next_id = int(torch.argmax(outputs.logits[0, -1, :]).item())

        layer = model.model.language_model.layers[int(target_layer)]
        original_forward = layer.self_attn.forward
        captures: dict[str, np.ndarray] = {}

        def attn_forward(hidden_states, position_embeddings=None, attention_mask=None, past_key_values=None, **kwargs):
            bsz, q_len, _ = hidden_states.size()
            captures["attn_input"] = hidden_states.detach().to(torch.float32).cpu().numpy()
            query_states = layer.self_attn.q_proj(hidden_states)
            key_states = layer.self_attn.k_proj(hidden_states)
            value_states = layer.self_attn.v_proj(hidden_states)
            captures["q_proj"] = query_states.detach().to(torch.float32).cpu().numpy()
            captures["k_proj"] = key_states.detach().to(torch.float32).cpu().numpy()
            captures["v_proj"] = value_states.detach().to(torch.float32).cpu().numpy()
            query_states = query_states.view(bsz, q_len, -1, layer.self_attn.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, -1, layer.self_attn.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, -1, layer.self_attn.head_dim).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
            captures["q_rope"] = query_states.transpose(1, 2).detach().to(torch.float32).cpu().numpy()
            captures["k_rope"] = key_states.transpose(1, 2).detach().to(torch.float32).cpu().numpy()
            captures["v_reshape"] = value_states.transpose(1, 2).detach().to(torch.float32).cpu().numpy()
            if past_key_values is not None:
                key_states, value_states = past_key_values.update(key_states, value_states, layer.self_attn.layer_idx)
            captures["q_flash"] = query_states.transpose(1, 2).detach().to(torch.float32).cpu().numpy()
            captures["attn_key"] = key_states.transpose(1, 2).detach().to(torch.float32).cpu().numpy()
            captures["attn_value"] = value_states.transpose(1, 2).detach().to(torch.float32).cpu().numpy()
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                layer.self_attn.config._attn_implementation, eager_attention_forward
            )
            attn_output, attn_weights = attention_interface(
                layer.self_attn,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0,
                scaling=layer.self_attn.scaling,
                **kwargs,
            )
            captures["flash_context"] = attn_output.detach().to(torch.float32).cpu().numpy()
            attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
            attn_output = layer.self_attn.o_proj(attn_output)
            captures["o_proj"] = attn_output.detach().to(torch.float32).cpu().numpy()
            return attn_output, attn_weights

        layer.self_attn.forward = attn_forward
        try:
            model(
                input_ids=torch.tensor([[int(shared_prefix[-1])]], dtype=torch.long, device=device),
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        finally:
            layer.self_attn.forward = original_forward

    return captures


def array_diff_summary(lhs: np.ndarray, rhs: np.ndarray) -> dict[str, float]:
    lhs32 = dtype_to_float32(lhs)
    rhs32 = dtype_to_float32(rhs)
    delta = np.abs(lhs32 - rhs32)
    return {
        "max_abs": float(delta.max(initial=0.0)),
        "mean_abs": float(delta.mean()),
        "median_abs": float(np.median(delta)),
    }


def cache_layer_summaries(
    *,
    dinoml_outputs: dict[str, np.ndarray],
    transformers_past_key_values: list[tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float]]:
    summaries: list[dict[str, float]] = []
    for layer_idx, (key_ref, value_ref) in enumerate(transformers_past_key_values):
        key_summary = array_diff_summary(dinoml_outputs[f"present_key_{layer_idx}"], key_ref)
        value_summary = array_diff_summary(dinoml_outputs[f"present_value_{layer_idx}"], value_ref)
        summaries.append(
            {
                "layer": layer_idx,
                "key_max_abs": key_summary["max_abs"],
                "key_mean_abs": key_summary["mean_abs"],
                "key_median_abs": key_summary["median_abs"],
                "value_max_abs": value_summary["max_abs"],
                "value_mean_abs": value_summary["mean_abs"],
                "value_median_abs": value_summary["median_abs"],
            }
        )
    return summaries


def run_debug_capture(
    *,
    artifact_path: Path,
    step_inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    module = runtime.load(artifact_path, load_constants=True)
    session = module.create_session()
    try:
        return session.run_numpy(step_inputs)
    finally:
        session.close()
        module.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    processor, processed_torch, processed_numpy, image_size, source_image_size, processor_image_size = (
        load_processor_and_inputs(
            args.snapshot,
            args.image,
            args.prompt,
            longest_side=args.longest_side,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
        )
    )
    config = load_glm_ocr_config(snapshot=args.snapshot, dtype=args.dtype)
    prefill_inputs, full_inputs, prefill_len, max_cache_len = build_pipeline_inputs(
        config,
        processed_numpy,
        args.max_new_tokens,
    )
    artifact_modes = validate_artifacts(
        prefill_artifact=args.prefill_artifact,
        decode_artifact=args.decode_artifact,
    )
    if str(artifact_modes["decode_mode"]) != "dynamic":
        raise ValueError("This parity tool currently supports dynamic decode artifacts only.")
    stop_token_ids = load_stop_token_ids(args.snapshot, processor)

    prefill_module = runtime.load(args.prefill_artifact, load_constants=True)
    decode_module = runtime.load(args.decode_artifact, load_constants=True)
    prefill_session = prefill_module.create_session()
    decode_session = decode_module.create_session()
    try:
        dinoml_prefill_outputs = prefill_session.run_numpy(prefill_inputs)
        dinoml_generated_ids: list[int]
        dinoml_generated_ids, _, _ = run_pipeline_once(
            prefill_session,
            decode_session,
            prefill_inputs=prefill_inputs,
            full_inputs=full_inputs,
            config=config,
            decode_mode="dynamic",
            use_decode_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
            prefill_len=prefill_len,
            max_cache_len=max_cache_len,
            max_new_tokens=args.max_new_tokens,
            stop_token_ids=stop_token_ids,
        )
    finally:
        decode_session.close()
        prefill_session.close()
        decode_module.close()
        prefill_module.close()

    transformers_run = run_transformers_greedy(
        snapshot=args.snapshot,
        processed_torch=processed_torch,
        stop_token_ids=stop_token_ids,
        max_new_tokens=args.max_new_tokens,
        device_name=args.transformers_device,
        dtype_name=args.dtype,
    )
    transformers_generated_ids = transformers_run["generated_ids"]
    diff_index = first_difference(dinoml_generated_ids, transformers_generated_ids)
    shared_prefix = (
        dinoml_generated_ids[:] if diff_index is None else dinoml_generated_ids[:diff_index]
    )

    prefill_diff = array_diff_summary(dinoml_prefill_outputs["logits"], transformers_run["prefill_logits"])
    prefill_cache_summaries = cache_layer_summaries(
        dinoml_outputs=dinoml_prefill_outputs,
        transformers_past_key_values=transformers_run["prefill_past_key_values"],
    )
    payload: dict[str, Any] = {
        "snapshot": str(args.snapshot),
        "image": str(args.image),
        "source_image_size": list(source_image_size),
        "image_size": list(image_size),
        "processor_image_size": processor_image_size,
        "prefill_artifact": str(args.prefill_artifact),
        "decode_artifact": str(args.decode_artifact),
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "stop_token_ids": list(stop_token_ids),
        "prefill_len": prefill_len,
        "prefill_parity": {
            "dinoml_next_token": int(np.argmax(dinoml_prefill_outputs["logits"][0, 0, :])),
            "transformers_next_token": int(np.argmax(transformers_run["prefill_logits"][0, 0, :])),
            "logit_diff": prefill_diff,
            "dinoml_top10": topk_tokens(dtype_to_float32(dinoml_prefill_outputs["logits"][0, 0, :]), k=10),
            "transformers_top10": topk_tokens(transformers_run["prefill_logits"][0, 0, :], k=10),
            "cache_first_layer": prefill_cache_summaries[0],
            "cache_middle_layer": prefill_cache_summaries[len(prefill_cache_summaries) // 2],
            "cache_last_layer": prefill_cache_summaries[-1],
            "cache_max_key_layer": max(prefill_cache_summaries, key=lambda item: item["key_max_abs"]),
            "cache_max_value_layer": max(prefill_cache_summaries, key=lambda item: item["value_max_abs"]),
        },
        "generation": {
            "dinoml_generated_tokens": len(dinoml_generated_ids),
            "transformers_generated_tokens": len(transformers_generated_ids),
            "first_difference_index": diff_index,
            "shared_prefix_len": len(shared_prefix),
            "dinoml_text": detokenize_window(processor, dinoml_generated_ids),
            "transformers_text": detokenize_window(processor, transformers_generated_ids),
        },
    }

    if diff_index is None:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        if args.output_json is not None:
            args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return

    window_start = max(0, diff_index - 16)
    window_end = min(max(len(dinoml_generated_ids), len(transformers_generated_ids)), diff_index + 16)
    payload["generation"]["divergence_window"] = {
        "start": window_start,
        "end": window_end,
        "dinoml_ids": dinoml_generated_ids[window_start:window_end],
        "transformers_ids": transformers_generated_ids[window_start:window_end],
        "dinoml_text": detokenize_window(processor, dinoml_generated_ids[window_start:window_end]),
        "transformers_text": detokenize_window(processor, transformers_generated_ids[window_start:window_end]),
    }

    prefill_module = runtime.load(args.prefill_artifact, load_constants=True)
    decode_module = runtime.load(args.decode_artifact, load_constants=True)
    prefill_session = prefill_module.create_session()
    decode_session = decode_module.create_session()
    try:
        dinoml_step = dinoml_state_before_step(
            prefill_session=prefill_session,
            decode_session=decode_session,
            prefill_inputs=prefill_inputs,
            full_inputs=full_inputs,
            config=config,
            prefill_len=prefill_len,
            shared_prefix=shared_prefix,
            use_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
        )
    finally:
        decode_session.close()
        prefill_session.close()
        decode_module.close()
        prefill_module.close()

    transformers_step = capture_transformers_step(
        snapshot=args.snapshot,
        processed_torch=processed_torch,
        shared_prefix=shared_prefix,
        dtype_name=args.dtype,
        device_name=args.transformers_device,
    )

    debug_artifact = build_decode_debug_artifact(
        snapshot=args.snapshot,
        target=artifact_target(args.decode_artifact),
        past_len=int(dinoml_step["past_len"]),
        dtype=args.dtype,
        use_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
    )
    try:
        dinoml_layer_outputs = run_debug_capture(
            artifact_path=debug_artifact,
            step_inputs=dinoml_step["step_inputs"],
        )
    finally:
        # Keep the artifact directory content available if the user wants to inspect it later.
        pass

    logits_diff = array_diff_summary(dinoml_step["step_outputs"]["logits"], transformers_step["step_logits"])
    logits_candidates = {
        token_id
        for token_id in (
            int(np.argmax(dinoml_step["step_outputs"]["logits"][0, 0, :])),
            int(np.argmax(transformers_step["step_logits"][0, 0, :])),
        )
    }
    layer_summaries = []
    for name in ["embed_hidden", *[f"layer_hidden_{i}" for i in range(int(config.text_config.num_hidden_layers))], "final_hidden"]:
        summary = array_diff_summary(dinoml_layer_outputs[name], transformers_step["captures"][name])
        summary["name"] = name
        layer_summaries.append(summary)
    payload["teacher_forced_decode"] = {
        "shared_prefix_last_token": int(shared_prefix[-1]),
        "differing_token_index": int(diff_index),
        "past_len_before_step": int(dinoml_step["past_len"]),
        "dinoml_argmax": int(np.argmax(dinoml_step["step_outputs"]["logits"][0, 0, :])),
        "transformers_argmax": int(np.argmax(transformers_step["step_logits"][0, 0, :])),
        "logit_diff": logits_diff,
        "candidate_logits": [
            {
                "token_id": int(token_id),
                "dinoml_logit": float(dtype_to_float32(dinoml_step["step_outputs"]["logits"][0, 0, :])[int(token_id)]),
                "transformers_logit": float(transformers_step["step_logits"][0, 0, int(token_id)]),
            }
            for token_id in sorted(logits_candidates)
        ],
        "layer_hidden_diffs": layer_summaries,
        "largest_hidden_diff": max(layer_summaries, key=lambda item: item["max_abs"]),
        "debug_artifact": str(debug_artifact),
    }

    if args.detail_layer is not None:
        detail_layer = int(args.detail_layer)
        detail_artifact = build_layer_stage_artifact(
            snapshot=args.snapshot,
            target=artifact_target(args.decode_artifact),
            past_len=int(dinoml_step["past_len"]),
            dtype=args.dtype,
            use_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
            target_layer=detail_layer,
        )
        dinoml_detail_outputs = run_debug_capture(
            artifact_path=detail_artifact,
            step_inputs=dinoml_step["step_inputs"],
        )
        transformers_detail = capture_transformers_layer_detail(
            snapshot=args.snapshot,
            processed_torch=processed_torch,
            shared_prefix=shared_prefix,
            dtype_name=args.dtype,
            device_name=args.transformers_device,
            target_layer=detail_layer,
        )
        stage_names = [
            "layer_input",
            "input_layernorm",
            "self_attn",
            "post_self_attn_layernorm",
            "after_attn_residual",
            "post_attention_layernorm",
            "mlp",
            "post_mlp_layernorm",
            "layer_output",
        ]
        payload["teacher_forced_decode"]["detail_layer"] = {
            "layer": detail_layer,
            "debug_artifact": str(detail_artifact),
            "stage_diffs": [
                {"name": name, **array_diff_summary(dinoml_detail_outputs[name], transformers_detail[name])}
                for name in stage_names
            ],
        }
    if args.attention_detail_layer is not None:
        detail_layer = int(args.attention_detail_layer)
        attention_artifact = build_attention_stage_artifact(
            snapshot=args.snapshot,
            target=artifact_target(args.decode_artifact),
            past_len=int(dinoml_step["past_len"]),
            dtype=args.dtype,
            use_attention_mask=bool(artifact_modes["use_decode_attention_mask"]),
            target_layer=detail_layer,
        )
        dinoml_attention_outputs = run_debug_capture(
            artifact_path=attention_artifact,
            step_inputs=dinoml_step["step_inputs"],
        )
        transformers_attention = capture_transformers_attention_detail(
            snapshot=args.snapshot,
            processed_torch=processed_torch,
            shared_prefix=shared_prefix,
            dtype_name=args.dtype,
            device_name=args.transformers_device,
            target_layer=detail_layer,
        )
        attention_stage_names = [
            "attn_input",
            "q_proj",
            "k_proj",
            "v_proj",
            "q_rope",
            "k_rope",
            "v_reshape",
            "q_flash",
            "attn_key",
            "attn_value",
            "flash_context",
            "o_proj",
        ]
        payload["teacher_forced_decode"]["attention_detail_layer"] = {
            "layer": detail_layer,
            "debug_artifact": str(attention_artifact),
            "stage_diffs": [
                {"name": name, **array_diff_summary(dinoml_attention_outputs[name], transformers_attention[name])}
                for name in attention_stage_names
            ],
        }

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
