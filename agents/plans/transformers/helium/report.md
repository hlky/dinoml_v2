# Transformers Family Audit: helium

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: kyutai/helium-1-preview and kyutai/helium-1-preview-2b
Config source: public Hugging Face raw config.json plus HeliumConfig defaults
Source files inspected:
- X:/H/transformers/src/transformers/models/helium/configuration_helium.py
- X:/H/transformers/src/transformers/models/helium/modeling_helium.py
- X:/H/transformers/src/transformers/models/helium/modular_helium.py
- X:/H/transformers/src/transformers/models/llama/modeling_llama.py
- X:/H/transformers/src/transformers/models/granite/modeling_granite.py
- X:/H/transformers/src/transformers/masking_utils.py
- X:/H/transformers/src/transformers/modeling_rope_utils.py
- X:/H/transformers/tests/models/helium/test_modeling_helium.py
Any missing files or assumptions: no processor/image/audio files exist for this family. The generated modeling_helium.py is the runtime file; modular_helium.py is authoritative for future Transformers source edits.
```

Small source/config snapshot: `config_sweep.json` in this folder.

Hub notes: `kyutai/helium-1-preview-2b` is public, ungated, and has Hub sha `645c94758493d06ba1b4706c5674d6510e3a84cb`. Newer Kyutai repos named `helium-1-2b*` are public but declare `model_type: llama` and `architectures: LlamaForCausalLM`; they should be audited under Llama, not this Helium source.

## 2. High-level architecture

Text-only decoder-only causal language model.

```text
tokenizer/input_ids + attention_mask -> token embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
```

The first useful DinoML target is `HeliumForCausalLM`: prefill and autoregressive decode with self-attention KV cache. `HeliumModel` is the base decoder. Sequence and token classification heads are implemented through generic Transformers helper classes and are optional/deferred for the causal-LM target.

No CPU/GPU multimodal stage split is present. Tokenization and padding are CPU/data-pipeline work. GPU/runtime stages are embedding lookup, causal mask handling, RoPE, decoder blocks, final norm, and logits.

## 3. Important config dimensions

| Field | In-scope Helium value | Source |
|---|---:|---|
| `vocab_size` | 48000 | checkpoint config |
| `hidden_size` | 2560 | checkpoint config |
| `intermediate_size` | 7040 | checkpoint config |
| `num_hidden_layers` | 24 | checkpoint config |
| `num_attention_heads` | 20 | checkpoint config |
| `num_key_value_heads` | 20 | checkpoint config |
| `head_dim` | 128 | checkpoint config |
| attention output width | 2560 | `num_attention_heads * head_dim` |
| KV projection width | 2560 | `num_key_value_heads * head_dim` |
| `max_position_embeddings` | 4096 | checkpoint config |
| RoPE theta | 100000.0 | checkpoint config legacy `rope_theta`; effective `rope_parameters.rope_theta` |
| RoPE type | `default` | HeliumConfig / rope defaults |
| activation | `silu` | checkpoint config |
| norm epsilon | `1e-8` | checkpoint config |
| attention bias | false | checkpoint config |
| MLP bias | false | checkpoint config |
| lm head bias | false | source |
| dtype | bfloat16 | checkpoint config metadata |
| cache support | true | checkpoint config/source |
| tied embeddings | false | checkpoint config; source still declares possible tied key alias |

Representative config sweep:

| Model | Scope | model_type | Layers | Hidden | Heads/KV | MLP | Vocab | RoPE theta | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| `kyutai/helium-1-preview` | in | helium | 24 | 2560 | 20/20 | 7040 | 48000 | 100000 | Test integration uses this id with revision `refs/pr/1`. |
| `kyutai/helium-1-preview-2b` | in | helium | 24 | 2560 | 20/20 | 7040 | 48000 | 100000 | Public current repo; same config as preview. |
| `kyutai/helium-1-2b` | out | llama | 28 | 2048 | 16/8 | 8192 | 64000 | 20000 | Helium-branded but Llama source. |
| `kyutai/helium-1-2b-books/science/stem/main/wiki` | out | llama | 28 | 2048 | 16/8 | 8192 | 64000 | 20000 | Same Llama-family structure. |

Checkpoint config omits `pad_token_id` and `rope_parameters`; HeliumConfig supplies `pad_token_id=3` and the RoPE config machinery supplies effective default RoPE with theta from `rope_theta` or `default_theta=100000.0`.

## 3a. Family variation traps

- Only `model_type: helium` checkpoints route to this source. Kyutai `helium-1-2b*` repos are Llama checkpoints with GQA and different tokenizer/vocab.
- Helium source supports `head_dim` explicitly. Do not infer projection widths from `hidden_size` alone; assert `o_proj` input width equals `hidden_size` for current checkpoints because `20 * 128 == 2560`.
- `num_key_value_heads` can be less than `num_attention_heads` by config, because source implements `repeat_kv`; the representative Helium checkpoint is MHA (`20/20`).
- RoPE application is not the ordinary half-split Llama form. Helium rotates even/odd pairs and repeats cos/sin interleaved before applying them.
- `attention_bias` and `mlp_bias` are config fields. Current checkpoint disables both, but loaders/lowering should keep guarded bias paths.
- `attention_dropout` exists but inference uses `0.0`.
- No sliding-window, ALiBi, MoE, cross-attention, multimodal placeholder scatter, packed projections, or quantized weight metadata are read by Helium source.
- The tokenizer may include `<|im_*|>` added tokens, but the model source treats all tokens as ordinary text IDs.
- No NCHW/NHWC or NCDHW layout translation applies; all runtime tensors are rank-2 token IDs/masks or rank-3/4 sequence/head tensors.

## 4. Operator coverage checklist

Tensor/layout ops:
- `Embedding(input_ids [B,S] -> [B,S,2560])`.
- `view/reshape [B,S,H*D] -> [B,S,heads,D]`, `transpose(1,2)`, `contiguous`, `slice` for `logits_to_keep`.
- Elementwise add for residuals and mask addition.
- Optional attention mask preparation from 2D padding mask and position IDs.

Neural primitives:
- RMSNorm over last dim with fp32 variance and fp32 scale multiply, output cast to input dtype.
- Linear projections: Q `2560 -> 2560`, K `2560 -> 2560`, V `2560 -> 2560`, O `2560 -> 2560`, no bias in current checkpoint.
- Gated MLP: gate `2560 -> 7040`, up `2560 -> 7040`, SiLU, elementwise multiply, down `7040 -> 2560`.
- LM head `2560 -> 48000`, bias false. Weights are not tied for current checkpoint.

Attention primitives:
- Causal self-attention, MHA for current checkpoint. Generic GQA path must support KV repeat from `[B,Hkv,S,D]` to `[B,Hq,S,D]`.
- MatMul QK^T scaled by `1/sqrt(head_dim)`, additive mask, fp32 softmax on last dim, optional dropout, MatMul AV.
- Backend dispatch can use eager, SDPA, FlashAttention, or FlexAttention through `ALL_ATTENTION_FUNCTIONS`.

Position/rotary:
- Default RoPE with inverse frequency length `head_dim/2`.
- Dynamic RoPE update hook exists for advanced `rope_type` values if admitted by config, though public Helium uses default.

Generation/cache:
- DynamicCache when `use_cache=True` and no cache is supplied.
- Per-layer key/value cache updated after RoPE, before attention. Current cache tensors are `[B,20,T,128]` each per layer.
- Position IDs default to `arange(S) + past_seen_tokens`.
- Last-token-only logits optimization via `logits_to_keep`.

Preprocessing-coupled ops:
- CPU tokenizer emits `input_ids` and `attention_mask`; right padding with pad ID 3 is observed in tokenizer.json.
- No processor-owned image/audio/video tensors.

Distributed/tensor-parallel:
- Config declares TP plan: q/k/v/gate/up columnwise, o/down rowwise, lm_head columnwise gather output. This is optional staging work.

## 5. Layer/block breakdown

Decoder block, repeated 24 times for the in-scope checkpoint:

```text
x: [B,S,2560]
r = x
x = RMSNorm(x, eps=1e-8)
q = Linear_q(x).view(B,S,20,128).transpose(1,2)
k = Linear_k(x).view(B,S,20,128).transpose(1,2)
v = Linear_v(x).view(B,S,20,128).transpose(1,2)
q,k = HeliumRoPE(q,k, cos[B,S,128], sin[B,S,128])
if cache: k,v = cache.update(k,v, layer_idx)
a = CausalAttention(q,k,v, mask, scale=1/sqrt(128))
a = a.transpose/reshape to [B,S,2560]
x = r + Linear_o(a)
r = x
x = RMSNorm(x, eps=1e-8)
x = Linear_down(SiLU(Linear_gate(x)) * Linear_up(x))
x = r + x
```

After all layers:

```text
h = RMSNorm(x)
logits = Linear_lm_head(h[:, slice_indices, :])
```

For current checkpoint all block projections are biasless. If `attention_bias` or `mlp_bias` is true, q/k/v and MLP projections gain biases; `o_proj` and `lm_head` remain biasless in Helium source.

## 6. Attention requirements

Required attention variant:

- Causal self-attention only.
- Current checkpoint: MHA, `num_attention_heads=20`, `num_key_value_heads=20`, `head_dim=128`.
- Config-capable path: GQA/MQA via `num_key_value_groups = num_attention_heads // num_key_value_heads`; admission should require divisibility.
- Query/key/value width: Q = `num_attention_heads * head_dim`; K/V = `num_key_value_heads * head_dim`; current all equal 2560.
- Masking: `create_causal_mask` combines lower-triangular causality, optional 2D padding mask, cache offsets, and packed-sequence detection when custom `position_ids` are supplied without an attention mask/cache.
- Cache: autoregressive self-attention KV cache only. Keys are stored after RoPE. Values are stored after V projection with no RoPE.
- Packed/varlen: no explicit `cu_seqlens` ABI in Helium. Transformers mask utilities can derive a packed sequence mask from discontinuous `position_ids`; DinoML can defer this unless packed prompts are admitted.
- Sliding/local attention: not used; no `sliding_window` field in HeliumConfig.
- Backend compatibility: source advertises FlashAttention, SDPA, FlexAttention, and attention-backend dispatch. Eager fallback is simple dense causal attention and likely too slow for production.

Decode cache shapes:

```text
Per layer key:   [B, num_key_value_heads, T_total, head_dim] = [B,20,T,128]
Per layer value: [B, num_key_value_heads, T_total, head_dim] = [B,20,T,128]
Attention expands to [B,20,T,128] only if num_key_value_heads < num_attention_heads.
```

## 7. Position encoding and custom math

RoPE generation:

```python
inv_freq = 1.0 / (theta ** (arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos, sin = emb.cos(), emb.sin()
```

Helium-specific application:

```python
def rotate_half_even_odd(x):
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return stack((-x2, x1), dim=-1).flatten(-2)

def helium_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    return q * cos + rotate_half_even_odd(q) * sin, k * cos + rotate_half_even_odd(k) * sin
```

`cos/sin` can be precomputed for default RoPE up to the admitted max context and indexed by `position_ids`. Dynamic/yarn/longrope variants are technically reachable through generic `rope_parameters`, but no in-scope checkpoint uses them; reject or separately test non-default RoPE initially.

## 8. Preprocessing and input packing

Runtime model inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,2560]`, exactly one required.
- `attention_mask` optional, normally `[B,S]` padding mask from tokenizer.
- `position_ids` optional; if absent source creates `[1,S]` plus cache offset.
- `past_key_values` optional DynamicCache-compatible object.
- `logits_to_keep`: integer or tensor controlling final hidden-state slice before LM head.

Tokenizer snapshot:

- `kyutai/helium-1-preview-2b` uses `PreTrainedTokenizerFast`, `model_input_names = ["input_ids", "attention_mask"]`.
- Tokenizer JSON has right padding with `pad_id=3`; config has BOS 1, EOS 2, and effective pad 3 from source default.
- Added `<|im_start|>`, `<|im_end|>`, and `<|im_sp_*>` tokens are ordinary vocabulary tokens for this source. No placeholder validation, scatter, grid metadata, or modality embedding stitch exists.

CPU/data pipeline owns tokenization, padding, truncation, and generation prompt formatting. GPU graph starts at embedding lookup or supplied embeddings.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split linear attention projections -> packed QKV GEMM

Source pattern:

```text
q = x @ Wq.T
k = x @ Wk.T
v = x @ Wv.T
```

Replacement:

```text
qkv = x @ concat_rows(Wq, Wk, Wv).T
split qkv as [Hq*D, Hkv*D, Hkv*D]
```

Preconditions: same input tensor, same dtype/device, no or compatible biases, no intervening side effects, source split order Q then K then V. Current shapes are `2560 -> 2560 + 2560 + 2560`. Failure cases: per-projection quant metadata, tensor-parallel sharding mismatch, `head_dim`/heads not matching output rows.

Parity test: compare q/k/v tensors before RoPE for random `[B,S,2560]` in fp32 and bf16.

### Rewrite: Helium RoPE canonical kernel

Source pattern: `cat(freqs,freqs)` followed by interleaved repeat and even/odd rotation.

Replacement: a dedicated even/odd-pair RoPE kernel that consumes compact `cos/sin [B,S,D/2]` and applies pair rotations directly.

Preconditions: `head_dim` even, `unsqueeze_dim=1`, q/k layout `[B,H,S,D]`, default or validated scaling. Failure cases: Llama half-split RoPE kernels are not parity-equivalent.

Parity test: random q/k with fixed position IDs across several positions including cache offsets.

### Rewrite: SwiGLU block fusion

Source pattern:

```text
down(silu(gate(x)) * up(x))
```

Replacement: fused two-input activation multiply feeding down GEMM, optionally with packed gate/up input GEMM.

Preconditions: same input to gate/up, activation exactly SiLU, bias handling preserved. Current shapes: `2560 -> 7040` for gate/up, then `7040 -> 2560`.

### Rewrite: last-token logits

Source pattern: `hidden_states[:, slice_indices, :]` before `lm_head`.

Replacement: for decode, pass only final hidden row to LM head.

Preconditions: `logits_to_keep == 1` or equivalent final-token index, no loss computation requiring full logits. Failure case: training/loss or caller requests full sequence logits.

### Layout notes

No image/channel layout rewrites apply. Sequence/head transposes can be eliminated inside a fused attention region if the provider owns projection output layout, RoPE, attention, and O projection contract together.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: used twice per block plus final norm; fp32 accumulation and cast behavior must match.
- QKV projection + reshape + Helium RoPE: removes memory traffic and preserves the even/odd RoPE variant.
- Dense causal attention with KV cache: main prefill/decode bottleneck; FlashAttention/GQA-compatible backend even though current checkpoint is MHA.
- SwiGLU: gate/up SiLU multiply is a standard hot MLP pattern.
- Last-token-only logits: avoids `[B,S,48000]` projection during decode.

Medium priority:

- Residual add + norm scheduling around pre-norm blocks.
- Packed gate/up GEMM.
- Cache append/update provider path with static addresses for decode.
- Tensor-parallel q/k/v/gate/up/o/down/lm_head plans.

Lower priority:

- FlexAttention and packed-sequence mask support.
- Non-default RoPE scaling variants.
- Classification heads.

## 11. Runtime staging plan

Stage 1: parse HeliumConfig, normalize legacy `rope_theta` to effective `rope_parameters`, and load weights for `HeliumForCausalLM`.

Stage 2: implement one-block fp32 parity with RMSNorm, projections, Helium RoPE, eager attention, MLP, and residuals.

Stage 3: full prefill parity for `kyutai/helium-1-preview-2b` with dense causal attention and full logits.

Stage 4: decode parity with DynamicCache-compatible KV layout and final-token logits.

Stage 5: replace eager attention with optimized prefill/decode backend; keep eager parity path.

Stage 6: add QKV packing, RoPE fusion, SwiGLU fusion, and last-token logits rewrite.

Stage 7: optional TP plan and classification heads.

Initially stub/defer training loss, gradient checkpointing, packed prompts, non-default RoPE, and generic classification heads.

## 12. Parity and validation plan

- RMSNorm random tensor parity in fp32/fp16/bf16; tolerance `1e-6` fp32, `1e-2` bf16 for full blocks.
- Helium RoPE parity against source for `position_ids` with offsets and nonzero cache lengths.
- Single attention layer parity before and after cache update; include `attention_mask=None` and padded 2D mask.
- Single decoder block parity with random weights or loaded checkpoint weights.
- After-N-layer hidden-state parity for 1, 2, 24 layers.
- Prefill logits parity on short prompts and max-ish context smoke (`S=1, 16, 512, 4096` as memory allows).
- Decode token parity for greedy generation using the integration-test prompt, comparing generated token IDs rather than only text.
- Weight alias check: current `tie_word_embeddings=false`; if a future config ties weights, preserve one logical embedding/lm_head parameter.

## 13. Performance probes

- Prefill throughput by sequence length: 128, 512, 2048, 4096.
- Decode tokens/sec by batch size and cache length.
- Attention backend comparison: eager vs SDPA vs FlashAttention provider.
- KV cache memory per layer and aggregate: `2 * B * layers * Hkv * T * D * dtype_size`.
- MLP throughput and gate/up packing speedup.
- LM head cost for full logits vs `logits_to_keep=1`.
- Tokenizer throughput separately from GPU runtime; tokenizer is CPU/data pipeline.
- Tensor-parallel sweep if TP plans are admitted.

## 14. Skip/defer list

- Training and loss parity.
- Gradient checkpointing.
- Beam search and advanced generation controllers beyond greedy/token sampling.
- Sequence classification and token classification heads.
- Non-default RoPE scaling variants unless a real Helium checkpoint requires them.
- Packed-sequence `position_ids` mask path.
- FlexAttention-specific block-mask lowering.
- Tensor parallel and pipeline parallel execution.
- Quantized/packed weights; source checkpoint inspected is safetensors bf16 with no source-coupled quant metadata.
- Kyutai `helium-1-2b*` Llama checkpoints; handle in Llama audit.

## 15. Final implementation checklist

- [ ] Parse `HeliumConfig` and normalize effective RoPE defaults.
- [ ] Load `HeliumForCausalLM` weights and preserve untied embedding/lm_head behavior.
- [ ] Implement RMSNorm with fp32 variance and scale multiply.
- [ ] Implement q/k/v/o Linear projections with explicit `head_dim`.
- [ ] Implement Helium even/odd RoPE, distinct from Llama half-split RoPE.
- [ ] Implement dense causal self-attention with optional padding mask.
- [ ] Implement KV cache update storing post-RoPE keys and projected values.
- [ ] Implement SwiGLU MLP.
- [ ] Implement final RMSNorm and biasless LM head.
- [ ] Add last-token logits path.
- [ ] Add one-block, prefill, and decode parity tests.
- [ ] Add QKV packing rewrite with Q,K,V split-order guard.
- [ ] Add RoPE/attention fusion candidate.
- [ ] Benchmark prefill, decode, MLP, LM head, and KV memory.
