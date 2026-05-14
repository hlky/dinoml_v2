# Persimmon Transformers Family Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from local checkout `X:/H/transformers`.

Model id: primary open checkpoints [`adept/persimmon-8b-base`](https://huggingface.co/adept/persimmon-8b-base) and [`adept/persimmon-8b-chat`](https://huggingface.co/adept/persimmon-8b-chat). Both Hub API records were open/not gated on 2026-05-13.

Config source: raw Hub `config.json`, tokenizer/generation configs for the two Adept checkpoints, plus [`optimum-intel-internal-testing/tiny-random-PersimmonForCausalLM`](https://huggingface.co/optimum-intel-internal-testing/tiny-random-PersimmonForCausalLM) for a small shape sweep. Compact snapshots are in `config_snapshots.md`.

Source files inspected:

- `X:/H/transformers/src/transformers/models/persimmon/configuration_persimmon.py`
- `X:/H/transformers/src/transformers/models/persimmon/modeling_persimmon.py`
- `X:/H/transformers/src/transformers/models/persimmon/convert_persimmon_weights_to_hf.py`
- `X:/H/transformers/src/transformers/activations.py` for `relu2`
- `X:/H/transformers/src/transformers/modeling_rope_utils.py` and `configuration_utils.py` for legacy `rope_theta`/`rope_scaling` conversion into `rope_parameters`

Any missing files or assumptions: there is no Persimmon-specific tokenizer implementation in the model directory; the official tokenizer config selects `LlamaTokenizer` with a SentencePiece `tokenizer.model`. No modular Persimmon source file was present, so `modeling_persimmon.py` is the authoritative in-library modeling file. No remote-code files are required for the official Persimmon checkpoints.

## 2. High-level architecture

Persimmon is a text-only decoder-only causal language model. The primary DinoML runtime target should be `PersimmonForCausalLM`: token embedding, repeated decoder blocks, final LayerNorm, and untied LM head.

Dataflow:

```text
LlamaTokenizer/SentencePiece -> input_ids/attention_mask -> token embedding
  -> N decoder blocks with causal self-attention and MLP
  -> final LayerNorm -> LM head -> logits -> generation controller/sampling
```

Stage decomposition:

- CPU/data pipeline: SentencePiece tokenization, BOS insertion, attention-mask construction policy, chat prompt formatting outside the neural graph.
- GPU prefill: embedding lookup, full causal self-attention, partial RoPE, Q/K LayerNorm, dense MLP.
- GPU decode: one or more new tokens with per-layer KV cache updates.
- Independently validatable pieces: tokenizer ABI, one decoder block, attention with cache, MLP `relu2`, final logits with `logits_to_keep`.

## 3. Important config dimensions

| Field | Adept 8B base/chat config | Source default | Tiny random |
|---|---:|---:|---:|
| `hidden_size` | 4096 | 4096 | 32 |
| `num_hidden_layers` | 36 | 36 | 2 |
| `num_attention_heads` | 64 | 64 | 4 |
| effective `num_key_value_heads` | 64 MHA | source ignores field | 4 MHA by inference from heads |
| `head_dim` | 64 | `hidden_size // heads` | 8 |
| `intermediate_size` | 16384 | 16384 | 37 |
| `vocab_size` | 262144 | 262144 | 262144 |
| `max_position_embeddings` | 16384 | 16384 | 512 |
| RoPE theta | 25000.0 | defaults to 10000 unless legacy config supplies `rope_theta` | 25000.0 |
| partial RoPE factor | effective 0.5 via config BC | 0.5 | 0.5 |
| rotary dims/head | 32 | 32 | 4 |
| activation | `relu2` | `relu2` | `gelu` |
| Q/K LayerNorm | true | true | true |
| attention bias | yes, fused QKV and output Linear biases | yes | yes |
| MLP bias | yes on both Linear layers | yes | yes |
| dtype | `bfloat16` from config metadata | not forced by source | `float32` |
| cache support | true | true | true |
| tied embeddings | false | false | false |

Representative checkpoint sweep:

| Checkpoint | Scope | Operator-significant notes |
|---|---|---|
| `adept/persimmon-8b-base` | official production/base | 36-layer bf16 CausalLM, `relu2`, Q/K LayerNorm, partial RoPE, 16k positions |
| `adept/persimmon-8b-chat` | official chat | Same neural config as base; generation/tokenizer metadata matches base |
| `optimum-intel-internal-testing/tiny-random-PersimmonForCausalLM` | small/debug | Same source class but tiny dimensions and `hidden_act="gelu"`; useful for CI shape parity but not activation coverage for official 8B |
| `OpenVINO/persimmon-8b-chat-fp16-ov` | derivative, out of scope | Raw config says `model_type="gpt_neox"` and `GPTNeoXForCausalLM`; do not route through Persimmon lowering |

## 3a. Family variation traps

- Official configs contain legacy fields `rope_theta` and `rope_scaling`; current source reads `config.rope_parameters`. `PreTrainedConfig` plus RoPE helpers standardize legacy fields into `rope_parameters`.
- `partial_rotary_factor` is omitted by official configs but `PersimmonConfig.__post_init__` supplies 0.5 for backwards compatibility.
- Official configs include `num_key_value_heads=64`, `pretraining_tp`, and `rms_norm_eps`, but inspected source does not read them. Treat Persimmon as MHA, not GQA/MQA, unless a future source basis changes.
- Fused QKV storage is head-major grouped as `[head, qkv, head_dim]`, not all-Q/all-K/all-V row blocks.
- Q/K LayerNorm is over each head vector after QKV split and before transpose/RoPE; the affine parameters have shape `[head_dim]` and are shared across heads by broadcasting.
- `hidden_dropout=0.6` in official configs is training-only; inference disables dropout.
- `tie_word_embeddings=false`, but the class declares `_tied_weights_keys`. Do not assume LM head and token embedding weights alias unless config/loaded weights explicitly tie them.
- The tiny random checkpoint changes activation to GELU and should not be used to prove `relu2` parity.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, S] -> [B, S, H]`.
- View QKV output `[B, S, 3H] -> [B, S, num_heads, 3, head_dim]`.
- Slice/select Q/K/V from fused QKV, slice rotary/pass-through dimensions, concatenate along head dim.
- Transpose `[B, S, heads, D] <-> [B, heads, S, D]`, contiguous before attention-output reshape.
- Causal mask construction and optional padding mask incorporation.
- Last-token or selected-token slice for `logits_to_keep`.

Neural primitives:

- Linear with bias: QKV `Linear(4096 -> 12288)`, attention output `Linear(4096 -> 4096)`, MLP up `Linear(4096 -> 16384)`, MLP down `Linear(16384 -> 4096)`.
- Linear without bias: LM head `Linear(4096 -> 262144)`.
- LayerNorm over hidden size 4096: pre-attention, post-attention, final.
- LayerNorm over head dim 64 for Q and K when `qk_layernorm=true`.
- Activation `relu2(x) = square(relu(x))`; GELU only for tiny/debug or non-official configs.
- Residual adds around attention and MLP.

Attention primitives:

- Dense causal self-attention, MHA, 64 heads, head dim 64.
- Softmax upcast to fp32 in eager attention, then cast back to query dtype.
- KV cache update per layer after RoPE has been applied to keys.
- SDPA/FlashAttention-compatible dispatch via Transformers attention backend, with eager fallback.

Position/rotary ops:

- Default RoPE with theta 25000 for official checkpoints.
- Partial RoPE on first half of each head, 32 of 64 dims for 8B.
- Cos/sin computed in fp32 and cast to hidden dtype.

Generation/cache ops:

- Dynamic cache allocation when `use_cache=true`.
- Per-layer key/value append and beam/cache reorder through Transformers cache ABI.
- Generation controller handles sampling/beam search; neural graph only emits logits.

Preprocessing-coupled ops:

- LlamaTokenizer/SentencePiece with `add_bos_token=true`, no tokenizer pad token by default, `|ENDOFTEXT|` as BOS content. Config IDs use `bos_token_id=eos_token_id=71013`.

## 5. Layer/block breakdown

Decoder block, repeated 36 times for official 8B:

```text
x0: [B, S, 4096]
a = LayerNorm_4096(x0)
fused = Linear_bias(a; 4096 -> 12288)
q,k,v = view(fused, [B,S,64,3,64]) select q/k/v
q = LayerNorm_64(q)           # if qk_layernorm
k = LayerNorm_64(k)
q,k,v = transpose to [B,64,S,64]
q_rot/q_pass = split q at 32 dims
k_rot/k_pass = split k at 32 dims
q_rot,k_rot = RoPE(q_rot,k_rot, cos/sin)
q,k = concat(rot, pass)
k,v = cache.update(k,v, layer_idx) if cache is present
attn = causal_attention(q,k,v, mask, scale=1/sqrt(64))
attn = reshape transpose result to [B,S,4096]
attn = Linear_bias(attn; 4096 -> 4096)
x1 = x0 + attn
m = LayerNorm_4096(x1)
m = Linear_bias(m; 4096 -> 16384)
m = relu2(m)
m = Linear_bias(m; 16384 -> 4096)
out = x1 + dropout_training_only(m)
```

Model tail:

```text
hidden = final LayerNorm_4096(out)
logits = Linear_no_bias(hidden[:, slice_indices, :]; 4096 -> 262144)
```

## 6. Attention requirements

Persimmon uses causal self-attention only for the primary CausalLM target. There is no cross-attention, encoder-decoder cache, sliding-window attention, sparse attention, ALiBi, or local/block pattern in the inspected source.

Attention details:

- MHA: `num_attention_heads=64`, effective KV heads also 64.
- Query/key/value width: each is 4096 total, `[64, 64]` per token.
- Scaling: `head_dim ** -0.5`.
- Masking: additive causal mask from `create_causal_mask`; optional caller attention mask is folded into that helper.
- Eager math order: matmul QK^T, multiply scale, add mask, softmax with fp32 accumulation, cast to query dtype, dropout if training, matmul with V.
- Backend compatibility: source advertises `_supports_sdpa=True`, `_supports_flash_attn=True`, and dispatches through `ALL_ATTENTION_FUNCTIONS`; DinoML should preserve eager parity first and then route to FlashAttention for supported prefill/decode shapes.
- Cache: key cached after partial RoPE, value cached directly after transpose. Cache tensors are logically `[B, heads, cached_seq, head_dim]` per layer.

## 7. Position encoding and custom math

RoPE can be precomputed per position bucket for default/static theta, but dynamic RoPE types from the common helper would require runtime updates. Official Persimmon configs use default RoPE semantics after legacy conversion.

Concise parity snippets:

```python
def relu2(x):
    return torch.square(torch.relu(x))
```

```python
def persimmon_rope_freqs(position_ids, head_dim, partial_rotary_factor, theta):
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()
```

```python
def apply_partial_rope(q, k, cos, sin, rotary_ndims):
    q_rot, q_pass = q[..., :rotary_ndims], q[..., rotary_ndims:]
    k_rot, k_pass = k[..., :rotary_ndims], k[..., rotary_ndims:]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_rot = q_rot * cos + rotate_half(q_rot) * sin
    k_rot = k_rot * cos + rotate_half(k_rot) * sin
    return torch.cat([q_rot, q_pass], -1), torch.cat([k_rot, k_pass], -1)
```

`position_ids` default to `arange(current_seq) + past_seen_tokens`, so decode parity must account for cache length.

## 8. Preprocessing and input packing

The neural graph consumes text tensors only:

- `input_ids`: `[B, S]` int token IDs.
- `attention_mask`: optional mask accepted by Transformers and folded into a causal additive mask.
- `position_ids`: optional `[B, S]`; generated from cache length when omitted.
- `inputs_embeds`: alternative to `input_ids`; exactly one of `input_ids` or `inputs_embeds` must be provided.

Tokenizer/data-pipeline contract:

- Official tokenizer config uses `LlamaTokenizer`, SentencePiece file `tokenizer.model`, `add_bos_token=true`, `add_eos_token=false`.
- `pad_token` is null in official tokenizer config and `pad_token_id` is null in source default. Batching with padding needs an explicit policy outside the model graph.
- Generation config from official checkpoints sets BOS/EOS IDs to 71013.

No image/audio/video/OCR/placeholder/scatter packing is applicable.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fused Persimmon QKV Linear

Source pattern:

```text
Linear(H -> 3H) -> view [B,S,num_heads,3,head_dim] -> select q,k,v
```

Replacement:

```text
single GEMM -> three logical views, or packed-QKV attention input
```

Preconditions:

- `hidden_size % num_attention_heads == 0`.
- Weight row order is `[head, qkv, head_dim]`.
- Bias row order follows the same layout.
- Consumers only need Q/K/V views or a backend supporting this packed layout.

Failure cases: all-Q/all-K/all-V packed weight assumptions will produce wrong heads. Parity test: compare split Q/K/V tensors against Transformers for random hidden states before RoPE.

### Rewrite: Q/K LayerNorm + partial RoPE into attention preparation kernel

Source pattern:

```text
q,k from packed QKV -> LayerNorm(head_dim) -> transpose -> split rotary/pass -> RoPE -> concat
```

Replacement:

```text
attention-prep kernel emitting Q,K,V in backend layout with post-RoPE K
```

Preconditions:

- `qk_layernorm=true`.
- LayerNorm affine shape is `[head_dim]` shared across heads.
- Rotary dims equal `int(head_dim * partial_rotary_factor)` and even.
- Cache stores post-RoPE K.

Failure cases: configs disabling Q/K LayerNorm, non-default/dynamic RoPE, or unexpected head_dim.

### Rewrite: `relu2` MLP epilogue

Source pattern:

```text
GEMM_up + bias -> relu -> square -> GEMM_down + bias
```

Replacement:

```text
CUTLASS/elementwise epilogue for relu-square before down projection
```

Preconditions:

- Activation is exactly `relu2`.
- No training dropout.
- Intermediate tensor shape `[B*S, intermediate_size]`.

Failure cases: tiny/debug config with `gelu`, or training mode.

### Rewrite: last-token-only logits

Source pattern:

```text
hidden[:, slice_indices, :] -> lm_head
```

Replacement:

```text
gather selected hidden rows -> GEMM to vocab
```

Preconditions:

- `logits_to_keep` is int or a concrete index tensor.
- For decode, keep only last token by default in generation loops.

Failure cases: callers requesting full-sequence logits for scoring/training.

## 10. Kernel fusion candidates

Highest priority:

- Dense MHA prefill/decode with KV cache, including partial RoPE and post-RoPE K cache storage.
- Fused packed QKV GEMM handling Persimmon row order.
- LayerNorm kernels for hidden-size LN and small head-dim Q/K LN.
- `relu2` activation fusion in MLP.
- Last-token-only LM head for decode to avoid full-sequence vocab GEMM.

Medium priority:

- Q/K LayerNorm + RoPE prep fusion into FlashAttention-friendly layout.
- Attention mask construction/canonicalization for padded batches.
- BF16 GEMM coverage for official checkpoints.

Lower priority:

- Sequence/token classification generic heads.
- Training dropout/loss paths.
- Non-default RoPE scaling variants; source supports common RoPE helper paths but official configs do not require them.

## 11. Runtime staging plan

Stage 1: parse `PersimmonConfig`, normalize legacy RoPE fields into an explicit DinoML config schema, and reject out-of-scope `model_type != "persimmon"` derivatives.

Stage 2: load weights and validate packed QKV row order, tokenizer/generation metadata, and untied embedding/LM-head identity.

Stage 3: implement one-block fp32/bf16 parity without cache using eager dense attention.

Stage 4: full prefill parity for base model with causal mask and partial RoPE.

Stage 5: decode parity with per-layer KV cache storing post-RoPE keys.

Stage 6: add optimized attention/GEMM fusions, `relu2` epilogue, and last-token logits.

Stage 7: production batching, padding-mask policy, and performance tuning.

Initially stub/defer sequence classification, token classification, training loss, dropout, and non-default RoPE variants.

## 12. Parity and validation plan

- Config parser tests: official 8B configs, tiny random config, and rejection of GPT-NeoX/OpenVINO derivative.
- Unit test `relu2` against PyTorch over fp32/bf16, including negative/zero/positive ranges.
- QKV split parity: compare query/key/value tensors immediately after `_split_heads`.
- Q/K LayerNorm parity: verify affine sharing over heads.
- RoPE parity: compare partial RoPE output for prefill and decode positions.
- Attention parity: eager attention with additive causal/padding masks, fp32 softmax, and dtype cast back.
- Single-layer parity: random weights and hidden states for one decoder block.
- N-layer smoke parity on tiny random Persimmon.
- Prefill logits parity on short prompts for official shape or a reduced synthetic config.
- Decode parity: one-token and multi-token incremental cache against full-prefix recompute.
- Tolerances: fp32 `rtol=1e-4, atol=1e-5`; bf16/fp16 attention/logits should use looser `rtol=2e-2, atol=2e-2` initially, tightened after backend choices stabilize.

## 13. Performance probes

- Prefill throughput over sequence lengths 128, 512, 2048, 8192, 16384.
- Decode tokens/sec over batch sizes 1, 4, 16, 64 with KV cache.
- QKV GEMM layout/projection benchmark: packed output plus split versus separate Q/K/V materialization.
- Q/K LayerNorm + RoPE prep bandwidth and temporary allocation size.
- FlashAttention versus eager/SDPA for prefill and decode.
- MLP `relu2` epilogue fusion versus separate relu and square kernels.
- LM head full logits versus last-token-only logits.
- KV cache memory: `layers * 2 * B * heads * seq * head_dim * dtype_size`.
- Tokenization throughput separately from neural runtime.

## 14. Skip/defer list

- Training loss and dropout behavior.
- Gradient checkpointing.
- Sequence and token classification heads for the first CausalLM target.
- Beam search and advanced generation processors beyond basic logits emission.
- Non-default/dynamic RoPE variants unless encountered in an official Persimmon config.
- Quantized/GGUF derivative loading; treat as a separate weight/provider contract.
- GPT-NeoX/OpenVINO derivatives that reuse the Persimmon name but not the Persimmon source class.
- Multi-GPU tensor parallel behavior; config field `pretraining_tp` is ignored by inspected source.

## 15. Final implementation checklist

- [ ] Parse Persimmon config and normalize legacy RoPE fields.
- [ ] Reject non-`persimmon` model types and unsupported remote-code/derivative configs.
- [ ] Load embedding, packed QKV, attention dense, MLP, final LN, and LM-head weights.
- [ ] Preserve QKV packed row order `[head, qkv, head_dim]`.
- [ ] Implement hidden-size LayerNorm and head-dim Q/K LayerNorm.
- [ ] Implement `relu2`.
- [ ] Implement partial RoPE with theta 25000 and factor 0.5 for official checkpoints.
- [ ] Implement causal MHA prefill with fp32 softmax parity.
- [ ] Implement decode KV cache with post-RoPE keys.
- [ ] Implement `logits_to_keep`/last-token LM-head slicing.
- [ ] Add one-block and tiny full-model parity tests.
- [ ] Add official-config prefill/decode parity tests.
- [ ] Benchmark prefill, decode, QKV prep, MLP, LM head, and KV memory.
