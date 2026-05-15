# Transformers GLM Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4 (2026-05-11)
Model id: THUDM/glm-4-9b-hf and THUDM/glm-4-9b-chat-hf for native model_type="glm"
Config source: Hugging Face raw config.json snapshots saved in this folder
Primary runtime target: causal language modeling, prefill + autoregressive decode
Source files inspected:
- transformers/src/transformers/models/glm/modeling_glm.py
- transformers/src/transformers/models/glm/modular_glm.py
- transformers/src/transformers/models/glm/configuration_glm.py
- transformers/tests/models/glm/test_modeling_glm.py
- transformers/docs/source/en/model_doc/glm.md
Any missing files or assumptions:
- modeling_glm.py is generated from modular_glm.py. Runtime facts below use generated modeling_glm.py; future Transformers source edits should target modular_glm.py.
- Native in-library scope is model_type="glm". THUDM/glm-4-9b and THUDM/glm-4-9b-chat currently expose model_type="chatglm" and ChatGLMModel configs; treat them as remote-code/out-of-scope for this report.
- THUDM/glm-4-9b-chat-128k config fetch returned HTTP 401; note as gated/unavailable.
```

Local snapshots:

- `THUDM__glm-4-9b-hf__config.json`
- `THUDM__glm-4-9b-chat-hf__config.json`
- `THUDM__glm-4-9b-chat__config.json` and `THUDM__glm-4-9b-chat-1m__config.json` as out-of-scope ChatGLM contrast configs
- tokenizer and generation config snapshots for the same repos where accessible
- `config_sweep_summary.json`

Relevant source URLs:

- [configuration_glm.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm/configuration_glm.py)
- [modeling_glm.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm/modeling_glm.py)
- [modular_glm.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/glm/modular_glm.py)

## 2. High-level architecture

GLM in this source is a text-only decoder-only Transformer with grouped-query causal self-attention, partial rotary position embeddings, RMSNorm, fused gated MLP projection, and an untied LM head.

```text
tokenizer/chat template -> input_ids/attention_mask -> token embedding
  -> repeated decoder blocks -> final RMSNorm -> LM head
  -> logits/sampling

prefill: full prompt causal attention + optional cache creation
decode: one or more new tokens + per-layer KV cache update + last-token logits
```

First useful DinoML target: `GlmForCausalLM` inference. `GlmModel` is required as the body. Sequence classification and token classification heads are implemented through generic wrappers but are optional/deferred for the causal LM target.

## 3. Important config dimensions

Native `GlmConfig` source defaults:

| Field | Default | Runtime meaning |
| --- | ---: | --- |
| `vocab_size` | 151552 | token embedding rows and LM head rows |
| `hidden_size` | 4096 | model width |
| `intermediate_size` | 13696 | MLP gated branch width |
| `num_hidden_layers` | 40 | decoder layers |
| `num_attention_heads` | 32 | query heads |
| `num_key_value_heads` | 2 | KV heads; GQA ratio 16 |
| `head_dim` | 128 | explicit per-head width |
| `max_position_embeddings` | 131072 | default context limit |
| `hidden_act` | `silu` | MLP gate activation |
| `rms_norm_eps` | 1.5625e-7 | RMSNorm epsilon |
| `attention_bias` | true | Q/K/V biases only; O projection is biasless |
| `tie_word_embeddings` | false | embeddings and LM head are separate parameters |
| `use_cache` | true | DynamicCache-compatible decode |
| RoPE | `rope_theta`, default rope type, `partial_rotary_factor=0.5` | only first 64 of 128 head dims rotate for default GLM configs |

Representative checkpoint sweep:

| Model id | Scope | Layers | Hidden | Heads/KV | Head dim | MLP | Max positions | Dtype | Notes |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `THUDM/glm-4-9b-hf` | native `glm` | 40 | 4096 | 32/2 | 128 | 13696 | 8192 | bf16 | base HF-native checkpoint |
| `THUDM/glm-4-9b-chat-hf` | native `glm` | 40 | 4096 | 32/2 | 128 | 13696 | 131072 | bf16 | chat tokenizer/template, native class |
| `THUDM/glm-4-9b-chat` | out-of-scope `chatglm` | 40 | 4096 | 32 groups/2 | 128 | 13696 | 131072 `seq_length` | bf16 | remote-code style `ChatGLMModel`, not this source |
| `THUDM/glm-4-9b-chat-1m` | out-of-scope `chatglm` | 40 | 4096 | 32 groups/4 | 128 | 13696 | 1048576 `seq_length` | bf16 | changes KV group count and context; separate audit |
| `THUDM/glm-4-9b-chat-128k` | unavailable | unknown | unknown | unknown | unknown | unknown | unknown | unknown | raw config returned HTTP 401 |

## 3a. Family variation traps

- `glm` vs `chatglm` is the largest routing trap. Native `GlmForCausalLM` uses `num_key_value_heads`; ChatGLM configs use fields such as `multi_query_group_num`, `kv_channels`, `padded_vocab_size`, `seq_length`, `add_qkv_bias`, and `add_bias_linear`.
- Native GLM uses separate Q, K, V projections, not a packed QKV weight. Fused projection rewrites must pack weights explicitly as `[Q_rows, K_rows, V_rows]` if DinoML chooses a packed kernel.
- `head_dim` is explicit; do not assume `hidden_size / num_attention_heads` without checking it.
- Query output width is `num_attention_heads * head_dim`; attention output width is the same and then projected back to `hidden_size`.
- KV projection width is `num_key_value_heads * head_dim`; GQA repeat is logical, not an extra parameter tensor.
- RoPE is partial by default: effective rotary dim is `head_dim * partial_rotary_factor = 64` for the 9B HF configs.
- GLM's rotary helper interleaves even/odd dimensions with `repeat_interleave(2)`, not the common split-half rotate layout.
- `attention_bias=True` adds bias to Q/K/V only. `o_proj`, MLP projections, and LM head are biasless.
- `tie_word_embeddings=False` in representative configs. The class declares tied-key metadata for loaders, but actual config says do not tie.
- `logits_to_keep` allows last-token-only or indexed logits. This is an important decode optimization and an output-shape variation.
- Tokenizer/chat behavior is ABI-level, not neural graph behavior. Chat HF repos use `PreTrainedTokenizerFast`; non-HF ChatGLM repos use `ChatGLM4Tokenizer` remote code.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup: `input_ids [B,S] -> [B,S,4096]`.
- `view`/reshape Q/K/V to `[B,S,H,D]` or `[B,S,KVH,D]`, then transpose to `[B,H,S,D]`.
- `chunk(2, dim=-1)` for fused `gate_up_proj`.
- `cat` for RoPE frequency duplication and concatenating rotated/pass-through head dims.
- `stack`, strided even/odd slicing, `flatten(-2)`, `repeat_interleave`, `expand`, `reshape`.
- Last-token or indexed logits slice: `hidden_states[:, slice_indices, :]`.

Neural primitives:

- Linear Q: `4096 -> 4096`, bias enabled.
- Linear K/V: `4096 -> 256`, bias enabled for `num_key_value_heads=2`, `head_dim=128`.
- Linear O: `4096 -> 4096`, no bias.
- MLP fused gate/up: `4096 -> 27392`, no bias, split into two `13696` halves.
- MLP down: `13696 -> 4096`, no bias.
- SiLU activation and elementwise multiply for SwiGLU-like MLP.
- RMSNorm over hidden dim, fp32 variance accumulation, affine weight.
- LM head: `4096 -> 151552`, no bias.

Attention primitives:

- Causal GQA self-attention.
- KV repeat from `[B,2,T,128]` to `[B,32,T,128]` for eager implementation; optimized backends should avoid materializing repeat.
- Scale by `head_dim ** -0.5`.
- Add causal/padding mask.
- Softmax in fp32, cast back to query dtype.
- Attention dropout is source-supported for training; inference uses zero dropout.
- Cache update through `Cache.update(key_states, value_states, layer_idx)`.

Position/rotary ops:

- RoPE frequency matmul from `inv_freq [32]` and `position_ids [B,S]`.
- `cos`/`sin` computed in fp32 with autocast disabled, then cast to activation dtype.
- Partial rotary apply over first 64 head dims; remaining 64 pass through unchanged.

Generation/cache ops:

- `DynamicCache` construction when `use_cache=True` and no cache is supplied.
- Position IDs default to `arange(S) + past_seen_tokens`.
- Per-layer cache tensors before GQA repeat: keys and values `[B, num_key_value_heads, cached_seq, head_dim]`.
- Cache reorder is inherited from Transformers cache utilities for generation/beam paths; first DinoML target can support greedy decode before beam reorder.

Preprocessing-coupled ops:

- Tokenizer emits `input_ids` and `attention_mask`; chat templates add GLM role/control tokens.
- No image/audio/video tensors in native `glm`.

Distributed/tensor-parallel ops:

- Source config includes TP hints: Q/K/V colwise, O rowwise, `gate_up_proj` colwise with gather because of chunk, `down_proj` rowwise split input, LM head colwise gather. This is optimization metadata, not required for single-GPU parity.

## 5. Layer/block breakdown

Model body:

```text
input_ids [B,S] -> embed_tokens -> x [B,S,4096]
position_ids [1 or B,S] -> rotary_emb -> cos/sin [B,S,64]
for layer in 40:
  residual = x
  x = RMSNorm(x)
  q = Linear(4096 -> 4096, bias)(x).view(B,S,32,128).transpose(1,2)
  k = Linear(4096 -> 256, bias)(x).view(B,S,2,128).transpose(1,2)
  v = Linear(4096 -> 256, bias)(x).view(B,S,2,128).transpose(1,2)
  q,k = partial_interleaved_RoPE(q,k, cos, sin)
  k,v = cache.update(k,v) if cache is enabled
  attn = causal_gqa_attention(q,k,v, mask, scale=1/sqrt(128))
  x = residual + Linear(4096 -> 4096, no bias)(attn)
  residual = x
  x = RMSNorm(x)
  gate_up = Linear(4096 -> 27392, no bias)(x)
  gate, up = chunk(gate_up, 2, dim=-1)
  x = residual + Linear(13696 -> 4096, no bias)(up * silu(gate))
x = final RMSNorm(x)
logits = Linear(4096 -> 151552, no bias)(selected sequence positions)
```

## 6. Attention requirements

Required attention is causal self-attention with GQA. There is no cross-attention, no encoder attention, no sliding-window/local attention in native `glm`, no ALiBi, and no block-sparse/hash attention path in the inspected source.

For representative native configs:

- Query heads: 32.
- KV heads: 2.
- GQA groups: 16 query heads per KV head.
- Head dim: 128.
- Query projection width: 4096.
- Key/value projection width: 256 each.
- Attention score shape in eager form: `[B, 32, Q, K_total]`.
- Cache stores un-repeated K/V with shape `[B, 2, T_total, 128]`; repeat to 32 heads happens inside eager attention only.
- Cached keys are stored after RoPE, because RoPE is applied before `past_key_values.update`.
- Mask is produced by `create_causal_mask` from Transformers masking utilities and passed to the backend attention function.
- Source advertises eager, SDPA, FlashAttention, and flex attention support via the generic attention registry.

Fused attention parity should preserve this math order:

```text
q_proj/k_proj/v_proj
-> reshape/transpose
-> partial interleaved RoPE on q,k
-> cache update
-> attention backend with scale=head_dim^-0.5 and additive causal mask
-> output transpose/contiguous/reshape
-> o_proj
```

## 7. Position encoding and custom math

GLM uses default RoPE frequencies unless a config supplies another standardized `rope_parameters["rope_type"]`. The audited HF configs omit `rope_parameters` but provide `rope_theta=10000`; the config standardization path supplies `rope_type="default"` and `partial_rotary_factor=0.5`.

Short reproduction of the source-specific rotary behavior:

```python
def glm_rope_inv_freq(head_dim=128, partial_rotary_factor=0.5, theta=10000.0):
    dim = int(head_dim * partial_rotary_factor)  # 64 for GLM-4 9B HF
    return 1.0 / (theta ** (arange(0, dim, 2).float() / dim))

def glm_apply_rope(q, k, cos, sin):
    # q: [B,H,S,D], cos/sin: [B,S,rotary_dim]
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    cos = cos[..., : cos.shape[-1] // 2].repeat_interleave(2, dim=-1)
    sin = sin[..., : sin.shape[-1] // 2].repeat_interleave(2, dim=-1)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    rotate_half = lambda x: stack((-x[..., 1::2], x[..., 0::2]), dim=-1).flatten(-2)
    return cat([q_rot * cos + rotate_half(q_rot) * sin, q_pass], -1), cat([k_rot * cos + rotate_half(k_rot) * sin, k_pass], -1)
```

Precompute opportunity: `inv_freq` is static per config. `cos/sin` can be cached by position range and dtype/device, but dynamic decode must account for `past_seen_tokens` and any caller-supplied `position_ids`.

## 8. Preprocessing and input packing

Native model inputs:

- `input_ids [B,S]` or `inputs_embeds [B,S,4096]`, exactly one required.
- Optional `attention_mask`, consumed by `create_causal_mask`.
- Optional `position_ids [B,S]`; if omitted, source creates `[1,S]` from current cache length.
- Optional `past_key_values`.

Tokenizer ABI observations from snapshots:

- `THUDM/glm-4-9b-hf`: `PreTrainedTokenizerFast`, no chat template in tokenizer config.
- `THUDM/glm-4-9b-chat-hf`: `PreTrainedTokenizerFast`, left padding, chat template present, model input names are `input_ids` and `attention_mask`.
- Special token IDs include `<|endoftext|>` 151329, `[MASK]` 151330, `[gMASK]` 151331, `[sMASK]` 151332, `<sop>` 151333, `<eop>` 151334, role tokens 151335-151338, and image/video delimiters 151339-151342. Native `glm` source does not implement multimodal handling for those image/video tokens.
- Generation config uses `pad_token_id=151329` and EOS IDs `[151329, 151336, 151338]`.

CPU/data-pipeline work: tokenization, chat-template rendering, padding, and generation-controller EOS handling. GPU/runtime work: embedding, causal mask handling, position ID/RoPE math, decoder, logits.

## 9. Graph rewrite / lowering opportunities

### Rewrite: fused gate/up projection

Source pattern:

```text
gate_up = Linear(hidden -> 2*intermediate, bias=False)(x)
gate, up = chunk(gate_up, 2, dim=-1)
down(silu(gate) * up)
```

Replacement:

```text
GEMM_RRR/GEMM_RCR -> split last dim -> SiLU-mul fused elementwise -> GEMM
```

Preconditions:

- Last dimension split exactly into two equal halves.
- Activation is source `hidden_act`; representative configs use `silu`.
- No bias in `gate_up_proj` or `down_proj`.
- Preserve split order: first half is gate, second half is up.

Parity test sketch: random `[B,S,4096]` bf16/fp16/fp32 through source MLP vs lowered MLP, including nontrivial `B*S`.

### Rewrite: Q/K/V projection packing

Source pattern:

```text
q = Linear(4096 -> 4096, bias=True)
k = Linear(4096 -> 256, bias=True)
v = Linear(4096 -> 256, bias=True)
```

Replacement:

```text
single packed GEMM 4096 -> 4608, then split [4096, 256, 256]
```

Preconditions:

- Same input tensor and dtype.
- All three projections have compatible bias setting.
- Packed weight row order must be Q rows, then K rows, then V rows.
- Must still reshape Q with 32 heads and K/V with 2 heads.

Failure cases: any checkpoint/source variant with packed remote-code layout, missing bias on one projection, or nonstandard per-projection quantization metadata should reject this rewrite.

### Rewrite: GQA attention without materialized KV repeat

Source eager path materializes repeat with expand/reshape. Replacement should call a GQA-aware attention kernel using KV heads directly.

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- Attention backend supports grouped KV, causal mask, cache append, and same scale.
- Output must match source score semantics and fp32 softmax tolerance.

### Rewrite: last-token logits

Source supports `logits_to_keep`. During decode, replace full `[B,S,V]` LM head with `[B,1,V]` or indexed rows.

Preconditions:

- Caller requests `logits_to_keep=1` or equivalent generation path.
- No loss computation requiring full shifted logits.

### Layout guards

No NCHW/NHWC vision layout translation is applicable. Protect decoder sequence tensors with a no-layout-translation guard around all `[B,S,H]`, `[B,H,S,D]`, and cache tensors; axis numbers in RoPE, softmax, chunk, and logits slicing are semantic.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm over 4096 with fp32 accumulation, because it appears twice per block plus final norm.
- Packed QKV projection + split, because it reduces launch overhead and exposes a single large GEMM.
- Partial RoPE + GQA FlashAttention prefill/decode, because 128K context makes attention backend choice decisive.
- SwiGLU MLP epilogue (`silu(gate) * up`) between two GEMMs.
- Last-token-only LM head for decode, because full-sequence vocab projection is wasteful.

Medium priority:

- Cache append/update fused with attention input layout.
- Add-residual fused with O projection and MLP down projection where numerically acceptable.
- Causal mask specialization for left-padded generation batches.
- Tensor-parallel lowering using source TP hints.

Lower priority:

- Generic eager attention reconstruction with materialized KV repeat, useful only as a fallback/parity reference.
- Sequence/token classification heads.
- Dynamic/non-default RoPE types beyond the representative default configs.

## 11. Runtime staging plan

Stage 1: parse `GlmConfig`, reject `model_type!="glm"`, load native HF weights, and run embedding + one decoder block parity.

Stage 2: full prefill without cache using eager/composed ops, including partial interleaved RoPE and causal mask parity.

Stage 3: causal LM logits with `logits_to_keep` support and final generation-controller EOS IDs handled outside the graph.

Stage 4: decode with per-layer KV cache `[B,2,T,128]`, position offset from cache length, and greedy generation parity.

Stage 5: optimized GQA attention backend for prefill/decode, avoiding materialized KV repeat.

Stage 6: packed QKV and MLP rewrites, then RMSNorm/SwiGLU fusions.

Stage 7: long-context production probes, batching, cache memory planning, and optional tensor parallelism.

Can be stubbed initially: training loss, dropout, output attentions/hidden states, sequence/token classification, beam cache reorder, remote-code ChatGLM configs, 1M context variants.

## 12. Parity and validation plan

- Config load tests: native `glm` HF configs load; `chatglm` configs reject with a clear route-to-separate-audit error.
- Custom RoPE tests: compare `glm_rope_inv_freq`, cos/sin, and `apply_rotary_pos_emb` for random `position_ids`, including partial rotary pass-through dims.
- RMSNorm tests: fp32/fp16/bf16 random tensors with fp32 variance accumulation; tolerance `1e-5` fp32, `2e-3` fp16/bf16.
- MLP tests: random `[1,3,4096]` and flattened larger `B*S` inputs against PyTorch source.
- Attention tests: one-layer no-cache and with-cache, including left padding masks and multiple decode steps.
- Full model tests: prefill logits for `THUDM/glm-4-9b-hf` small prompt if weights are available locally.
- Decode tests: greedy next-token parity for 1, 2, and N incremental tokens; validate cache shapes and position IDs.
- End-to-end text: tokenizer/chat-template CPU path plus DinoML graph logits, comparing generated IDs for deterministic greedy generation.

## 13. Performance probes

- Prefill throughput sweep: sequence lengths 128, 1024, 8192, 32768, 131072 where memory permits.
- Decode tokens/sec sweep: batch sizes 1, 4, 16, 64 with cache lengths 128 to 131072.
- GQA attention backend comparison: eager repeat fallback vs GQA-aware FlashAttention-style kernel.
- KV cache memory usage: 40 layers x 2 tensors x `[B,2,T,128]` x dtype bytes.
- MLP GEMM throughput: fused gate/up GEMM and down GEMM by `B*S`.
- LM head cost: full sequence logits vs last-token-only logits.
- RoPE overhead: precomputed cos/sin cache vs per-call generation.
- Weight load/dequant probe if GGUF or other quantized checkpoints are introduced later; no native source-coupled quantized format is present in this audit.

## 14. Skip/defer list

- Training, labels/loss, gradient checkpointing, and dropout behavior.
- Sequence classification and token classification heads.
- Beam search and cache reorder in the first greedy decode target.
- Remote-code `chatglm` checkpoints, including `THUDM/glm-4-9b-chat` and `THUDM/glm-4-9b-chat-1m`.
- Gated/unavailable `THUDM/glm-4-9b-chat-128k` until access is resolved.
- 1M context behavior and `multi_query_group_num=4` ChatGLM variant.
- Multimodal image/video special tokens; native `glm` has no multimodal branch.
- Tensor parallel/distributed execution.
- Non-default RoPE scaling types unless a native `glm` config requiring them is admitted.

## 15. Final implementation checklist

- [ ] Parse `GlmConfig` and standardize RoPE fields, including `partial_rotary_factor=0.5`.
- [ ] Reject or reroute `model_type="chatglm"` configs.
- [ ] Load embeddings, separate Q/K/V/O projections, MLP projections, RMSNorm weights, and LM head.
- [ ] Implement GLM RMSNorm parity.
- [ ] Implement partial interleaved RoPE parity.
- [ ] Implement GQA causal self-attention with un-repeated KV cache storage.
- [ ] Implement `DynamicCache`-equivalent per-layer K/V update and position offset.
- [ ] Implement fused gate/up MLP split order and SiLU multiply.
- [ ] Implement final LM head with `logits_to_keep`.
- [ ] Add one-block, full-prefill, and incremental-decode parity tests.
- [ ] Add packed QKV rewrite with strict weight/bias preconditions.
- [ ] Add GQA attention backend probe and long-context memory benchmark.
- [ ] Add last-token logits benchmark.
