# NanoChat Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: karpathy/nanochat-d32, nanochat-students/nanochat-d20, pankajmathur/nanochat-d34-sft-hf, viswamaicoe/nanochat-telugu-560M-sft-smoltalk
Config source: Hugging Face raw config.json; see config_sweep.json
Source files inspected: src/transformers/models/nanochat/modeling_nanochat.py, modular_nanochat.py, configuration_nanochat.py, convert_nanochat_checkpoints.py, tests/models/nanochat/test_modeling_nanochat.py, src/transformers/activations.py
Any missing files or assumptions: karpathy/nanochat-d32 config.json is on refs/pr/1 in Transformers tests, not main. KandirResearch/Nanochat-Moroccan-Instruct-0.7B is gated and returned 401. Custom-code nanochat repos are deferred unless separately audited.
```

Primary runtime target: `NanoChatForCausalLM` causal text generation, CUDA inference first, prefill and decode with KV cache. The generated `modeling_nanochat.py` is runtime-authoritative; `modular_nanochat.py` is the Transformers edit source.

Source URLs: [modeling_nanochat.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nanochat/modeling_nanochat.py), [configuration_nanochat.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nanochat/configuration_nanochat.py), [modular_nanochat.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/nanochat/modular_nanochat.py), [d20 config](https://huggingface.co/nanochat-students/nanochat-d20/raw/main/config.json), [d32 config PR](https://huggingface.co/karpathy/nanochat-d32/raw/refs%2Fpr%2F1/config.json), [d34 config](https://huggingface.co/pankajmathur/nanochat-d34-sft-hf/raw/main/config.json).

## 2. High-level architecture

NanoChat is a text-only decoder transformer: token embedding -> extra parameter-free RMSNorm -> repeated causal decoder blocks -> final RMSNorm -> untied LM head -> optional final logit tanh softcap.

```text
tokenizer/chat template -> input_ids/attention_mask -> decoder prefill -> KV cache -> decode -> logits/sampling
```

There are no vision/audio/projector branches. CPU/data-pipeline work is tokenizer, chat template, padding, and generation control. GPU/runtime work is embedding lookup, RoPE, attention, MLP, norm, logits, cache update, and optional softcap. `NanoChatModel` base hidden-state output is optional; `NanoChatForCausalLM` is required for the target. Training loss, gradient checkpointing, and output attentions/hidden states are deferred.

## 3. Important config dimensions

| Field | Source behavior / defaults |
| --- | --- |
| `vocab_size` | Default 50304; representative native checkpoints use 65536. |
| `hidden_size` | Projection input/output width. |
| `num_hidden_layers` | Decoder block repeat count. |
| `num_attention_heads` | Query head count. |
| `num_key_value_heads` | Defaults to `num_attention_heads`; native sweep uses MHA, not GQA. |
| `head_dim` | Optional; otherwise `hidden_size // num_attention_heads`. Sweep infers 128. |
| `intermediate_size` | MLP `fc1` output / `fc2` input. |
| `hidden_act` | `relu2`, implemented as squared ReLU. |
| `max_position_embeddings` | RoPE cache/default maximum; sweep uses 2048. |
| `rope_parameters` | Required by runtime source; default type is `{"rope_type":"default","rope_theta":10000.0}` in configs. |
| `attention_bias` | Optional q/k/v/o projection bias; sweep uses false. |
| `final_logit_softcapping` | Source-read field, default 15.0. Some configs instead contain legacy `logits_soft_cap`; for current source this is ignored while the class default still supplies 15.0. |
| `tie_word_embeddings` | False; LM head is a separate logical weight. |
| `use_cache` | True in native representative checkpoints. |

Representative checkpoint sweep:

| Model | Native source? | Layers | H | Heads/KV | D | MLP | Vocab | Softcap | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `nanochat-students/nanochat-d20` | yes | 20 | 1280 | 10/10 | 128 | 5120 | 65536 | effective 15 | HF metadata: 561M params, Apache-2.0. |
| `karpathy/nanochat-d32` `refs/pr/1` | yes | 32 | 2048 | 16/16 | 128 | 8192 | 65536 | effective 15 | Main config 404; Transformers tests pin PR revision. |
| `pankajmathur/nanochat-d34-sft-hf` | yes | 34 | 2176 | 17/17 | 128 | 8704 | 65536 | explicit 15 | 17 heads is unusual but divides hidden size. |
| `viswamaicoe/nanochat-telugu-560M-sft-smoltalk` | yes | 20 | 1280 | 10/10 | 128 | 5120 | 65536 | explicit 15 | Same operator shape as d20, different token IDs/language. |
| `Lyte/nanochat-darija-73m-*` | no | custom | 384 | 3/3 | inferred 128 | not native | 32768 | remote-code | Uses `auto_map`, `n_embd`, `window_pattern`, extra gates. |

## 3a. Family variation traps

- Native source supports MHA, GQA, and MQA structurally through `num_key_value_heads`, even though representative native configs use MHA.
- `head_dim` may be explicit; do not assume `hidden_size == num_heads * head_dim` without checking.
- Projection bias is config-dependent via `attention_bias`; MLP and LM head remain bias-free in source.
- RoPE is applied before q/k RMSNorm. This order is unusual and fusion-sensitive.
- RMSNorm has no learned scale parameter. Do not lower it as ordinary learned RMSNorm.
- The model has an extra norm before the first decoder block and a final norm after all blocks.
- `relu2` is not SwiGLU/GEGLU; MLP is ungated `Linear -> ReLU^2 -> Linear`.
- `final_logit_softcapping` is the current source field. Legacy `logits_soft_cap` in d20/d32 configs is not read by this source.
- `tie_word_embeddings=False`; `_tied_weights_keys` exists, but config says the embedding and LM head are separate weights.
- Custom-code repos with `auto_map` and original nanochat fields are out of scope for this native-source audit.
- No NCHW/NHWC layout translation is relevant for the text-only native model.

## 4. Operator coverage checklist

Tensor/layout ops: token embedding gather `[B,T] -> [B,T,H]`, arange/add/unsqueeze position IDs, 2D/4D causal mask creation, view/reshape, transpose, contiguous, slice for `logits_to_keep`, optional indexed logits slice tensor, residual add.

Neural primitives: parameter-free RMSNorm over last dim with fp32 internal math, Linear/GEMM for q/k/v/o, MLP `Linear(H -> I)`, squared ReLU, `Linear(I -> H)`, untied LM head `Linear(H -> V)`, tanh final softcap.

Attention primitives: causal self-attention, MHA/GQA/MQA head layout, q/k RoPE, q/k post-RoPE RMSNorm, qk matmul scaling by `head_dim**-0.5`, mask add, fp32 softmax, value matmul, output projection.

Position/rotary ops: default RoPE inverse frequency, cos/sin in fp32, NanoChat `rotate_half(x) = cat(x[...,D/2:], -x[...,:D/2])`, dynamic RoPE update hook for non-default rope types.

Generation/cache ops: `DynamicCache` allocation when `use_cache`; per-layer key/value update after RoPE and q/k norm; cache shape before repeat is `[B, num_key_value_heads, T_cache, head_dim]`; eager fallback repeats KV to query heads only for attention computation.

Preprocessing-coupled ops: tokenizer/chat template emits `input_ids` and `attention_mask`. Native d20/d32 tokenizer config uses `PreTrainedTokenizerFast`, `<|bos|>`, `<|assistant_end|>` as EOS/PAD, and chat template on d32 PR.

Distributed/tensor parallel metadata: config has TP plans for q/k/v colwise, o rowwise, MLP fc1 colwise, fc2 rowwise, LM head colwise gather. First DinoML integration can reject distributed TP.

## 5. Layer/block breakdown

For representative d20 (`H=1280`, `A=10`, `KV=10`, `D=128`, `I=5120`) and d32 (`H=2048`, `A=16`, `KV=16`, `D=128`, `I=8192`):

```text
input_ids -> Embedding(V,H)
x = RMSNorm_no_weight(x)                       # extra pre-stack norm
repeat N times:
  residual = x
  x = RMSNorm_no_weight(x)
  q = Linear(H, A*D, bias=attention_bias)(x)
  k = Linear(H, KV*D, bias=attention_bias)(x)
  v = Linear(H, KV*D, bias=attention_bias)(x)
  q,k,v -> view [B,T,heads,D] -> transpose [B,heads,T,D]
  q,k = RoPE(q,k)
  q = RMSNorm_no_weight(q)                     # over head_dim
  k = RMSNorm_no_weight(k)
  k,v = cache_update(k,v, layer_idx) if cache
  attn = causal_attention(q,k,v, mask, scale=D^-0.5)
  attn -> transpose/reshape [B,T,A*D]
  x = residual + Linear(A*D, H, bias=attention_bias)(attn)
  residual = x
  x = RMSNorm_no_weight(x)
  x = residual + Linear(I,H,bias=False)(relu(Linear(H,I,bias=False)(x)) ** 2)
x = RMSNorm_no_weight(x)
logits = Linear(H,V,bias=False)(x[:, logits_slice, :])
logits = softcap * tanh(logits / softcap) if enabled
```

## 6. Attention requirements

Attention is causal self-attention. There is no cross-attention, no sliding window in generated source, no block sparse pattern, and no encoder cache. Native source advertises FlashAttention, SDPA, and flex attention support via Transformers attention interfaces, while eager fallback is dense matmul attention.

Head shape: q `[B,A,Tq,D]`, k/v before repeat `[B,KV,Tkv,D]`. GQA repeat factor is `A // KV`; representative native configs have factor 1. Cached keys are stored after RoPE and after the q/k RMSNorm analog for keys; values are cached after projection/reshape and before any repeat. Masking uses `create_causal_mask` and any caller attention mask. Softmax is explicitly fp32 in eager attention, cast back to query dtype. Dropout is zero for inference.

For DinoML, first admission can require `_attn_implementation` compatible with dense causal attention or a known fused backend, `A % KV == 0`, same key/value head dim, no output attentions, no training dropout, and contiguous dense input embeddings.

## 7. Position encoding and custom math

Default RoPE can precompute `inv_freq`, and cos/sin can be generated per batch/position IDs. Position IDs are `[1,T]` in default generation and offset by `past_key_values.get_seq_length()` during decode.

```python
def nanochat_rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((x2, -x1), dim=-1)

def nanochat_rope_then_norm(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q = q * cos + nanochat_rotate_half(q) * sin
    k = k * cos + nanochat_rotate_half(k) * sin
    return rms_no_weight(q), rms_no_weight(k)

def relu2(x):
    return relu(x) * relu(x)

def final_softcap(logits, cap):
    return tanh(logits / cap) * cap
```

Dynamic or non-default RoPE types flow through `ROPE_INIT_FUNCTIONS` and `dynamic_rope_update`; first integration should admit only `rope_type="default"` unless a separate RoPE variant is tested.

## 8. Preprocessing and input packing

The model consumes token IDs and optional attention mask only. Tokenizer/chat behavior is ABI-level, not neural graph work. Native tokenizer configs observed for d20/d32 use `PreTrainedTokenizerFast`, special tokens `<|bos|>`, `<|assistant_end|>` as EOS/PAD, and user/assistant/python/output special tokens. d32 PR tokenizer config names `chat_template.jinja`; d20 tokenizer config did not include that field in the raw snippet inspected.

GPU inputs: `input_ids [B,T]` or `inputs_embeds [B,T,H]`, exactly one required; optional `attention_mask`, optional `position_ids`, optional cache. For first DinoML path, prefer owning `input_ids`, `attention_mask`, `position_ids`, and cache; treat direct `inputs_embeds` as a later expert ABI.

## 9. Graph rewrite / lowering opportunities

### Rewrite: parameter-free RMSNorm

Source pattern: `x.float(); x * rsqrt(mean(x*x, dim=-1, keepdim=True) + eps); cast back`.

Replacement: fused no-weight RMSNorm kernel. Preconditions: reduce over final dim, no affine weight, fp32 accumulation, preserve output dtype. Failure cases: configs or remote code with learned scale.

Parity sketch: random `[B,T,H]` and `[B,A,T,D]`, compare fp32/fp16/bf16 to PyTorch with fp32 reduce.

### Rewrite: QKV separate GEMMs to packed projection

Source pattern: separate q/k/v Linear from same normalized hidden state.

Replacement: packed GEMM producing `[q,k,v]` slices, or grouped GEMM. Preconditions: identical input, compatible bias setting, weight packing preserves row order `q_proj`, `k_proj`, `v_proj`; split widths are `A*D`, `KV*D`, `KV*D`. Failure cases: tensor-parallel sharding, nonuniform explicit head dims, quantized packed checkpoints.

### Rewrite: RoPE + q/k norm + attention

Source pattern: q/k reshape-transpose, RoPE, parameter-free q/k RMSNorm, cache update, causal attention.

Replacement: fused pre-attention transform plus FlashAttention-compatible kernel. Preconditions: default RoPE, q/k cached after norm, no output attentions, dropout 0, supported mask. Failure cases: non-default dynamic RoPE, requested dense attention weights, unsupported GQA.

### Rewrite: relu2 MLP

Source pattern: `fc2(relu(fc1(x)) ** 2)`.

Replacement: GEMM + fused squared-ReLU epilogue + GEMM. Preconditions: `hidden_act=="relu2"`, bias-free MLP. Failure cases: remote-code MLP variants or extra gates.

### Rewrite: last-token-only logits

Source pattern: `hidden_states[:, slice_indices, :]` before LM head.

Replacement: during decode/generation set `logits_to_keep=1` and launch LM head for last token only. Preconditions: no loss, caller does not need full-prefix logits. Failure cases: perplexity/eval needing all logits or tensor index slice requiring arbitrary positions.

## 10. Kernel fusion candidates

Highest priority:

- No-weight RMSNorm, because it appears before the stack, twice per layer, after the stack, and on q/k heads.
- Causal attention with RoPE-before-qk-norm and KV cache, because decode throughput depends on preserving the exact cached representation.
- GEMM epilogues for `relu2` and final tanh softcap.
- Last-token-only LM head for decode.

Medium priority:

- QKV packed projection and optional bias handling.
- Fused residual add around attention/MLP.
- Mask construction/cached position ID generation for decode.
- GQA repeat elimination inside attention kernels.

Lower priority:

- Tensor-parallel plan support.
- Dynamic/non-default RoPE variants.
- Output attentions and hidden-state capture.
- Full-prefix LM head optimization for evaluation workloads.

## 11. Runtime staging plan

Stage 1: parse native `NanoChatConfig`, reject remote-code/original-field configs, load dense bf16/fp32 weights, and run embedding plus one block parity.

Stage 2: implement full prefill without cache using eager dense causal attention; support d20 dimensions first.

Stage 3: add KV cache ABI where cached k is post-RoPE/post-norm and v is projected value; validate one-token decode.

Stage 4: add optimized causal attention backend for MHA, then GQA/MQA if configs appear.

Stage 5: enable last-token-only logits and final softcap; validate greedy generation.

Stage 6: add packed QKV, MLP epilogue fusion, and RMSNorm fusion.

Stage 7: consider tensor parallel and quantized/GGUF loading as separate provider work.

## 12. Parity and validation plan

- Unit tests: no-weight RMSNorm on `[2,7,H]` and `[2,A,7,D]`; `rotate_half`; default RoPE cos/sin; `relu2`; final softcap.
- Projection tests: q/k/v/o and MLP weights copied from PyTorch, compare per-op outputs.
- Single-layer parity: d20-sized random config with cache disabled, fp32 tolerance `rtol=1e-5`, bf16 tolerance around `rtol=3e-2` depending on backend.
- Full prefill logits parity: `"Hello world"` using d20/d32 checkpoints; Transformers tests include expected logits means/slices for d20 and d32.
- Decode parity: prefill a prompt, decode one greedy token with cache, compare logits and cache lengths.
- End-to-end parity: d20 greedy chat test from Transformers expects "The capital of France is Paris." for the first prompt.
- Rejection tests: custom-code Lyte-style config, gated/unavailable config, `rope_type` non-default if unsupported, `attention_bias=True` if the first build omits bias.

## 13. Performance probes

- Prefill throughput sweep: batch `{1,4,8}`, sequence `{128,512,2048}`.
- Decode tokens/sec sweep: batch `{1,8,32}`, cache length `{128,1024,2048}`.
- Attention backend comparison: eager dense, SDPA-like, FlashAttention-like with post-RoPE q/k norm.
- RMSNorm microbench: no-weight norm over `H` and over `D=128` q/k heads.
- MLP microbench: GEMM + `relu2` epilogue versus unfused activation.
- LM head probe: full-prefix logits versus `logits_to_keep=1`.
- KV cache memory: layers * 2 * B * KV * T * D * dtype bytes; d20 bf16 MHA is about `20*2*B*10*T*128*2` bytes.

## 14. Skip/defer list

Training, labels/loss, gradient checkpointing, dropout, output attentions, hidden-state capture, beam search parity, distributed tensor parallel, non-default/dynamic RoPE, remote-code nanochat variants, original checkpoint conversion, gated custom-code repos, quantized/packed loading, and arbitrary `inputs_embeds` entrypoints can be deferred for first integration.

## 15. Final implementation checklist

- [ ] Parse native `NanoChatConfig` and reject remote-code/original-field configs.
- [ ] Load token embedding, per-layer q/k/v/o, MLP, norms, and untied LM head weights.
- [ ] Implement no-weight RMSNorm with fp32 reduction.
- [ ] Implement default RoPE with NanoChat `rotate_half`.
- [ ] Implement q/k post-RoPE no-weight norm.
- [ ] Implement causal MHA first, then admit GQA/MQA with `A % KV == 0`.
- [ ] Implement KV cache storing post-RoPE/post-norm keys and projected values.
- [ ] Implement `relu2` MLP.
- [ ] Implement `logits_to_keep` and final tanh softcap.
- [ ] Add single-op, single-layer, prefill-logits, and one-token decode parity tests.
- [ ] Add d20/d32 greedy generation smoke tests.
- [ ] Benchmark prefill, decode, MLP, RMSNorm, attention, LM head, and KV memory.
