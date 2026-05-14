# HyperCLOVAX Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Primary model id: naver-hyperclovax/HyperCLOVAX-SEED-Think-14B
Config source: https://huggingface.co/naver-hyperclovax/HyperCLOVAX-SEED-Think-14B/raw/main/config.json
Hub revision observed: ddcb423f12db90bf4a27f97028c4b3f8b6c3c25c
Source files inspected:
- X:/H/transformers/src/transformers/models/hyperclovax/configuration_hyperclovax.py
- X:/H/transformers/src/transformers/models/hyperclovax/modeling_hyperclovax.py
- X:/H/transformers/src/transformers/models/hyperclovax/modular_hyperclovax.py
- X:/H/transformers/docs/source/en/model_doc/hyperclovax.md
- X:/H/transformers/tests/models/hyperclovax/test_modeling_hyperclovax.py
Local snapshots:
- config_hub.json
- hub_model_api.json
- generation_config_hub.json
- special_tokens_map_hub.json
- config_think_32b.json
- config_omni_8b.json
- config_text_0_5b.json
Any missing files or assumptions:
- The audited built-in Transformers source covers model_type="hyperclovax" text-only causal LM.
- The public 14B Hub repo is not gated. Safetensors metadata reports 14,748,112,896 F32 parameters.
- naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B returned restricted-access for config.json; gated/restricted access would be needed to inspect it.
- naver-hyperclovax/HyperCLOVAX-SEED-Think-32B and Omni-8B are remote-code VLM wrapper configs, not the built-in HyperCLOVAX model body, but the 32B text_config is HyperCLOVAX-shaped and useful for family variation.
```

## 2. High-level architecture

HyperCLOVAX in the inspected source is a text-only decoder-only causal language model. It is Granite-like, with HyperCLOVAX-specific MuP scaling and optional Peri-Layer post-normalization after attention and MLP sublayer outputs.

```text
token ids / optional input embeddings -> token embedding scale -> repeated decoder blocks -> final RMSNorm -> LM head -> logits scale -> generation
```

Generation decomposition:

```text
CPU tokenizer + ChatML template -> prefill decoder with causal mask and RoPE -> autoregressive decode with per-layer KV cache -> logits/sampling
```

The tokenizer/chat-template work is CPU/data-pipeline work. Prefill, decode, RMSNorm, RoPE, attention, MLP, and logits projection are GPU/runtime work. The 32B VLM wrapper config includes independently cacheable vision/projector stages, but those are outside the built-in `hyperclovax` source audited here.

## 3. Important config dimensions

Primary public checkpoint, from `config_hub.json`:

| Field | Value | Source |
|---|---:|---|
| `model_type` | `hyperclovax` | config.json |
| `hidden_size` | 6144 | config.json |
| `num_hidden_layers` | 38 | config.json |
| `num_attention_heads` | 48 | config.json |
| `num_key_value_heads` | 8 | config.json |
| `head_dim` | 128 | config.json |
| Q width | 6144 | source + config |
| KV width | 1024 each | source + config |
| `intermediate_size` | 14336 | config.json |
| `vocab_size` | 110592 | config.json |
| `max_position_embeddings` | 131072 | config.json |
| RoPE theta | 100000000 | config.json legacy `rope_theta` |
| `hidden_act` | `silu` | config.json |
| `attention_bias` / `mlp_bias` | false / false | config.json |
| `rms_norm_eps` | 1e-5 | config.json |
| `embedding_multiplier` | 10.0 | config.json |
| `attention_multiplier` | 0.0078125 | config.json |
| `residual_multiplier` | 1.0 | config.json |
| `logits_scaling` | 0.125 | config.json |
| `use_post_norm` | true | config.json |
| `use_cache` | false | config/generation_config; source supports cache |
| `tie_word_embeddings` | false | config.json |
| Hub safetensors dtype/count | F32, 14.748B params | Hub API metadata |

Representative sweep:

| Model/config | Built-in source target? | Text model | Layers | Hidden | Heads/KV | Head dim | MLP | Context | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| HyperCLOVAX-SEED-Think-14B | yes | HyperCLOVAX | 38 | 6144 | 48/8 | 128 | 14336 | 131072 | `use_post_norm=true`, `embedding_multiplier=10`, `attention_multiplier=1/128` |
| HyperCLOVAX-SEED-Think-32B VLM `text_config` | remote VLM wrapper; text branch is HyperCLOVAX-shaped | HyperCLOVAX | 72 | 5120 | 40/8 | 128 | 24192 | 131072 | `use_post_norm=false`, `attention_multiplier=1/sqrt(128)` |
| HyperCLOVAX-SEED-Text-Instruct-0.5B | no, Llama config | Llama | 24 | 1024 | 16/8 | 128 | 4096 | 8192 | Not `hyperclovax`; useful name trap |
| HyperCLOVAX-SEED-Omni-8B `text_config` | no, VLM/Llama config | Llama | 36 | 4096 | 32/8 | 128 | 12288 | 8192 | Multimodal remote-code wrapper, not this source |
| HyperCLOVAX-SEED-Text-Instruct-1.5B | not inspected | unknown | unknown | unknown | unknown | unknown | unknown | unknown | restricted config access |

## 3a. Family variation traps

- `hidden_size == num_attention_heads * head_dim` for inspected 14B, but source has explicit `head_dim`; do not infer projection widths only from hidden size.
- GQA is required: 14B has 48 query heads and 8 KV heads, so KV must repeat by 6 for eager attention.
- `attention_multiplier` is a config value. 14B uses `0.0078125`, not the default `1 / sqrt(128)`.
- `embedding_multiplier` and `logits_scaling` are required math, not training-only annotations.
- `use_post_norm` changes the block graph. 14B enables post attention/MLP RMSNorm; 32B text_config disables it.
- Hub 14B config uses legacy `rope_theta`/`rope_scaling`, while the generated source reads `config.rope_parameters["rope_type"]` and `["rope_theta"]`. DinoML should normalize both config spellings.
- Hub config contains historical fields not read by the inspected source, including `attn_pdrop`, `embd_pdrop`, `resid_pdrop`, `summary_first_dropout`, `pretraining_tp`, and `end_token_id`.
- `use_cache=false` in config/generation_config does not mean the source lacks cache support; `DynamicCache` is created when `use_cache` is requested.
- `tie_word_embeddings=false` for 14B, but `HyperCLOVAXForCausalLM` declares tied-weight keys for compatibility. Respect the effective config.
- HyperCLOVAX-branded Hub repos include VLM and Llama-backed models; do not route all `naver-hyperclovax/*` names to this source.
- ChatML special tokens and `force_reasoning` / `skip_reasoning` are tokenizer/controller behavior, not decoder graph ops.

## 4. Operator coverage checklist

Tensor/layout ops:
- Token embedding lookup `[batch, seq] -> [batch, seq, hidden]`.
- Multiply embeddings by scalar `embedding_multiplier`.
- View/reshape Q/K/V to `[batch, seq, heads, head_dim]`, transpose to `[batch, heads, seq, head_dim]`.
- Contiguous/reshape after attention back to `[batch, seq, hidden]`.
- Slice logits with `logits_to_keep`, including int tail slice and tensor index modes.

Neural network primitives:
- RMSNorm over last axis with fp32 variance accumulation and learned weight.
- Bias-free Linear for 14B: Q `6144 -> 6144`, K `6144 -> 1024`, V `6144 -> 1024`, O `6144 -> 6144`.
- SwiGLU MLP: gate/up `6144 -> 14336`, SiLU, multiply, down `14336 -> 6144`.
- LM head `6144 -> 110592`, bias false.
- Residual add with scalar `residual_multiplier`.
- Optional `post_norm1` and `post_norm2` RMSNorm.

Attention primitives:
- Causal self-attention.
- GQA repeat KV from 8 heads to 48 heads.
- RoPE on Q/K before cache update.
- Mask add before softmax.
- Softmax in fp32 then cast back to query dtype.
- Attention dropout only during training.
- Backend dispatch through eager, SDPA, FlashAttention, or FlexAttention interfaces.

Position/rotary ops:
- Default RoPE inverse-frequency generation from `rope_theta` and `head_dim`.
- Dynamic RoPE update wrapper is present for advanced rope types.
- `rotate_half` with first-half/second-half split.

Generation/cache ops:
- Dynamic per-layer KV cache update.
- Position IDs default to `arange(seq) + past_seen_tokens`.
- Causal mask creation using `create_causal_mask`.

Preprocessing-coupled ops:
- ChatML template inserts role tokens, optional tool list, optional assistant thinking prefix, and special end tokens.
- Tokenizer special token ids use `<|endoftext|>` id 100257 for BOS/EOS/PAD in 14B.

Distributed/tensor-parallel ops:
- Source declares a tensor-parallel plan: Q/K/V/gate/up columnwise, O/down rowwise, LM head colwise gather.

No sparse/local/block attention, hash/sort attention, multimodal scatter, recurrent/state-space cache, or quantized packed-weight metadata is required by the built-in text-only source.

## 5. Layer/block breakdown

Decoder block, repeated `num_hidden_layers` times:

```text
residual = x
x = RMSNorm(x)
q = Linear(hidden -> num_attention_heads * head_dim, bias=attention_bias)
k = Linear(hidden -> num_key_value_heads * head_dim, bias=attention_bias)
v = Linear(hidden -> num_key_value_heads * head_dim, bias=attention_bias)
q,k = RoPE(q,k, cos, sin)
k,v = cache.update(k,v) if cache is present
attn = causal_attention(q,k,v, mask, scaling=attention_multiplier)
x = Linear(attn, num_attention_heads * head_dim -> hidden, bias=attention_bias)
x = post_norm1(x) if use_post_norm else x
x = residual + x * residual_multiplier

residual = x
x = RMSNorm(x)
x = Linear(SiLU(Linear(x)) * Linear(x), intermediate -> hidden)
x = post_norm2(x) if use_post_norm else x
x = residual + x * residual_multiplier
```

For 14B, all attention and MLP projections are bias-free. Shapes are `hidden=6144`, `head_dim=128`, Q heads 48, KV heads 8, intermediate 14336.

## 6. Attention requirements

- Type: causal self-attention only for built-in HyperCLOVAX.
- Head pattern: GQA, with `num_attention_heads / num_key_value_heads` repeat groups.
- 14B shapes: Q `[B, 48, Q, 128]`, K/V before repeat `[B, 8, K, 128]`, K/V after repeat `[B, 48, K, 128]`.
- Cache shape before repeat stores KV heads, not expanded query heads: per layer K/V `[B, 8, cached_seq, 128]`.
- RoPE is applied before cache update, so cached keys are already position encoded.
- Masking: causal mask from `create_causal_mask`; eager path adds mask to attention scores before fp32 softmax.
- Rectangular prefill/decode: query length can be 1 during decode while key/value length includes past tokens.
- Sliding window/local attention: not present in inspected HyperCLOVAX source.
- ALiBi/relative bias: not present.
- Packed/varlen support: not explicit in the model body, but backend attention interfaces may accept backend-specific kwargs.
- Optimized backends: source advertises FlashAttention, SDPA, and FlexAttention support. Parity must preserve scaling, RoPE-before-cache, mask order, fp32 softmax for eager reference, and dropout behavior.

## 7. Position encoding and custom math

Default RoPE can precompute inverse frequencies for a config/head_dim/theta pair. Cos/sin depend on runtime `position_ids` and are computed in fp32, then cast to the hidden-state dtype.

```python
def hyperclovax_default_rope(config, position_ids, dtype):
    dim = config.head_dim
    theta = config.rope_parameters["rope_theta"]
    inv_freq = 1.0 / (theta ** (arange(0, dim, 2).float() / dim))
    freqs = inv_freq[None, :, None] @ position_ids[:, None, :].float()
    emb = cat([freqs.transpose(1, 2), freqs.transpose(1, 2)], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def apply_hyperclovax_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    rotate = lambda x: cat([-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]], dim=-1)
    return q * cos + rotate(q) * sin, k * cos + rotate(k) * sin
```

Custom scalar math:
- `inputs_embeds *= embedding_multiplier`.
- Attention logits use `attention_multiplier`.
- Sublayer residuals add `sublayer_output * residual_multiplier`.
- LM logits multiply by `logits_scaling`. This differs from Granite, which divides in the referenced comment.

## 8. Preprocessing and input packing

Text-only preprocessing:
- Tokenizer emits `input_ids` and optional `attention_mask`; these enter the GPU graph.
- If `position_ids` are not supplied, source builds contiguous positions offset by cache length.
- Exactly one of `input_ids` or `inputs_embeds` must be supplied.
- 14B special token map uses `<|endoftext|>` as BOS/EOS/PAD/UNK content.

Chat/controller behavior:
- Docs state a ChatML-based format with `<|im_start|>`, `<|im_end|>`, `<|endofturn|>`, and `<|stop|>`.
- `apply_chat_template` supports `force_reasoning=True` and `skip_reasoning=True`; this selects assistant prefix mode, not a decoder-graph branch.
- Tool/function-call formatting lives in tokenizer template metadata and can be stubbed for first decoder parity if raw token IDs are supplied.

No image/audio/video preprocessing is required by the built-in `hyperclovax` source. VLM wrapper configs under the same organization are separate remote-code families and should get separate audits.

## 9. Graph rewrite / lowering opportunities

### Rewrite: bias-free Linear to GEMM

Source pattern:
```text
y = nn.Linear(in, out, bias=False)(x)
```

Replacement:
```text
reshape batch tokens -> GEMM(x, weight.T) -> restore [B, S, out]
```

Preconditions:
- Weight is dense 2D `[out_features, in_features]`.
- No bias or bias explicitly handled.
- Last-axis input contiguous or lowered with correct strides.

Failure cases:
- Quantized/packed remote-code weights, tensor-parallel shards without gather/partition plan, or nonstandard tied aliases.

Parity test sketch:
- Compare each projection and full block against PyTorch with fp32 and bf16 tolerances.

### Rewrite: QKV projection grouping

Source pattern:
```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:
```text
one grouped/fused GEMM producing [Q, K, V] outputs, then split as Q all rows, K all rows, V all rows
```

Preconditions:
- Separate source weights remain logically separate or are packed with explicit metadata.
- Split order is exactly Q projection output, K projection output, V projection output.
- Bias presence matches `attention_bias`.

Failure cases:
- Existing checkpoints are not source-packed QKV; do not assume a packed on-disk layout.

### Rewrite: SwiGLU MLP fusion

Source pattern:
```text
down_proj(silu(gate_proj(x)) * up_proj(x))
```

Replacement:
```text
dual GEMM -> fused SiLU multiply -> GEMM
```

Preconditions:
- `hidden_act == "silu"`.
- Gate and up outputs have identical shape `[B, S, intermediate_size]`.

Failure cases:
- Other activations or MLP bias variants need separate kernels.

### Rewrite: last-token-only logits

Source pattern:
```text
lm_head(hidden_states[:, -logits_to_keep:, :])
```

Replacement:
```text
slice hidden before GEMM; for decode use only final token hidden state
```

Preconditions:
- `logits_to_keep` is an int or tensor index with known semantics.
- Loss computation is not requesting all logits.

Failure cases:
- Training/loss path with labels over full sequence.

## 10. Kernel fusion candidates

Highest priority:
- RMSNorm, because it appears before every attention and MLP plus final norm and optional post norms.
- GQA FlashAttention with RoPE-before-cache and KV cache, because prefill/decode throughput is attention-bound at 131k context.
- SwiGLU activation multiply, because each block has two large input GEMMs feeding an elementwise SiLU multiply.
- Last-token-only logits, because 110592 vocab projection is expensive during decode.

Medium priority:
- Q/K/V grouped projection plus RoPE application.
- Residual multiplier fused with residual add.
- Embedding multiplier fused with embedding output or first norm input.
- LM head logits scaling fused into projection epilogue.

Lower priority:
- Training dropout paths.
- Tensor-parallel collectives, unless multi-GPU serving is in scope.
- Dynamic/advanced RoPE variants beyond default/legacy theta normalization.

## 11. Runtime staging plan

Stage 1: parse config and normalize Hub legacy RoPE fields into DinoML config state. Load weights for embeddings, one decoder block, final norm, and LM head.

Stage 2: implement one-block parity with RMSNorm, Q/K/V/O Linear, default RoPE, causal eager attention, SwiGLU, optional post norms, and MuP scalars.

Stage 3: full prefill parity for fixed batch/sequence without cache. Stub tokenizer by accepting token IDs directly.

Stage 4: decode parity with DynamicCache-equivalent per-layer KV storage, position offset, and last-token logits.

Stage 5: optimized attention path using SDPA/FlashAttention-compatible lowering with GQA KV storage, RoPE-before-cache, and mask/scaling parity.

Stage 6: add graph fusions for RMSNorm, QKV, SwiGLU, residual scalars, and logits slice/projection.

Stage 7: production scheduling for long context and batching. Chat template, function calling, and reasoning-mode prefixes can stay in CPU/controller code.

## 12. Parity and validation plan

- Config roundtrip: 14B config plus source defaults, including legacy `rope_theta` normalization and ignored historical fields.
- Unit tests: RMSNorm fp32 accumulation, RoPE cos/sin and `rotate_half`, GQA repeat, attention scaling, residual/logit/embedding multipliers.
- Single-layer parity: random tiny HyperCLOVAX config with and without `use_post_norm`, with GQA and MHA variants.
- Full small-model parity: generated small config through `HyperCLOVAXForCausalLM` for prefill logits.
- 14B smoke parity: selected logits from integration test input IDs `[105319, 21028, 107115, 16969, 102949, 80052, 13]` if weights and hardware are available.
- Decode parity: one prefill step then several single-token decode steps, verifying cache length and logits.
- Tolerances: fp32 `rtol=1e-5, atol=1e-5` for isolated ops; bf16/fp16 block/logit parity around `rtol=1e-2, atol=1e-2`, matching upstream integration-test scale.

## 13. Performance probes

- Tokenizer/chat-template throughput separately from model runtime.
- Prefill tokens/sec sweep over sequence length, including long-context cases.
- Decode tokens/sec sweep over batch size and cache length.
- KV cache memory per layer and total: `2 * layers * batch * kv_heads * seq * head_dim * dtype_size`.
- Attention backend comparison: eager reference, SDPA, FlashAttention/FlexAttention candidates.
- MLP GEMM throughput for `6144 -> 14336 -> 6144`.
- LM head throughput and last-token-only projection savings.
- RMSNorm fusion impact with and without `use_post_norm`.
- Weight load bandwidth and optional dense/quantized provider experiments, clearly separate from source-required behavior.

## 14. Skip/defer list

- Training, gradients, dropout, and gradient checkpointing.
- Tensor-parallel execution beyond preserving plan metadata.
- Beam search, sampling policies, and function-calling controller behavior.
- VLM, audio, video, and discrete-token Omni paths from remote-code wrapper configs.
- Restricted 1.5B checkpoint support until access is available.
- Quantization and float8 tensor-parallel generation skipped upstream.
- Advanced RoPE variants unless a public HyperCLOVAX checkpoint uses them.

## 15. Final implementation checklist

- [ ] Parse `HyperCLOVAXConfig`, including `head_dim`, GQA heads, MuP scalars, `use_post_norm`, and legacy RoPE fields.
- [ ] Load dense weights with correct aliases for embeddings and LM head.
- [ ] Implement fp32-accumulating RMSNorm.
- [ ] Implement default RoPE and `rotate_half`.
- [ ] Implement causal GQA attention with KV cache stored before repeat and after RoPE.
- [ ] Implement bias-optional Q/K/V/O projections.
- [ ] Implement SwiGLU MLP with `hidden_act="silu"`.
- [ ] Implement embedding, residual, attention, and logits scaling.
- [ ] Implement `logits_to_keep` slicing before LM head.
- [ ] Add one-block parity tests for `use_post_norm=true/false`.
- [ ] Add prefill logits parity against a tiny source model.
- [ ] Add decode/cache parity with position offset.
- [ ] Add 14B config smoke test and gated/restricted-link handling.
- [ ] Benchmark prefill, decode, RMSNorm, MLP, attention backend, and LM head slices.
