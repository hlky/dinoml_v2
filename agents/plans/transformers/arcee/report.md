# Arcee Transformers Family Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: arcee-ai/AFM-4.5B-Base, arcee-ai/AFM-4.5B, arcee-ai/AFM-4.5B-Preview, arcee-ai/AFM-4.5B-Base-Pre-Anneal
Config source: raw Hugging Face config.json snapshots saved beside this report
Source files inspected:
- X:/H/transformers/src/transformers/models/arcee/modular_arcee.py
- X:/H/transformers/src/transformers/models/arcee/configuration_arcee.py
- X:/H/transformers/src/transformers/models/arcee/modeling_arcee.py
- X:/H/transformers/tests/models/arcee/test_modeling_arcee.py
- X:/H/transformers/src/transformers/modeling_rope_utils.py
- X:/H/transformers/src/transformers/activations.py
Any missing files or assumptions:
- modeling_arcee.py and configuration_arcee.py are generated from modular_arcee.py; future source edits should target modular_arcee.py.
- No arcee tokenizer source is model-specific in Transformers. Tokenization is ordinary AutoTokenizer/repo metadata.
- arcee-ai/AFM-4.5B-GGUF is an official quantized repo, but https://huggingface.co/arcee-ai/AFM-4.5B-GGUF/raw/main/config.json returned "Entry not found"; treat it as a weight packaging/loading target derived from arcee-ai/AFM-4.5B, not as a separate native config snapshot.
- arcee-ai/AFM-4.5B-Base-KDA-* repos advertise model_type arcee_kda/custom_code and are out of scope for this arcee native-source report.
```

Source URLs:
- [modular_arcee.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/arcee/modular_arcee.py)
- [modeling_arcee.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/arcee/modeling_arcee.py)
- [configuration_arcee.py](https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/arcee/configuration_arcee.py)

Saved config snapshots:
- `config_AFM-4.5B-Base.json`
- `config_AFM-4.5B.json`
- `config_AFM-4.5B-Preview.json`
- `config_AFM-4.5B-Base-Pre-Anneal.json`

## 2. High-level architecture

Primary DinoML target: text-only causal LM, inference first, CUDA target.

Dataflow:

```text
tokenizer/input_ids -> token embedding -> N decoder blocks -> final RMSNorm -> lm_head -> logits/sampling
```

Stage decomposition:
- CPU/data pipeline: tokenize text, construct `input_ids`, optional `attention_mask`, optional generation-controller choices.
- Prefill: embed `[B, S]`, build causal mask, compute RoPE cos/sin for positions, run all decoder layers, optionally return per-layer KV cache.
- Decode: embed one or more new tokens, compute position ids from cache length, update per-layer KV cache, run causal self-attention against cached keys/values, compute only needed logits via `logits_to_keep`.
- Optional heads: sequence classification, token classification, and QA wrappers exist through generic HF heads, but causal LM is the required first target.

Arcee is Llama-like in block structure but not a direct Llama operator clone: the MLP is ungated `Linear -> relu2 -> Linear`, while sampled production configs use GQA and YaRN long-context RoPE.

## 3. Important config dimensions

Source defaults from `ArceeConfig`:

| field | default | operator impact |
|---|---:|---|
| `vocab_size` | 32000 | embedding/lm_head rows |
| `hidden_size` | 2560 | residual width |
| `intermediate_size` | 18432 | MLP up/down width |
| `num_hidden_layers` | 32 | decoder repeat count |
| `num_attention_heads` | 32 | Q heads |
| `num_key_value_heads` | defaults to heads | MHA by default; configs may use GQA |
| `head_dim` | `hidden_size // num_attention_heads` | projection and RoPE width |
| `hidden_act` | `relu2` | ReLU squared, not SiLU/SwiGLU |
| `max_position_embeddings` | 4096 | RoPE/cache max unless config scales |
| `rms_norm_eps` | 1e-5 | RMSNorm epsilon |
| `attention_bias` | false | Q/K/V/O bias absent in sampled configs |
| `mlp_bias` | false | MLP bias absent in sampled configs |
| `tie_word_embeddings` | false | lm_head not tied in config, despite `_tied_weights_keys` alias support |
| `use_cache` | true | model supports DynamicCache |

Representative checkpoint sweep, from saved `config.json` snapshots:

| checkpoint | task/repo metadata | vocab | layers | hidden | Q heads | KV heads | head dim | intermediate | max pos | RoPE | dtype | `use_cache` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| arcee-ai/AFM-4.5B-Base-Pre-Anneal | text-generation | 128256 | 36 | 2560 | 20 | 4 | 128 | 18432 | 4096 | default | bfloat16 | false |
| arcee-ai/AFM-4.5B-Base | text-generation | 128004 | 36 | 2560 | 20 | 4 | 128 | 18432 | 65536 | YaRN factor 20, original 4096 | bfloat16 | false |
| arcee-ai/AFM-4.5B | conversational/text-generation | 128005 | 36 | 2560 | 20 | 4 | 128 | 18432 | 65536 | YaRN factor 20, original 4096 | bfloat16 | false |
| arcee-ai/AFM-4.5B-Preview | conversational/text-generation | 128064 | 36 | 2560 | 20 | 4 | 128 | 18432 | 65536 | YaRN factor 20, original 4096 | bfloat16 | false |

HF Hub metadata reports all four as Apache-2.0 arcee text-generation models. Parameter counts in Hub metadata are about 4.619B for Base/instruct and 4.6195B for Preview.

## 3a. Family variation traps

- Checkpoint configs use `num_attention_heads=20`, `num_key_value_heads=4`, `head_dim=128`, so GQA repeat factor is 5. Do not assume MHA.
- Source defaults use 32 heads and default KV heads, while production configs use 20/4. Shape code must honor explicit config, not defaults.
- `hidden_size == num_heads * head_dim` in sampled configs, but source defines projections from `num_attention_heads * head_dim` and only validates `hidden_size % num_attention_heads`; keep `head_dim` explicit.
- MLP is not gated. It has no `gate_proj` and no activation multiply. It is `down_proj(relu(up_proj(x)) ** 2)`.
- Long-context checkpoints use legacy `rope_scaling` in config JSON. Current Transformers normalizes it to `rope_parameters`; DinoML config loading should do the same.
- YaRN RoPE changes both inverse-frequency mixing and `attention_scaling`; default RoPE is still needed for pre-anneal configs.
- Sampled configs set `use_cache=false`, but source supports `DynamicCache` when `use_cache` is enabled. First prefill parity can ignore cache; production decode needs cache support or an explicit no-cache route.
- Vocab and EOS differ: pre-anneal/base use EOS 128001, instruct/preview use EOS 128003, and vocab ranges 128004-128256.
- `pretraining_tp` appears in some configs, but native modular config sets `pretraining_tp = AttributeError()` and modeling source does not implement tensor-parallel slicing logic around it. Treat it as ignored for this native source basis.
- No image/audio/video branches, no NHWC/NCHW layout concerns, no placeholder scatter, no MoE in native arcee.
- KDA repos with `arcee_kda` and custom code are separate audits; route or reject for this report.

## 4. Operator coverage checklist

Tensor/layout ops:
- Embedding lookup: `input_ids [B,S] -> [B,S,H]`.
- `view`, `reshape`, `transpose(1,2)`, `contiguous`, slice for `logits_to_keep`.
- Residual adds over `[B,S,H]`.
- Optional `attention_mask` to causal mask conversion outside core block.

Neural primitives:
- Linear projections:
  - Q: `Linear(2560 -> 20*128=2560)`, no bias in sampled configs.
  - K: `Linear(2560 -> 4*128=512)`, no bias.
  - V: `Linear(2560 -> 512)`, no bias.
  - O: `Linear(2560 -> 2560)`, no bias.
  - MLP up: `Linear(2560 -> 18432)`, no bias.
  - MLP down: `Linear(18432 -> 2560)`, no bias.
  - LM head: `Linear(2560 -> vocab_size)`, no bias.
- RMSNorm over last dim, fp32 variance, output cast back to input dtype.
- ReLU squared activation.

Attention primitives:
- Causal self-attention.
- GQA repeat of K/V from 4 KV heads to 20 Q heads.
- Attention score matmul, scale by `head_dim ** -0.5`, add mask, fp32 softmax, dropout only for training, value matmul.
- Backend dispatch through HF `ALL_ATTENTION_FUNCTIONS`: eager, SDPA, FlashAttention 2, and flex attention are advertised by source.

Position/rotary ops:
- RoPE cos/sin generation in fp32.
- `rotate_half`, broadcast cos/sin at `unsqueeze_dim=1`, apply to Q and K before cache update.
- Default RoPE and YaRN RoPE required by sampled configs.

Generation/cache ops:
- DynamicCache allocation when `use_cache and past_key_values is None`.
- Per-layer cache update after RoPE.
- Position ids default to `arange(current_seq) + past_seen_tokens`.
- Optional last-token-only or selected-token logits via `logits_to_keep`.

Quantized/packed weight metadata:
- Native arcee source has ordinary dense Linear weights.
- AFM-4.5B-GGUF is a packaging/loading concern, not a source-level quantized operator family. DinoML can map it to existing GGUF encoded constant flows if tensor names/shapes are verified separately.

Optional heads:
- Sequence classification: optional/deferred for causal LM target.
- Token classification: optional/deferred.
- Question answering: optional/deferred; has BC `base_model_prefix="transformer"`.

## 5. Layer/block breakdown

For sampled 4.5B configs, repeated 36 times:

```text
x: [B, S, 2560]
residual = x
x = RMSNorm(x)
q = Linear(2560 -> 2560)(x).view(B,S,20,128).transpose(1,2)
k = Linear(2560 -> 512)(x).view(B,S,4,128).transpose(1,2)
v = Linear(2560 -> 512)(x).view(B,S,4,128).transpose(1,2)
q,k = RoPE(q,k, cos[position_ids], sin[position_ids])
k,v = cache.update(k,v, layer_idx) if cache is enabled
attn = causal_attention(q,k,v, repeat_kv=5, scale=1/sqrt(128), mask)
x = residual + Linear(2560 -> 2560)(attn.transpose_to_BSH)
residual = x
x = RMSNorm(x)
x = Linear(18432 -> 2560)(relu(Linear(2560 -> 18432)(x)) ** 2)
x = residual + x
```

After all blocks:

```text
x = final RMSNorm(x)
logits = lm_head(x[:, selected_positions, :])
```

Biases are disabled in all sampled checkpoint configs for attention and MLP projections.

## 6. Attention requirements

Required variant: causal decoder self-attention with GQA.

Details:
- Query heads: 20.
- KV heads: 4.
- KV repeat groups: 5.
- Head dim: 128.
- Query width: 2560.
- Key/value projection width: 512 each.
- Cached key/value shape per layer before repeat: `[B, 4, cache_seq, 128]`.
- Attention computation shape after repeat: query `[B,20,Q,128]`, key/value `[B,20,K,128]`.
- Masking: `create_causal_mask` combines causal structure with optional caller attention mask.
- Cache stores keys after RoPE, because `past_key_values.update` happens after `apply_rotary_pos_emb`.
- Eager attention upcasts softmax to fp32, then casts probabilities to query dtype before value matmul.
- Dropout is effectively zero for inference.
- SDPA/FlashAttention/flex attention should be usable if they preserve this math and mask/cache semantics; eager is the parity fallback.
- No cross-attention, no encoder-decoder cache, no sliding window/local attention, no ALiBi, no packed/varlen metadata in native arcee source.

## 7. Position encoding and custom math

Default RoPE:

```python
inv_freq = 1.0 / (rope_theta ** (arange(0, head_dim, 2).float() / head_dim))
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
emb = cat([freqs, freqs], dim=-1)
cos = emb.cos()
sin = emb.sin()
```

Application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat([-x2, x1], dim=-1)

q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

YaRN requirements from sampled long-context configs:
- `rope_type`: `yarn`
- `factor`: 20.0
- `original_max_position_embeddings`: 4096
- `beta_fast`: 32.0
- `beta_slow`: 1.0
- `mscale`: 1.0 on Base/instruct; omitted on Preview but source can infer attention factor from `factor`.
- `rope_theta`: 10000.0

The inverse-frequency table and attention scaling are config-dependent and can be precomputed for static maximum context, but position-id ranges and cache offsets are runtime inputs. Dynamic YaRN variants in `modeling_rope_utils` are not used by sampled configs, but source keeps `dynamic_rope_update` around the rotary module, so config admission should reject unsupported RoPE types explicitly.

ReLU squared:

```python
relu2(x) = square(relu(x))
```

## 8. Preprocessing and input packing

Primary inputs:
- `input_ids`: `[B,S]` integer token ids.
- `attention_mask`: optional `[B,S]` mask consumed by `create_causal_mask`.
- `position_ids`: optional `[B,S]`; if omitted, source creates a row vector from current sequence length plus cache length.
- `inputs_embeds`: alternative to `input_ids`; exactly one of `input_ids` or `inputs_embeds` must be supplied.

Generation-controller/runtime ABI:
- Tokenizer controls BOS/EOS/pad conventions. Checkpoint configs carry BOS 128000 and varied EOS ids.
- `logits_to_keep=0` keeps all logits in source because `slice(-0, None)` is `slice(0, None)`; for efficient decode, use `logits_to_keep=1`.
- No multimodal embedding stitch, no packed sequence descriptors, no `cu_seqlens`, no image/audio preprocessing.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Arcee MLP to fused relu2 FFN

Source pattern:

```text
down_proj(square(relu(up_proj(x))))
```

Replacement:

```text
GEMM up -> fused ReLU+square -> GEMM down
```

Preconditions:
- `hidden_act == "relu2"`.
- No `gate_proj` present.
- Up/down weights are dense Linear weights.
- Bias handling follows `mlp_bias`; sampled configs have no bias.

Shape equations:
- `x [B,S,H]`, `up [H,I]`, `tmp [B,S,I]`, `down [I,H]`.
- For sampled configs: `H=2560`, `I=18432`.

Failure cases:
- Any checkpoint with different `hidden_act` needs a separate activation path.
- Do not rewrite to SwiGLU.

Parity test sketch:
- Compare single MLP output for random bf16/fp32 tensors against PyTorch at tolerances listed below.

### Rewrite: Q/K/V projection packing

Source pattern:

```text
q_proj(x), k_proj(x), v_proj(x)
```

Replacement:

```text
one packed GEMM producing [Q | K | V], then split widths [2560, 512, 512]
```

Preconditions:
- Same input tensor and dtype.
- Compatible bias policy across projections.
- Preserve split order exactly Q, K, V.
- Packed weight layout must be explicitly generated from source weights; source stores separate Linear modules.

Failure cases:
- Tensor parallel sharding or external weight formats that already pack differently.
- Bias mismatch.

Parity test sketch:
- Pack weights offline and verify Q/K/V tensors match separate projections bitwise or within dtype tolerance before RoPE.

### Rewrite: last-token-only logits

Source pattern:

```text
lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice hidden states to decode positions before GEMM
```

Preconditions:
- Generation path only needs the final token or explicit selected positions.
- Preserve full logits for loss/evaluation if requested.

Failure cases:
- Training/loss path, perplexity evaluation, or caller requests all logits.

### Rewrite: RoPE precompute/cache

Source pattern:

```text
rotary_emb(hidden_states, position_ids) -> cos/sin
```

Replacement:

```text
precomputed inv_freq/cos/sin table or generated kernel indexed by position_ids
```

Preconditions:
- Fixed RoPE parameters and max context.
- Cache offset handled in position ids.
- YaRN attention scaling included.

Failure cases:
- Unsupported `rope_type`, dynamic RoPE growth behavior not reproduced, or position ids not monotonic.

## 10. Kernel fusion candidates

Highest priority:
- RMSNorm over 2560: appears twice per block plus final norm; fp32 variance and bf16 output need exact parity.
- GQA attention with RoPE and KV cache: 36 layers, 20 Q heads, 4 KV heads, long context up to 65536.
- MLP relu2 activation fusion: intermediate 18432 dominates memory traffic if activation is materialized naively.
- QKV packed GEMM: reduces launch overhead and improves prefill throughput.
- Last-token-only logits: vocab is about 128k, so full sequence logits are expensive.

Medium priority:
- RoPE apply fused with Q/K layout transform.
- Causal mask construction/caching for long contexts.
- GEMM epilogue residual add where safe.
- GGUF dense fallback and runtime dequant path for AFM-4.5B-GGUF packaging, after tensor-name verification.

Lower priority:
- Generic classification/QA/token-classification heads.
- Training dropout/loss and gradient checkpointing.
- Flex attention parity beyond default generation.

## 11. Runtime staging plan

Stage 1: config and weight loading.
- Parse arcee config, normalize legacy `rope_scaling` to `rope_parameters`, reject `arcee_kda` and unsupported RoPE types.
- Load dense safetensors for one small layer slice or full Base config.

Stage 2: single-block parity.
- Implement embedding, RMSNorm, separate Q/K/V/O Linear, default/YaRN RoPE, eager GQA attention, relu2 MLP.
- Validate one layer with random tensors and copied HF weights.

Stage 3: prefill causal LM parity.
- Run all layers without KV cache first, using causal masks and final lm_head.
- Prefer `logits_to_keep=1` path after full-logits parity is proven.

Stage 4: decode with KV cache.
- Add per-layer `[B,4,T,128]` KV cache, RoPE-before-cache update, cache length driven position ids.
- Validate one-token and multi-token decode against HF.

Stage 5: optimized attention and GEMM.
- Replace eager attention with DinoML fused/GQA attention where masks and RoPE order match.
- Add packed QKV and fused relu2 FFN.

Stage 6: quantized/package variants.
- Admit AFM-4.5B-GGUF only through explicit GGUF encoded constant metadata and verified tensor map.
- Keep dense safetensors as the reference path.

## 12. Parity and validation plan

Concrete tests:
- Config parsing: verify defaults, sampled configs, `rope_scaling` normalization, and rejection of `arcee_kda`.
- Activation test: compare `relu2(x)` against `square(relu(x))`.
- RMSNorm test: random fp32/bf16 tensors over width 2560, fp32 variance.
- RoPE test: default and YaRN cos/sin against HF for positions near 0, 4095, 4096, and long-context positions.
- Attention unit test: one layer, GQA repeat factor 5, causal mask, no cache.
- Cache test: prefill `S=N` then decode one token; compare logits with full forward over `N+1`.
- Full prefill logits: compare AFM-4.5B-Base or a reduced synthetic config initialized from HF for short sequences.
- Last-token logits: verify `logits_to_keep=1` equals slicing full logits.
- GGUF loading follow-up: verify dequantized weights from GGUF match dense reference or documented quant tolerance.

Suggested tolerances:
- fp32: `rtol=1e-4`, `atol=1e-5` for blocks; tighter for isolated ops where possible.
- bf16/fp16: `rtol=2e-2`, `atol=2e-2` for full-layer logits; use op-specific tighter thresholds for RMSNorm/Linear.
- Attention parity should compare pre-softmax scores and outputs separately when debugging.

## 13. Performance probes

- Prefill throughput by sequence length: 512, 4096, 8192, 32768, 65536.
- Decode tokens/sec by batch size and cache length.
- KV cache memory: `layers * 2 * B * kv_heads * T * head_dim * dtype_size`.
- RMSNorm bandwidth and relu2 MLP activation materialization cost.
- Separate QKV GEMM versus packed QKV GEMM.
- Eager GQA repeat materialization versus fused GQA attention with native KV heads.
- LM head cost with all logits versus last-token-only logits.
- GGUF load/dequant time and memory footprint versus dense safetensors.

## 14. Skip/defer list

- Training, labels/loss, dropout, gradient checkpointing.
- Sequence classification, token classification, and QA heads for first causal LM target.
- Beam search and advanced generation processors; greedy/sampling controller can live outside core graph initially.
- Tensor parallel/distributed `_tp_plan` execution.
- `arcee_kda` custom-code models.
- Flex attention unless needed by a concrete deployment.
- GGUF direct quantized-RHS kernels; start with dense fallback or existing DinoML GGUF runtime-dequant contracts.

## 15. Final implementation checklist

- [ ] Parse `ArceeConfig` and normalize legacy `rope_scaling` to `rope_parameters`.
- [ ] Reject unsupported `model_type` values such as `arcee_kda` for this path.
- [ ] Load dense embedding, decoder, norm, and lm_head weights.
- [ ] Implement/validate RMSNorm with fp32 variance.
- [ ] Implement/validate ReLU-squared MLP.
- [ ] Implement default RoPE and YaRN RoPE.
- [ ] Implement causal GQA attention with KV repeat factor from config.
- [ ] Implement per-layer KV cache storing post-RoPE K/V.
- [ ] Add last-token-only logits lowering.
- [ ] Add QKV packing rewrite with Q/K/V split order `[q, k, v]`.
- [ ] Add single-block parity tests.
- [ ] Add prefill logits parity tests.
- [ ] Add decode cache parity tests.
- [ ] Benchmark prefill, decode, LM head, and MLP kernels.
- [ ] Verify AFM-4.5B-GGUF tensor mapping before admitting GGUF packaged weights.
