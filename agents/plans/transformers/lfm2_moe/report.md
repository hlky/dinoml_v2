# LFM2-MoE Transformers Family Audit

Primary target: `Lfm2MoeForCausalLM` inference and generation on CUDA. This is
a source/config audit only; no DinoML runtime code was edited, no DinoML tests
or imports were run, and no commit was made.

## 1. Source basis

```text
Transformers commit/version: local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: lfm2_moe
Primary task: causal LM prefill/decode/generation
Config source: local Lfm2MoeConfig plus HF config.json snapshots saved under _sources/
```

Source files inspected:

- `X:/H/transformers/src/transformers/models/lfm2_moe/configuration_lfm2_moe.py`
- `X:/H/transformers/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py`
- `X:/H/transformers/src/transformers/models/lfm2_moe/modular_lfm2_moe.py`
- Cross-checks: `src/transformers/models/lfm2/modeling_lfm2.py`,
  `src/transformers/cache_utils.py`, `docs/source/en/model_doc/lfm2_moe.md`,
  `tests/models/lfm2_moe/test_modeling_lfm2_moe.py`

Source URLs at the inspected commit:

- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lfm2_moe/configuration_lfm2_moe.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lfm2_moe/modeling_lfm2_moe.py`
- `https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/lfm2_moe/modular_lfm2_moe.py`

Representative HF snapshots inspected:

- `https://huggingface.co/LiquidAI/LFM2-8B-A1B/raw/main/config.json`
- `https://huggingface.co/LiquidAI/LFM2-8B-A1B/raw/main/generation_config.json`
- `https://huggingface.co/LiquidAI/LFM2-8B-A1B/raw/main/tokenizer_config.json`
- `https://huggingface.co/LiquidAI/LFM2-24B-A2B/raw/main/config.json`
- `https://huggingface.co/LiquidAI/LFM2-24B-A2B/raw/main/generation_config.json`
- `https://huggingface.co/LiquidAI/LFM2-24B-A2B/raw/main/tokenizer_config.json`
- `https://huggingface.co/tiny-random/lfm2-moe/raw/main/config.json`
- ONNX config snapshots for 8B and 24B, used only to identify export/JS metadata
  that the native PyTorch source does not consume.

Authoritative source note: `modeling_lfm2_moe.py` is generated from
`modular_lfm2_moe.py`. This report uses the generated file for exact expanded
behavior and the modular file to identify intended inheritance: LFM2 attention
and short-conv blocks plus Qwen2-MoE-style packed experts.

Missing files or assumptions: no gated source gap was encountered for the
Python model/config/tokenizer metadata. No GGUF, MLX, ONNX graph, or remote-code
neural implementation was audited; those are separate loader/export surfaces.

## 2. High-level architecture

LFM2-MoE is a text-only hybrid decoder language model. It is not a plain
Transformer stack: each decoder layer chooses either a full causal GQA
self-attention operator or a gated depthwise short-convolution mixer, then runs
either dense SwiGLU for the first shallow layers or sparse top-k MoE SwiGLU for
the remaining layers.

```text
token ids / input embeddings
  -> embedding
  -> repeated hybrid decoder blocks
       operator path: RMSNorm -> full causal GQA attention OR gated short-conv
       ffn path: RMSNorm -> dense SwiGLU for early layers OR sparse MoE SwiGLU
  -> final RMSNorm
  -> tied lm_head logits
  -> generation controller / sampling
```

Stage decomposition:

- CPU/data pipeline: tokenizer, chat template, attention-mask construction,
  generation settings.
- GPU prefill: embedding, shared RoPE cos/sin generation, full sequence
  attention layers, full sequence short-conv layers, dense early FFNs, MoE
  routing and expert execution, final norm/logits.
- GPU decode: one or few new tokens, position IDs from cache length, attention
  KV append in attention layers, fixed-size conv-state update in conv layers,
  per-token MoE routing/expert execution, last-token logits.
- Independently optimizable units: RoPE/GQA attention, short-conv mixer,
  conv-state cache update, top-k router, grouped expert GEMMs, last-token-only
  LM head.

Implemented heads:

- Required: `Lfm2MoeForCausalLM`.
- Useful internal target: `Lfm2MoeModel` for hidden-state parity.
- Deferred: training loss and any export-specific ONNX/Transformers.js path.

## 3. Important config dimensions

Source defaults from `Lfm2MoeConfig`:

| Field | Source default / behavior |
| --- | --- |
| `vocab_size` | 65536 |
| `hidden_size` | 2048 |
| `intermediate_size` | 7168 dense SwiGLU hidden width |
| `moe_intermediate_size` | 1792 per expert by default |
| `num_hidden_layers` | 32 default; official snapshots override |
| `num_attention_heads` | 32 |
| `num_key_value_heads` | 8 |
| `head_dim` | Source uses explicit `config.head_dim` if present, otherwise `hidden_size // num_attention_heads` |
| `max_position_embeddings` | 128000 |
| `rope_parameters` | Standardized dict; default theta is 1000000.0 |
| `conv_L_cache` | 3 |
| `conv_bias` | false |
| `num_dense_layers` | 2 dense FFN layers before MoE FFNs |
| `num_experts` | 32 default |
| `num_experts_per_tok` | 4 |
| `use_expert_bias` | true; routing adds an fp32 expert-bias buffer before top-k selection |
| `norm_topk_prob` | true |
| `routed_scaling_factor` | 1.0 |
| `tie_word_embeddings` | true; `tie_embedding` is accepted as a config alias |
| `use_cache` | true |

Representative checkpoint sweep:

| Model id | Layers | Conv/Attn | H | Q/KV heads | Head dim | Dense I | MoE I | Experts/top-k | Max pos | RoPE theta | Dtype/notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `tiny-random/lfm2-moe` | 3 | 2/1 | 64 | 2/1 | 32 | 128 | 128 | 32/4 | 128000 | 1e6 | bf16 metadata |
| `LiquidAI/LFM2-8B-A1B` | 24 | 18/6 | 2048 | 32/8 | 64 | 7168 | 1792 | 32/4 | 128000 | 1e6 | bf16; raw config uses legacy top-level `rope_theta` |
| `LiquidAI/LFM2-8B-A1B-ONNX` | 24 | 18/6 | 2048 | 32/8 | 64 | 7168 | 1792 | 32/4 | 128000 | 1e6 | adds `transformers.js_config` |
| `LiquidAI/LFM2-24B-A2B` | 40 | 30/10 | 2048 | 32/8 | 64 | 11776 | 1536 | 64/4 | 128000 | 1e6 | bf16 |
| `LiquidAI/LFM2-24B-A2B-ONNX` | 40 | 30/10 | 2048 | 32/8 | 64 | 11776 | 1536 | 64/4 | 128000 | 1e6 | adds `transformers.js_config` |

## 3a. Family variation traps

- `layer_types` is topology-defining. DinoML must not assume every layer is
  attention or every layer has KV cache. Official 8B is 18 conv + 6 attention;
  official 24B is 30 conv + 10 attention.
- Short-conv layers are causal stateful mixers with fixed-size conv caches, not
  ordinary full-sequence attention.
- Attention layers use GQA: 32 query heads and 8 KV heads in official snapshots,
  with 4 query groups per KV head.
- `head_dim` is source-configurable. Do not hard-code
  `hidden_size == num_attention_heads * head_dim`, even though sampled
  checkpoints satisfy it.
- First `num_dense_layers` use dense SwiGLU FFNs. Later layers use sparse MoE.
  The official value is 2, so the first two layers are dense even when their
  operator mixer is conv.
- The MoE router uses sigmoid probabilities, optional fp32 expert bias for
  top-k selection, optional top-k renormalization, and a routed scaling factor.
  This is not softmax router math.
- Expert weights are packed as 3D tensors:
  `gate_up_proj[E, 2 * moe_intermediate_size, hidden_size]` and
  `down_proj[E, hidden_size, moe_intermediate_size]`.
- The packed expert projection splits into `gate, up` in that order after the
  linear projection.
- `conv_bias` controls both short-conv projection biases and depthwise conv
  bias. Official snapshots set it false.
- The 8B raw config uses legacy `rope_theta` while newer configs use
  `rope_parameters`. Current config utilities standardize this, but loaders
  should record the normalized effective parameters.
- ONNX/Transformers.js configs add export and quantization metadata. Native
  `modeling_lfm2_moe.py` does not read those fields.
- Tokenizer configs include image-like special tokens, but this model source has
  no image processor or multimodal embedding stitch. Treat those as tokenizer
  vocabulary/controller metadata, not neural graph inputs.

## 4. Operator coverage checklist

Tensor/layout ops:

- Embedding lookup `[B, S] -> [B, S, H]`.
- Reshape/view/transpose/contiguous for Q/K/V `[B, S, heads, D] ->
  [B, heads, S, D]`, short-conv `[B, S, 3H] -> [B, 3H, S]`, expert token
  flattening `[B, S, H] -> [B*S, H]`, and logits slicing.
- `chunk(3, dim=channels)` for short-conv input projection and `chunk(2)` for
  packed expert gate/up output.
- Padding, slice, roll/window update, index select for cache reorder.
- Top-k, gather, one-hot or equivalent token bucketing, where/nonzero,
  index-add accumulation for sparse experts.

Neural network primitives:

- RMSNorm over last dim, fp32 variance/rsqrt, output cast to input dtype.
- Dense linear projections, all bias-free except optional `conv_bias`.
- Dense SwiGLU: `w2(silu(w1(x)) * w3(x))`.
- Depthwise causal Conv1d with `groups=hidden_size`, kernel `conv_L_cache`,
  padding `L_cache - 1`, output sliced to original sequence length.
- Elementwise multiply gates in short-conv: `Bx = B * x`, then `y = C * conv(Bx)`.
- Tied embedding/LM head weight alias for causal LM.

Attention primitives:

- Causal self-attention only, no cross-attention.
- GQA repeat of KV heads by reshape/expand when using eager path.
- Backend-dispatched attention through Transformers `ALL_ATTENTION_FUNCTIONS`;
  source advertises flash attention, SDPA, and flex attention support.
- Attention mask from `create_causal_mask` for attention layers only.

Position/rotary ops:

- Default RoPE with theta 1e6, full head dimension, fp32 freqs/cos/sin, cast to
  hidden dtype.
- Dynamic RoPE update wrapper is present for non-default rope types, but sampled
  configs use `rope_type="default"`.

Generation/cache ops:

- Dynamic KV cache for attention layers, storing post-RoPE K/V in shape
  `[B, num_key_value_heads, cache_seq, head_dim]`.
- Linear-attention cache layers for conv layers, storing `conv_states` with
  shape `[B, H, conv_L_cache]`.
- Cache reorder for beam search must reorder both KV and conv states.
- Position IDs default to `arange(S) + past_seen_tokens` using attention-cache
  sequence length.

MoE routing/expert ops:

- Router linear `H -> E` with no bias.
- Sigmoid router probabilities.
- Optional expert-bias addition before top-k only.
- Top-k over experts per flattened token.
- Gather selected true sigmoid weights, optional renorm by selected sum plus
  `1e-6`, multiply `routed_scaling_factor`.
- Expert dispatch by selected expert and top-k slot, per-expert packed gate/up
  GEMM, SiLU multiply, down GEMM, route-weight multiply, index-add into output.

Preprocessing-coupled ops:

- Tokenizer and chat template produce `input_ids` and `attention_mask`.
- No multimodal tensors are consumed by the audited source.

Quantized/packed metadata:

- Native source has no quantized weight kernel path. ONNX/JS metadata and
  external GGUF/MLX variants should be separate loader/provider audits.

## 5. Layer/block breakdown

Common shapes use `B=batch`, `S=query length`, `H=hidden_size`, `D=head_dim`,
`QH=num_attention_heads`, `KVH=num_key_value_heads`, `E=num_experts`,
`K=num_experts_per_tok`.

Decoder block, repeated according to `layer_types`:

```text
residual = x
u = RMSNorm(x)
if layer_types[i] == "full_attention":
  q = Linear(H -> QH*D, bias=False)(u).view(B,S,QH,D).transpose(1,2)
  k = Linear(H -> KVH*D, bias=False)(u).view(B,S,KVH,D).transpose(1,2)
  v = Linear(H -> KVH*D, bias=False)(u).view(B,S,KVH,D).transpose(1,2)
  q = RMSNorm(D)(q); k = RMSNorm(D)(k)
  q,k = RoPE(q,k, cos, sin)
  k,v = cache.update(k,v, layer=i) if cache is enabled
  attn = causal_gqa_attention(q,k,v, mask, scale=D**-0.5)
  y = Linear(QH*D -> H, bias=False)(attn)
else:
  y = short_conv_mixer(u, attention_mask, conv_cache)
x = residual + y
x = x + feed_forward(RMSNorm(x))
```

Short-conv mixer:

```text
u = apply_padding_mask(u, attention_mask)
BCx = Linear(H -> 3H, bias=conv_bias)(u).transpose(-1,-2)
B_gate, C_gate, x_gate = chunk(BCx, 3, channel_dim)
Bx = B_gate * x_gate
conv_out = depthwise_causal_conv1d(Bx, weight[H, 1, L], bias=conv_bias)
y = C_gate * conv_out
y = Linear(H -> H, bias=conv_bias)(y.transpose(-1,-2))
```

Dense FFN for `layer_idx < num_dense_layers`:

```text
y = w2(silu(w1(x)) * w3(x))
```

Sparse MoE FFN for later layers:

```text
tokens = x.view(B*S, H)
router_logits = gate(tokens)                         # [T, E]
scores = sigmoid(router_logits)
topk_indices = topk(scores + expert_bias if enabled else scores, K)
topk_weights = gather(scores, topk_indices) or topk values
if norm_topk_prob: topk_weights /= sum(topk_weights) + 1e-6
topk_weights *= routed_scaling_factor
for each hit expert e:
  selected = tokens[token_idx_for_e]
  gate, up = linear(selected, gate_up_proj[e]).chunk(2, -1)
  h = linear(silu(gate) * up, down_proj[e])
  output.index_add_(token_idx_for_e, h * selected_route_weight)
```

Final model path:

```text
hidden = embedding(input_ids)
position_embeddings = RoPE(hidden, position_ids)
hidden = all decoder layers
hidden = final RMSNorm(hidden)
logits = lm_head(hidden[:, logits_to_keep_slice, :])
```

## 6. Attention requirements

Attention is present only in `layer_types == "full_attention"` layers.

- Type: causal self-attention.
- Form: GQA, with sampled official configs using Q heads 32, KV heads 8,
  `head_dim=64`, and 4 query groups per KV head.
- Projection widths for official configs: Q `2048 -> 2048`, K/V
  `2048 -> 512`, output `2048 -> 2048`, all bias-free.
- Q/K post-projection RMSNorm is required before transpose/RoPE/attention.
- RoPE is applied before KV cache update, so cached keys are post-RoPE.
- Eager attention repeats KV heads to Q heads, computes
  `matmul(q, k.T) * head_dim**-0.5`, adds mask, softmaxes in fp32, casts to
  query dtype, then computes `matmul(weights, v)`.
- Masking: `create_causal_mask` builds attention-layer masks. Conv layers
  receive a 2D `attention_mask` except during one-token decode where the source
  passes `None` for compile-friendly masking skip.
- Packed/varlen support: not explicit in the audited source; backend attention
  implementations may optimize internally, but DinoML should first model dense
  causal masks and normal cache lengths.
- Sliding/local attention: not implemented by `lfm2_moe` layer types sampled
  here; only `"full_attention"` and `"conv"` were observed.
- KV cache shape before GQA repeat: `[B, KVH, cache_seq, D]`. After repeat in
  eager attention: `[B, QH, cache_seq, D]`.
- Cache manifest must be per layer: attention layers own growing KV cache;
  conv layers own fixed `[B, H, conv_L_cache]` conv state.

## 7. Position encoding and custom math

RoPE construction:

```python
def lfm2_moe_rope(config, position_ids, dtype):
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    base = config.rope_parameters["rope_theta"]
    inv_freq = 1.0 / (base ** (arange(0, dim, 2).float() / dim))
    freqs = (inv_freq[None, :, None] @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = cat([freqs, freqs], dim=-1)
    return cos(emb).to(dtype), sin(emb).to(dtype)
```

RoPE application:

```python
def apply_lfm2_moe_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return cat([-x2, x1], dim=-1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunities:

- `inv_freq` is static per config.
- Cos/sin depend on runtime `position_ids` and dtype/device. For decode, the
  one-token position slice can be generated or gathered from a bounded table.
- Dynamic/non-default RoPE should be gated until source configs requiring it are
  audited. The wrapper supports non-default `ROPE_INIT_FUNCTIONS`, but sampled
  configs use default RoPE.

## 8. Preprocessing and input packing

Runtime graph inputs:

- `input_ids` or `inputs_embeds`, exactly one required.
- Optional 2D `attention_mask`.
- Optional `position_ids`; otherwise generated from cache length.
- Optional `past_key_values`.

Tokenizer/controller facts from snapshots:

- `bos_token_id=1`, `pad_token_id=0`, `eos_token_id=7` for official snapshots.
- `generation_config.json` carries only basic BOS/EOS/PAD defaults in sampled
  official repos.
- 8B tokenizer metadata includes tool, FIM, chain-of-thought, file, and
  image-like special tokens. The audited neural source does not consume
  modality tensors or perform embedding replacement, so image-like token IDs are
  text vocabulary entries for this report.

CPU/data-pipeline work: tokenization, chat template, padding, and sampling
policy. GPU/runtime work starts at token embedding or provided `inputs_embeds`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: GQA projection cluster

Source pattern:

```text
q_proj/k_proj/v_proj -> view -> q/k RMSNorm -> transpose -> RoPE -> attention
```

Replacement pattern: keep separate logical weights initially, then optionally
fuse projection loads and reshapes into one QKV preparation kernel.

Preconditions:

- All three projections are bias-free.
- Output split order remains Q, K, V as separate HF modules, not a packed
  checkpoint tensor.
- Q/K RMSNorm is per-head over `head_dim` and must happen before RoPE.

Failure cases: custom `head_dim`, enabled projection bias in a future config,
or a loader that packs QKV in a different row order.

Parity sketch: compare q/k/v tensors after RMSNorm and after RoPE for random
hidden states and position IDs.

### Rewrite: short-conv full prefill to depthwise causal conv provider

Source pattern:

```text
Linear(H -> 3H) -> transpose/chunk -> B*x -> depthwise Conv1d(pad=L-1)[:S] -> C*conv -> Linear(H -> H)
```

Replacement pattern:

```text
GatedProjection -> CausalDepthwiseConv1d -> GateMultiply -> OutLinear
```

Preconditions:

- `groups == hidden_size`.
- `kernel_size == conv_L_cache`.
- Padding is left-causal equivalent and output is sliced to sequence length.
- Input layout is `[B, S, H]` at graph boundary.
- `conv_bias` must match both projection and conv bias handling.

Failure cases: non-depthwise conv, alternate padding, noncontiguous sequence
layout, or trying to treat this as ordinary static Conv1d without preserving
causal crop and cache update semantics.

Parity sketch: compare slow PyTorch conv path against fused prefill for
multiple `S`, including `S < conv_L_cache` and padded attention masks.

### Rewrite: short-conv decode to ring-buffer update

Source pattern:

```text
conv_state = cache.update_conv_state(Bx)
conv_out = sum(conv_state * conv.weight[:,0,:], dim=-1) + optional bias
```

Replacement pattern: fixed-size per-layer rolling state plus one depthwise dot
per decode token.

Preconditions:

- Decode step length is 1 or a bounded small `num_new_tokens`.
- Cache state is `[B, H, L]` and reorder semantics are preserved for beams.
- State updates use copy/static-address semantics when CUDA graphs matter.

Failure cases: multi-token decode without correct roll/window semantics, beam
reorder ignored for conv states, or prefill shorter than kernel length without
matching source padding behavior.

Parity sketch: prefill then decode token-by-token and compare with a full
sequence slow-forward reference.

### Rewrite: sparse expert dispatch to grouped expert GEMM

Source pattern: top-k routing, per-expert token gathers, packed gate/up GEMM,
SwiGLU, down GEMM, weighted index-add.

Replacement pattern: sort/bucket `(token, topk_slot)` by expert, run grouped
GEMM per expert, scatter-add weighted outputs.

Preconditions:

- Router top-k exactly matches sigmoid plus optional expert bias semantics.
- Top-k tie behavior is accepted or covered by a deterministic tolerance policy.
- Expert weights retain packed layout `[E, 2I, H]` and split order
  `gate, up`.
- `index_add` accumulation order differences are within dtype tolerance, or a
  deterministic accumulation mode is used for parity tests.

Failure cases: treating router as softmax, applying expert bias to gathered
weights rather than selection scores, skipping `norm_topk_prob`, or using an
all-token dense expert matmul without route guards.

Parity sketch: seed router logits with ties/non-ties, verify selected experts,
route weights, per-expert token counts, and final hidden states.

### Rewrite: last-token-only logits

Source pattern: `logits_to_keep` slices hidden states before `lm_head`.

Replacement: for decode, lower only the last hidden row through the tied LM
head.

Preconditions: generation requests only last-token logits or an explicit
integer/tensor slice that DinoML supports.

Failure cases: callers ask for full-sequence logits for scoring or loss.

Parity sketch: compare full logits versus sliced logits for `logits_to_keep=1`
and tensor index cases.

## 10. Kernel fusion candidates

Highest priority:

- RMSNorm: used before every operator and FFN plus final embedding norm.
- Short-conv mixer prefill/decode: most official layers are conv layers, so
  this dominates layer count and has custom cache semantics.
- MoE router and expert grouped GEMMs: all post-dense layers use sparse experts;
  efficient dispatch is essential.
- GQA attention with Q/K RMSNorm + RoPE + KV cache: attention layer count is
  lower than conv layers but still controls long-context quality and cache size.

Medium priority:

- Dense SwiGLU for the first two layers.
- Fused projection/chunk/gate multiply in short-conv.
- Last-token-only tied LM head.
- Cache reorder kernels covering both KV and conv states for beam search.

Lower priority:

- Dynamic/non-default RoPE variants beyond sampled default RoPE.
- ONNX/Transformers.js quantized metadata ingestion.
- Training loss and output attentions.

## 11. Runtime staging plan

Stage 1: parse config and load weights for tiny and 8B configs. Reject configs
without `layer_types`, unsupported rope types, or nonstandard quantized export
metadata unless routed to a separate loader.

Stage 2: implement one dense early layer in isolation: RMSNorm, short-conv or
attention by `layer_types`, dense SwiGLU, residuals.

Stage 3: implement short-conv prefill and decode cache parity, including
attention-mask zeroing and conv-state reorder/reset.

Stage 4: implement attention-layer prefill/decode with GQA, Q/K RMSNorm, RoPE,
and KV cache.

Stage 5: implement sparse MoE routing exactly, initially with eager per-expert
dispatch, then grouped GEMM optimization.

Stage 6: full model prefill logits, then decode logits with `logits_to_keep=1`.

Stage 7: optimize fusions: short-conv fused kernels, MoE grouped provider,
attention backend selection, and last-token LM head.

Stubbable initially: generation sampling, output attentions, labels/loss,
ONNX/JS metadata, quantized loaders, and non-default RoPE.

## 12. Parity and validation plan

- Config loader tests: source defaults, 8B legacy `rope_theta` normalization,
  24B `rope_parameters`, layer count/type validation, tied embedding alias.
- RMSNorm random tensor parity in fp32/fp16/bf16, tolerance about `1e-5` fp32
  and `1e-2` reduced precision.
- RoPE parity for random position IDs and decode position offsets.
- Short-conv prefill parity for `S=1`, `S=2`, `S=3`, `S>3`, with and without
  padding masks.
- Short-conv cache parity: full-sequence reference versus prefill plus
  token-by-token decode; include beam reorder.
- Attention parity: one attention layer, GQA repeat, causal masks, cache append,
  fp32 softmax.
- Router parity: sigmoid, expert-bias top-k selection, gathered weights,
  renormalization, routed scaling.
- Expert parity: packed gate/up split, SiLU multiply, down projection,
  weighted index-add for random token/expert assignments.
- Single-block parity for conv+dense, attention+dense, conv+MoE, attention+MoE.
- End-to-end tiny model parity with random weights, then official 8B logits
  smoke if weights are available.
- Decode token parity: compare generated greedy tokens for a short prompt after
  prefill/decode cache is implemented.

## 13. Performance probes

- Prefill throughput by sequence length, separating conv layers, attention
  layers, and MoE layers.
- Decode tokens/sec with cache enabled, separating conv-state update, attention
  KV read, router, expert GEMM, and LM head.
- MoE routing histogram: expert load balance, top-k slot distribution, empty
  experts, grouped GEMM occupancy.
- Short-conv kernel comparison: PyTorch-style depthwise conv, causal-conv1d
  equivalent, and DinoML fused prefill/decode.
- Attention backend comparison: eager baseline, SDPA/FlashAttention-compatible
  fused GQA path, cache memory usage at 32K/128K contexts.
- Batch-size sweep for on-device style small batches and server-style batches.
- Quantized loader/provider comparison only after native dense parity is stable:
  dense bf16, GGUF dequant-before-GEMM, and any direct quantized expert path.

## 14. Skip/defer list

- Training loss and gradient checkpointing.
- Output attentions and hidden-state captures beyond debug parity.
- Multimodal/image special-token behavior; no audited neural source consumes
  image tensors.
- ONNX, Transformers.js, MLX, and GGUF graph/runtime parity.
- Non-default/dynamic RoPE variants until a checkpoint requiring them is in
  scope.
- General sliding/local attention; sampled configs use full attention plus conv.
- Multi-GPU/tensor parallel plans.
- Speculative decoding and advanced generation controllers.

## 15. Final implementation checklist

- [ ] Parse `Lfm2MoeConfig`, including legacy `rope_theta` normalization.
- [ ] Validate `layer_types` length and admit only `conv` and `full_attention`.
- [ ] Load tied embedding / LM head as one logical parameter.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement default RoPE and apply-to-QK helper.
- [ ] Implement GQA attention with Q/K RMSNorm and KV cache.
- [ ] Implement short-conv prefill path.
- [ ] Implement short-conv decode conv-state cache and beam reorder.
- [ ] Implement dense early SwiGLU FFN.
- [ ] Implement sigmoid top-k router with expert bias, renorm, and scaling.
- [ ] Implement packed expert gate/up and down projections.
- [ ] Add grouped expert GEMM rewrite with exact routing preconditions.
- [ ] Add last-token-only logits lowering.
- [ ] Add single-layer and whole-tiny-model parity tests.
- [ ] Add 8B/24B config admission tests.
- [ ] Benchmark prefill/decode split by conv, attention, and MoE expert work.
