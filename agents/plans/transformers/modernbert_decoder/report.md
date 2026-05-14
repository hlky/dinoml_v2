# ModernBERT Decoder Transformers Audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: modernbert-decoder family; representative checkpoints include jhu-clsp/ettin-decoder-{17m,32m,68m,150m,400m,1b}, blab-jhu/test-32m-dec, and onnx-internal-testing/tiny-random-ModernBertDecoderForCausalLM.
Config source: local Transformers configuration source plus HF Hub config.json files listed in config_sweep.md.
Source files inspected:
- X:/H/transformers/src/transformers/models/modernbert_decoder/modeling_modernbert_decoder.py
- X:/H/transformers/src/transformers/models/modernbert_decoder/configuration_modernbert_decoder.py
- X:/H/transformers/src/transformers/models/modernbert_decoder/modular_modernbert_decoder.py
- X:/H/transformers/src/transformers/masking_utils.py
- X:/H/transformers/src/transformers/cache_utils.py
Any missing files or assumptions: no remote-code neural body is required for in-library ModernBertDecoderForCausalLM. `modeling_modernbert_decoder.py` and `configuration_modernbert_decoder.py` are generated from `modular_modernbert_decoder.py`; future source edits should target the modular file. No DinoML tests were run by request.
```

The primary runtime target for this report is `ModernBertDecoderForCausalLM`: text-only causal LM prefill and decode with hybrid full/sliding-window self-attention. `ModernBertDecoderForSequenceClassification` is implemented in source but optional/deferred for the first CLM target.

## 2. High-level architecture

ModernBERT Decoder is a decoder-only Transformer with token embedding, embedding LayerNorm, alternating full and sliding-window causal self-attention layers, GEGLU-style MLP blocks, final LayerNorm, a prediction head, and a tied LM decoder projection.

```text
token ids / input embeds -> embedding + LayerNorm -> N decoder layers -> final LayerNorm -> prediction head -> vocab logits -> generation
```

Per-stage decomposition:

```text
CPU/data pipeline: tokenizer emits input_ids, attention_mask, optional position_ids
GPU prefill: embeddings, full/sliding masks, RoPE tables, decoder blocks, logits
GPU decode: one or more new tokens, DynamicCache update, full layers grow KV, sliding layers keep bounded KV state
Generation controller: sampling/beam/search policy outside the core module graph
```

Independent validation units are embedding+norm, RoPE generation/application, one full-attention block, one sliding-attention block, cache update behavior, final head/logits, and last-token-only logits slicing.

## 3. Important config dimensions

Source defaults from `ModernBertDecoderConfig`:

| Field | Default / rule | Runtime impact |
|---|---|---|
| `model_type` | `modernbert-decoder` | AutoModel routing. |
| `vocab_size` | 50368 | Embedding rows and LM output rows. |
| `hidden_size` | 768 | Main residual width. |
| `intermediate_size` | 1152 | MLP inner width after GLU split. |
| `num_hidden_layers` | 22 | Number of decoder blocks. |
| `num_attention_heads` | 12 | MHA heads. |
| `head_dim` | `hidden_size // num_attention_heads`; source rejects non-divisible configs | Q/K/V per-head width. |
| `max_position_embeddings` | 8192 default; Ettin configs use 7999 | RoPE cache/init max. |
| `hidden_activation` | `gelu` | MLP gate activation and prediction-head activation by default. |
| `attention_bias` | false | Q/K/V/O projection bias toggle. |
| `mlp_bias` | false | Wi/Wo bias toggle. |
| `norm_eps`, `norm_bias` | `1e-5`, false | LayerNorm epsilon and bias. |
| `decoder_bias` | true | LM vocab projection bias. |
| `local_attention` | 128 | Source computes effective `sliding_window=64`. |
| `layer_types` | if absent, every third layer full, others sliding | Hybrid attention/cache topology. |
| RoPE theta | full 160000, sliding 10000 by source defaults; Ettin configs set both to 160000 | Separate RoPE parameter sets by layer type. |
| `use_cache` | true | Creates `DynamicCache(config)` when requested. |
| `tie_word_embeddings` | true | LM decoder weight aliases token embedding weight. |

Representative sweep is in `config_sweep.md`. Operator-significant variation: scale changes `H/L/heads/intermediate`, but public Ettin decoder checkpoints keep `head_dim=64`, no attention/MLP/norm biases, `vocab_size=50368`, explicit full/sliding layer patterns, and `torch_dtype=float32` in config metadata.

## 3a. Family variation traps

- This is MHA, not GQA/MQA: source has separate `q_proj`, `k_proj`, and `v_proj`, all `hidden_size -> hidden_size`; no `num_key_value_heads` field is read.
- `head_dim` is inferred and must divide `hidden_size`; do not infer support for arbitrary `head_dim` configs.
- `layer_types` changes both masks and cache layer classes. Full layers use unbounded dynamic KV; sliding layers use `DynamicSlidingWindowLayer`.
- Source `local_attention` is half-window compatible with ModernBERT comments: `local_attention=128` becomes `sliding_window=64`.
- Some configs contain a literal `sliding_window` field. Current config source recomputes `self.sliding_window` from `local_attention`; treat checkpoint `sliding_window` as legacy unless validated through config construction.
- Source defaults set sliding RoPE theta to 10000, while Ettin decoder configs use `local_rope_theta=160000`; load config values, do not assume defaults.
- MLP is gated: `Wi` produces `2 * intermediate_size`, split order is `(input, gate)`, output is `act(input) * gate`.
- Prediction head is extra before logits: `dense(hidden->hidden) -> activation -> LayerNorm`, then decoder projection to vocab. First CLM parity cannot skip it.
- First decoder layer has `attn_norm = Identity`; later layers apply LayerNorm before attention. All layers apply `mlp_norm`.
- `logits_to_keep=0` in source becomes `slice(0, None)`, so all logits are computed. Positive integers keep last K tokens.
- Tied weight contract: `decoder.weight` aliases `model.embeddings.tok_embeddings.weight` when `tie_word_embeddings=True`; keep one logical parameter.
- Legacy config fields `is_causal`, `masked_prediction`, `causal_mask`, `position_embedding_type`, `deterministic_flash_attn`, and `reference_compile` were observed but are not required by the inspected source path.
- No image/audio/video tensors are present. NHWC/NCHW guidance is mostly negative: there is no channel layout translation region; keep semantic tensor axes `[B, S, H]` and attention `[B, heads, S, D]`.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `input_ids [B,S] -> hidden [B,S,H]`.
- Optional `inputs_embeds [B,S,H]` bypass.
- LayerNorm over last dimension for embeddings, per-block norms, final norm, and prediction head norm.
- View/reshape: `[B,S,H] -> [B,S,num_heads,head_dim]`.
- Transpose: `[B,S,heads,D] -> [B,heads,S,D]` for attention, then back to `[B,S,heads,D]`.
- Chunk/split: `Wi(...).chunk(2, dim=-1)`.
- Slice/gather: CLM `hidden_states[:, slice_indices, :]`; classification last non-pad token gather.
- Contiguous/reshape materialization after attention transpose.
- Argmax and index arithmetic only for optional sequence classification pooling.

Neural network primitives:

- `Embedding(vocab_size -> H)`.
- `Linear(H -> H)` for Q, K, V, O, prediction head dense.
- `Linear(H -> 2I)` for MLP `Wi`.
- `Linear(I -> H)` for MLP `Wo`.
- `Linear(H -> vocab_size)` LM decoder, bias controlled by `decoder_bias`.
- `Linear(H -> num_labels)` only for sequence classification.
- GELU activation by default; `ACT2FN[hidden_activation]` and `ACT2FN[classifier_activation]` are config-controlled.
- Dropout exists in source but is zero in representative inference configs and should be disabled in eval.
- Residual add after attention and after MLP.

Attention primitives:

- Dense causal MHA for full layers.
- Sliding-window causal MHA for sliding layers.
- QK matmul with scaling `head_dim ** -0.5`.
- Additive attention mask, softmax upcast to fp32 in eager path, dropout, AV matmul.
- SDPA/Flash/Flex dispatch through `ALL_ATTENTION_FUNCTIONS` when configured.

Position/rotary ops:

- Per-layer-type RoPE cos/sin generation using inverse frequencies.
- `rotate_half` implemented as split last dim into halves, concatenate `(-second_half, first_half)`.
- Apply RoPE to Q and K after projection/transpose and before cache update.

Generation/cache ops:

- `DynamicCache(config)` construction.
- Per-layer KV update with shapes `[B, heads, S, D]`.
- Sliding layers keep bounded resident cache while returning full current attention states for the step.
- Beam cache reorder is inherited from generic cache.

Packed/varlen metadata:

- `masking_utils` can detect packed sequence format from `position_ids` when `attention_mask is None and past_key_values is None`; first integration can reject packed position patterns and require ordinary monotonic positions.

Preprocessing-coupled ops:

- Tokenization is ordinary `PreTrainedTokenizerFast` metadata in representative configs. No model-owned special packing beyond `input_ids`, `attention_mask`, and optional `position_ids`.

## 5. Layer/block breakdown

For a representative Ettin 150M decoder (`H=768`, `heads=12`, `D=64`, `I=1152`, `L=22`):

```text
Embedding:
  input_ids [B,S] -> tok_embeddings [B,S,768]
  hidden = Dropout(LayerNorm(hidden))

Layer 0:
  residual = x
  y = Identity(x)
  q = Linear(768 -> 768, bias=false)(y).view[B,S,12,64].transpose -> [B,12,S,64]
  k = Linear(768 -> 768, bias=false)(y).view[B,S,12,64].transpose -> [B,12,S,64]
  v = Linear(768 -> 768, bias=false)(y).view[B,S,12,64].transpose -> [B,12,S,64]
  q,k = RoPE(q,k, cos/sin for layer type)
  k,v = cache.update(k,v, layer_idx=0) if cache exists
  a = causal/sliding attention(q,k,v, mask)
  a = Linear(768 -> 768, bias=false)(a.reshape[B,S,768])
  x = residual + Dropout(a)
  residual = x
  m = LayerNorm(x)
  wi = Linear(768 -> 2304, bias=false)(m)
  input, gate = chunk(wi, 2, dim=-1)
  m = Linear(1152 -> 768, bias=false)(Dropout(GELU(input) * gate))
  x = residual + m

Layers 1..21:
  same as layer 0, except attn_norm is LayerNorm rather than Identity.

Final:
  x = LayerNorm(x)
  h = LayerNorm(GELU(Linear(768 -> 768, classifier_bias=false)(selected x)))
  logits = Linear(768 -> 50368, decoder_bias=true)(h)
```

For 400M and 1B, shapes become `1024 -> 5248 -> 1024` and `1792 -> 7680 -> 1792` through the MLP `Wi` split, respectively.

## 6. Attention requirements

Required attention variants:

- Causal self-attention only; no cross-attention.
- MHA with equal Q/K/V head counts.
- Full causal layers and sliding-window causal layers in the same stack.
- Head layout entering backend: query/key/value `[B, num_heads, q_len, head_dim]`.
- Eager attention scores: `[B, heads, q_len, kv_len] = q @ k.transpose(-2,-1) * scaling`.
- Masking style: additive mask supplied by `create_causal_mask` or `create_sliding_window_causal_mask`; eager path adds mask before fp32 softmax.
- Sliding mask predicate is causal plus `kv_idx > q_idx - sliding_window`.
- Packed sequence masks may be overlaid by `masking_utils` when `position_ids` encode packed sequences; reject initially unless explicitly implemented.
- RoPE is applied before cache update, so cached keys are already RoPE-encoded.
- Cache tensors are logically `[B, heads, seq_len, head_dim]`. Dynamic full layers grow with generated sequence length.
- Sliding cache resident storage is `[B, heads, min(total_seen, sliding_window - 1), head_dim]` in `DynamicSlidingWindowLayer`; attention for the current call receives `full_key_states = concat(cache, new)`.
- FlashAttention/SDPA compatibility: source declares support and dispatches through `ALL_ATTENTION_FUNCTIONS`; DinoML should first implement an eager-equivalent dense/sliding attention path, then substitute optimized kernels under strict mask/cache guards.

Cache manifest for Ettin-style configs:

```text
layer_types repeat: full, sliding, sliding, ...
full_attention layer: K,V grow to [B,Hd,total_seq,D]
sliding_attention layer: resident K,V bounded to [B,Hd,63,D] for sliding_window=64, but current attention sees up to 64 keys during one-token decode
```

## 7. Position encoding and custom math

RoPE parameters are separate for `"full_attention"` and `"sliding_attention"`. The source computes inverse frequencies as:

```python
base = config.rope_parameters[layer_type]["rope_theta"]
dim = config.head_dim if present else config.hidden_size // config.num_attention_heads
inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))
```

At runtime:

```python
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
emb = cat((freqs, freqs), dim=-1)
cos = cos(emb) * attention_scaling
sin = sin(emb) * attention_scaling
```

Application:

```python
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precompute opportunity: `inv_freq` is static per layer type. Cos/sin depend on `position_ids`, batch, sequence length, dtype, and device. For ordinary non-packed decode, position ids are contiguous offsets from `past_key_values.get_seq_length()`.

## 8. Preprocessing and input packing

Runtime inputs:

- `input_ids [B,S]` int tokens, or exactly one `inputs_embeds [B,S,H]`.
- `attention_mask` may be a 2D padding mask `[B, seen+S]`, an already prepared mapping keyed by `"full_attention"`/`"sliding_attention"`, or absent.
- `position_ids [B,S]` optional. If absent, source creates `arange(S) + past_seen_tokens`, expanded across batch.
- Token IDs in representative configs: `bos/cls=50281`, `eos/sep=50282`, `pad=50283`, vocab 50368.

CPU/data pipeline owns tokenization. GPU/runtime owns embedding lookup, position id defaulting if admitted, mask construction or consumption, RoPE, and cache updates. No modality placeholder scatter, image preprocessing, or NHWC/NCHW axis work exists.

First integration guard: require ordinary contiguous `position_ids` and either no padding or a standard 2D padding mask. Defer packed-sequence detection and custom `or_mask_function`/`and_mask_function` overlays.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split Q/K/V projections -> packed QKV GEMM

Source pattern:

```text
q = Linear(H -> H)(x)
k = Linear(H -> H)(x)
v = Linear(H -> H)(x)
view/transpose each to [B,heads,S,D]
```

Replacement:

```text
Linear(H -> 3H) with row-block packed weight [Q; K; V] -> split last dim into Q,K,V -> view/transpose
```

Preconditions: same input tensor, same dtype, same bias policy, all output widths equal `H`, no observers between projections, and no quantized storage requiring separate materialization. Weight transform concatenates source PyTorch linear weights along output rows in Q,K,V order; concatenate biases the same way when `attention_bias=true`.

Failure cases: configs with missing or mismatched Q/K/V modules, future GQA/MQA variants, or checkpoint formats that already pack projections differently.

Parity test: compare packed projection split outputs against three source projections before RoPE for random `[B,S,H]`.

### Rewrite: MLP Wi chunk -> gated activation GEMM epilogue

Source pattern:

```text
u, gate = Linear(H -> 2I)(x).chunk(2, dim=-1)
y = Linear(I -> H)(dropout(act(u) * gate))
```

Replacement: one GEMM producing `2I`, then fused `GELU(u) * gate` into the consumer GEMM input when memory planning allows, or a dedicated GEGLU elementwise kernel.

Preconditions: eval mode/dropout disabled, chunk exactly into equal halves, activation known and supported, dense row-major `[B*S,I]` flattening legal.

Parity test: random fp32/fp16 block-level MLP comparison with source module.

### Rewrite: last-token-only logits

Source pattern:

```text
selected = hidden_states[:, slice_indices, :]
logits = decoder(lm_head(selected))
```

Replacement: during decode, set `logits_to_keep=1` or compile a graph whose prediction head and vocab GEMM run only on `[B,1,H]`.

Preconditions: generation only needs next-token logits; no training loss; no caller requests full logits or arbitrary tensor `logits_to_keep`.

Failure cases: perplexity/evaluation requiring all sequence logits.

### Rewrite: attention layout fusion

Source pattern:

```text
Linear -> view[B,S,heads,D] -> transpose(1,2) -> attention -> transpose(1,2) -> reshape[B,S,H]
```

Replacement: generate Q/K/V directly in backend-preferred attention layout, or fuse projection output layout with RoPE/attention.

Layout constraints: semantic graph remains `[B,S,H]` externally. This is not NHWC/NCHW translation; it is a local sequence/head layout optimization. Consumers outside the fused attention region must see `[B,S,H]`.

Failure cases: exposing Q/K/V tensors to debug outputs, unsupported attention backend layout, or packed-sequence mask paths requiring alternate metadata.

### Rewrite: tied embedding/decoder storage alias

Source pattern: `_tied_weights_keys = {"decoder.weight": "model.embeddings.tok_embeddings.weight"}`.

Replacement: one constant storage object with two logical uses: embedding lookup rows and LM projection RHS.

Preconditions: `tie_word_embeddings=true` and checkpoint actually ties or shares values.

Failure cases: untied fine-tunes or adapters that intentionally separate embeddings and output projection.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B*S,H]`, including no-bias norm. It appears at embeddings, most attention inputs, every MLP input, final norm, and prediction head.
- QKV packed GEMM + view/transpose. Three separate GEMMs per layer are an obvious launch and memory bandwidth cost.
- RoPE + attention prefill/decode, preserving fp32 position math and cached RoPE-encoded keys.
- Sliding-window causal attention with bounded KV cache. Ettin has more sliding layers than full layers.
- GEGLU elementwise `GELU(input) * gate`, ideally fused between GEMMs.
- Last-token-only prediction head + vocab projection for decode.

Medium priority:

- Residual add + following LayerNorm fusion.
- Prediction head `Linear + GELU + LayerNorm`.
- Attention output projection plus residual add where epilogue support exists.
- Mask construction specialized for fixed full/sliding layer patterns.

Lower priority:

- Dropout kernels for training; inference configs use zero dropout/eval.
- Sequence classification pooling/gather and classifier head.
- Packed sequence mask overlays and custom block/or/and mask functions.

## 11. Runtime staging plan

Stage 1: parse config and load weights for `ModernBertDecoderForCausalLM`, preserving tied embedding/decoder alias metadata.

Stage 2: implement embedding, LayerNorm, prediction head, and LM projection parity for random hidden states and one small checkpoint.

Stage 3: one decoder layer without cache, first full-attention layer, eager dense causal attention, RoPE parity.

Stage 4: one sliding-attention layer, sliding mask parity, no cache.

Stage 5: full prefill through all layers for tiny and 32M/150M configs, with full logits.

Stage 6: DynamicCache decode parity: full layers grow KV, sliding layers keep bounded resident KV, cached keys stored after RoPE.

Stage 7: optimized attention and projection rewrites under guards.

Stage 8: production decode features: last-token logits, batching, cache memory probes, and optional quantized weight loading.

Initially stub/defer training losses, dropout, sequence classification, beam search cache reorder, packed sequence masks, and remote/downstream custom heads.

## 12. Parity and validation plan

- Config construction tests for source defaults, legacy `global_rope_theta`/`local_rope_theta`, explicit `layer_types`, and effective `sliding_window`.
- RoPE unit tests: compare inv_freq, cos/sin, `rotate_half`, and Q/K application for both full and sliding theta values.
- Q/K/V projection shape tests for `H=32`, `384`, `768`, `1024`, `1792`.
- MLP GEGLU parity tests for `I=32`, `576`, `1152`, `2624`, `3840`.
- One full-attention layer parity in fp32 with no cache.
- One sliding-attention layer parity in fp32 with `sliding_window=64`.
- Tiny full-model prefill logits parity against `onnx-internal-testing/tiny-random-ModernBertDecoderForCausalLM`.
- 32M/150M checkpoint prefill hidden-state/logit parity if weights are accessible.
- Decode parity: prefill N tokens, decode one token with `use_cache=True`, compare logits and cache shapes.
- Last-token logits parity for `logits_to_keep=1`.

Suggested tolerances: fp32 `atol=1e-4, rtol=1e-4`; fp16/bf16 `atol=2e-2, rtol=2e-2` for full-model logits, tighter for isolated GEMM/norm where accumulation policy matches.

## 13. Performance probes

- Prefill tokens/sec by model scale and sequence length: 128, 512, 2048, 7999.
- Decode tokens/sec for batch sizes 1, 4, 16, with cache enabled.
- Full-vs-sliding layer attention timing split.
- KV cache memory by layer type: full layers grow with sequence; sliding layers bounded.
- QKV separate GEMMs versus packed QKV GEMM.
- Eager attention versus SDPA/Flash/sliding-window specialized backend.
- LayerNorm and GEGLU kernel time share.
- Last-token-only logits versus all-token logits.
- Vocab projection bandwidth for tied output weight.
- Weight dtype/quantization load probes if DinoML adds GGUF or other encoded storage later.

## 14. Skip/defer list

- Training losses and gradient checkpointing.
- Dropout behavior outside eval mode.
- `ModernBertDecoderForSequenceClassification` and downstream custom heads such as `EttinTokenClassifier`.
- Packed sequence mask detection and block/or/and mask overlays.
- Beam search cache reorder beyond generic cache support.
- FlexAttention-specific BlockMask lowering.
- Dynamic or scaled RoPE variants beyond observed default RoPE unless a checkpoint config requires them.
- Multi-GPU/tensor parallel.
- NHWC/NCHW layout translation; not applicable to this text-only family.
- Gated/401 gaps: none encountered for the inspected JHU CLM configs; `yosefw/SPLADE-Ettin-32m-decoder` had no accessible `config.json` at the searched URL and is excluded.

## 15. Final implementation checklist

- [ ] Parse `ModernBertDecoderConfig`, including legacy RoPE theta keys and `layer_types`.
- [ ] Preserve tied `decoder.weight` / token embedding alias when `tie_word_embeddings=true`.
- [ ] Load embedding, LayerNorm, Q/K/V/O, MLP, prediction-head, and decoder weights.
- [ ] Implement token embedding + embedding LayerNorm.
- [ ] Implement last-dim LayerNorm with optional bias.
- [ ] Implement RoPE cos/sin generation per layer type.
- [ ] Implement `rotate_half` and Q/K RoPE application.
- [ ] Implement full causal attention prefill.
- [ ] Implement sliding-window causal attention prefill.
- [ ] Implement DynamicCache decode for full layers.
- [ ] Implement bounded sliding cache for sliding layers.
- [ ] Implement GEGLU MLP.
- [ ] Implement prediction head and tied vocab projection.
- [ ] Add packed QKV projection rewrite with Q,K,V row-block order.
- [ ] Add last-token-only logits rewrite.
- [ ] Add one-layer full/sliding parity tests.
- [ ] Add tiny and 32M prefill/decode parity tests.
- [ ] Benchmark prefill, decode, attention backend, QKV packing, and cache memory.
