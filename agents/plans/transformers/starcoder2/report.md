# StarCoder2 Transformers audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model family: starcoder2
Primary runtime target: causal language modeling / code generation
Config source: official Hugging Face config.json snapshots under _sources/
Source files inspected:
- transformers/src/transformers/models/starcoder2/configuration_starcoder2.py
- transformers/src/transformers/models/starcoder2/modeling_starcoder2.py
- transformers/src/transformers/models/starcoder2/modular_starcoder2.py
- transformers/src/transformers/masking_utils.py
- transformers/src/transformers/cache_utils.py
- transformers/src/transformers/modeling_rope_utils.py
- transformers/src/transformers/integrations/sdpa_attention.py
- transformers/src/transformers/integrations/flash_attention.py
Any missing files or assumptions:
- `modeling_starcoder2.py` is generated from `modular_starcoder2.py`; inspect the generated file for exact pinned runtime behavior, but future source edits should target the modular file.
- This report covers native in-library Transformers source, not remote code.
- No DinoML runtime code was edited and no DinoML tests were run.
```

Representative checkpoints/configs inspected:

- `bigcode/starcoder2-3b`: `_sources/bigcode__starcoder2-3b.config.json`
- `bigcode/starcoder2-7b`: `_sources/bigcode__starcoder2-7b.config.json`
- `bigcode/starcoder2-15b`: `_sources/bigcode__starcoder2-15b.config.json`
- `bigcode/starcoder2-15b-instruct-v0.1`: `_sources/bigcode__starcoder2-15b-instruct-v0.1.config.json`
- `hf-internal-testing/tiny-random-Starcoder2ForCausalLM`: `_sources/hf-internal-testing__tiny-random-Starcoder2ForCausalLM.config.json`

Tokenizer/generation snapshots were fetched only to document input/generation coupling.

## 2. High-level architecture

StarCoder2 is a text-only decoder transformer for causal language modeling:

```text
GPT-2-style tokenization -> token embedding -> embedding dropout
  -> repeated decoder blocks with LayerNorm + GQA self-attention + MLP
  -> final LayerNorm -> tied or untied LM head -> logits/sampling
```

The primary stage split for DinoML:

```text
CPU/data pipeline: GPT2Tokenizer, special tokens, optional chat template for instruct
GPU/runtime prefill: embedding, RoPE, sliding-window causal GQA, MLP, final logits
GPU/runtime decode: one or more new tokens, KV cache update, sliding-window attention window
Generation controller: sampling/beam/etc. outside the core module graph
```

Independently stageable pieces:

- Tokenization and prompt/chat formatting are CPU/data-pipeline work.
- RoPE tables can be precomputed or generated per batch from `position_ids`.
- Prefill and decode should be validated separately because the official checkpoints use `sliding_window=4096`; decode cache shape is bounded differently from full-context position IDs.
- Classification heads exist in source but are optional/deferred for the causal-LM target.

## 3. Important config dimensions

Source defaults from `Starcoder2Config`:

| Field | Source default | Runtime relevance |
| --- | ---: | --- |
| `vocab_size` | 49152 | embedding rows and LM head rows |
| `hidden_size` | 3072 | residual width |
| `intermediate_size` | 12288 | MLP expansion |
| `num_hidden_layers` | 30 | decoder blocks |
| `num_attention_heads` | 24 | Q heads |
| `num_key_value_heads` | 2 | K/V heads, GQA |
| `hidden_act` | `gelu_pytorch_tanh` | MLP activation |
| `max_position_embeddings` | 4096 | RoPE cache/default context metadata |
| `rope_parameters` | `None` before normalization | runtime reads `rope_parameters["rope_type"]` and `["rope_theta"]` |
| `sliding_window` | `None` | full causal when absent; checkpoint configs set 4096 |
| `norm_epsilon` | `1e-5` | LayerNorm epsilon |
| `use_bias` | `true` | all attention and MLP linears have bias |
| `tie_word_embeddings` | `true` | LM head aliasing unless checkpoint overrides |
| `use_cache` | `true` | DynamicCache default |

Checkpoint sweep:

| Model | Layers | Hidden | Heads / KV | Head dim | MLP | Vocab | Max pos | Sliding | RoPE theta | Act | Bias | Tie embeddings | dtype metadata |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| `bigcode/starcoder2-3b` | 30 | 3072 | 24 / 2 | 128 | 12288 | 49152 | 16384 | 4096 | 999999.4420358813 | `gelu_pytorch_tanh` | true | omitted, effective true from source default | omitted |
| `bigcode/starcoder2-7b` | 32 | 4608 | 36 / 4 | 128 | 18432 | 49152 | 16384 | 4096 | 1000000 | `gelu_pytorch_tanh` | true | omitted, effective true from source default | `bfloat16` |
| `bigcode/starcoder2-15b` | 40 | 6144 | 48 / 4 | 128 | 24576 | 49152 | 16384 | 4096 | 100000 | `gelu_pytorch_tanh` | true | false | `float32` |
| `bigcode/starcoder2-15b-instruct-v0.1` | 40 | 6144 | 48 / 4 | 128 | 24576 | 49152 | 16384 | 4096 | 100000 | `gelu_pytorch_tanh` | true | false | `bfloat16` |
| `hf-internal-testing/tiny-random-Starcoder2ForCausalLM` | 2 | 32 | 4 / 2 | 8 | 37 | 1024 | 512 | null | 10000 | `gelu` | true | omitted, effective true | `float32` |

The official 3B/7B/15B code checkpoints share the same operator structure. The meaningful production differences are size, KV-head count, RoPE theta, weight tying, and dtype metadata. The instruct 15B keeps the 15B architecture but sets dropout fields to 0 and carries a chat template in tokenizer config.

## 3a. Family variation traps

- GQA is always present in official production checkpoints: `num_key_value_heads < num_attention_heads`. Do not lower as plain MHA unless KV repetition or native GQA attention is explicit.
- `hidden_size == num_attention_heads * head_dim` in inspected configs, but source computes `head_dim = config.head_dim or hidden_size // num_attention_heads`; a future config with explicit `head_dim` can change projection output widths.
- All source linears use `config.use_bias`; official configs set `use_bias=true`.
- MLP is ungated: one up projection, activation, one down projection. Historical config fields such as `mlp_type` are not read by current source.
- Production configs set `sliding_window=4096`, so the current source uses sliding-window causal mask and `DynamicSlidingWindowLayer` cache. Tiny random has no sliding window and should not be used to infer production cache behavior.
- Checkpoints use legacy top-level `rope_theta`; current `PreTrainedConfig` normalizes this into `config.rope_parameters`, and `Starcoder2RotaryEmbedding` reads `config.rope_parameters`.
- Production max position is 16384, while local attention window is 4096. Positions can grow past the cache window; RoPE is still based on absolute `position_ids`.
- 15B and 15B-instruct set `tie_word_embeddings=false`; 3B and 7B omit it and inherit the source default `true`.
- Config-advertised legacy fields observed but not used by current StarCoder2 source include `activation_function`, `attention_softmax_in_fp32`, `scale_attention_softmax_in_fp32`, `scale_attn_weights`, `layer_norm_epsilon`, `norm_type`, `mlp_type`, `hidden_dropout_prob`, `attention_probs_dropout_prob`, `type_vocab_size`, and `is_decoder`.
- No NHWC/channel-last layout rewrite is relevant for the text decoder. Protect attention head/layout reshapes and transposes from image-style layout translation.

## 4. Operator coverage checklist

Tensor/layout ops:

- Token embedding lookup: `Embedding(vocab_size -> hidden_size)`.
- View/reshape: `[B, S, hidden] -> [B, S, heads, head_dim] -> [B, heads, S, head_dim]`.
- Transpose/contiguous around attention output.
- KV repeat/expand for GQA fallback: `[B, kv_heads, S, D] -> [B, q_heads, S, D]`.
- Slice/index for `logits_to_keep`: hidden states can be sliced before LM head.
- Optional tied-weight alias between embedding table and LM head.

Neural network primitives:

- LayerNorm over last dim, epsilon `1e-5`.
- Biased Linear for Q, K, V, O, MLP up/down.
- `gelu_pytorch_tanh` for production MLP; tiny random uses `gelu`.
- Dropout appears in source but is zero or disabled for inference.
- Residual adds.

Attention primitives:

- Causal self-attention with GQA.
- Sliding-window causal mask for official checkpoints.
- Dense eager fallback: QK matmul, additive mask, fp32 softmax, dropout, AV matmul.
- SDPA/Flash/Flex dispatch is advertised by source; DinoML can start with its own semantic attention and later map to optimized kernels.

Position/cache ops:

- RoPE on Q and K before cache update.
- `DynamicCache` / `DynamicSlidingWindowLayer` update, reorder, crop semantics.
- Position ID generation from `past_key_values.get_seq_length()`.

Preprocessing-coupled ops:

- GPT-2-style BPE tokenization and attention mask construction are CPU/data-pipeline work.
- Instruct checkpoint chat template is generation-controller/prompt construction, not model graph.

Distributed/tensor-parallel hints:

- Source config has TP plans: Q/K/V/c_fc colwise, O/c_proj rowwise, LM head colwise gather. Treat as optional optimization, not first-integration requirement.

## 5. Layer/block breakdown

For each production decoder block, repeated `num_hidden_layers`:

```text
x: [B, S, H]
residual = x
x = LayerNorm(H, eps=1e-5)(x)

q = Linear(H -> num_attention_heads * head_dim, bias=True)(x)
k = Linear(H -> num_key_value_heads * head_dim, bias=True)(x)
v = Linear(H -> num_key_value_heads * head_dim, bias=True)(x)

q = view(q, [B, S, QH, D]).transpose(1, 2)  # [B, QH, S, D]
k = view(k, [B, S, KVH, D]).transpose(1, 2) # [B, KVH, S, D]
v = view(v, [B, S, KVH, D]).transpose(1, 2) # [B, KVH, S, D]

q, k = RoPE(q, k, cos[position_ids], sin[position_ids])
k_cache, v_cache = cache.update(k, v, layer_idx)
attn = causal_sliding_gqa_attention(q, k_cache, v_cache, mask)
attn = transpose/reshape(attn, [B, S, QH * D])
attn = Linear(QH * D -> H, bias=True)(attn)
x = residual + attn

residual = x
x = LayerNorm(H, eps=1e-5)(x)
x = Linear(H -> I, bias=True)(x)
x = gelu_pytorch_tanh(x)
x = Linear(I -> H, bias=True)(x)
x = residual + x
```

Production dimensions:

| Model | Q projection | K/V projection | O projection | MLP |
| --- | --- | --- | --- | --- |
| 3B | `3072 -> 3072` | `3072 -> 256` each | `3072 -> 3072` | `3072 -> 12288 -> 3072` |
| 7B | `4608 -> 4608` | `4608 -> 512` each | `4608 -> 4608` | `4608 -> 18432 -> 4608` |
| 15B / instruct | `6144 -> 6144` | `6144 -> 512` each | `6144 -> 6144` | `6144 -> 24576 -> 6144` |

Final head:

```text
hidden = final LayerNorm(H)(x)
logits = Linear(H -> vocab_size, bias=False)(hidden[:, logits_to_keep, :])
```

## 6. Attention requirements

Required attention for official production checkpoints:

- Type: causal self-attention.
- Structure: GQA, not full MHA.
- Q heads: 24 / 36 / 48 for 3B / 7B / 15B.
- KV heads: 2 / 4 / 4 for 3B / 7B / 15B.
- Head dim: 128 in official production checkpoints.
- Scaling: `head_dim ** -0.5`.
- Mask: sliding-window causal mask with local size 4096, plus optional padding mask. Tiny random uses full causal mask because `sliding_window=null`.
- RoPE: applied to Q and K before cache update.
- Cache storage before repeat expansion: `[B, num_key_value_heads, cached_seq, head_dim]`.
- Eager fallback expands cached K/V to Q heads using `repeat_kv`.
- SDPA path may use native `enable_gqa=True` only when supported and `attention_mask is None`; otherwise it repeats K/V.
- FlashAttention path receives unexpanded K/V and a `sliding_window` argument.

Cache details:

- With full attention, `DynamicLayer.update` concatenates along sequence and returns full `[B, KVH, total_seq, D]`.
- With sliding attention, `DynamicSlidingWindowLayer.update` retains only the last `sliding_window - 1` cached states but returns full states for the current call. Mask sizes account for a moving `kv_offset`.
- `get_seq_length()` for sliding cache returns cumulative sequence length, not resident tensor length.
- Beam reorder/select operations index the batch dimension of cached keys/values.

Fused attention parity needs to preserve:

- RoPE before cache update.
- Sliding-window offset behavior.
- Additive mask before softmax in eager path.
- Softmax computed in fp32 in eager path, then cast back to query dtype.
- Dropout disabled in inference.

## 7. Position encoding and custom math

StarCoder2 uses standard RoPE with model-specific theta from config. Current source normalizes legacy `rope_theta` into `config.rope_parameters`, then computes:

```python
def starcoder2_inv_freq(config):
    base = config.rope_parameters["rope_theta"]
    dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    return 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
```

Runtime forward:

```python
freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
emb = torch.cat((freqs, freqs), dim=-1)
cos = emb.cos().to(x.dtype)
sin = emb.sin().to(x.dtype)
q = q * cos[:, None, :, :] + rotate_half(q) * sin[:, None, :, :]
k = k * cos[:, None, :, :] + rotate_half(k) * sin[:, None, :, :]
```

`rotate_half(x)` splits the last dimension in half and returns `[-x2, x1]`. Cos/sin can be precomputed up to max position for fixed theta, but dynamic or advanced RoPE types in `ROPE_INIT_FUNCTIONS` would require separate admission. Inspected official configs use default RoPE only.

## 8. Preprocessing and input packing

Text preprocessing:

- Tokenizer class from official snapshots: `GPT2Tokenizer`.
- Special token ID 0 is `<|endoftext|>` for BOS/EOS/UNK.
- Added code/FIM/repository special tokens are tokenizer-level tokens and do not require model-side scatter/stitch ops.
- Model graph accepts either `input_ids` or `inputs_embeds`, exactly one.
- `attention_mask` is optional; when present it participates in causal/sliding mask construction.
- If `position_ids` is omitted, source builds `[0..S-1] + past_seen_tokens` and unsqueezes to `[1, S]`.

Instruct preprocessing:

- `bigcode/starcoder2-15b-instruct-v0.1` tokenizer config carries a chat template that formats user/assistant messages with `### Instruction` / `### Response`.
- The template rejects system messages. This belongs in the prompt/generation controller, not in the compiled model graph.

No multimodal placeholders, image/audio processors, packed varlen metadata, or codebook constraints are required by native StarCoder2 source.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Biased Linear -> GEMM bias epilogue

Source pattern:

```text
y = torch.nn.Linear(in, out, bias=True)(x)
```

Replacement:

```text
flatten leading dims -> GEMM_RCR(x, weight) + bias -> restore leading dims
```

Preconditions:

- Input is dense row-major with last dim equal to `in`.
- Weight is PyTorch linear storage `[out, in]`; use RCR-style RHS as needed.
- Bias shape is `[out]`.
- Leading dimensions are flattened into M and restored after GEMM.

Failure cases:

- Non-contiguous/strided input without explicit accessor lowering.
- Quantized or offloaded weights unless represented through DinoML constant policies.

Parity test sketch:

- Compare each projection and MLP linear against PyTorch for fp32/fp16/bf16 with representative StarCoder2 shapes.

### Rewrite: GQA attention canonicalization

Source pattern:

```text
k = repeat_kv(k, q_heads // kv_heads)
v = repeat_kv(v, q_heads // kv_heads)
softmax(q @ k.T * scale + mask) @ v
```

Replacement:

```text
native_gqa_attention(q, k, v, group_size=q_heads // kv_heads, mask, scale)
```

Preconditions:

- `num_attention_heads % num_key_value_heads == 0`.
- Q/K/V shapes are `[B, heads, S, D]` and `[B, kv_heads, S_cache, D]`.
- RoPE has already been applied to Q/K.
- Sliding-window and padding mask semantics are preserved.

Failure cases:

- Requesting `output_attentions=True` if optimized backend cannot return dense attention weights.
- Mask/offset combinations not covered by the backend.

Parity test sketch:

- Compare eager repeated-KV path and native GQA path for prefill, one-token decode, and multi-token decode with a filled sliding cache.

### Rewrite: RoPE fusion into Q/K projection or attention

Source pattern:

```text
q_proj/k_proj -> view/transpose -> RoPE -> attention
```

Replacement:

```text
QK projection epilogue or attention prologue applies RoPE from position_ids
```

Preconditions:

- Default RoPE type and known theta.
- Position IDs are monotonic or supplied as an explicit tensor.
- Head dim even.

Failure cases:

- Advanced `rope_type` values from future configs.
- Packed/non-monotonic position IDs unless kernel accepts explicit position IDs.

Parity test sketch:

- Random Q/K and position IDs, compare fused and unfused RoPE for all production head dims.

### Rewrite: Last-token-only logits

Source pattern:

```text
logits = lm_head(hidden_states[:, slice_indices, :])
```

Replacement:

```text
slice hidden before GEMM; for decode use only [B, 1, H]
```

Preconditions:

- `logits_to_keep` is an int or static/dynamic index tensor that DinoML can represent.
- Generation only needs last-token logits.

Failure cases:

- Training/loss path requires all shifted logits.
- Caller requests full logits for analysis.

Parity test sketch:

- Prefill with `logits_to_keep=1` and compare with full-logits slice.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm over `[B*S, H]`: every block has two LayerNorms plus final norm.
- Biased GEMM for all projections and MLP/LM head: dominant compute.
- GQA FlashAttention/SDPA equivalent with sliding-window support and KV cache.
- RoPE + attention prefill/decode: avoid extra memory traffic on Q/K.
- Last-token-only LM head for decode/prefill sampling.

Medium priority:

- Q/K/V projection scheduling: separate K/V projection outputs are much smaller than Q; a fused multi-output projection is useful only if weight layout and bias handling remain explicit.
- GELU tanh approximation fused with MLP down-projection input path.
- Residual-add + following LayerNorm fusion if DinoML has a stable norm kernel.
- KV cache update + attention for decode, especially for sliding-window eviction.

Lower priority:

- Dropout elimination in inference graphs.
- Tensor-parallel sharding plans from config.
- Returning attention weights for debug/evaluation; optimized paths can defer it.

## 11. Runtime staging plan

Stage 1: config/weight loading.

- Parse StarCoder2 config including legacy `rope_theta -> rope_parameters`.
- Load embeddings, biased linears, LayerNorms, and optional tied LM head alias.
- Reject unsupported advanced RoPE types initially.

Stage 2: one-block eager parity.

- Implement embedding, LayerNorm, biased linear, RoPE, repeated-KV attention, MLP, residuals.
- Validate 3B-like and tiny random dimensions.

Stage 3: full prefill parity.

- Add full stack execution with sliding-window causal mask.
- Compare final hidden states and logits for short and >4096-token synthetic contexts if feasible.

Stage 4: decode with KV cache.

- Implement cache ABI as `[B, KVH, resident_seq, D]` plus cumulative position length for sliding windows.
- Validate prefill+single-token decode and multi-token decode.

Stage 5: optimized attention.

- Add native GQA attention and sliding-window mask support.
- Add backend fallbacks for full causal tiny/random configs.

Stage 6: graph rewrites/fusions.

- Fuse linear+bias, RoPE prologue, attention, GELU/MLP, last-token logits.

Stage 7: production scheduling.

- Continuous batching, cache memory planning, tensor parallelism, GGUF/offload policy if needed.

Initially stub/defer classification heads, training loss, dropout, beam-search-specific cache mutations beyond reorder, and attention-weight outputs.

## 12. Parity and validation plan

Concrete tests:

- Config normalization tests for official configs: top-level `rope_theta` must produce default `rope_parameters` with the expected theta.
- Operator tests: LayerNorm, GELU tanh, RoPE, repeat_kv, biased linear.
- Single decoder layer parity against Transformers with random weights for fp32.
- Full tiny random model logits parity for short full-causal sequences.
- Production-shape synthetic block parity for 3B/7B/15B dimensions without loading full weights.
- Prefill logits parity for a small real checkpoint slice if weights are available.
- Decode parity: prefill then one token; compare logits and cache shapes.
- Sliding-window parity: sequence length around 4095, 4096, 4097, and longer than window; verify resident cache length and cumulative `position_ids`.
- Weight tying parity: 3B/7B effective tied embeddings versus 15B untied LM head.
- Instruct prompt parity outside runtime: tokenizer template output IDs match HF tokenizer.

Suggested tolerances:

- fp32: absolute/relative around `1e-5` to `1e-4` for one block, looser across full stack.
- fp16/bf16: compare against HF reduced-precision outputs with attention backend fixed; expect `1e-2`-level logits tolerance depending on accumulation.

## 13. Performance probes

- Prefill throughput by batch and sequence length: 1k, 4k, 8k, 16k.
- Decode tokens/sec by batch size and resident cache length.
- Sliding-window boundary probe at 4096 tokens versus full-cache emulation.
- Attention backend comparison: repeated-KV dense attention vs native GQA vs sliding-window FlashAttention-style kernel.
- KV cache memory: resident `[layers, B, 2, KVH, min(seq, 4095), D]` plus cumulative-length metadata.
- LM head cost with full logits versus last-token-only logits.
- MLP GEMM throughput by model size, especially `H -> 4H -> H`.
- Weight-loading/offload probes for 15B if GGUF or CPU-to-GPU staged constants are used.

## 14. Skip/defer list

Safe to defer for first causal-LM integration:

- Training loss and labels path.
- Dropout behavior in training.
- Sequence/token classification heads.
- `output_attentions=True` dense attention tensors.
- FlexAttention and paged attention interfaces.
- Advanced RoPE types beyond inspected default RoPE.
- Tensor parallel and pipeline parallel plans.
- Quantization/GGUF/offload unless selected as a separate weight-ingestion target.
- Chat template execution inside DinoML runtime; keep it in CPU/generation pipeline.
- Beam search and speculative decoding beyond minimal cache reorder support.

## 15. Final implementation checklist

- [ ] Parse StarCoder2 config and normalize legacy `rope_theta`.
- [ ] Load embedding, LayerNorm, biased projection/MLP weights, and LM head.
- [ ] Preserve tied embedding/LM-head alias when `tie_word_embeddings=true`.
- [ ] Implement `gelu_pytorch_tanh`.
- [ ] Implement default RoPE with explicit `position_ids`.
- [ ] Implement GQA attention with pre-RoPE cached K/V.
- [ ] Implement sliding-window causal mask and cache metadata.
- [ ] Add full-causal fallback for configs with `sliding_window=null`.
- [ ] Add one-block PyTorch parity tests.
- [ ] Add tiny-random full-model logits parity.
- [ ] Add prefill/decode cache parity, including sliding-window boundaries.
- [ ] Add last-token-only logits lowering.
- [ ] Add optimized GQA/sliding-window attention backend.
- [ ] Benchmark prefill, decode, LM head, and KV memory.
