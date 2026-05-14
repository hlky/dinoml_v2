# Gemma4 Assistant DinoML Audit

## 1. Source Basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4` from `X:/H/transformers`.

Model id: representative official checkpoints `google/gemma-4-E2B-it-assistant`, `google/gemma-4-E4B-it-assistant`, `google/gemma-4-26B-A4B-it-assistant`, `google/gemma-4-31B-it-assistant`.

Config source: raw Hugging Face Hub `config.json`, `generation_config.json`, and `tokenizer_config.json` snapshots saved in this folder. Metadata files were public at audit time; weights are large Xet-backed model files and may require the normal Google/HF access flow.

Source files inspected:

- `src/transformers/models/gemma4_assistant/configuration_gemma4_assistant.py`
- `src/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py`
- `src/transformers/models/gemma4/configuration_gemma4.py`
- `src/transformers/models/gemma4/modeling_gemma4.py`
- `src/transformers/models/gemma4/modular_gemma4.py`
- `src/transformers/masking_utils.py`

Source URLs at the inspected commit:

- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4_assistant/configuration_gemma4_assistant.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4_assistant/modeling_gemma4_assistant.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/configuration_gemma4.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/models/gemma4/modeling_gemma4.py
- https://github.com/huggingface/transformers/blob/b75feb2af64c3e29cbbc1bd859958c5432cc7ed4/src/transformers/masking_utils.py

Representative model/config URLs:

- https://huggingface.co/google/gemma-4-E2B-it-assistant
- https://huggingface.co/google/gemma-4-E4B-it-assistant
- https://huggingface.co/google/gemma-4-26B-A4B-it-assistant
- https://huggingface.co/google/gemma-4-31B-it-assistant

Any missing files or assumptions: `gemma4/modeling_gemma4.py` is generated from `modular_gemma4.py`; future source edits should inspect the modular file first. `gemma4_assistant` has no modular source file in this checkout. This report targets inference-time assisted decoding, not standalone text generation from `input_ids`.

## 2. High-Level Architecture

Gemma4 Assistant is a text-only assistant drafter for Gemma 4 assisted decoding. It is not the main multimodal Gemma4 model: it consumes hidden inputs and shared KV states produced by the target/backbone model, runs a small four-layer Gemma4 text stack whose K/V projections are entirely shared from the backbone, and returns draft logits plus a projected hidden state.

Dataflow:

```text
main Gemma4 generation step -> assistant inputs_embeds + shared_kv_states
  -> pre_projection(2 * backbone_hidden_size -> assistant_hidden_size)
  -> 4-layer Gemma4TextModel with shared external KV states
  -> post_projection(assistant_hidden_size -> backbone_hidden_size)
  -> dense lm_head or centroid/top-k masked vocabulary head
  -> assistant logits for speculative/assisted decode controller
```

Independently stageable pieces: config/weight loading, input projection, one shared-KV attention block, four-layer drafter parity, logits head parity, and generation-controller integration with the main Gemma4 model.

## 3. Important Config Dimensions

| checkpoint | backbone hidden | assistant hidden | layers | heads | sliding KV | full KV | head dim | full head dim | MLP | max pos | window | ordered head |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `gemma-4-E2B-it-assistant` | 1536 | 256 | 4 | 4 | 1 | 1 | 256 | 512 | 2048 | 131072 | 512 | yes |
| `gemma-4-E4B-it-assistant` | 2560 | 256 | 4 | 4 | 2 | 2 | 256 | 512 | 2048 | 131072 | 512 | yes |
| `gemma-4-26B-A4B-it-assistant` | 2816 | 1024 | 4 | 16 | 8 | 2 | 256 | 512 | 8192 | 262144 | 1024 | no |
| `gemma-4-31B-it-assistant` | 5376 | 1024 | 4 | 32 | 16 | 4 | 256 | 512 | 8192 | 262144 | 1024 | no |

Common fields from configs: `vocab_size=262144`, `num_centroids=2048`, `centroid_intermediate_top_k=32`, dtype `bfloat16`, activation `gelu_pytorch_tanh`, RMSNorm epsilon `1e-6`, layer types `[sliding, sliding, sliding, full]`, no MoE, no per-layer embeddings, no double-wide MLP, and `num_kv_shared_layers == num_hidden_layers`.

Generation config snapshot: all four use `bos_token_id=2`, `pad_token_id=0`, `eos_token_id=[1,106,50]`, `do_sample=true`, `temperature=1`, `top_p=0.95`. Tokenizer config reports `GemmaTokenizer` with no assistant-specific chat template.

## 3a. Family Variation Traps

- The assistant forward requires `inputs_embeds` and `shared_kv_states`; `input_ids` and `use_cache` are signature compatibility fields and are not used.
- `inputs_embeds` must have last dimension `2 * backbone_hidden_size` before `pre_projection`; this is not ordinary token embedding input.
- All assistant layers are KV-shared. They do not own K/V projection weights, K norm, or V norm; lowering must source keys/values from `shared_kv_states`.
- Full-attention layers use `global_head_dim=512`; sliding layers use `head_dim=256`. Projection widths are layer-type dependent.
- For 26B/31B assistants, full attention has `attention_k_eq_v=true` and `num_global_key_value_heads` smaller than sliding KV heads.
- E2B/E4B assistants set `use_ordered_embeddings=true` and require the centroid/top-k masked vocabulary head with `token_ordering`. 26B/31B use the dense LM head.
- The assistant config validation rejects MoE, per-layer input embeddings, double-wide MLP, and partial KV sharing.
- The last layer is full attention. A naive 5:1 default Gemma4 sliding pattern is not the assistant pattern here.
- No vision/audio branch belongs to this assistant target, despite token IDs for image/audio being present in checkpoint configs.

## 4. Operator Coverage Checklist

Tensor/layout ops: embedding lookup for tied token table, reshape/view, transpose, contiguous, slice, dict-select by layer type, `flip` for sliding masks, `where`/mask additions from shared mask builders, scalar fill, indexed gather, scatter along vocab axis.

Neural primitives: bias-free Linear, RMSNorm, elementwise residual add, GELU tanh approximation, activation multiply for gated MLP, final dense LM projection, optional `tanh` softcap only if inherited configs enable it. Current inspected assistant configs have `final_logit_softcapping=null`.

Attention primitives: GQA/MQA-style repeat-KV attention, external K/V state reads, RoPE on Q only for shared-KV layers, bidirectional full attention mask, bidirectional sliding-window mask, fp32 softmax, dropout disabled for inference, output projection.

Position/rotary ops: two RoPE parameter families: sliding default RoPE with theta 10000 and full proportional RoPE with theta 1000000, partial factor 0.25, and `global_head_dim` for full attention.

Generation/cache ops: assistant-specific shared KV state ABI keyed by `"sliding_attention"` and `"full_attention"`; no assistant-owned growing KV cache for this target.

Scatter/index update ops: optional masked vocabulary head uses centroid top-k, flattened gather from `lm_head.weight`, batched dot product, full-vocab fill with `min(selected_logits)-1`, and scatter into canonical token positions.

Distributed/tensor-parallel notes: source declares `lm_head` as `colwise_gather_output`; DinoML can defer TP.

## 5. Layer/Block Breakdown

Assistant wrapper:

```text
inputs_embeds[B,L,2*backbone_H]
  -> Linear(2*backbone_H -> H, bias=False)
  -> Gemma4TextModel(H)
  -> Linear(H -> backbone_H, bias=False) for returned hidden state
  -> logits head
```

Decoder block, repeated 4 times:

```text
residual = x
x = RMSNorm(x)
q = Linear(H -> num_heads * layer_head_dim, bias=False)
q = RMSNorm(q per head)
q = RoPE(q)
k,v = shared_kv_states[layer_type]
x = Attention(q, k, v, bidirectional mask)
x = Linear(num_heads * layer_head_dim -> H, bias=False)
x = RMSNorm(x)
x = residual + x

residual = x
x = RMSNorm(x)
x = Linear(H -> intermediate, bias=False)
x = gelu_pytorch_tanh(x) * Linear(H -> intermediate, bias=False)
x = Linear(intermediate -> H, bias=False)
x = RMSNorm(x)
x = residual + x
```

Dense logits head: `Linear(H -> vocab_size, bias=False)`.

Ordered logits head: `Linear(H -> 2048)` centroids, `topk(k=32)`, gather `32 * (262144/2048) = 4096` candidate token embeddings, dot against hidden state, fill full vocab tensor, scatter selected logits.

## 6. Attention Requirements

Required attention is bidirectional cross/self-style attention over externally provided shared KV states, not normal causal decode inside the assistant.

- Layer pattern: 3 sliding-window layers followed by 1 full-attention layer.
- MHA/GQA: query heads vary by checkpoint; KV heads are smaller and repeated by the attention backend.
- Query shape: `[B, q_len, num_heads, head_dim]` before transpose; attention backend consumes `[B, heads, q_len, head_dim]`.
- KV state shape expected by attention: `[B, num_kv_heads, kv_len, head_dim]`, selected from `shared_kv_states[layer_type]`.
- Sliding attention uses `sliding_window` 512 or 1024 and flips the generated bidirectional sliding mask on the KV axis in assistant mask creation.
- Cache: no assistant-owned `past_key_values` in the target path. Shared KV states are produced by the main model and reused across all assistant layers.
- Backend compatibility: source marks FlashAttention and SDPA support, but DinoML first parity can use explicit matmul-softmax-matmul with the exact mask semantics.

## 7. Position Encoding And Custom Math

Text RoPE is computed once per layer type, not per layer. Sliding layers use default RoPE; full layer uses proportional RoPE and `global_head_dim`.

Short reproduction sketch:

```python
def gemma4_text_rope(x, position_ids, inv_freq, attention_scaling):
    freqs = (inv_freq[None, :, None].float() @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    return (emb.cos() * attention_scaling).to(x.dtype), (emb.sin() * attention_scaling).to(x.dtype)
```

Assistant shared-KV layers apply RoPE only to Q in the assistant. K has already been produced and position-encoded by the backbone path that created `shared_kv_states`.

## 8. Preprocessing And Input Packing

CPU/data-pipeline: tokenizer is ordinary `GemmaTokenizer`; no assistant-specific processor or chat template was found in tokenizer configs.

GPU/runtime ABI:

- `inputs_embeds`: dense tensor `[B, L, 2 * backbone_hidden_size]`.
- `shared_kv_states`: dict with `"full_attention"` and `"sliding_attention"` entries; each entry is `(key, value)`.
- `attention_mask`: optional 2D mask over the main/backbone KV length. Assistant code truncates full masks to full KV length and slices/flips the sliding mask to match sliding KV state length.
- `position_ids`: optional; if omitted, underlying text model can create positions, but assistant integration should pass positions aligned to the main model.

No image/audio/video tensors are required for `gemma4_assistant`.

## 9. Graph Rewrite / Lowering Opportunities

### Rewrite: Shared-KV Assistant Attention

Source pattern: attention layers with `num_kv_shared_layers == num_hidden_layers`, no K/V projection modules, and `shared_kv_states[layer_type]`.

Replacement: `QProjection -> QRMSNorm -> QRoPE -> GQAAttention(Q, external_K, external_V, mask) -> OProjection`.

Preconditions: assistant config validation passes, `shared_kv_states` contains both layer types, K/V head count and head dimension match the checkpoint config, and K states are already RoPE-applied by the backbone.

Failure cases: missing shared states, wrong full/sliding head dim, or attempting to treat the assistant as standalone causal LM.

### Rewrite: Ordered Masked Head To Bounded Top-K Vocabulary Projection

Source pattern: centroid projection, top-k centroids, gather `lm_head.weight`, batched dot, full tensor fill, scatter.

Replacement: top-k centroid selection plus candidate-token GEMM. For first parity, materialize full `[B,L,V]` logits and scatter. Later, expose sparse candidate logits to the assisted-generation controller if DinoML owns that controller.

Preconditions: `vocab_size % num_centroids == 0`, `token_ordering` is resident int64 metadata, and selected candidate count is `centroid_top_k * vocab_per_centroid`.

Failure cases: arbitrary scatter admission without bounded vocab-axis semantics, or preserving `.min().item()` as a CPU sync in a CUDA graph.

### Rewrite: Dense Head Last-Token Slice

Source pattern: dense LM head over all assistant hidden states.

Replacement: when the generation controller only needs current draft positions, slice hidden states before the LM head.

Preconditions: logits consumer does not require all positions.

Failure cases: parity tests requesting full sequence logits.

## 10. Kernel Fusion Candidates

Highest priority:

- RMSNorm, including head-wise Q RMSNorm.
- Gated MLP `gate_proj -> gelu_tanh -> multiply up_proj -> down_proj`.
- GQA attention with external KV and bidirectional/sliding masks.
- Ordered masked head top-k/gather/dot/scatter for E2B/E4B.

Medium priority:

- Q projection + Q RMSNorm + RoPE.
- Dense last-token LM head for 26B/31B assistants.
- Mask creation and flip/slice logic as graph-visible shape/mask preprocessing.

Lower priority:

- Tensor parallel LM-head output gather.
- Sparse candidate-logits output ABI that avoids full-vocab materialization.

## 11. Runtime Staging Plan

Stage 1: parse assistant configs and reject unsupported variants explicitly.

Stage 2: load weights and verify tied `lm_head.weight` / `model.embed_tokens.weight` aliasing.

Stage 3: run one block with synthetic `shared_kv_states` and explicit masks.

Stage 4: run the full four-layer assistant with dense LM head configs.

Stage 5: add ordered masked head for E2B/E4B.

Stage 6: integrate with a Gemma4 backbone run that returns shared KV states.

Stage 7: optimize attention and logits head for assisted decoding throughput.

Initially stub: generation sampling, tokenizer chat formatting, TP, and full sparse-candidate controller integration.

## 12. Parity And Validation Plan

- Config validation tests for all four snapshots.
- Random tensor parity for `pre_projection` and `post_projection`.
- RMSNorm parity in bf16/fp32 with tolerance around `1e-3` bf16 and `1e-5` fp32.
- Single attention-layer parity for sliding and full layer types with synthetic shared KV states.
- Mask parity for full bidirectional and flipped bidirectional sliding masks.
- Four-layer assistant hidden-state parity with `use_ordered_embeddings=false`.
- Ordered masked head parity: verify top-k centroid indices, gathered token positions, scatter locations, and fill value.
- End-to-end assisted decode smoke test with a Gemma4 backbone: compare accepted/rejected draft tokens against Transformers for a short prompt.

## 13. Performance Probes

- Assistant-only latency by batch, draft length, and KV length.
- Sliding versus full attention time, separated by layer type.
- Dense LM head versus ordered masked head time and memory bandwidth.
- KV state read bandwidth from main model to assistant.
- Full-vocab materialization cost for E2B/E4B ordered head.
- Prefill/decode throughput of main model with and without assistant.
- Candidate acceptance rate versus assistant cost for representative prompts.

## 14. Skip/Defer List

- Training, loss, gradient checkpointing.
- Vision/audio/video Gemma4 paths; they belong to the main `gemma4` audit.
- MoE, per-layer embeddings, double-wide MLP, and partial KV sharing; assistant config rejects them.
- Tensor parallel and pipeline parallel execution.
- General boolean scatter admission beyond the bounded ordered-vocab head.
- Sparse logits ABI until full-logit parity exists.

## 15. Final Implementation Checklist

- [ ] Parse `Gemma4AssistantConfig` and nested `Gemma4TextConfig`.
- [ ] Reject non-assistant variants: MoE, PLE, double-wide MLP, partial KV sharing.
- [ ] Load tied token embedding / LM head weights with alias preservation.
- [ ] Implement `pre_projection` and `post_projection`.
- [ ] Implement Gemma4 RMSNorm and scaled word embedding.
- [ ] Implement assistant shared-KV attention for sliding and full layer types.
- [ ] Implement Gemma4 text RoPE with per-layer-type parameters.
- [ ] Implement bidirectional full and flipped bidirectional sliding masks.
- [ ] Implement gated GELU MLP.
- [ ] Implement dense LM head path.
- [ ] Implement ordered centroid/top-k masked vocabulary head.
- [ ] Add single-layer and four-layer parity tests.
- [ ] Add integrated assisted-decoding parity with a Gemma4 backbone.
- [ ] Benchmark assistant-only and main-model-plus-assistant throughput.
