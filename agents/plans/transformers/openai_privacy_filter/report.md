# Transformers Family Audit: `openai_privacy_filter`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: openai/privacy-filter
Config source: https://huggingface.co/openai/privacy-filter/raw/main/config.json
Source files inspected:
- transformers/src/transformers/models/openai_privacy_filter/configuration_openai_privacy_filter.py
- transformers/src/transformers/models/openai_privacy_filter/modeling_openai_privacy_filter.py
- transformers/src/transformers/models/openai_privacy_filter/modular_openai_privacy_filter.py
- transformers/src/transformers/models/openai_privacy_filter/convert_openai_privacy_filter_weights_to_hf.py
- transformers/src/transformers/modeling_layers.py
- transformers/src/transformers/masking_utils.py
Any missing files or assumptions:
- `modeling_openai_privacy_filter.py` and `configuration_openai_privacy_filter.py` are generated from `modular_openai_privacy_filter.py`; the generated files are the shipped runtime source, while the modular file is authoritative for future Transformers edits.
- The official Hub repo is public, not gated. No remote-code files are required.
- Only small Hub files were inspected: `config.json`, `tokenizer_config.json`, `original/config.json`, `original/dtypes.json`, `viterbi_calibration.json`, README, and model API metadata. Weights were not downloaded.
- Representative structural sweep is limited because the official repo exposes one canonical checkpoint. ONNX fp16/q4/q4f16/quantized artifacts share the same source graph and are deployment artifacts, not distinct model configs.
```

Small snapshots are stored beside this report:

- `hf_config_snapshot.json`
- `original_config_snapshot.json`
- `tokenizer_config_snapshot.json`
- `viterbi_calibration_snapshot.json`

## 2. High-level architecture

Primary runtime target: token classification for PII span labeling. This is not an autoregressive text-generation target.

Architecture:

```text
CPU tokenizer/o200k -> input_ids + attention_mask
  -> token embedding
  -> 8 bidirectional sliding-window transformer encoder blocks
  -> final RMSNorm
  -> dropout(0.0) + token classification Linear(640 -> 33)
  -> per-token BIOES logits
  -> optional CPU/controller Viterbi span decoding
```

The body is encoder-style even though it descends from GPT-OSS code. Attention is noncausal, bidirectional, and local with left/right radius 128. The head emits 33 token logits for `O` plus BIOES variants of 8 privacy span classes. End-to-end product parity may require the constrained Viterbi decoder described by the model card and calibrated by `viterbi_calibration.json`; that decoder is postprocessing, not part of the Transformers model graph inspected here.

Independently stageable pieces:

- CPU/data pipeline: o200k tokenizer, padding mask, token-to-character span recovery.
- GPU/runtime graph: embedding, local bidirectional GQA blocks, MoE FFN, token-classification head.
- CPU/controller postprocess: logits to labels/spans, aggregation, optional constrained Viterbi, masking/redaction policy.

## 3. Important config dimensions

| Field | Official `openai/privacy-filter` value | Provenance | Runtime significance |
|---|---:|---|---|
| `hidden_size` | 640 | HF `config.json` | Residual width, classifier input |
| `num_hidden_layers` | 8 | HF `config.json` | Encoder block count |
| `num_attention_heads` | 14 | HF `config.json` | Query heads |
| `num_key_value_heads` | 2 | HF `config.json` | GQA KV heads, 7 query heads per KV head |
| `head_dim` | 64 | HF `config.json` | Q width 896, K/V width 128 |
| `num_attention_heads * head_dim` | 896 | Inference from config | Larger than hidden size; do not infer attention width from hidden size |
| `intermediate_size` | 640 | HF `config.json` | Expert hidden width |
| `num_local_experts` | 128 | HF `config.json` | Sparse MoE expert count |
| `num_experts_per_tok` | 4 | HF `config.json` | Top-k router width |
| `vocab_size` | 200064 | HF `config.json` | Token embedding rows |
| `num_labels` | 33 | Derived from `id2label` | Token-classification output width |
| `sliding_window` | 128 | HF `config.json` | Bidirectional local attention radius in source mask; original config reports effective 257 |
| `max_position_embeddings` | 131072 | HF `config.json` | RoPE/position guard |
| `default_n_ctx` | 128000 | HF `config.json` | Tokenizer model max length |
| `rope_parameters` | YaRN, theta 150000, factor 32, original max 4096 | HF `config.json` | RoPE frequency/scaling |
| `attention_bias` | true | HF `config.json` | Q/K/V/O projections have bias |
| `classifier_dropout` | 0.0 | HF `config.json` | Head dropout is identity at inference |
| `dtype` | bfloat16 | HF `config.json`; safetensors metadata says BF16 plus 112 F32 params | Main dense weights BF16, sinks F32 |
| `use_cache` | true | HF `config.json` | Source base accepts it through generic head, but inspected model has no KV-cache path and `_skip_keys_device_placement = None  # No cache` |

Representative checkpoint sweep:

| Checkpoint/artifact | Config structure | Dtype/format | Notes |
|---|---|---|---|
| `openai/privacy-filter` `model.safetensors` | Canonical 8-layer MoE encoder | BF16 weights plus F32 sinks | Official source target, public Apache-2.0 |
| `openai/privacy-filter` `onnx/model.onnx` | Same graph | ONNX external data | Deployment artifact; not a distinct architecture |
| `openai/privacy-filter` `onnx/model_fp16.onnx` | Same graph | FP16 ONNX | Dtype artifact only |
| `openai/privacy-filter` `onnx/model_q4*.onnx` | Same graph | Quantized ONNX | Quantized deployment artifact; Transformers source does not implement q4 kernels |
| `original/*` checkpoint metadata | Pre-conversion source format | BF16, original packed QKV | Conversion source; not a separate HF model class |

## 3a. Family variation traps

- `hidden_size != num_attention_heads * head_dim`: residual width is 640, attention Q/O inner width is 896. Q projection is `Linear(640 -> 896)`, K/V are `Linear(640 -> 128)`, O is `Linear(896 -> 640)`.
- GQA is mandatory for the official config: `num_key_value_heads=2`, `num_attention_heads=14`, repeat factor 7.
- Attention is noncausal bidirectional sliding-window, not decoder causal attention.
- Source scales Q and K separately by `head_dim**-0.25`, then calls attention with `scaling=1.0`. A fused attention kernel must not also apply `1/sqrt(head_dim)`.
- Attention has one learnable sink scalar per query head. Eager attention concatenates sink logits, softmaxes over `K+1`, then drops the sink probability before `P @ V`.
- RoPE uses interleaved even/odd rotation, not half-split rotation.
- MoE expert projections are stored transposed relative to normal `nn.Linear`: `gate_up_proj` shape `[E, H, 2I]`, `down_proj` shape `[E, I, H]`, both with bias.
- Expert activation is not plain SwiGLU: `gate=clamp(gate,max=7)`, `up=clamp(up,-7,7)`, `glu=gate*sigmoid(1.702*gate)`, output `(up+1)*glu`.
- Router computes in fp32, top-k over all 128 experts, softmaxes only top-k logits, divides by top-k, and expert output is later multiplied by top-k. The product cancels scale algebraically but affects intermediate accumulation if fused incorrectly.
- Official converted source splits original packed QKV tensors in order `[Q, K, V]` by output rows. Original attention sink tensors are multiplied by `log(2.0)` during conversion.
- `hidden_act` appears in HF config as `"silu"` but generated `OpenAIPrivacyFilterConfig` marks `hidden_act` unused in the modular source. Runtime should ignore it for FFN semantics.
- `use_cache=true` appears in config but the inspected model does not implement autoregressive cache update or decode. DinoML should reject generation/KV-cache assumptions for this family.
- There is no vision/audio layout concern. All runtime tensors are token sequences `[B, T, C]`; no NCHW/NHWC translation is applicable.

## 4. Operator coverage checklist

Tensor/layout ops:

- Integer token embedding lookup: `[B,T] -> [B,T,640]`, padding id 199999.
- Reshape/view/transpose/contiguous for attention: `[B,T,896] -> [B,14,T,64]`; `[B,T,128] -> [B,2,T,64]`; output transpose back to `[B,T,896]`.
- `repeat_kv`/head expansion: `[B,2,T,64] -> [B,14,T,64]` for eager attention, or native GQA in fused attention.
- `torch.cat` on attention logits with sink column: `[B,14,T,K] + [B,14,T,1] -> [B,14,T,K+1]`.
- Flatten tokens for MoE: `[B,T,640] -> [B*T,640]`, restore after expert accumulation.
- `one_hot`, `permute`, `nonzero`, `where`, `index_add_` for eager MoE dispatch. For optimized lowering, replace with grouped token buckets and segmented scatter-add.

Neural primitives:

- RMSNorm with fp32 variance and output cast back to input dtype.
- Dense GEMMs with bias:
  - Q: `Linear(640 -> 896)`
  - K: `Linear(640 -> 128)`
  - V: `Linear(640 -> 128)`
  - O: `Linear(896 -> 640)`
  - Classifier: `Linear(640 -> 33)`
  - Router: `Linear(640 -> 128)` in fp32
- MoE grouped expert GEMM:
  - gate/up: per expert `[tokens_e,640] @ [640,1280] + [1280]`
  - down: per expert `[tokens_e,640] @ [640,640] + [640]`
  - weighted accumulation into `[B*T,640]`
- Elementwise: add, multiply, clamp min/max, sigmoid, rsqrt, mean, pow, softmax, topk, greater, zeros_like.

Attention primitives:

- Noncausal self-attention.
- Bidirectional local/sliding mask with inclusive rule `abs(q_idx - kv_idx) <= 128`.
- GQA/MQA-style KV head repeat factor 7.
- RoPE on Q/K before scaling.
- Attention sink support, including stable max-subtract before softmax in eager path.
- FlashAttention-compatible path if kernel supports symmetric sliding window and `s_aux`/`learnable_sink`; otherwise eager dense-local fallback.

Position/rotary/custom math:

- YaRN RoPE parameterization from Transformers `ROPE_INIT_FUNCTIONS`.
- Interleaved even/odd RoPE application.
- Dynamic `position_ids` default as `[0..T-1]` broadcast with batch 1 unless caller supplies positions.

Preprocessing-coupled ops:

- Tokenizer emits `input_ids` and `attention_mask` only.
- EOS and pad share `<|endoftext|>` id 199999.
- Token-to-character span recovery and aggregation are outside the GPU graph.

Postprocessing:

- Per-token logits `[B,T,33]`.
- Optional softmax/argmax for basic token class prediction.
- Optional constrained BIOES Viterbi span decoding using calibration biases for end-to-end model-card parity.

Quantized/packed weight metadata:

- Transformers source uses dense BF16/F32 parameters.
- Official Hub includes quantized ONNX artifacts, but the inspected PyTorch source has no q4 weight loading or kernels.
- Original checkpoint has packed QKV keys; conversion splits them. DinoML should load converted HF weights first unless explicitly adding original-checkpoint ingestion.

## 5. Layer/block breakdown

Base model:

```text
input_ids: [B,T] int64
inputs_embeds = Embedding(200064,640)(input_ids)
position_ids = arange(T)[None,:] if absent
cos,sin = YaRNRoPE(position_ids, head_dim=64)
attention_mask = bidirectional sliding-window mask plus padding mask

repeat 8 blocks:
  residual = x
  h = RMSNorm(x)
  q = Linear(640 -> 896, bias=True)(h).view(B,T,14,64).transpose(1,2)
  k = Linear(640 -> 128, bias=True)(h).view(B,T,2,64).transpose(1,2)
  v = Linear(640 -> 128, bias=True)(h).view(B,T,2,64).transpose(1,2)
  q,k = interleaved_RoPE(q,k,cos,sin)
  q = q * 64**-0.25
  k = k * 64**-0.25
  a = bidirectional_local_attention_with_sink(q,k,v,mask,sinks)
  x = residual + Linear(896 -> 640, bias=True)(a.reshape(B,T,896))

  residual = x
  h = RMSNorm(x)
  flat = h.reshape(B*T,640)
  logits = RouterLinear(640 -> 128, bias=True, fp32)(flat)
  topv, topi = topk(logits, k=4)
  scores = softmax(topv, dim=-1) / 4
  moe = sum_over_top4(expert_down(expert_gate_up(flat), expert=topi) * scores)
  x = residual + moe.reshape(B,T,640) * 4

x = RMSNorm(x)
logits = Linear(640 -> 33, bias=True)(x)
```

All attention projections have bias. Expert projections also have bias. Dropout is present in generic token classification but `classifier_dropout=0.0`, and attention dropout is 0.0 in inference.

## 6. Attention requirements

Required variant:

- Type: self-attention, encoder-style, noncausal.
- Pattern: bidirectional sliding window with inclusive radius 128 from HF config. Original config calls this effective window 257 including self.
- Heads: 14 query heads, 2 KV heads, 64 head dim, GQA repeat factor 7.
- Projection widths: Q width 896; K width 128; V width 128; output input width 896 and output width 640.
- Masking: source creates a mask from padding attention mask plus local bidirectional rule. Already-prepared 4D masks may pass through the Transformers utility.
- Packed/varlen: not in model source. Flash backends may accept packed metadata internally, but DinoML should treat that as an attention-provider optimization, not a model ABI.
- Sliding/local: yes, symmetric local window. For FA-style kernels the source passes `sliding_window=config.sliding_window + 1` from the attention module to account for FA symmetric window conversion. Eager mask construction uses config value 128 directly.
- RoPE: applied to Q/K before scaling.
- Attention sink: per-head scalar, treated as an extra logit column for softmax, then omitted from the value matmul.
- KV cache: not applicable for primary target. The model has no causal decode path and deletes GPT-OSS cache-specific fields in modular attention. Generic token-classification forward accepts `past_key_values`/`use_cache`, but the base model ignores growing-cache semantics.
- SDPA compatibility: source marks `_supports_sdpa = False`. FlashAttention and FlexAttention support are advertised, with compatible FA implementations including `kernels-community/vllm-flash-attn3` and `flash_attention_4`.

Eager attention math:

```python
q, k = rope(q, k, cos, sin)
q = q * head_dim**-0.25
k = k * head_dim**-0.25
scores = matmul(q, repeat_kv(k).transpose(-2, -1))
scores = scores + mask
combined = cat([scores, sinks[..., None]], dim=-1)
combined = combined - max(combined, dim=-1, keepdim=True)
probs = softmax(combined, dim=-1, dtype=float32)
out = matmul(probs[..., :-1].to(v.dtype), repeat_kv(v))
```

## 7. Position encoding and custom math

RoPE is YaRN by default:

```text
rope_type = yarn
rope_theta = 150000
factor = 32
beta_fast = 32
beta_slow = 1
original_max_position_embeddings = 4096
truncate = false
```

The custom source-specific part is the interleaved rotation:

```python
def openai_privacy_filter_rope(x, cos, sin):
    even = x[..., ::2]
    odd = x[..., 1::2]
    first = even * cos - odd * sin
    second = odd * cos + even * sin
    return stack((first, second), dim=-1).flatten(-2)
```

Cos/sin can be precomputed for static or bucketed sequence lengths up to 131072, but the source path accepts caller-provided `position_ids`, so full parity needs dynamic gather or generation by positions. RoPE computation is forced to fp32 before casting cos/sin to the hidden dtype.

Other custom math:

```python
def expert_gate(gate_up):
    gate, up = chunk(gate_up, 2, dim=-1)
    gate = clamp(gate, max=7.0)
    up = clamp(up, min=-7.0, max=7.0)
    return (up + 1.0) * gate * sigmoid(1.702 * gate)
```

## 8. Preprocessing and input packing

Tokenizer contract:

- Tokenizer class in `tokenizer_config.json`: `TokenizersBackend`.
- Encoding: original config says `o200k_base`.
- Inputs: `input_ids`, `attention_mask`.
- Max tokenizer length: 128000.
- EOS token and pad token are both `<|endoftext|>` with id 199999.
- No token type ids, image/audio tensors, packed sequence descriptors, or placeholder tokens.

GPU/runtime inputs:

- `input_ids`: `[B,T]`, integer token ids.
- `attention_mask`: optional `[B,T]` padding mask or prepared `[B,1,T,T]` mask.
- `position_ids`: optional `[B,T]` or source default `[1,T]`.
- `inputs_embeds`: alternative to `input_ids`; exactly one of `input_ids` or `inputs_embeds` must be supplied.

Postprocessing:

- Basic Transformers path returns token logits and the pipeline may aggregate entities.
- Model card describes constrained Viterbi decoding for coherent BIOES spans. The shipped `viterbi_calibration.json` has a `default` operating point with all transition biases zero. DinoML can initially expose logits and argmax labels, then add CPU/controller Viterbi parity as a separate stage.

## 9. Graph rewrite / lowering opportunities

### Rewrite: split independent Q/K/V projections into one packed GEMM

Source pattern:

```text
q = Linear(H -> 14*D)
k = Linear(H -> 2*D)
v = Linear(H -> 2*D)
```

Replacement:

```text
qkv = Linear(640 -> 1152) with row-concatenated weights [Q,K,V]
split qkv as [896,128,128]
```

Preconditions:

- All three projections share the same input tensor.
- Bias enabled consistently for Q/K/V.
- Split order must be `[Q,K,V]`, matching conversion code.
- Output reshapes must preserve Q as 14 heads and K/V as 2 heads.

Failure cases:

- Future configs with different bias flags per projection.
- Provider cannot return split views without extra copies.

Parity test sketch:

- Compare per-projection tensors before RoPE for random `[B,T,640]` in fp32 and bf16.

### Rewrite: GQA local attention with sink

Source pattern:

```text
repeat_kv -> matmul -> mask add -> cat sink -> max subtract -> fp32 softmax -> drop sink -> matmul V
```

Replacement:

```text
single fused local GQA attention kernel with symmetric window, per-head sink logits, fp32 softmax accumulation
```

Preconditions:

- `is_causal == False`.
- Symmetric inclusive local window radius 128.
- Sink semantics exactly match source: sink participates in softmax denominator but not value accumulation.
- Q/K scaling has already been applied as `head_dim**-0.25` to both Q and K, or fused kernel applies that exact split scaling and no extra scale.

Failure cases:

- Backend lacks sink-logit support.
- Full dense attention requested for debugging with prepared 4D masks.
- Sequence length/window shapes unsupported by local attention provider.

Parity test sketch:

- Compare eager and fused attention on short sequences where dense local mask can be materialized, including padded tokens and sink values.

### Rewrite: eager MoE loop to grouped expert GEMM

Source pattern:

```text
topk router -> one_hot/where per expert -> per-expert GEMM -> weighted index_add
```

Replacement:

```text
router topk -> token/expert bucket sort -> grouped GEMM gate_up -> fused gate -> grouped GEMM down -> segmented weighted scatter-add
```

Preconditions:

- Top-k is 4.
- Expert count is 128.
- Expert weights use transposed storage `[E,H,2I]` and `[E,I,H]`.
- Accumulation for expert projections and weighted output remains fp32 as source requests.
- Stable tie behavior for `torch.topk` should be tested, but exact tie determinism is usually not meaningful for trained non-equal logits.

Failure cases:

- Need exact eager accumulation order for numerically sensitive cases. Source comments say the model is sensitive and defaults experts implementation to eager.
- Dynamic expert bucket sizes exceed provider limits.

Parity test sketch:

- Test top-k, expert routing, and final MoE output against source for synthetic logits with known expert hits, including repeated tokens per expert and empty experts.

### Rewrite: token-classification head as last-dim GEMM

Source pattern:

```text
dropout(0.0)([B,T,640]) -> Linear(640 -> 33)
```

Replacement:

```text
flatten [B*T,640] -> GEMM + bias -> reshape [B,T,33]
```

Preconditions:

- Inference only or `classifier_dropout == 0.0`.
- Last dimension is contiguous or provider handles leading dimensions.

Failure cases:

- Training mode with nonzero dropout in non-official config.

## 10. Kernel fusion candidates

Highest priority:

- Local bidirectional GQA attention with sink. This is the main nonstandard attention requirement and avoids materializing `[B,14,T,257]` local score bands as dense `[B,14,T,T]`.
- MoE grouped GEMM plus scatter-add. The model has 128 experts and only 50M active parameters per token; sparse expert execution is the core performance story.
- RMSNorm. Used twice per block plus final norm; straightforward and latency-sensitive.

Medium priority:

- Packed QKV GEMM plus split views. Saves launches and memory traffic before attention.
- RoPE plus Q/K scaling fusion. Interleaved RoPE and split scaling are small but repeated in every block.
- Expert activation fusion: clamp, sigmoid, multiply, `(up+1)` in one kernel between expert GEMMs.
- Token-classification head GEMM with optional logits-to-label argmax for simple inference.

Lower priority:

- CPU/controller Viterbi decoder. Important for product parity, but outside the neural graph and cheap relative to model compute for long batches.
- ONNX q4 artifact parity. Useful for deployment comparison, but not part of the PyTorch source contract.

## 11. Runtime staging plan

Stage 1: config and weight admission.

- Parse `OpenAIPrivacyFilterConfig`.
- Reject generation/KV-cache mode for this family.
- Load converted HF dense BF16/F32 weights and preserve attention sink F32.
- Verify labels and tokenizer metadata.

Stage 2: single-block eager-equivalent parity.

- Implement RMSNorm, Q/K/V/O projections, interleaved RoPE, local attention with sink using a dense/eager reference path for short sequences.
- Implement router and one expert slice in a correctness-first path.

Stage 3: full encoder token-logit parity.

- Run all 8 blocks plus classifier head.
- Support `input_ids`, `attention_mask`, and optional `position_ids`.
- Return `[B,T,33]` logits.

Stage 4: optimized providers.

- Add fused local GQA attention with sink.
- Add grouped MoE GEMM with fp32 accumulation.
- Add packed QKV rewrite.

Stage 5: end-to-end privacy-filter parity.

- Add tokenizer-driven token span mapping in the host pipeline.
- Add optional constrained BIOES Viterbi decoder and calibration operating points.
- Validate span outputs, not just logits.

Stage 6: deployment/quantization follow-up.

- Compare BF16 PyTorch-source graph to official ONNX fp16/q4 artifacts.
- Decide whether q4 ONNX is a separate runtime import path or out of scope for DinoML.

## 12. Parity and validation plan

Recommended tests:

- Config parse test for official `config.json`, including `hidden_size != q_width`.
- Weight mapping smoke test for converted HF tensor names and F32 sink dtype.
- RMSNorm random tensor parity in fp32/bf16 with fp32 variance.
- Interleaved RoPE parity against Transformers for fixed `position_ids`, including nonzero offsets.
- Local mask parity for small `T`, with and without padding mask.
- Attention parity for one block on `T <= 16` where dense eager masking is feasible, including sink logits.
- Router parity: fp32 router logits, top-k indices/scores, score division by 4.
- Expert activation parity for clamp edge cases around `-7`, `7`, and large positive gate.
- MoE parity with random and crafted routing, comparing final `index_add_` accumulation.
- Full block parity after 1, 2, and 8 layers.
- Full model logits parity for tokenizer examples from the model card.
- Optional end-to-end span parity against Transformers pipeline/simple aggregation and against a Viterbi decoder once implemented.

Tolerance guidance:

- fp32 operator tests: `rtol=1e-5`, `atol=1e-5` where deterministic.
- bf16 full-model logits: begin with `rtol=3e-2`, `atol=3e-2`, then tighten per provider. MoE accumulation order may require wider tolerances unless DinoML exactly matches source eager order.
- Attention softmax/sink tests should compare fp32 probabilities before value matmul for diagnosis.

## 13. Performance probes

- Tokenization throughput: chars/sec and tokens/sec for long documents.
- Encoder throughput: `[B,T]` sweep with `T` in 128, 512, 4096, 32768, 128000 where memory allows.
- Local attention provider comparison: eager dense mask, banded local, FlashAttention sink-enabled backend.
- Sliding-window radius sweep around 128 to validate provider scaling.
- MoE routing distribution probe: active experts per batch, bucket sizes, empty expert count.
- Grouped GEMM versus eager expert loop, including small-batch and long-sequence cases.
- Router/topk overhead at `[B*T,128]`.
- End-to-end logits/sec and spans/sec with Viterbi enabled/disabled.
- Memory usage for max-context RoPE tables, masks, local attention temporaries, and MoE buckets.
- BF16 versus fp32 accumulation sensitivity for MoE and attention.

## 14. Skip/defer list

- Training, loss, gradient checkpointing, and router auxiliary loss.
- Autoregressive generation and KV cache.
- Beam search, sampling, speculative decode, and continuous batching decode schedulers.
- Multi-GPU tensor parallel or expert parallel execution, despite config EP plan metadata.
- Original checkpoint ingestion, unless DinoML chooses to own OpenAI source-format conversion.
- Official ONNX q4/q4f16 runtime import, until dense BF16 parity is complete.
- Nonzero classifier dropout or attention dropout.
- Dynamic runtime label-taxonomy changes. The label set is fixed by config/fine-tuning.
- Full model-card constrained Viterbi span decoding can be deferred after logits parity, but should be included before claiming end-to-end redaction parity.

## 15. Final implementation checklist

- [ ] Parse `OpenAIPrivacyFilterConfig` and admit only token-classification runtime.
- [ ] Load converted HF weights with Q/K/V split widths `[896,128,128]`.
- [ ] Preserve F32 attention sink parameters.
- [ ] Implement RMSNorm with fp32 variance.
- [ ] Implement interleaved RoPE with YaRN parameters.
- [ ] Implement bidirectional sliding-window mask with padding mask.
- [ ] Implement GQA attention with per-head sink logits and no causal cache.
- [ ] Implement fp32 router `Linear(640 -> 128)` plus top-k 4.
- [ ] Implement custom expert gate activation.
- [ ] Implement MoE grouped expert GEMMs with fp32 accumulation and weighted scatter-add.
- [ ] Implement token-classification head `Linear(640 -> 33)`.
- [ ] Add single-block and full-model logits parity tests.
- [ ] Add tokenizer/input ABI test for `input_ids`, `attention_mask`, `position_ids`.
- [ ] Add optional constrained BIOES Viterbi decoder or document host-pipeline ownership.
- [ ] Benchmark local attention, MoE grouped GEMM, and end-to-end tokens/sec.
