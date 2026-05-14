# InstructBlipVideo audit

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: no public Hub checkpoint with model_type=instructblipvideo found by name search; representative Salesforce InstructBLIP configs inspected because the video docs/examples reuse those repos.
Config source: local source defaults plus Hub config/processor snapshots copied into this folder.
Source files inspected:
- X:/H/transformers/src/transformers/models/instructblipvideo/configuration_instructblipvideo.py
- X:/H/transformers/src/transformers/models/instructblipvideo/modeling_instructblipvideo.py
- X:/H/transformers/src/transformers/models/instructblipvideo/modular_instructblipvideo.py
- X:/H/transformers/src/transformers/models/instructblipvideo/processing_instructblipvideo.py
- X:/H/transformers/src/transformers/models/instructblipvideo/video_processing_instructblipvideo.py
Any missing files or assumptions: modeling/config files are generated from modular_instructblipvideo.py; future source edits should inspect the modular file first. The public Salesforce checkpoints are mostly model_type=instructblip and use image processors, not native instructblipvideo configs; treat them as representative dimensional anchors, not proof of native video checkpoint coverage.
```

Hub snapshots in this folder:

- `Salesforce__instructblip-vicuna-7b__config.json`, `preprocessor_config.json`, `processor_config.json`, `tokenizer_config.json`, `generation_config.json`
- `Salesforce__instructblip-vicuna-13b__config.json`, same side files
- `Salesforce__instructblip-flan-t5-xl__config.json`, same side files
- `Salesforce__instructblip-flan-t5-xxl__config.json`, same side files

No gated checkpoint was required for these snapshots. Representative Hub URLs: [Salesforce/instructblip-vicuna-7b](https://huggingface.co/Salesforce/instructblip-vicuna-7b), [Salesforce/instructblip-vicuna-13b](https://huggingface.co/Salesforce/instructblip-vicuna-13b), [Salesforce/instructblip-flan-t5-xl](https://huggingface.co/Salesforce/instructblip-flan-t5-xl), [Salesforce/instructblip-flan-t5-xxl](https://huggingface.co/Salesforce/instructblip-flan-t5-xxl). Source docs were cross-checked against [HF InstructBlipVideo docs](https://huggingface.co/docs/transformers/model_doc/instructblipvideo), but operator claims below come from local source.

## 2. High-level architecture

InstructBlipVideo is a composite multimodal conditional generator:

```text
video decode/frame sampling in data pipeline
-> video processor emits pixel_values [B,T,C,H,W]
-> per-frame ViT-style vision encoder over [B*T,C,H,W]
-> Q-Former with learned query tokens and cross-attention into frame tokens
-> Linear projector from Q-Former hidden to language hidden
-> video-feature embeddings replace <video> placeholders in text embeddings
-> delegated language model prefill/decode/generation
```

Stage decomposition:

- CPU/data pipeline: video decode, frame sampling, resize/RGB/rescale/normalize, tokenization for both language tokenizer and Q-Former tokenizer.
- Independently cacheable encoder/projector: vision encoder + Q-Former + `language_projection` produce `[B, T * num_query_tokens, text_hidden]` video prefix features. This can be cached per video/prompt pair before language prefill.
- Prefix construction: token embeddings are created by the delegated language model embedding table; video features are stitched into positions whose ids equal `config.video_token_id`.
- Prefill/decode: fully delegated to `AutoModelForCausalLM` for LLaMA/Vicuna-style configs or `AutoModelForSeq2SeqLM` for T5-style configs. DinoML should compose separate audits for those language model families.

First useful DinoML target: `InstructBlipVideoForConditionalGeneration.generate` with fixed four-frame processor convention, video feature extraction parity, placeholder embedding stitch, then delegated language-model generation.

## 3. Important config dimensions

Source defaults from `configuration_instructblipvideo.py`:

| Component | Field | Default / effective value |
|---|---:|---:|
| top | `model_type` | `instructblipvideo` |
| top | `num_query_tokens` | 32 |
| top | `video_token_index` | `None` by source default; real checkpoints must set it |
| vision | `hidden_size` | 1408 |
| vision | `intermediate_size` | 6144 |
| vision | `num_hidden_layers` | 39 |
| vision | `num_attention_heads` | 16 |
| vision | `head_dim` | 88 |
| vision | `image_size` / `patch_size` | 224 / 14 |
| vision | patches | 16 x 16 + CLS = 257 tokens at default source size |
| vision | activation / norm eps | GELU / 1e-6 |
| vision | `qkv_bias` | true, with Q and V bias only; K bias is an inserted zero segment |
| Q-Former | `hidden_size` | 768 |
| Q-Former | `intermediate_size` | 3072 |
| Q-Former | `num_hidden_layers` | 12 |
| Q-Former | `num_attention_heads` / `head_dim` | 12 / 64 |
| Q-Former | `cross_attention_frequency` | 2, layers 0,2,4,6,8,10 cross-attend |
| Q-Former | `max_position_embeddings` | 512 |
| Q-Former | `vocab_size` | 30522 default; Salesforce snapshots use 30523 |
| language | type | delegated by `text_config.model_type`; CausalLM mapping decides decoder-only vs seq2seq |

Representative checkpoint sweep from copied Hub `config.json` files:

| Repo snapshot | Native config type | LM family | LM hidden | LM layers | LM heads | LM FFN | Vocab | Decoder-only | Query tokens | Placeholder ids |
|---|---|---|---:|---:|---:|---:|---:|---|---:|---|
| Salesforce/instructblip-vicuna-7b | `instructblip` | LLaMA/Vicuna | 4096 | 32 | 32 KV=32 | 11008 | 32064 | true | 32 | image 32001, video 32002 |
| Salesforce/instructblip-vicuna-13b | `instructblip` | LLaMA/Vicuna | 5120 | 40 | 40 KV=40 | 13824 | 32064 | true | 32 | image 32001, video 32002 |
| Salesforce/instructblip-flan-t5-xl | `instructblip` | T5 | 2048 | 24 encoder/decoder | 32 | 5120 | 32128 | false | 32 | image 32100, video 32101 |
| Salesforce/instructblip-flan-t5-xxl | `instructblip` | T5 | 4096 | 24 encoder/decoder | 64 | 10240 | 32128 | false | 32 | image 32100, video 32101 |

The Salesforce configs omit detailed `vision_config` and most Q-Former dimensions, relying on source defaults from the InstructBLIP class. For this video audit, treat the source defaults above as the effective vision/Q-Former dimensions unless a native `instructblipvideo` checkpoint supplies explicit overrides.

## 3a. Family variation traps

- Native source supports both decoder-only and encoder-decoder language models. DinoML should route `text_config.model_type` to an audited LLaMA/OPT-like CausalLM or T5-like Seq2SeqLM rather than hard-code one decode ABI.
- The language model is delegated by `AutoModelForCausalLM` / `AutoModelForSeq2SeqLM`; InstructBlipVideo owns the vision/Q-Former/projector/stitch ABI, not the internal LM operators.
- Source comments and processor logic assume four frames: processor prepends `<video>` repeated `num_query_tokens * 4`, and `generate` creates the same fixed count when no `input_ids` are supplied. The model forward itself reshapes by runtime `frames`; if `frames != 4`, placeholder count must still equal `frames * num_query_tokens`.
- Video processor source default is 384x384, while source config default is 224x224 and public Salesforce preprocessor snapshots are 224x224 `BlipImageProcessor` configs. This is an admission trap: processor/config pairing must be explicit.
- The checked Salesforce configs use `model_type=instructblip`, not `instructblipvideo`; they include `video_token_index` but are not native video-family configs.
- Vision QKV storage uses one fused `Linear(hidden -> 3*hidden)` with a special bias layout `[q_bias, zero_k_bias, v_bias]`.
- Q-Former has two feed-forward paths: query tokens use `intermediate_query/output_query`, text tokens use `intermediate/output`. Cross-attention applies only to the query prefix on every `cross_attention_frequency` layer.
- Q-Former attention is eager-only in source flags; no SDPA/Flash/Flex support for Q-Former because it adds masks directly and has output-recording behavior.
- Vision attention advertises attention backend support, but eager parity does not upcast attention weights to fp32.
- Placeholder stitch uses broad `masked_scatter`. Processor-generated inputs make this a bounded prefix row-copy opportunity, but raw callers can pass arbitrary placeholder positions.
- Layout-sensitive region: video input is source semantic `[B,T,C,H,W]`, immediately reshaped to `[B*T,C,H,W]` for Conv2d. NTHWC/NHWC is only a guarded local optimization around processor + patch embedding, not the semantic graph.

## 4. Operator coverage checklist

Tensor/layout ops:

- `reshape [B,T,C,H,W] -> [B*T,C,H,W]`
- Conv patch output flatten/transpose: `[N,1408,H/14,W/14] -> [N,patches,1408]`
- CLS expand, concat along sequence axis, position embedding slice/add
- Q-Former `repeat_interleave` over batch for text ids/masks by `frames`
- `cat` query mask + Q-Former text mask along dim 1
- slice first query tokens: `hidden[:, :num_query_tokens, :]`
- projector output reshape `[B*T,32,text_hidden] -> [B,32*T,text_hidden]`
- equality masks, unsqueeze, expand_as, dtype/device cast, `masked_scatter`

Neural primitives:

- Vision Conv2d patch embedding: `Conv2d(3 -> 1408, kernel=14, stride=14, bias=True)` by source default.
- Vision LayerNorm(1408), Linear(1408 -> 3*1408) fused QKV, Linear(1408 -> 1408), Linear(1408 -> 6144), GELU, Linear(6144 -> 1408), residual adds.
- Q-Former embeddings: token embedding, position embedding, LayerNorm(768), dropout as inference identity.
- Q-Former Linear Q/K/V: self-attn `768 -> 768`, cross-attn K/V `1408 -> 768`, output `768 -> 768`.
- Q-Former FFN: Linear(768 -> 3072), GELU, Linear(3072 -> 768), residual + LayerNorm.
- Language projection: Linear(768 -> text_hidden), e.g. 4096/5120 for Vicuna, 2048/4096 for Flan-T5.
- Delegated LM operators: covered by separate LLaMA/T5 audits.

Attention primitives:

- Vision noncausal MHA, 16 heads, head dim 88, no attention mask.
- Q-Former noncausal self-attention over `[query_tokens + qformer_text_len]`, 12 heads, head dim 64, additive mask.
- Q-Former cross-attention on query prefix only, rectangular Q length 32 and K/V length vision tokens, every other layer by default.
- Delegated LM causal self-attention and optional encoder-decoder cross-attention are owned by the LM family.

Position/custom math:

- Vision learned absolute position table, optional bicubic interpolation for dynamic image size.
- Q-Former learned absolute position embeddings for text tokens only; query embeddings are concatenated before token embeddings after text positions are added.
- No RoPE/ALiBi in InstructBlipVideo-owned vision/Q-Former modules.

Preprocessing-coupled ops:

- RGB conversion, bicubic resize, rescale by 1/255, CLIP mean/std normalization.
- Video frame decode/sampling is not owned by the model graph; source video processor default has `do_sample_frames=False`.
- `<video>` placeholder expansion count and tokenizer id must match projected feature count.

Scatter/indexed update:

- Source `masked_scatter(mask, language_model_inputs)` is required for parity.
- Admission can lower processor-produced prefix placeholders to row copy if mask is exactly the first `T*num_query_tokens` text positions and count matches.

## 5. Layer/block breakdown

Vision embeddings:

```text
pixel_values [B*T,3,H,W]
patch = Conv2d(3 -> 1408, k=14, s=14)(pixel_values)
patch = flatten spatial then transpose to [B*T, Npatch, 1408]
cls = class_embedding.expand(B*T,1,1408)
x = cat([cls, patch], dim=1)
x = x + position_embedding[:, :seq, :] or interpolated position table
```

Vision encoder layer, repeated 39 times by source default:

```text
r = x
x = LayerNorm(1408, eps=1e-6)(x)
qkv = Linear(1408 -> 4224, bias=[q,0,v])(x)
q,k,v = reshape/split to [B*T,16,seq,88]
a = softmax((q @ k^T) * 88^-0.5)
x = Linear(1408 -> 1408)(a @ v) + r
r = x
x = LayerNorm(1408)(x)
x = Linear(6144 -> 1408)(GELU(Linear(1408 -> 6144)(x))) + r
```

Q-Former embeddings:

```text
text = word_embedding(qformer_input_ids) + position_embedding(position_ids)
x = cat([query_tokens.expand(B*T,32,768), text], dim=1)
x = LayerNorm(768, eps=1e-12)(x)
```

Q-Former layer, repeated 12 times:

```text
self_attn = dense attention over query+text tokens with additive padding mask
x = LayerNorm(Linear(self_attn) + x)
if layer_idx % cross_attention_frequency == 0:
    query_part = cross_attn(query_part, K/V=image_embeds [B*T,Vseq,1408])
query_part = LayerNorm(Linear(GELU(Linear(query_part))) + query_part) using query FFN weights
text_part = LayerNorm(Linear(GELU(Linear(text_part))) + text_part) using text FFN weights
x = cat([query_part, text_part], dim=1)
```

Projection and stitch:

```text
query_output = qformer_last_hidden[:, :32, :]
video_features = Linear(768 -> text_hidden)(query_output)
video_features = reshape [B, 32*T, text_hidden]
inputs_embeds = language_embedding(input_ids)
mask = input_ids == video_token_id, expanded over hidden
inputs_embeds = masked_scatter(inputs_embeds, mask, video_features)
language_model(inputs_embeds=inputs_embeds, attention_mask=...)
```

## 6. Attention requirements

Vision attention:

- Noncausal self-attention.
- MHA, not GQA/MQA: 16 Q/K/V heads, head dim 88, Q/K/V all width 1408.
- No attention mask in the source call.
- Eager math is `matmul -> scale -> optional mask add -> softmax -> dropout -> matmul`; eager path does not force fp32 softmax weights.
- Backend dispatch can use Transformers `ALL_ATTENTION_FUNCTIONS`, but a DinoML parity path should start with dense MHA.

Q-Former self-attention:

- Noncausal dense self-attention over query and text tokens.
- MHA: 12 heads, head dim 64, all-head width 768.
- Additive attention mask shape is `[B*T,1,1,S]` or broadcasted from 3D masks, with mask values `(1 - mask) * -10000.0` in model dtype.
- No KV cache is implemented for Q-Former in this source path.

Q-Former cross-attention:

- Query source is learned query table after self-attention, shape `[B*T,32,768]`.
- K/V source is vision encoder sequence `[B*T,Vseq,1408]`; K and V project `1408 -> 768`.
- Rectangular attention shape: Q length 32, K/V length `1 + (H/patch)*(W/patch)`.
- Cross-attention mask comes from all-ones image attention mask unless caller changes it.
- Applies only on layers where `layer_idx % cross_attention_frequency == 0`; default 6 of 12 layers.

Language model attention:

- Decoder-only LLaMA/Vicuna configs need causal prefill/decode KV cache.
- T5 configs need encoder-decoder generation with encoder outputs and decoder KV/cross-attention cache.
- These requirements should be imported from separate `llama` and `t5` reports, with InstructBlipVideo only responsible for building `inputs_embeds`.

## 7. Position encoding and custom math

Vision position interpolation is the only InstructBlipVideo-owned dynamic position math:

```python
def interpolate_video_pos(pos, embeddings, height, width, patch):
    cls = pos[:, :1]
    patch_pos = pos[:, 1:]
    base = int((patch_pos.shape[1]) ** 0.5)
    patch_pos = patch_pos.reshape(1, base, base, -1).permute(0, 3, 1, 2)
    patch_pos = bicubic_interpolate(
        patch_pos,
        size=(height // patch, width // patch),
        align_corners=False,
    )
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, pos.shape[-1])
    return concat([cls, patch_pos], dim=1)
```

Static 224x224 or 384x384 deployments can precompute/slice the learned absolute table when no interpolation is requested. Interpolation depends on runtime height and width and should be guarded by patch divisibility.

Q-Former text position ids are `arange(max_position_embeddings)` sliced to the text sequence length. Query embeddings do not receive learned position ids in this embedding module.

## 8. Preprocessing and input packing

Video processor source ABI:

- Input videos are handled by `BaseVideoProcessor`; model input name is `pixel_values`.
- Source default output layout for tensors is `[B,T,C,H,W]`.
- Source defaults: bicubic resize, CLIP mean/std, RGB conversion, rescale, normalize, `size={"height":384,"width":384}`, `do_sample_frames=False`.
- Public Salesforce snapshots use `BlipImageProcessor` with size 224x224 and no native video sampling fields; label those as legacy image-processor configs when using them.

Tokenizer/placeholder ABI:

- Processor adds `<video>` as a special token if the language tokenizer lacks `video_token`.
- For text+video calls, it prepends `<video>` repeated `num_query_tokens * 4` before the normal text tokens.
- Q-Former gets a separate tokenized prompt as `qformer_input_ids` and `qformer_attention_mask`.
- Model forward repeats Q-Former ids and masks by `frames`, so each frame cross-attends separately before projected features are reshaped back to one video prefix.
- Generate with no `input_ids` creates `[video_token_index] * (num_query_tokens * 4) + [bos_token_id]`.

Stitch contract:

- Required count is `frames * num_query_tokens` rows, each of width `text_hidden`.
- Processor-generated mask is a contiguous prefix before BOS/text, row-major by frame then query token because `reshape(B, T*Q, H)` follows source frame-major batching.
- Raw callers can pass placeholders anywhere; DinoML optimized lowering should reject non-prefix or count-mismatch patterns unless it admits a general boolean scatter.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embed -> Linear

Source pattern: `Conv2d(3 -> D, kernel=patch, stride=patch, padding=0)` followed by flatten spatial and transpose.

Replacement:

```text
NCHW PatchExtract row-major [N, Nh*Nw, 3*patch*patch]
-> MatMul(weight_flat.T)
-> BiasAdd
```

Preconditions:

- `kernel_size == stride == patch`
- `padding == 0`, `dilation == 1`, `groups == 1`
- `H % patch == 0`, `W % patch == 0`
- Preserve PyTorch NCHW flatten order unless a guarded NHWC layout pass owns the full region.

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * patch * patch)
```

Failure cases: dynamic non-divisible frame sizes, non-default conv attrs, or downstream layout consumers outside the controlled patch region. Parity test: compare embedding output before position add for random `[B*T,3,H,W]`.

### Rewrite: processor-prefix masked_scatter -> indexed row copy

Source pattern: `inputs_embeds.masked_scatter(expanded(input_ids == video_token_id), video_features)`.

Replacement:

```text
inputs_embeds[:, 0:T*Q, :] = video_features
```

Preconditions:

- `input_ids[:, 0:T*Q] == video_token_id`
- no `video_token_id` outside the prefix
- `video_features.shape == [B,T*Q,H]`
- `inputs_embeds.dtype == video_features.dtype` after source cast

Failure cases: caller-supplied arbitrary placeholders, missing `video_token_id`, `inputs_embeds` path where mask is inferred by embedding equality. Parity test: random embeddings plus generated prefix ids, compare full language inputs.

### Rewrite: Q-Former attention GEMM/softmax/BMM chain

Source pattern: separate Linear Q/K/V, reshape/permute, `matmul`, scale, additive mask, softmax, `matmul`, output Linear.

Replacement: dense MHA primitive for Q-Former self-attention and rectangular cross-attention.

Preconditions: no training dropout, no attention output capture required, additive mask exactly broadcastable to backend contract, fp behavior matches source dtype softmax.

Failure cases: attention map recording, gradient paths, or backend that silently upcasts differently from source.

### Rewrite: frame flatten/unflatten layout elimination

Source pattern: `[B,T,C,H,W] -> [B*T,C,H,W] -> per-frame encoder/Q-Former -> [B*T,Q,H] -> [B,T*Q,H]`.

Replacement: treat `B*T` as a flattened batch dimension in graph metadata and avoid material copies.

Preconditions: input is contiguous in source layout and no op observes separate `B` and `T` until final reshape. Failure cases: external layout pass changes temporal order or the processor emits NTHWC without an explicit axis rewrite.

## 10. Kernel fusion candidates

Highest priority:

- Vision patch embedding lowered to GEMM: large first-stage cost, regular non-overlap conv.
- LayerNorm + QKV projection for vision and Q-Former: repeated in 39 vision and 12 Q-Former layers.
- Dense attention prefill kernels for vision and Q-Former, including rectangular Q-Former cross-attention.
- FFN GELU GEMM pairs: `1408 -> 6144 -> 1408` and `768 -> 3072 -> 768`.
- Placeholder prefix row-copy instead of generic `masked_scatter`.

Medium priority:

- Vision QKV packed projection with special `[q, zero-k, v]` bias handling.
- Q-Former query/text FFN split handling in one layer schedule.
- Projector `Linear(768 -> text_hidden)` batched over `B*T*Q`.
- Last-token-only logits and KV-cache decode in delegated LM once composed.

Lower priority:

- Bicubic position interpolation; important for non-default resolutions but can be guarded/deferred.
- Full generic boolean scatter for arbitrary caller placeholders.
- Attention-map output materialization and training dropout.

## 11. Runtime staging plan

Stage 1: parse `InstructBlipVideoConfig`, load source-default vision/Q-Former dims, and reject missing `video_token_id` for generation.

Stage 2: implement vision embeddings + one vision block parity for fixed `[B,T,3,224,224]` or `[B,T,3,384,384]`.

Stage 3: implement full vision encoder and Q-Former without LM; validate `get_video_features` output `[B,T*32,text_hidden]`.

Stage 4: implement placeholder prefix copy and feed delegated LM `inputs_embeds` for prefill parity.

Stage 5: compose with audited LLaMA/Vicuna or T5 language backend for generation; support decoder-only first because Vicuna is the doc example.

Stage 6: add optimized patch GEMM, fused attention, LayerNorm/GEMM, and prefix-copy rewrites behind guards.

Stage 7: add dynamic frame count/resolution only after placeholder count and position interpolation guards are tested.

Initially stub/defer: training loss, attention outputs, arbitrary masked scatter, non-prefix placeholders, processor-owned video decode, beam-search controller details beyond delegated LM support.

## 12. Parity and validation plan

- Config parse tests for source defaults and four copied Salesforce snapshots, including explicit classification as native video vs legacy instructblip config.
- Processor contract test: text+4-frame input produces `pixel_values [B,4,C,H,W]`, `qformer_input_ids`, `qformer_attention_mask`, and `num_query_tokens*4` video placeholders.
- Vision embedding parity: random fixed-size frames, compare Conv2d patch + position add in fp32.
- Vision one-layer and full-encoder parity with fp32 tolerance around `1e-5` absolute/relative.
- Q-Former self-attention and cross-attention parity with small text length and fake vision tokens.
- `get_video_features` parity for `[B,4,3,H,W]`, checking output order by frame then query.
- Stitch parity: compare `masked_scatter` to guarded prefix copy for processor-generated ids; negative tests for wrong placeholder count and non-prefix masks.
- Delegated LM prefill logits parity for Vicuna-style and T5-style tiny/random configs if available.
- Decode token parity should use the delegated LM audit tolerances; InstructBlipVideo adds only the prefix embedding construction before generation.

Recommended tolerances: fp32 `1e-5` to `1e-4`; fp16/bf16 `1e-2` for full composite, with tighter per-op tolerances for GEMM/LayerNorm when using fp32 accumulation.

## 13. Performance probes

- Video processor throughput: decode/frame sampling separated from resize/normalize.
- Vision encoder throughput over flattened batch `B*T`, sweep frame count and resolution.
- Q-Former throughput, sweep Q-Former prompt length and vision token count.
- Projector + stitch microbenchmark, compare generic boolean scatter vs prefix row copy.
- Delegated LM prefill-only with video prefix length `T*32 + text_len`.
- Delegated LM decode tokens/sec with cached video-prefill context.
- Memory probes: vision activations, Q-Former activations, LM KV cache as prefix length changes.
- Attention backend comparison: eager dense vs fused attention for vision, eager Q-Former parity backend, delegated LM Flash/SDPA path.
- Batch-size sweep for video feature cache reuse, especially repeated text prompts over same video.

## 14. Skip/defer list

- Training, labels/loss, dropout, gradient checkpointing, attention gradient capture.
- General arbitrary `masked_scatter`; use guarded prefix copy first.
- Non-four-frame processor behavior unless caller supplies matching placeholders and tests cover it.
- Video decode and frame sampling inside DinoML runtime; keep in CPU/data pipeline initially.
- Bicubic position interpolation for dynamic resolutions if fixed processor size is admitted first.
- Multi-GPU/Accelerate device-map hooks.
- Language model internals not covered by separate LLaMA/T5 audits.
- Beam search and generation penalties unless the delegated generation controller already owns them.

## 15. Final implementation checklist

- [ ] Parse `InstructBlipVideoConfig` and subconfigs; require `video_token_id`/`video_token_index` for generation.
- [ ] Classify delegated `text_config` and route to audited CausalLM or Seq2SeqLM backend.
- [ ] Load vision, Q-Former, query token, projector, and delegated LM weights with correct aliases.
- [ ] Implement video ABI `[B,T,C,H,W] -> [B*T,C,H,W]` and guarded frame-count metadata.
- [ ] Implement Conv2d patch embedding or guarded Conv2d-to-Linear rewrite.
- [ ] Implement vision encoder dense MHA/LayerNorm/GELU FFN.
- [ ] Implement Q-Former embeddings, self-attention, cross-attention every `cross_attention_frequency`, and split query/text FFNs.
- [ ] Implement `get_video_features` parity and cacheable output `[B,T*num_query_tokens,text_hidden]`.
- [ ] Implement placeholder mask validation and prefix row-copy lowering for processor-generated inputs.
- [ ] Compose `inputs_embeds` prefill with delegated LM.
- [ ] Add parity tests for processor ABI, vision, Q-Former, projector, stitch, and end-to-end generation smoke.
- [ ] Benchmark processor, encoder/projector, LM prefill, LM decode, and stitch path separately.
