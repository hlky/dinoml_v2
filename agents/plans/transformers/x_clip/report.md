# X-CLIP Transformers Audit

## 1. Source basis

Transformers commit/version: `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

Model id: primary `microsoft/xclip-base-patch32`; sweep also inspected `microsoft/xclip-base-patch16`, `microsoft/xclip-base-patch32-16-frames`, `microsoft/xclip-base-patch16-zero-shot`, and `microsoft/xclip-large-patch14-16-frames`.

Config source: official Hugging Face raw `config.json`, `preprocessor_config.json`, tokenizer metadata snapshots under `_sources/`.

Source files inspected:
- `X:/H/transformers/src/transformers/models/x_clip/modeling_x_clip.py`
- `X:/H/transformers/src/transformers/models/x_clip/modular_x_clip.py`
- `X:/H/transformers/src/transformers/models/x_clip/configuration_x_clip.py`
- `X:/H/transformers/src/transformers/models/x_clip/processing_x_clip.py`
- `X:/H/transformers/src/transformers/models/x_clip/convert_x_clip_original_pytorch_to_hf.py`
- `X:/H/transformers/src/transformers/models/videomae/image_processing_videomae.py`
- auto mappings for model, processor, tokenizer, and image processor.

`modeling_x_clip.py` is generated from `modular_x_clip.py`; runtime behavior was read from generated modeling code, while future upstream edits should target modular source. No official config was gated, 401, or 404 during this audit.

## 2. High-level architecture

X-CLIP is a dual-encoder text-video contrastive model, not an autoregressive generator.

Dataflow:

```text
video frames + text -> processor -> frame ViT + causal CLIP text encoder
frame CLS sequence -> Multiframe Integration Transformer -> video embedding
frame patch tokens -> visual prompt features -> prompt cross-attention over text labels
normalized text/video embeddings -> scaled cosine similarity logits
```

Stage decomposition:
- CPU/data pipeline: caller samples frames; `XCLIPProcessor` maps `videos` to the image/video processor and CLIP tokenizer.
- Frame encoder: each video is packed as `[B, T, C, H, W]`, then reshaped to `[B*T, C, H, W]`.
- Temporal encoder: projected frame CLS embeddings `[B, T, D]` pass through one MIT encoder and mean pool over frames.
- Prompt conditioning: patch tokens from all frames are projected, averaged over time, then used as visual keys/values for cross-attention into expanded text embeddings.
- Contrastive head: video embeddings and per-video prompted text embeddings are L2-normalized and compared.

Frame vision embeddings, MIT video embeddings, text embeddings, and prompted text embeddings are independently useful cache points. Text branch embeddings before visual prompting are cacheable across videos for the same label set; final prompted text embeddings are video-dependent.

## 3. Important config dimensions

Source defaults:

| Field | Default |
| --- | --- |
| text hidden/layers/heads/mlp | 512 / 12 / 8 / 2048 |
| text vocab/max positions | 49408 / 77 |
| text activation | `quick_gelu` |
| vision hidden/layers/heads/mlp | 768 / 12 / 12 / 3072 |
| image/patch/frames | 224 / 32 / 8 |
| MIT hidden/layers/heads/mlp | 512 / 1 / 8 / 2048 |
| projection dim | 512 |
| prompt layers/heads/alpha | 2 / 8 / 0.1 |
| logit scale init | 2.6592 |

Checkpoint sweep:

| Model | Text | Vision | Image/patch/frames | MIT | Projection |
| --- | --- | --- | --- | --- | --- |
| `microsoft/xclip-base-patch32` | 512, 12L, 8H | 768, 12L, 12H | 224, p32, 8f | 512, 1L, 8H | 512 |
| `microsoft/xclip-base-patch32-16-frames` | same | same | 224, p32, 16f | 512, 1L, 8H | 512 |
| `microsoft/xclip-base-patch16` | same | same | 224, p16, 8f | 512, 1L, 8H | 512 |
| `microsoft/xclip-base-patch16-zero-shot` | same | same | 224, p16, 32f | 512, 1L, 8H | 512 |
| `microsoft/xclip-large-patch14-16-frames` | 768, 12L, 12H | 1024, 24L, 16H | 336, p14, 16f | 768, 1L, 8H | 768 |

Preprocessor configs use `VideoMAEFeatureExtractor`/`XCLIPProcessor`, resize and center crop, ImageNet mean/std, and size 224 for base variants or 336 for large.

## 3a. Family variation traps

- `pixel_values` enters the model as `[B,T,C,H,W]`; frame ViT consumes NCHW after reshape.
- `num_frames` is baked into cross-frame message-token reshapes and MIT positional embedding length. Runtime `T` must equal config `num_frames` unless DinoML adds explicit guarding/fallback.
- Base patch32 old config has architecture string `XClipModel`, while current class is `XCLIPModel`; do not treat casing as operator behavior.
- Text pooling is CLIP legacy EOT pooling: source forces `self.eos_token_id = 2`, then selects `argmax(input_ids)` instead of reading current config `eos_token_id`.
- Full forward text embeddings are video-conditioned by prompt generator; standalone `get_text_features` returns only unprompted text projection.
- `get_video_features` returns MIT-pooled video output but not normalized contrastive embeddings and does not run prompt generation.
- Vision encoder layers add a per-frame message token into spatial self-attention. Sequence length is `1 + patches + 1` inside attention, then the extra message token is sliced away.
- Prompt cross-attention uses projection dimension, not vision hidden size, and uses no mask.
- `interpolate_pos_encoding=True` enables bicubic resize of learned 2D patch position embeddings; normal config-sized inference should keep static position embedding.
- Layout pass trap: Conv2d patch embedding and VideoMAE processor are source NCHW. NHWC/channel-last fusion is an optimization candidate only if frame packing, `flatten(2).transpose(1,2)`, patch-position order, and downstream token consumers remain equivalent.

## 4. Operator coverage checklist

Tensor/layout ops:
- reshape `[B,T,C,H,W] -> [B*T,C,H,W]`; view `[B*T,D] -> [B,T,D]`
- flatten spatial patches, transpose, concatenate CLS/message tokens, slice, unsqueeze, expand, mean over frames, L2 norm
- argmax/equality-based text pooling and advanced gather over `[batch, seq]`

Neural primitives:
- token and learned position embeddings
- Conv2d patch embedding with `kernel=stride=patch`, bias false
- Linear projections with and without bias, LayerNorm, residual add, `quick_gelu`
- dropout/drop-path are inference no-ops

Attention primitives:
- CLIP-style dense MHA for text, vision, MIT, and prompt cross-attention
- text uses causal mask; vision/MIT/prompt are noncausal
- eager math is matmul, additive mask, softmax in fp32, matmul V

Preprocessing-coupled ops:
- frame resize, center crop, rescale, normalize, stack to `[B,T,C,H,W]`
- CLIP BPE tokenizer and 77-token context

Contrastive/postprocessing:
- L2 normalize along projection dim
- `exp(logit_scale)` scalar multiply
- `einsum("bd,bkd->bk")` yielding `logits_per_video [B_video, B_text]`
- transpose to `logits_per_text [B_text, B_video]`

## 5. Layer/block breakdown

Text encoder, repeated 12 layers:

```text
input_ids [N,L] -> token_emb + position_emb
causal mask from attention_mask
x = LN(x)
x = residual + MHA(q,k,v all D -> D)
x = LN(x)
x = residual + Linear(D -> 4D) -> quick_gelu -> Linear(4D -> D)
final LN
pooled = hidden[batch, argmax(input_ids)]
text_proj: Linear(text_hidden -> projection_dim, bias=False)
```

Frame vision encoder:

```text
pixel_values [B*T,3,H,W]
Conv2d(3 -> V, kernel=stride=patch, bias=False)
flatten patches -> [B*T,P,V], prepend CLS, add learned positions
for each of vision_layers:
  msg = Linear(CLS) -> view [B,T,V]
  msg = msg + self_attention(LN(msg)) over frames
  append msg token to each frame's spatial sequence
  x = residual + spatial self_attention(LN(x))
  drop appended msg token
  x = residual + MLP(LN(x))
pooler = LN(CLS)
visual_projection: Linear(V -> projection_dim, bias=False)
```

MIT:

```text
cls_features [B,T,projection_dim]
add learned temporal position [1,T,projection_dim]
1-layer encoder with noncausal MHA + MLP
residual add to input
video_embeds = mean over T
```

Prompt generator:

```text
patch tokens [B*T,P,V] -> LN(V) -> matmul prompts_visual_projection [V,projection_dim]
view [B,T,P,D] -> mean over T -> [B,P,D]
text_embeds [N_text,D] -> expand [B,N_text,D]
for prompt_layers:
  text = text + CrossAttention(LN(text), visual, visual)
  text = text + MLP(LN(text))
text = text + alpha * prompt_delta
```

## 6. Attention requirements

Text attention is causal self-attention over up to 77 tokens, MHA with `num_attention_heads`, `head_dim = hidden_size / heads`, additive causal/padding mask, no KV cache. It is an encoder-style CLIP text branch, not generation prefill/decode.

Vision spatial attention is noncausal self-attention over `1 + patches + 1 message` tokens per frame. The message token itself is built by noncausal temporal self-attention over `[B,T,V]` CLS messages before each spatial block.

MIT attention is noncausal self-attention over `T` projected frame CLS tokens.

Prompt attention is noncausal cross-attention: queries are per-video expanded text label embeddings `[B,N_text,D]`; keys/values are visual patch summaries `[B,P,D]`. Q/K/V are bias-free; output projection has bias. No prompt attention mask is supplied.

No sliding-window, sparse, block, RoPE, ALiBi, relative bias, varlen packing, or autoregressive KV cache is present.

## 7. Position encoding and custom math

Text uses learned absolute positions `[77,D_text]`. Vision uses learned absolute CLS + 2D patch positions; optional interpolation reshapes patch positions to square grid, bicubic interpolates, then flattens back.

MIT uses a learned temporal parameter `[1,num_frames,D_proj]`.

Custom parity snippets:

```python
def xclip_contrast(video_embeds, text_embeds, logit_scale):
    video = video_embeds / video_embeds.norm(p=2, dim=-1, keepdim=True)
    text = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
    logits_per_video = torch.einsum("bd,bkd->bk", video, logit_scale.exp() * text)
    return logits_per_video, logits_per_video.T
```

```python
def legacy_text_pool(last_hidden_state, input_ids):
    idx = input_ids.to(dtype=torch.int).argmax(dim=-1)
    return last_hidden_state[torch.arange(last_hidden_state.shape[0]), idx]
```

`quick_gelu` comes through Transformers `ACT2FN`; DinoML should validate the exact approximation used by shared activation support.

## 8. Preprocessing and input packing

Processor behavior:
- `XCLIPProcessor(videos=...)` maps `videos` to `images` and delegates to image/video processor plus tokenizer.
- Conversion script used `VideoMAEImageProcessor(size=224|336)` and CLIP tokenizer from `openai/clip-vit-base-patch32`.
- Official preprocessor configs resize, center crop, normalize with ImageNet mean/std, and emit frame tensors.

Model input ABI:
- `pixel_values`: `[B_video, T, C, H, W]`, source expects `C=3`.
- `input_ids`: `[B_text, L]`; `attention_mask` optional but relevant for causal mask.
- `position_ids`: optional text positions.

Frame sampling is not owned by the model. Examples show PyAV decode and sampled frame indices in user code, so first DinoML integration can require callers to provide already sampled frames or preprocessed tensors.

Text-video output ABI:
- `text_embeds`: full forward returns `[B_video, B_text, D]` after prompt conditioning and normalization.
- `video_embeds`: `[B_video, D]` after MIT and normalization.
- `logits_per_video`: `[B_video, B_text]`.
- `logits_per_text`: transpose `[B_text, B_video]`.

## 9. Graph rewrite / lowering opportunities

### Rewrite: non-overlap Conv2d patch embedding -> Linear/GEMM

Source pattern: `Conv2d(3,V,kernel=patch,stride=patch,padding=0,bias=False)` followed by `flatten(2).transpose(1,2)`.

Replacement: extract non-overlapping patches in row-major H/W order, flatten to `[B*T, P, 3*patch*patch]`, GEMM with weight reshaped to `[V, 3*patch*patch]`, output `[B*T,P,V]`.

Preconditions: fixed NCHW semantics, input H/W divisible by patch, no interpolation mismatch, groups=1, dilation=1, padding=0, bias absent. Failure cases: dynamic non-divisible sizes, channel-last graph without verified weight/layout transform.

### Rewrite: contrastive einsum -> batched GEMM

Source pattern: `einsum("bd,bkd->bk", video, scale * text)`.

Replacement: per-video dot of `[1,D]` against `[B_text,D]`, or batched GEMM over B videos. If text prompts are disabled or precomputed for one video batch, this can reduce to regular matrix multiply.

Preconditions: both branches L2-normalized exactly once; preserve output orientation `[B_video,B_text]`.

### Rewrite: prompt visual projection matmul -> Linear

Source pattern: `img_features @ prompts_visual_projection`, where parameter is `[vision_hidden, projection_dim]`.

Replacement: bias-free Linear over patch tokens.

Preconditions: no implicit transpose; preserve frame then patch packing before temporal mean.

### Layout guard: NTHWC/NHWC fusion opportunity

Source semantic layout is `[B,T,C,H,W]` then NCHW Conv2d. A guarded optimized path could keep decoded frames as `[B,T,H,W,C]` and fuse normalize + patch extraction + projection into a channel-last kernel.

Required guards: processor/data pipeline supplies contiguous channel-last frames; patch flatten order matches NCHW Conv2d order; position IDs still align with row-major patch sequence; all consumers after patch projection are token-major and layout independent.

## 10. Kernel fusion candidates

Highest priority:
- Conv patch embedding as im2col/GEMM or direct patch-projection kernel, especially p16/p14 large variants.
- Dense MHA + LayerNorm/MLP encoder blocks for text, frame vision, MIT, and prompt cross-attention.
- L2 normalization + scaled similarity GEMM with fixed output orientation.

Medium priority:
- Vision message-token temporal attention inside every vision block; small T but repeated per layer.
- Prompt generator cross-attention over label count x patch count; important for zero-shot classification with many labels.
- Processor-side normalize + channel/layout packing fused into patch projection.

Lower priority:
- Bicubic position interpolation, only for non-native resolutions.
- Training-only contrastive loss and dropout/drop-path.

## 11. Runtime staging plan

Stage 1: parse config, load base patch32 weights, and run text branch parity including legacy EOT pooling.

Stage 2: run frame vision encoder parity for one frame batch with static 224/p32 and no position interpolation.

Stage 3: add cross-frame message token and MIT parity for `[B,T]` packed videos.

Stage 4: implement full contrastive ABI: prompted text, normalized embeddings, `logits_per_video`, and transposed `logits_per_text`.

Stage 5: add processor/tensor ABI integration for preprocessed `[B,T,C,H,W]`; leave frame sampling outside DinoML initially.

Stage 6: enable optimized patch projection, attention, and similarity kernels with strict layout guards.

Stub initially: training loss, position interpolation, video decode/frame sampling, and pipeline-specific softmax over labels.

## 12. Parity and validation plan

- Config round-trip tests for base p32, p16, p32-16f, p16-zero-shot 32f, and large p14/336.
- Random tensor tests for Conv2d patch rewrite against PyTorch NCHW.
- Text branch parity: token embeddings through pooled projection; verify argmax pooling on CLIP-style token IDs.
- Single vision block parity including message token append/slice.
- MIT parity on random `[B,T,D]`.
- Prompt generator parity on random text/video patch features.
- End-to-end checkpoint parity for `microsoft/xclip-base-patch32`: compare `logits_per_video`, `logits_per_text`, normalized embeddings. Suggested tolerances: fp32 `1e-4` absolute, fp16 `1e-2` for full graph, tighter for isolated GEMMs.
- ABI test: multiple text labels with one video must produce `[1,N_text]`; multiple videos must produce `[B_video,N_text]`.

## 13. Performance probes

- Processor throughput: decode/sample excluded vs resize/crop/normalize/stack included.
- Frame encoder throughput by `B*T`, patch size, and image size.
- Vision message-attention overhead by frame count T and vision layer count.
- MIT throughput by T and projection dim.
- Prompt generator throughput by text label count and patch count.
- Similarity head throughput by `[B_video,B_text,D]`.
- End-to-end zero-shot video classification batch sweep over videos and candidate labels.
- Layout probe: NCHW processor output vs guarded NTHWC/channel-last fused patch path.

## 14. Skip/defer list

- Training loss and contrastive loss gradients.
- Dropout/drop-path stochastic behavior.
- Frame decode and sampling policy beyond documenting caller contract.
- `interpolate_pos_encoding=True` for non-native resolution.
- Image-only `get_image_features`; modular source explicitly rejects it.
- Autoregressive generation, KV cache, beam search, speculative decoding: not applicable.
- Sparse/local attention and quantized/packed weights: not present in official source.

## 15. Final implementation checklist

- [ ] Parse `XCLIPConfig`, nested text/vision configs, prompt config, and MIT dimensions.
- [ ] Load CLIP tokenizer metadata and VideoMAE/XCLIP processor metadata.
- [ ] Implement text branch with causal attention and legacy argmax EOT pooling.
- [ ] Implement NCHW patch embedding and learned position embeddings.
- [ ] Implement vision cross-frame message token attention inside each vision layer.
- [ ] Implement MIT temporal encoder and mean pooling.
- [ ] Implement visual prompt projection and prompt cross-attention.
- [ ] Implement normalized text/video embeddings and similarity logits with exact orientation.
- [ ] Add static guards for `num_frames`, image size, patch divisibility, and text max length.
- [ ] Add parity tests for branch features, full logits, and output shapes.
- [ ] Add guarded Conv2d patch -> GEMM rewrite.
- [ ] Benchmark processor, frame encoder, MIT, prompt generator, and similarity head separately.
