# Transformers audit: `mgp_str`

## 1. Source basis

```text
Transformers commit/version: b75feb2af64c3e29cbbc1bd859958c5432cc7ed4
Model id: alibaba-damo/mgp-str-base
Config source: Hugging Face Hub config.json and preprocessor_config.json; see config_snapshots.md
Source files inspected:
- transformers/src/transformers/models/mgp_str/configuration_mgp_str.py
- transformers/src/transformers/models/mgp_str/modeling_mgp_str.py
- transformers/src/transformers/models/mgp_str/processing_mgp_str.py
- transformers/src/transformers/models/mgp_str/tokenization_mgp_str.py
- transformers/src/transformers/models/vit/image_processing_vit.py
- transformers/tests/models/mgp_str/test_modeling_mgp_str.py
- transformers/tests/models/mgp_str/test_processing_mgp_str.py
Any missing files or assumptions: no family-local image processor exists; AutoImageProcessor maps mgp-str to ViT image processors. No gated official links were encountered. Only one official production checkpoint was found; tiny/random and ONNX mirrors were used only to expose dimension variation.
```

Primary DinoML target for this report: `MgpstrForSceneTextRecognition` image-to-text inference. This is not autoregressive generation: the model emits three fixed-length logit tensors and processor-side decoding chooses the best of character, GPT-2 BPE, and BERT WordPiece decodes.

## 2. High-level architecture

MGP-STR is a ViT-style image encoder plus three parallel A3 recognition heads:

```text
image resize/rescale -> pixel_values NCHW -> non-overlap Conv2d patch embed
-> prepend cls token + learned absolute pos embed
-> encoder-only dense self-attention blocks
-> three A3 token learner heads
-> char/BPE/WordPiece logits
-> CPU decode/top1 confidence fusion
```

Stage decomposition:

- CPU/data pipeline: image load, resize to `32x128`, optional rescale, no normalization for the official config, output `pixel_values` as `[B, 3, 32, 128]`.
- GPU/runtime encoder: patch projection, token concat, position add, 12 noncausal self-attention blocks for base.
- Independently stageable heads: three identical A3 modules with separate weights, each producing `[B, max_token_length, hidden_size]`.
- CPU postprocessing: top-1 per time step, softmax confidence, EOS truncation, tokenizer decode, choose highest cumulative confidence among the three heads.

## 3. Important config dimensions

| Field | Source default / base value | Tiny/random value | Runtime meaning |
|---|---:|---:|---|
| `image_size` | `[32, 128]` | `[32, 128]` | Required input H/W after preprocessing |
| `patch_size` | `4` | `4` | Conv kernel/stride, grid `8x32` |
| `num_channels` | `3` | `3` | NCHW input channels |
| `hidden_size` | `768` | `32` | Token width |
| `num_hidden_layers` | `12` | `5` | Encoder block count |
| `num_attention_heads` | `12` | `4` | Dense MHA heads |
| `head_dim` | `64` | `8` | Inferred as `hidden_size // heads` |
| `mlp_ratio` | `4` | `4` | FFN hidden width `hidden_size * mlp_ratio` |
| `max_token_length` | `27` | `27` | Per-head output positions |
| `num_character_labels` | `38` | `38` | Char classifier classes |
| `num_bpe_labels` | `50257` | `99` | GPT-2 BPE classifier classes |
| `num_wordpiece_labels` | `30522` | `99` | BERT WordPiece classifier classes |
| `qkv_bias` | `true` | `true` | Fused QKV Linear has bias |
| `layer_norm_eps` | `1e-5` | `1e-5` | LayerNorm epsilon |
| `torch_dtype` | `float32` | `float32` | Config metadata |
| cache support | none | none | Encoder-only; no KV cache |

Representative checkpoint sweep:

| Repo | Status | Config role | Operator-significant notes |
|---|---|---|---|
| `alibaba-damo/mgp-str-base` | official | production | `D=768`, `L=12`, `H=12`, label heads `38/50257/30522`, 148M params from Hub metadata |
| `hf-tiny-model-private/tiny-random-MgpstrForSceneTextRecognition` | public raw config despite private namespace | test/debug | `D=32`, `L=5`, `H=4`, small BPE/WP heads `99` |
| `onnx-community/mgp-str-base` | mirror | deployment mirror | Same neural dims as base; ONNX/preprocessor metadata adds explicit `do_rescale` |
| `onnx-community/tiny-random-MgpstrForSceneTextRecognition` | mirror | debug mirror | Matches tiny/random dimensions |
| `onnx-internal-testing/tiny-random-MgpstrForSceneTextRecognition-ONNX` | mirror | debug mirror | Matches tiny/random; `_attn_implementation_autoset` ignored by native source |

Official base `config.json` omits fields that the current config class supplies as defaults: `distilled=False`, `layer_norm_eps=1e-5`, and `initializer_range=0.02`. The official preprocessor omits `do_rescale`; with the current mapped ViT image processor this defaults to `true`, while the ONNX mirror records `do_rescale=true` explicitly.

## 3a. Family variation traps

- `distilled=True` changes positional length to `num_patches + 2`, but source only prepends `cls_token`; no distillation token is actually concatenated. Reject or separately audit `distilled=True`.
- `image_size` must match runtime H/W exactly; no dynamic image shape is accepted by the model forward.
- `hidden_size % num_attention_heads` is assumed. Source does not validate it before reshape.
- A3 `tokenLearner` uses grouped `1x1` Conv2d with `groups=8`; `hidden_size` must be divisible by 8.
- `mlp_ratio` is cast through `int(hidden_size * mlp_ratio)`, so non-integer products silently truncate.
- Base config `architectures` says `MGPSTRModel`, but inspected native class exports `MgpstrForSceneTextRecognition`; treat this as historical naming drift, not remote code.
- `config.output_a3_attentions` exists, but `MgpstrForSceneTextRecognition.forward` only returns A3 attentions when the call argument is truthy; `None` is not replaced from config in this inspected source.
- No RoPE, ALiBi, causal mask, local attention, packed sequence metadata, or KV cache.
- Source modeling is NCHW at image and A3 Conv2d boundaries. NHWC/channel-last is only a guarded optimization region around Conv2d and must rewrite flatten/transposes consistently.

## 4. Operator coverage checklist

Tensor/layout ops:

- NCHW rank-4 input validation, exact H/W guard.
- Conv2d patch embed: base `Conv2d(3 -> 768, kernel=4x4, stride=4x4, padding=0)`.
- Flatten spatial axes and transpose: `[B,D,8,32] -> [B,256,D]`.
- `expand` cls token, `cat` on token axis, add learned pos embed.
- Reshape/permute for QKV: `[B,N,3D] -> [3,B,H,N,head_dim]`.
- A3 transpose/unsqueeze/flatten: `[B,N,D] -> [B,D,N,1] -> [B,T,N]`.

Neural primitives:

- Linear QKV: base `Linear(768 -> 2304, bias=True)`, packed split order `[q,k,v]`.
- Linear output: `Linear(768 -> 768, bias=True)`.
- MLP: `Linear(768 -> 3072) -> GELU(exact torch default) -> Linear(3072 -> 768)`.
- LayerNorm over last dim with eps `1e-5`.
- A3 grouped Conv2d `1x1`: `768 -> 768, groups=8, bias=False`, then `768 -> 27, bias=False`.
- A3 feature grouped Conv2d `1x1`: `768 -> 768, groups=8, bias=False`.
- Heads: `Linear(768 -> 38)`, `Linear(768 -> 50257)`, `Linear(768 -> 30522)`.

Attention primitives:

- Dense noncausal self-attention over all `257` tokens for base.
- Softmax over key axis; dropout is identity in eval for base because rates are zero.
- No attention masks in model forward.

Pre/postprocessing-coupled ops:

- ViT image preprocessing to `pixel_values`; official config: resize to `32x128`, no normalize, current ViT processor source default implies rescale unless config overrides.
- Decode-side `topk(k=1)`, softmax over class axis, max probability, EOS search/truncation, cumulative product confidence, string decode.

## 5. Layer/block breakdown

Base patch and encoder shapes:

```text
pixel_values: [B,3,32,128]
patch = Conv2d(3,768,k=4,s=4)(pixel_values)  # [B,768,8,32]
tokens = transpose(flatten(patch, spatial), 1, 2)  # [B,256,768]
x = cat(cls[expand B,1,768], tokens, dim=1) + pos_embed[1,257,768]
```

Encoder block, repeated `num_hidden_layers`:

```text
y = LayerNorm(x)
qkv = Linear(D -> 3D, bias=qkv_bias)(y)
q,k,v = reshape(qkv, [B,N,3,H,D/H]).permute([2,0,3,1,4])
attn = softmax((q @ k.T) * (D/H)^-0.5, dim=-1)
y = Linear(D -> D)(reshape(attn @ v, [B,N,D]))
x = x + DropPath(y)  # identity in eval
y = LayerNorm(x)
y = Linear(D -> int(D*mlp_ratio)) -> GELU -> Linear(int(D*mlp_ratio) -> D)
x = x + DropPath(y)
```

A3 head, repeated independently for char/BPE/WP:

```text
u = LayerNorm(x)                         # [B,N,D]
u4 = transpose(u, 1, 2).unsqueeze(-1)     # [B,D,N,1]
selected = Conv1x1_group8(D->D) -> Conv1x1(D->T)
attn = softmax(flatten(selected, 2), dim=-1)  # [B,T,N]
feat = Conv1x1_group8(D->D)(u4).flatten(2).transpose(1,2)  # [B,N,D]
a3 = LayerNorm(einsum("...si,...id->...sd", attn, feat))   # [B,T,D]
logits = Linear(D -> labels)(a3)
```

## 6. Attention requirements

Required attention is encoder-only dense MHA:

- Noncausal self-attention, no cross-attention.
- MHA only; no MQA/GQA.
- Base heads/head dim: `12 x 64`; tiny: `4 x 8`.
- Query/key/value widths are equal.
- Query length equals key/value length: `257` for base image shape with one cls token.
- No source attention mask or padding mask.
- No packed/varlen sequence metadata.
- No sliding-window/local/sparse variants.
- No positional bias in attention scores.
- No KV cache or decode cache. A3 attentions are output diagnostics and weighted pooling, not generation cache.
- SDPA/FlashAttention compatibility is conceptually possible for the self-attention matmul-softmax-matmul, but source uses eager matmul and returns attention probs when requested.

## 7. Position encoding and custom math

Position encoding is a learned absolute table `pos_embed` with shape `[1, num_patches + num_tokens, hidden_size]`. For base this is `[1,257,768]`. It is added after cls-token concatenation and before the encoder.

Custom A3 pooling math:

```python
def a3_pool(hidden_states, token_learner, feat_proj, norm):
    u = layer_norm(hidden_states)
    u4 = u.transpose(1, 2).unsqueeze(-1)
    selected = token_learner(u4).flatten(2)
    weights = softmax(selected, dim=-1)
    feat = feat_proj(u4).flatten(2).transpose(1, 2)
    return norm(einsum("...si,...id->...sd", weights, feat)), weights
```

Precomputable: position table and cls token are weights. Dynamic: batch expansion, attention softmaxes, and A3 softmax weights.

## 8. Preprocessing and input packing

Official preprocessing:

- `MgpstrProcessor` delegates images to an image processor and text to the char tokenizer.
- Auto image processor maps `mgp-str` to ViT image processors.
- Official preprocessor config has `do_resize=true`, `size={height:32,width:128}`, `resample=3`, `do_normalize=false`.
- Current ViT image processor source defaults `do_rescale=true`; ONNX mirror records `do_rescale=true` and `rescale_factor=1/255`.
- Runtime tensor is `pixel_values` in channels-first `[B,3,32,128]`.

Tokenizer/decode ABI:

- Character tokenizer vocab is 38 tokens: `[GO]`, `[s]`, digits, lowercase letters.
- Processor creates GPT-2 and BERT-base-uncased tokenizers inside `__init__` for BPE/WP decode.
- Decode ignores position 0 by slicing predictions `[:,1:]`.
- EOS conventions: char token id `1` / string `[s]`; BPE token id `2` / string `#`; WordPiece token id `102` / string `[SEP]`.
- End-to-end parity requires CPU-side decode and confidence fusion; this is not part of the neural graph.

## 9. Graph rewrite / lowering opportunities

### Rewrite: patch Conv2d -> Linear

Preconditions:

- `kernel_size == stride == patch_size`.
- `padding == 0`, `dilation == 1`, `groups == 1`.
- Input is NCHW and H/W match config; H/W divisible by patch size.
- Flatten order preserves PyTorch Conv2d spatial order: height-major then width.

Replacement:

```text
NCHW image -> im2col non-overlap patches [B,256,3*4*4]
-> MatMul(weight_flat.T) + bias -> [B,256,D]
```

Weight transform:

```python
w = conv.weight.reshape(out_channels, in_channels * kh * kw)
```

Failure cases: dynamic or non-divisible image shapes, nonzero padding/dilation/groups, NHWC inputs without a complete axis rewrite.

Parity sketch: compare patch embeddings before cls/pos add for random fp32/fp16 inputs.

### Rewrite: A3 grouped 1x1 Conv2d -> grouped Linear

Preconditions:

- Input has shape `[B,D,N,1]`.
- Kernel `1x1`, stride 1, padding 0.
- `D % groups == 0`.

Replacement: reshape to per-position matrix and apply groupwise GEMM. The second `tokenLearner` conv is ordinary `Linear(D -> T, bias=False)` over each token position.

Failure cases: unsupported grouped GEMM, hidden size not divisible by 8, layout-translated region without matching group channel semantics.

### Rewrite: A3 einsum -> BMM

Pattern:

```text
attn [B,T,N] @ feat [B,N,D] -> [B,T,D]
```

Replacement: `bmm_rrr(attn, feat)`.

Preconditions: contiguous row-major or explicit layout-aware BMM lowering; same batch and token count.

### Layout guard: NCHW Conv regions

Patch embedding and A3 Conv2d source semantics are channel-first. A guarded NHWC optimization may be worthwhile, but initial translation should preserve NCHW. Axis-sensitive rewrites include Conv channel axis, `flatten(2)`, `transpose(1,2)`, LayerNorm last axis, A3 softmax over token axis `-1`, and classifier Linear over last dim.

## 10. Kernel fusion candidates

Highest priority:

- LayerNorm + Linear QKV for `[B,257,768]`.
- Dense attention softmax chain, optionally through SDPA/FlashAttention for noncausal encoder attention when attention tensors are not requested.
- GEMM + bias for large classifier heads, especially BPE `768 -> 50257`.
- A3 einsum as BMM plus possible softmax/BMM fusion.

Medium priority:

- Patch Conv2d lowered to GEMM.
- MLP `Linear/GELU/Linear` with activation epilogue.
- A3 grouped `1x1` Conv2d kernels or grouped GEMM.

Lower priority:

- Dropout/DropPath elimination in eval.
- Decode-side top-1/softmax confidence fusion; usually CPU-side and small except BPE/WP vocab softmax.

## 11. Runtime staging plan

Stage 1: parse config, load weights, and run patch embedding plus one encoder block parity.

Stage 2: full `MgpstrModel` encoder parity for `pixel_values -> last_hidden_state`.

Stage 3: add one A3 head and classifier parity, then share the pattern across char/BPE/WP heads.

Stage 4: end-to-end logits parity for `MgpstrForSceneTextRecognition`.

Stage 5: CPU decode parity for official sample: expected generated text `ticket` from Transformers integration test.

Stage 6: optimized lowering: Conv-to-GEMM, A3 BMM, attention backend dispatch, classifier GEMM profiling.

Initially stub/defer: training, dropout randomness, hidden-state/attention optional outputs, and processor-side tokenizer downloads inside runtime.

## 12. Parity and validation plan

- Random tensor parity for patch Conv2d rewrite in fp32, then fp16/bf16 with relaxed tolerance.
- Single encoder block parity with fixed random weights and `drop_rate=0`, `attn_drop_rate=0`, `drop_path_rate=0`.
- Full encoder parity for base dimensions and tiny dimensions.
- A3 module parity: compare attention weights `[B,27,257]` and pooled output `[B,27,D]`.
- Head parity for all three logits: `[B,27,38]`, `[B,27,50257]`, `[B,27,30522]`.
- Processor parity: official image should decode to `ticket`; Transformers test also checks a char-logit slice with `rtol=1e-4, atol=1e-4`.
- Recommended tolerances: fp32 `1e-4`; fp16/bf16 attention/classifier paths start with `1e-2` absolute around logits, then tune after accumulation policy is fixed.

## 13. Performance probes

- Image preprocessing throughput: resize/rescale to `[B,3,32,128]`.
- Patch embedding throughput as Conv2d vs lowered GEMM.
- Encoder-only throughput over batch sizes `1, 8, 32, 128`.
- Attention backend comparison for fixed `N=257`, `D=768`, `H=12`.
- Classifier-head GEMM cost split, especially BPE/WP heads.
- A3 grouped Conv2d and BMM throughput.
- End-to-end images/sec including CPU decode.
- Memory probes for logits: BPE/WP logits dominate output size at batch and fixed token length.

## 14. Skip/defer list

- Training, losses, gradients, dropout randomness, stochastic depth training behavior.
- Autoregressive generation, beam search, KV cache, sampling.
- `distilled=True` until source/model behavior is reconciled.
- Dynamic image resolutions.
- Attention/hidden-state optional outputs for first optimized runtime.
- Tokenizer download/management inside GPU runtime; keep decode as a host-side ABI.
- ONNX mirror-specific deployment metadata.

## 15. Final implementation checklist

- [ ] Parse `MgpstrConfig` and reject unsupported `distilled=True`.
- [ ] Load patch, encoder, A3, and three classifier weight groups.
- [ ] Implement NCHW image tensor contract with exact `32x128` guard.
- [ ] Implement Conv2d patch embedding or guarded Conv-to-GEMM rewrite.
- [ ] Implement cls-token concat and learned absolute position add.
- [ ] Implement encoder LayerNorm, fused QKV, dense noncausal attention, MLP.
- [ ] Implement A3 grouped `1x1` Conv2d, softmax, BMM/einsum, LayerNorm.
- [ ] Implement char/BPE/WP classifier heads.
- [ ] Add tiny/random single-block and full-model parity tests.
- [ ] Add official checkpoint logits and decode parity test.
- [ ] Benchmark patch, encoder, A3, classifier, decode, and end-to-end throughput.
