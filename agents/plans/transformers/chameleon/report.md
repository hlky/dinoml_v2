# Transformers Chameleon Operator and Integration Report

## 1. Source basis

```text
Transformers commit/version:
  Local checkout X:/H/transformers at b75feb2af64c3e29cbbc1bd859958c5432cc7ed4.

Model id:
  Primary target: facebook/chameleon-7b, scoped to ChameleonForConditionalGeneration
  multimodal image/text-to-text generation.
  Additional sizing references: facebook/chameleon-30b repository metadata,
  open original-parameter mirrors listed below, and local tiny test configs.

Config source:
  Source defaults from:
    X:/H/transformers/src/transformers/models/chameleon/configuration_chameleon.py
  Local test/debug config from:
    X:/H/transformers/tests/models/chameleon/test_modeling_chameleon.py
  Official facebook/chameleon-7b and facebook/chameleon-30b repos are gated in
  this environment. Hugging Face repo file listings were visible, but direct
  config/preprocessor/tokenizer/generation config download returned 403.
  Open mirrors used for original params/VQGAN sizing, clearly not treated as
  official HF config.json:
    H:/dinoml_v2/agents/plans/transformers/chameleon/_sources/ZeroWw__chameleon-7b/models/7b/params.json
    H:/dinoml_v2/agents/plans/transformers/chameleon/_sources/ZeroWw__chameleon-7b/tokenizer/vqgan.yaml
    H:/dinoml_v2/agents/plans/transformers/chameleon/_sources/Tom9000__not-chameleon-7b/params.json
    H:/dinoml_v2/agents/plans/transformers/chameleon/_sources/Tom9000__not-chameleon-30b/params.json
    H:/dinoml_v2/agents/plans/transformers/chameleon/_sources/lodestones__meta-chameleon-7b/models/7b/params.json
    H:/dinoml_v2/agents/plans/transformers/chameleon/_sources/lodestones__meta-chameleon-7b/tokenizer/vqgan.yaml
    H:/dinoml_v2/agents/plans/transformers/chameleon/_sources/alandao__open-chameleon/make_a_scene/img_config.yaml

Source files inspected:
  X:/H/transformers/src/transformers/models/chameleon/modeling_chameleon.py
  X:/H/transformers/src/transformers/models/chameleon/configuration_chameleon.py
  X:/H/transformers/src/transformers/models/chameleon/processing_chameleon.py
  X:/H/transformers/src/transformers/models/chameleon/image_processing_chameleon.py
  X:/H/transformers/src/transformers/models/chameleon/image_processing_pil_chameleon.py
  X:/H/transformers/src/transformers/models/chameleon/convert_chameleon_weights_to_hf.py
  X:/H/transformers/tests/models/chameleon/test_modeling_chameleon.py
  X:/H/transformers/tests/models/chameleon/test_processing_chameleon.py
  X:/H/transformers/tests/models/chameleon/test_image_processing_chameleon.py
  X:/H/transformers/docs/source/en/model_doc/chameleon.md

Any missing files or assumptions:
  Official gated config.json/preprocessor_config.json/tokenizer_config.json were
  not accessible without authorization. The report assumes inference-only CUDA
  GPU execution, with preprocessing allowed to remain in the CPU/data pipeline
  initially. The current Transformers source is authoritative for runtime graph
  behavior. The converter file is used only to understand original checkpoint
  parameter mapping and historical field names.
```

## 2. High-level architecture

Chameleon is an early-fusion tokenized multimodal causal decoder. Images are not
fed through a CLIP-style vision encoder/projector. Instead, a frozen VQ-VAE/VQGAN
image tokenizer converts each processed 512x512 image into 1024 discrete image
code indices. Those indices are mapped into existing BPE token IDs and embedded
with the same token embedding table used by text. The causal decoder then runs
over one mixed token sequence.

```text
CPU image/text preprocessing
  -> text prompt with expanded image placeholders
  -> VQ-VAE image tokenizer -> image code indices -> BPE image token IDs
  -> token embedding lookup and masked scatter into placeholder slots
  -> causal decoder prefill with RoPE and KV cache
  -> cached decode
  -> lm_head logits
  -> logits mask blocks raw image-code token generation
  -> text sampling
```

Stage decomposition:

- CPU/data-pipeline: PIL/RGB conversion, resize, center crop, rescale,
  normalize, tokenizer prompt expansion, chat-mode separator append.
- Independently cacheable image tokenizer: `pixel_values[B,3,512,512]` through
  VQ-VAE encoder/quantizer to `image_tokens[B,1024]`, then BPE image token IDs
  and embeddings. These image embeddings can be cached per image before prefill.
- Prefix construction: text embeddings are looked up, then image embeddings are
  inserted into `<image>` placeholder positions with `masked_scatter`.
- Prefill: full mixed text/image-token sequence runs through a causal decoder.
- Decode: subsequent generation iterations should not forward `pixel_values`
  when cache is active; image embeddings are already represented in the cache.

Implemented heads:

- Required for the target: `ChameleonForConditionalGeneration` with bias-free
  `lm_head`.
- Required internal modules: `ChameleonModel`, `ChameleonVQVAE`.
- Optional/deferred: standalone `ChameleonModel` feature extraction,
  `get_image_tokens`, `get_image_features`, loss/training paths, hidden-state
  and attention debug outputs.

## 3. Important config dimensions

Symbols: `B=batch`, `T=sequence length`, `H=hidden_size`,
`A=num_attention_heads`, `KvH=num_key_value_heads`, `D=H/A`,
`I=intermediate_size`, `V=vocab_size`, `G=A/KvH`.

Source-default ChameleonConfig:

| Field | Value | Source / notes |
|---|---:|---|
| model_type | `chameleon` | Source default |
| vocab_size / V | 65536 | Source default and original params mirrors |
| hidden_size / H | 4096 | Source default, 7B mirrors |
| intermediate_size / I | 11008 | Source default, converter formula for 7B |
| num_hidden_layers | 32 | Source default, 7B mirrors |
| num_attention_heads / A | 32 | Source default, 7B mirrors |
| num_key_value_heads / KvH | 32 | Source default; converter maps missing original `n_kv_heads` to MHA |
| head_dim / D | 128 inferred | `hidden_size // num_attention_heads` |
| hidden_act | `silu` | Gated MLP |
| max_position_embeddings | 4096 | Source default; converter uses 4096 for version 1 with theta 10000 |
| RoPE theta | 10000 inferred from conversion/default rope params | Converter passes `rope_theta=10000.0` for mirrors |
| rms_norm_eps | 1e-5 | Source default and mirrors |
| attention_bias | false | Source default |
| mlp_bias | false | Source default |
| tie_word_embeddings | false | Source default, but class declares tied key for `lm_head.weight` |
| use_cache | true | Source default |
| torch/checkpoint dtype | bf16 | Original params mirror metadata, not source default |
| attention backend | eager fallback plus SDPA/Flash/Flex support | Source advertises `_supports_*` attention backends |

VQ-VAE source defaults and mirrored VQGAN YAML:

| Field | Value | Notes |
|---|---:|---|
| input resolution | 512 | Image processor crops to 512x512 |
| input channels | 3 | RGB |
| embed_dim | 256 | Codebook vector dim |
| num_embeddings | 8192 | Image codebook size |
| latent_channels | 256 | Encoder output channels before quant conv |
| base_channels | 128 | GroupNorm uses 32 groups |
| channel_multiplier | `[1,1,2,2,4]` | 5 resolution levels |
| num_res_blocks | 2 | Per level |
| attn_resolutions | `[]` / `None` | Mid attention still exists when `attn_type="vanilla"` |
| attn_type | `vanilla` | Uses VQ attention block |
| latent grid | `32x32` inferred | Four stride-2 downsamples from 512 -> 32 |
| image_seq_length | 1024 | Processor default; matches 32*32 codes |

Representative sweep:

| Config source | Model | H | I | layers | A | KvH | D | G | V | max pos | swin_norm | model_parallel_size | dtype |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|
| Local test config | tiny/debug | 32 | 37 | 2 | 2 | 2 | 16 | 1 | 99 | 512 | false | 1 | test dtype |
| Source defaults | ChameleonConfig | 4096 | 11008 | 32 | 32 | 32 | 128 | 1 | 65536 | 4096 | false | 1 | unspecified |
| Open original params mirror | 7B | 4096 | 11008 inferred | 32 | 32 | 32 inferred | 128 | 1 | 65536 | 4096 inferred | false | 1 | bf16 |
| Open original params mirror | 30B | 8192 | 22016 inferred | 48 | 64 | 8 | 128 | 8 | 65536 | 4096 inferred | true | 4 | bf16 |

Effective default omissions:

- Original 7B params omit `n_kv_heads`; converter treats this as `KvH=A`.
- Original params use `dim`, `n_layers`, `n_heads`, `n_kv_heads`, `rope_theta`,
  and `swin_norm`; HF config uses `hidden_size`, `num_hidden_layers`,
  `num_attention_heads`, `num_key_value_heads`, `rope_parameters`, and
  `swin_norm`.
- Current config source has `rope_parameters`; the converter still passes
  historical `rope_theta`. DinoML should inspect the instantiated config object
  rather than raw historical fields.

## 3a. Family variation traps

- 7B is MHA in the available original params, while 30B is GQA with `A=64`,
  `KvH=8`, `G=8`. Do not assume Chameleon is always MHA.
- `swin_norm=True` switches the decoder block order to `ChameleonSwinDecoderLayer`
  for 30B-like params. This changes residual/norm ordering and must be a
  config guard.
- Q and K have a per-head `LayerNorm((num_heads, head_dim))` / `LayerNorm((KvH,
  head_dim))` after projection and before RoPE. This breaks naive packed QKV
  assumptions and is called out in local tests as incompatible with some
  packing/offload assumptions.
- Attention and MLP projections are biasless by source default, but Q/K
  normalization has affine weights and biases.
- The image path is a VQ tokenizer plus shared token embedding, not a dense
  visual feature projector. Image-token order and codebook mapping are part of
  correctness.
- The processor expands every `<image>` text marker into:
  `boi + 1024 * <image> + eoi`. The model scatters only over `<image>`
  placeholder IDs, not the begin/end image IDs.
- The model's LM head explicitly sets all raw `IMGIMG*` image-code token logits
  to dtype minimum, so current HF generation is text-only even though the paper
  architecture can model image tokens.
- Image-code vocabulary mapping relies on token names beginning with `IMGIMG`.
  It remaps letters `A` through `J` back to digits and strips a suffix before
  converting to VQ indices.
- `convert_img2bpe` builds the mapping tensor on CPU and indexes `img_batch` on
  CPU before moving results back. A runtime implementation should replace this
  with a device-resident gather table.
- `get_placeholder_mask` has a strict count guard: placeholder embedding element
  count must equal image feature element count.
- The image processor fast path falls back from LANCZOS to BICUBIC for tensor
  resizing; PIL path can use LANCZOS. Exact end-to-end parity depends on
  processor backend choice.
- Source tensor layout for the VQ-VAE is NCHW. Any NHWC/channel-last layout pass
  needs guarded axis rewrites for Conv2d, GroupNorm channel axis, VQ quantizer
  permutes, and spatial attention reshapes. First semantic translation should
  preserve NCHW.

## 4. Operator coverage checklist

### Tensor/layout ops

- Token embedding gather: `input_ids[B,T] -> inputs_embeds[B,T,H]`.
- Image placeholder mask: compare `input_ids == image_token_id`.
- Boolean expand to `[B,T,H]`, masked scatter/image embedding stitch.
- Reshape/view/transpose/contiguous for Q/K/V:
  - Q `[B,T,A*D] -> [B,A,T,D]`.
  - K/V `[B,T,KvH*D] -> [B,KvH,T,D]`.
- Q/K norm reshape through `[-1, heads, D]`.
- `repeat_kv` fallback: `[B,KvH,T,D] -> [B,A,T,D]`.
- Causal mask construction from attention mask, positions, and cache.
- `logits_to_keep` slicing before LM head.
- VQ-VAE NCHW layout ops: pad, permute NCHW<->NHWC around quantizer, flatten
  `[B,H,W,C] -> [B*H*W,C]`, view code indices `[B,1024]`.

### Neural network primitives

- Bias-free Linear/GEMM:
  - 7B Q/K/V/O: `4096 -> 4096`.
  - 30B Q: `8192 -> 8192`; K/V: `8192 -> 1024`; O: `8192 -> 8192`.
  - 7B MLP gate/up: `4096 -> 11008`; down `11008 -> 4096`.
  - 30B MLP gate/up: `8192 -> 22016`; down `22016 -> 8192`.
  - LM head: `H -> 65536`, bias false.
- RMSNorm over hidden dim with fp32 variance, scale only.
- Per-head LayerNorm for Q and K with affine weight/bias, stats over `head_dim`.
- SiLU and elementwise multiply for gated MLP.
- Residual adds.
- VQ-VAE Conv2d, GroupNorm, Swish-like `x * sigmoid(x)`, Dropout disabled in
  eval, residual blocks, 1x1 and 3x3 convs, asymmetric pad before stride-2
  downsample.
- VQ-VAE vector quantization: squared L2 distance to codebook, `argmin`,
  embedding gather.
- VQ-VAE spatial attention block: 1x1 conv Q/K/V/proj, BMM, softmax over
  spatial keys.

### Attention primitives

- Causal decoder self-attention with MHA/GQA.
- RoPE applied to Q/K before cache update.
- KV cache append/update, storing only `KvH` heads.
- Eager fallback: repeat K/V, QK matmul, scale, additive mask, fp32 softmax,
  dropout in training only, AV matmul.
- Optional SDPA/Flash/Flex backend dispatch through `ALL_ATTENTION_FUNCTIONS`.
- VQ-VAE noncausal spatial attention over `H*W` positions in selected blocks.

### Position/rotary ops

- Default RoPE inverse frequency from `rope_theta` and `head_dim`.
- `position_ids = arange(T) + past_seen_tokens` when not supplied.
- Cos/sin computed in fp32 and cast to activation dtype.
- Dynamic rope update decorator exists for non-default RoPE types; first
  integration can guard to default RoPE unless a target config proves otherwise.

### Generation/cache ops

- `DynamicCache(config)` allocation when `use_cache=True` and no cache is
  supplied.
- Per-layer cache update after RoPE.
- Cache length query for default `position_ids`.
- `prepare_inputs_for_generation` drops `pixel_values` after the first cached
  iteration.
- LM logits image-code suppression: set `logits[:,:,image_tokens]` to dtype min.
- `logits_to_keep` to avoid full prefill logits.

### Preprocessing-coupled ops

- RGB conversion with alpha blending over white for transparent PIL images.
- Resize shortest edge to 512, center crop to 512x512, rescale by `0.0078`,
  normalize with mean/std `[1,1,1]`.
- Processor prompt expansion of each `<image>` marker to 1026 special tokens.
- Optional `mm_token_type_ids` generation in processor, but current model
  forward does not consume it.

### Scatter/indexed update ops

- Image features are inserted with `inputs_embeds.masked_scatter`.
- VQ code index to BPE token ID uses a gather table derived from vocabulary
  names.
- Count guard compares placeholder tokens to `image_features.shape[0] *
  image_features.shape[1]`.

## 5. Layer/block breakdown

Image preprocessing:

```text
PIL/array image
  -> RGB conversion, alpha composited over white if needed
  -> resize shortest_edge=512
  -> center crop 512x512
  -> rescale by 0.0078
  -> normalize: (x - [1,1,1]) / [1,1,1]
  -> pixel_values[B,3,512,512]
```

VQ-VAE encoder/tokenizer:

```text
x = Conv2d(3 -> 128, k=3, s=1, p=1)(pixel_values)
for each of 5 resolution levels:
  repeat num_res_blocks:
    y = GroupNorm32(x); y = y * sigmoid(y); y = Conv2d(k=3,p=1)(y)
    y = GroupNorm32(y); y = y * sigmoid(y); y = Dropout(y); y = Conv2d(k=3,p=1)(y)
    x = shortcut(x) + y
    optional spatial attention if curr_res in attn_resolutions
  if not final level:
    x = pad right/bottom by 1; x = Conv2d(ch -> ch, k=3, s=2, p=0)(x)

x = mid ResBlock -> mid spatial attention -> mid ResBlock
x = GroupNorm32(x); x = x * sigmoid(x); x = Conv2d(ch -> 256, k=3,p=1)(x)
x = quant_conv Conv2d(256 -> 256, k=1)(x)
```

VQ quantizer:

```text
x_nhwc = x.permute(0,2,3,1).contiguous()              # [B,32,32,256]
flat = x_nhwc.view(-1,256)
dist = sum(flat**2, dim=1, keepdim=True)
     + sum(codebook**2, dim=1)
     - 2 * einsum("bd,dn->bn", flat, codebook.T)
indices = argmin(dist, dim=1)                         # [B*1024]
bpe_tokens = img2bpe_mapping[indices].view(B,1024)
image_features = embed_tokens(bpe_tokens)             # [B,1024,H]
```

Prefix stitch:

```text
inputs_embeds = embed_tokens(input_ids)                # [B,T,H]
mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
assert masked elements == image_features elements
inputs_embeds = masked_scatter(inputs_embeds, mask, image_features)
```

Standard decoder block, repeated `N` times when `swin_norm=False`:

```text
res = x
y = RMSNorm(x)
q = Linear(H -> A*D, bias=False)(y)
k = Linear(H -> KvH*D, bias=False)(y)
v = Linear(H -> KvH*D, bias=False)(y)
q = LayerNorm((A,D), affine=True)(q.view(-1,A,D)).view(B,A,T,D)
k = LayerNorm((KvH,D), affine=True)(k.view(-1,KvH,D)).view(B,KvH,T,D)
q,k = RoPE(q,k,cos,sin)
k,v = cache.update(k,v,layer_idx) if cache enabled
attn = CausalAttention(q,k,v,mask,scale=D**-0.5,GQA)
x = res + Linear(A*D -> H, bias=False)(attn)

res = x
y = RMSNorm(x)
mlp = Linear(I -> H, bias=False)(SiLU(Linear(H -> I)(y)) * Linear(H -> I)(y))
x = res + mlp
```

Swin-norm decoder block when `swin_norm=True`:

```text
res = x
y = SelfAttention(x)                                  # no input RMSNorm first
y = RMSNorm(y)
x = res + y
res = x
y = MLP(x)
y = RMSNorm(y)
x = res + y
```

Final LM:

```text
x = RMSNorm(x)
logits = lm_head(x[:, slice_indices, :])               # [B,K,V]
logits[:, :, image_code_token_ids] = dtype_min
```

## 6. Attention requirements

Decoder attention:

- Type: causal self-attention.
- Variants: MHA for 7B/default, GQA for 30B-like params.
- Head shapes before repeat:
  - Q `[B,A,Q,D]`.
  - K/V `[B,KvH,K,D]`.
- KV cache stores K/V before repeat and after RoPE on K. Values are unrotated.
- 7B cache per layer: K and V each `[B,32,T,128]`.
- 30B cache per layer: K and V each `[B,8,T,128]`; logical repeat to 64 heads
  only in fallback.
- Masking: additive causal mask produced by common `create_causal_mask`; eager
  attention adds it before softmax.
- Attention math order in eager fallback: repeat K/V, `query @ key.T`, multiply
  by `D**-0.5`, add mask, softmax in fp32, cast to query dtype, dropout,
  `weights @ value`.
- Backend dispatch: `ALL_ATTENTION_FUNCTIONS.get_interface` chooses SDPA,
  FlashAttention, FlexAttention, or eager fallback. Source advertises support
  for FlashAttention, SDPA, FlexAttention, and generic attention backends.
- Packed/varlen support is not Chameleon-specific in this file. It depends on
  the common attention interface.
- Sliding-window/local attention is not implemented as a Chameleon-specific
  config behavior in the inspected source.

VQ-VAE spatial attention:

- Noncausal full attention over image spatial positions.
- Q/K/V produced by 1x1 Conv2d, then reshaped:
  - Q `[B,C,H,W] -> [B,H*W,C]`.
  - K `[B,C,H,W] -> [B,C,H*W]`.
  - attention weights `[B,H*W,H*W]`, scale `C**-0.5`, softmax over keys.
  - V `[B,C,H*W]`, output BMM returns `[B,C,H,W]`.
- For default VQ config, mid attention happens at latent `32x32`, so the
  attention matrix is `1024x1024` per image.

## 7. Position encoding and custom math

Default Chameleon RoPE:

```python
def chameleon_inv_freq(head_dim, rope_theta, device):
    i = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    return 1.0 / (rope_theta ** (i / head_dim))

def chameleon_rope(position_ids, inv_freq, dtype):
    freqs = (inv_freq[None, :, None].float()
             @ position_ids[:, None, :].float()).transpose(1, 2)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_chameleon_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin
```

Precomputable:

- Default `inv_freq` is static for a fixed `head_dim` and `rope_theta`.
- Cos/sin tables can be cached up to max position for default RoPE.
- Decode only needs the current cache-length-derived position.

Dynamic:

- `position_ids` depend on `past_key_values.get_seq_length()` unless supplied.
- The `dynamic_rope_update` decorator means future/non-default `rope_parameters`
  can update inverse frequencies; guard unsupported rope types explicitly.

## 8. Preprocessing and input packing

Processor outputs:

| Tensor/key | Shape | Produced by | Notes |
|---|---|---|---|
| `input_ids` | `[B,T]` | tokenizer | Text with expanded image placeholder runs |
| `attention_mask` | `[B,T]` | tokenizer | Optional padding mask |
| `pixel_values` | `[N_img,3,512,512]` | image processor | NCHW image tensor |
| `mm_token_type_ids` | `[B,T]` if requested | processor helper | Current model forward does not consume it |

Text packing:

- `ChameleonProcessor` requires text and/or images. In practice the current
  code assumes `text` is a string/list before iterating, so image-only processor
  calls should be treated carefully.
- Each `<image>` string is replaced with
  `image_start_token + image_token * image_seq_length + image_end_token`.
- Defaults:
  - `image_seq_length=1024`.
  - `image_token="<image>"`.
  - start token from tokenizer `boi_token` or fallback `<racm3:break>`.
  - end token from tokenizer `eoi_token` or fallback `<eoss>`.
- Unless `return_for_text_completion=True`, the processor appends
  `tokenizer.sep_token` to every prompt for chat formatting.
- Local docs state the HF implementation aliases `<image>` to a reserved token
  rather than adding a new token; the converter maps `<reserved08707>` to
  `<image>` and sets `sep_token_id=8710`.

Image processing:

- Convert PIL images to RGB; if an alpha channel has transparency, blend over a
  white background.
- Resize shortest edge to 512, with `default_to_square=False`.
- Center crop to 512x512.
- Rescale by `0.0078`, then normalize with mean/std `[1,1,1]`.
- Fast tensor path warns that LANCZOS is not supported and uses BICUBIC.

Runtime stitch:

- The VQ-VAE produces 1024 code indices per 512x512 image.
- `ChameleonImageVocabularyMapping` maps code indices to BPE token IDs using
  the `IMGIMG*` vocabulary entries from `config.vocabulary_map`.
- `get_image_features` embeds those BPE IDs with `embed_tokens`.
- `get_placeholder_mask` enforces exact placeholder count before
  `masked_scatter`.
- With cache enabled, image inputs are only needed in the first iteration.

CPU/data-pipeline versus GPU/runtime:

- First integration can keep tokenizer/image preprocessing outside DinoML.
- For best prefill performance, DinoML should stage VQ-VAE image tokenization
  as an independently cacheable subgraph, then feed image embeddings into the
  decoder prefix.
- A minimal text-only Chameleon path can bypass `pixel_values` and
  `masked_scatter`, but should still apply image-code logits suppression.

## 9. Graph rewrite / lowering opportunities

### Rewrite: Chameleon RMSNorm -> RMSNormScaleOnly

Source pattern:

```text
x_fp32 = x.float()
var = mean(x_fp32**2, dim=-1, keepdim=True)
y = x_fp32 * rsqrt(var + eps)
out = weight * y.to(input_dtype)
```

Replacement:

```text
RMSNorm(x, weight, axis=-1, eps=rms_norm_eps, fp32_accum=True)
```

Preconditions:

- Weight shape `[H]`, no bias.
- Reduction axis is last dim.

Failure cases:

- Do not rewrite the per-head Q/K `LayerNorm`, which has mean subtraction and
  affine bias.

Parity test sketch:

- Random `[B,T,H]` tensors for `H=4096` and `8192`, fp32/bf16/fp16.

### Rewrite: bias-free Linear -> GEMM_RCR

Source pattern:

```text
nn.Linear(in_features, out_features, bias=False)
```

Replacement:

```text
Flatten leading dims -> GEMM_RCR(A=[M,K], B=[N,K]) -> reshape
```

Preconditions:

- Dense row-major activation after flattening.
- PyTorch weight layout `[out_features, in_features]`.
- Bias absent. If future config sets a bias, use a bias epilogue.

Failure cases:

- Tensor-parallel or sharded original checkpoint weights must be consolidated or
  explicitly represented.

Parity test sketch:

- Compare Q/K/V/O, gate/up/down, and LM head projections independently.

### Rewrite: Q/K projection plus per-head norm

Source pattern:

```text
q = q_proj(x).reshape(-1, A, D)
q = LayerNorm((A,D) but stats over D)(q)
k = k_proj(x).reshape(-1, KvH, D)
k = LayerNorm((KvH,D) but stats over D)(k)
```

Replacement:

```text
GEMM -> reshape heads -> PerHeadLayerNorm(axis=-1, affine per head)
```

Preconditions:

- `hidden_size % num_attention_heads == 0`.
- LayerNorm weights/biases have shape `[heads,D]`.
- Stats are computed only over `D`.

Failure cases:

- Do not fuse Q/K/V into a simple packed QKV projection without preserving the
  distinct Q/K norms before RoPE.
- 30B GQA has different Q and K head counts.

Parity test sketch:

- Compare Q/K tensors after norm and before RoPE for MHA and GQA configs.

### Rewrite: native GQA attention

Source pattern:

```text
k_rep = repeat_kv(k, A // KvH)
v_rep = repeat_kv(v, A // KvH)
attn = softmax((q @ k_rep.T) * scale + mask) @ v_rep
```

Replacement:

```text
GQAAttention(q[B,A,Q,D], k[B,KvH,K,D], v[B,KvH,K,D], group_size=A/KvH)
```

Preconditions:

- `A % KvH == 0`.
- KV grouping order matches `repeat_kv`: each KV head is expanded into a
  contiguous group of query heads.
- Cache stores unexpanded K/V.

Failure cases:

- Backend that materializes repeated K/V cannot claim memory savings.
- Incorrect grouping order breaks 30B parity.

Parity test sketch:

- Compare native GQA with eager repeat fallback for `G=1` and `G=8`.

### Rewrite: VQ codebook distance -> argmin codebook lookup

Source pattern:

```text
dist = sum(z**2) + sum(e**2) - 2 * z @ e.T
indices = argmin(dist, dim=1)
quantized = embedding(indices)
```

Replacement:

```text
CodebookNearestNeighbor(z_flat[B*1024,256], codebook[8192,256])
```

Preconditions:

- Codebook size and embedding dim match config.
- Inference only; embedding loss and straight-through gradient are not needed.

Failure cases:

- Approximate nearest neighbor is not parity-safe unless explicitly admitted.
- Training path needs embedding loss/gradient behavior.

Parity test sketch:

- Compare indices and quantized vectors against HF for random latent features
  and real/synthetic codebooks.

### Rewrite: VQ image-token BPE mapping -> device gather

Source pattern:

```text
mapping = img2bpe_mapping_tensor.cpu()
bpe = mapping[image_indices.cpu()].to(device)
emb = embed_tokens(bpe)
```

Replacement:

```text
bpe = Gather(img2bpe_mapping_device, image_indices)
emb = Embedding(embed_tokens, bpe)
```

Preconditions:

- `vocabulary_map` has all `IMGIMG*` tokens for `[0,num_embeddings)`.
- Mapping tensor is immutable for the loaded tokenizer/config.

Failure cases:

- Missing vocabulary map or incomplete codebook token names should reject load.

Parity test sketch:

- Validate all 8192 VQ indices map to expected BPE token IDs from the HF
  vocabulary mapping.

### Rewrite: non-overlap Conv2d downsample remains Conv2d, not Linear globally

The VQ-VAE downsample is `pad right/bottom by 1` then `Conv2d(k=3,s=2,p=0)`.
This is not a simple non-overlapping patch projection because windows overlap.
Keep it as Conv2d unless a convolution backend handles it directly.

### Layout pass guards

Candidate safe regions:

- Decoder hidden states `[B,T,H]` with last-dim norm/linear/MLP.
- LM head with optional last-token slice.

No-layout-translation guard regions:

- Entire VQ-VAE NCHW image tokenizer until Conv2d/GroupNorm/attention rewrites
  are layout-aware.
- VQ quantizer `permute(0,2,3,1)` and flatten order.
- `IMGIMG*` code index ordering and placeholder `masked_scatter`.

Axis-sensitive attrs:

- GroupNorm channel axis is NCHW `C`.
- VQ spatial attention flattens `H*W` in row-major spatial order.
- Q/K LayerNorm reduces only over `head_dim`, not over all `heads*head_dim`.

## 10. Kernel fusion candidates

Highest priority:

- Decoder RMSNorm: two or more per layer, bandwidth-sensitive, easy parity.
- Q/K projection + per-head LayerNorm + RoPE staging: Chameleon-specific and
  blocks naive QKV fusion.
- Native GQA/paged attention with rotated-K cache for 30B-like configs.
- SwiGLU MLP fusion: `SiLU(gate) * up` between large GEMMs.
- Last-token-only LM head plus image-code logits suppression.
- Placeholder indexed copy/masked scatter for image embedding stitch.

Medium priority:

- VQ codebook nearest-neighbor kernel for `[B*1024,256] x [8192,256]`.
- Device-resident image-index to BPE-ID gather table.
- VQ-VAE Conv2d/GroupNorm/Swish residual block kernels or library-backed
  conv/groupnorm path.
- VQ-VAE mid spatial attention at 1024 tokens.
- Image-tokenizer result cache keyed by image/preprocess hash.

Lower priority:

- Full GPU image resize/crop/rescale/normalize pipeline.
- Training-only VQ embedding loss and straight-through estimator.
- Beam search and advanced generation controllers.
- Tensor/model parallel execution from original 30B `model_parallel_size`.
- Non-default dynamic RoPE variants unless a target checkpoint requires them.

## 11. Runtime staging plan

Stage 1: text-only decoder parity.

- Parse ChameleonConfig, load embeddings, decoder layers, final norm, LM head.
- Implement RMSNorm, per-head Q/K LayerNorm, RoPE, MHA attention, SwiGLU, and
  image-code logits suppression.
- Run text-only prefill and one-token decode with cache.

Stage 2: GQA and `swin_norm` variation.

- Add 30B-like GQA shape support.
- Add `ChameleonSwinDecoderLayer` residual/norm ordering guard.

Stage 3: externalized image embeddings.

- Accept precomputed `image_features[B,1024,H]` or image BPE tokens as explicit
  runtime inputs.
- Implement placeholder count guard and indexed scatter.

Stage 4: VQ-VAE image tokenizer parity.

- Implement NCHW VQ-VAE encoder, quant conv, codebook argmin, and image-index
  to BPE mapping.
- Validate `get_image_tokens` and `get_image_features` independently.

Stage 5: multimodal prefill.

- Use external Transformers processor to produce `input_ids`, `attention_mask`,
  and `pixel_values`.
- Run image tokenizer + stitch + full decoder prefill, compare logits.

Stage 6: cached multimodal decode.

- Ensure subsequent decode omits image tokenizer and consumes only KV cache plus
  new token.

Stage 7: optimization.

- Add native GQA/paged attention, fused norms/MLP, VQ codebook kernel, and image
  embedding cache.

Stub initially:

- Tokenizer/chat formatting and PIL image preprocessing can stay outside
  DinoML.
- Training loss and VQ embedding loss can be omitted.
- Full image generation should be rejected because current HF LM masks raw image
  code token logits.

## 12. Parity and validation plan

Random tensor/unit tests:

- RMSNorm fp32/fp16/bf16.
- Per-head Q/K LayerNorm with `[A,D]` and `[KvH,D]` affine parameters.
- Default RoPE cos/sin and apply_rope for sampled positions and cache offsets.
- `repeat_kv` grouping for `G=1` and `G=8`.
- SwiGLU MLP.
- VQ quantizer distance/argmin/gather.
- Image-index to BPE mapping table.
- Placeholder count guard and masked scatter.

Single-layer parity:

- One standard decoder block without cache, MHA config.
- One standard decoder block with cache decode step.
- One GQA decoder block, 30B-like shapes scaled down if needed.
- One `swin_norm=True` block to lock residual/norm order.
- VQ-VAE resnet block, downsample block, and spatial attention block.

Subsystem parity:

- `get_image_tokens(pixel_values)` for synthetic or real 512x512 preprocessed
  image tensors.
- `get_image_features(pixel_values).pooler_output`.
- Text-only `ChameleonForConditionalGeneration` logits with image-code logits
  masked.
- Multimodal prefill logits for one image prompt.
- Cached decode after multimodal prefill, verifying pixel values are not reused.

End-to-end parity:

- Use HF processor externally for one-image and two-image prompts.
- Compare greedy decode token IDs for a short generation after logits parity is
  stable.

Suggested tolerances:

- fp32 unit ops: `rtol=1e-5`, `atol=1e-6`.
- fp16/bf16 graph logits: start with `rtol=3e-2`, `atol=3e-2`, then tighten
  per backend.
- VQ token indices should match exactly; approximate codebook selection is not
  acceptable for parity.

## 13. Performance probes

- Processor-only throughput: images/sec for PIL path and tensor fast path.
- VQ-VAE tokenizer throughput: images/sec, split by conv/resblock/attention and
  codebook argmin.
- Codebook search cost: `[B*1024,256] x [8192,256]` nearest-neighbor latency.
- Image embedding cache hit/miss latency.
- Prefill throughput versus mixed sequence length:
  `text_tokens + 1026 * num_images`.
- Decode tokens/sec versus batch size and cache length.
- KV cache memory:
  `2 * layers * B * KvH * T * D * dtype_bytes`.
- GQA backend comparison: eager repeat vs native GQA/paged attention.
- Q/K norm + RoPE staging cost per layer.
- LM head cost full logits vs `logits_to_keep=1`; vocab 65536.
- Image-code logits suppression cost over 8192 token IDs.

No benchmark measurements are included; these are source-derived probes.

## 14. Skip/defer list

- Training, labels/loss, dropout behavior, and gradient checkpointing.
- VQ embedding loss and straight-through gradient.
- Full tokenizer/chat template implementation inside DinoML.
- Full PIL/image preprocessing kernels inside DinoML.
- Image generation/multimodal output generation; current HF forward masks raw
  image-code token logits for text generation.
- Beam search, speculative decoding, and advanced sampling processors.
- Tensor/model parallelism for first integration.
- Non-default/dynamic RoPE variants until a checkpoint requires them.
- Attention/hidden-state debug outputs.
- Offload/quantization-specific paths.

## 15. Final implementation checklist

- [ ] Parse ChameleonConfig and VQ config, including historical mirror fields.
- [ ] Reject or separately route configs without `vocabulary_map`.
- [ ] Load embeddings, decoder layers, final norm, LM head, and VQ-VAE weights.
- [ ] Preserve tied/aliased weight intent for `lm_head.weight` and embeddings if a checkpoint ties them.
- [ ] Implement RMSNormScaleOnly.
- [ ] Implement per-head Q/K LayerNorm with affine weight and bias.
- [ ] Implement default RoPE and rotated-K KV cache.
- [ ] Implement MHA and native/fallback GQA attention.
- [ ] Implement standard and `swin_norm` decoder block order.
- [ ] Implement SwiGLU MLP.
- [ ] Implement `logits_to_keep` and image-code logits suppression.
- [ ] Implement placeholder count validation and indexed image embedding stitch.
- [ ] Implement VQ-VAE NCHW conv/resblock/downsample/mid-attention path.
- [ ] Implement VQ codebook nearest-neighbor argmin.
- [ ] Implement device image-index to BPE-ID gather table.
- [ ] Add text-only one-block, prefill, and decode parity tests.
- [ ] Add GQA and `swin_norm` parity tests.
- [ ] Add VQ tokenizer and image feature parity tests.
- [ ] Add multimodal prefill and cached decode parity with external HF processor.
- [ ] Benchmark preprocessing, VQ tokenizer, prefill, decode, LM head, and KV memory separately.
