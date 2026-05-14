# GLM46V source notes

Scope: local Transformers checkout `X:/H/transformers` at commit
`b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Local source files inspected

- `src/transformers/models/glm46v/configuration_glm46v.py`
- `src/transformers/models/glm46v/modeling_glm46v.py`
- `src/transformers/models/glm46v/modular_glm46v.py`
- `src/transformers/models/glm46v/processing_glm46v.py`
- `src/transformers/models/glm46v/image_processing_glm46v.py`
- `src/transformers/models/glm46v/image_processing_pil_glm46v.py`
- `src/transformers/models/glm46v/video_processing_glm46v.py`
- Delegated implementation files:
  - `src/transformers/models/glm4v/configuration_glm4v.py`
  - `src/transformers/models/glm4v/modeling_glm4v.py`

`modeling_glm46v.py`, `configuration_glm46v.py`, and processor files are
generated from `modular_glm46v.py`. Future Transformers edits should be made in
the modular source, but this audit uses the generated file to inspect the
expanded ABI hooks.

## Checkpoint/config snapshots

Fetched into this directory on 2026-05-13:

- `hf_config_GLM-4.6V.json` from `https://huggingface.co/zai-org/GLM-4.6V/raw/main/config.json`
- `hf_config_GLM-4.6V-Flash.json` from `https://huggingface.co/zai-org/GLM-4.6V-Flash/raw/main/config.json`
- `hf_config_GLM-4.6V-FP8.json` from `https://huggingface.co/zai-org/GLM-4.6V-FP8/raw/main/config.json`
- `hf_config_GLM-4.1V-9B-Thinking.json` from `https://huggingface.co/zai-org/GLM-4.1V-9B-Thinking/raw/main/config.json`
- `hf_preprocessor_GLM-4.6V.json` from `https://huggingface.co/zai-org/GLM-4.6V/raw/main/preprocessor_config.json`
- `hf_generation_GLM-4.6V.json` from `https://huggingface.co/zai-org/GLM-4.6V/raw/main/generation_config.json`

Representative config caveat: official GLM-4.6V configs inspected here use
`model_type` values `glm4v` or `glm4v_moe`, not `glm46v`. The checked-in
`glm46v` source is therefore treated as the model-family wrapper/processor
surface requested by the task, while GLM-4.6V MoE text-layer parity is gated on
a separate `glm4v_moe` audit.

## Key source anchors

- `glm46v/modeling_glm46v.py:85`: `Glm46VModel` composes
  `AutoModel.from_config(config.vision_config)` and
  `AutoModel.from_config(config.text_config)`.
- `glm46v/modeling_glm46v.py:106`: `get_vision_position_ids` builds temporal,
  height, and width indices for one image/video grid.
- `glm46v/modeling_glm46v.py:164`: `get_rope_index` groups contiguous
  `mm_token_type_ids`, repeats video grids per frame, and computes M-RoPE
  deltas.
- `glm46v/modeling_glm46v.py:259`: video features flatten `video_grid_thw` to
  per-frame `[1, h, w]` grids for the vision model, then split pooled outputs
  back by original video token counts.
- `glm46v/modeling_glm46v.py:310`: placeholder mask checks feature/token
  counts. With `input_ids`, both image and video masks are keyed on
  `image_token_id`; video disambiguation relies on processor token-type/RoPE
  metadata and video start/end wrappers.
- `glm46v/modeling_glm46v.py:436` and `:442`: multimodal features are stitched
  into token embeddings with `masked_scatter`.
- `glm46v/modeling_glm46v.py:642`: generation drops image/video pixel inputs
  after the first cached iteration.
- `glm46v/modeling_glm46v.py:680`: generation position ids prepend text
  positions to 3D vision positions, yielding `[4, batch, seq]` when packed mask
  construction needs text positions.
- `glm46v/modeling_glm46v.py:774`: beam/expand generation repeats packed
  visual tensors using image/video counts derived from start/end tokens.
- `glm46v/processing_glm46v.py:118`: image placeholders expand to
  `prod(image_grid_thw) / merge_size^2` `<|image|>` tokens.
- `glm46v/processing_glm46v.py:128`: each video placeholder expands into per
  frame `<|begin_of_image|><|image|><|end_of_image|>{timestamp} seconds` blocks.
- `glm46v/processing_glm46v.py:246`: `create_mm_token_type_ids` marks
  `<|image|>` tokens inside video spans as type 2 and standalone image tokens
  as type 1.
- `glm46v/image_processing_glm46v.py:50`: `smart_resize` clamps aspect ratio
  and returns dimensions divisible by `patch_size * merge_size`.
- `glm46v/image_processing_glm46v.py:194`: image patch pack order:
  `[B,T,C,H,W] -> [B, grid_t, gh/merge, gw/merge, merge_h, merge_w, C, tp, ph, pw]`
  then flatten to `[sum_patches, C*temporal_patch*patch*patch]`.
- `glm46v/video_processing_glm46v.py:97`: video frame sampling requires
  metadata with fps, uses duration-dependent fps thresholds, caps extracted
  frames at 640, and pads to an even count.
- `glm4v/modeling_glm4v.py:83`: vision patch embedding is a non-overlap
  `Conv3d(in=3, out=hidden, kernel=stride=[temporal_patch, patch, patch])` over
  processor-flattened patches.
- `glm4v/modeling_glm4v.py:136`: vision learned 2D positional table is
  bicubically sampled with `grid_sample(..., align_corners=False,
  padding_mode="border")`.
- `glm4v/modeling_glm4v.py:269`: vision attention is packed noncausal MHA;
  qkv is one linear and FlashAttention receives `cu_seqlens`.
- `glm4v/modeling_glm4v.py:386`: text rotary embedding applies M-RoPE over
  3D position ids; default rotary dimension is `head_dim * partial_rotary_factor`.
- `glm4v/modeling_glm4v.py:510`: text attention is causal GQA with biased
  Q/K/V projections and biasless output projection.
- `glm4v/modeling_glm4v.py:580`: text MLP uses packed `gate_up_proj`, chunked
  as gate then up, followed by SiLU-gated multiply and down projection.
- `glm4v/modeling_glm4v.py:693`: vision model builds patch embeddings,
  positional interpolation, packed attention blocks, spatial downsample
  `Conv2d`, and a merger MLP to text hidden width.
- `glm4v/modeling_glm4v.py:903`: text mask construction uses
  `create_causal_mask` with optional packed text position ids.

## Gated/source gaps

- Official GLM-4.6V main and FP8 configs point at `Glm4vMoeForConditionalGeneration`
  / `glm4v_moe`, which is not implemented by the checked `glm46v` source.
- Official GLM-4.6V-Flash config points at `Glm4vForConditionalGeneration` /
  `glm4v`, again not `glm46v`, though the operator surface overlaps heavily.
- The report does not inspect remote-code-only files because the in-library
  source was present locally at the pinned commit.
- No code imports, tests, or model execution were run.
