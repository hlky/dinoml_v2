# UDOP source snippets

Source basis: `transformers`, commit `b75feb2af64c3e29cbbc1bd859958c5432cc7ed4`.

## Config defaults

- `UdopConfig` defaults to `vocab_size=33201`, `d_model=1024`, `d_kv=64`, `d_ff=4096`, `num_layers=24`, `num_heads=16`, `image_size=224`, `patch_size=16`, `max_2d_position_embeddings=1024`, `feed_forward_proj="relu"`, `use_cache=True`, `tie_word_embeddings=True`.
- If `relative_bias_args` is omitted, it becomes `[{"type": "1d"}, {"type": "horizontal"}, {"type": "vertical"}]`.
- `num_decoder_layers` defaults to `num_layers`.
- `feed_forward_proj` parses `gated-*` as gated MLP; official inspected configs use ungated `relu`.

## Patch and image/text merge path

- `UdopPatchEmbeddings` is `Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)` followed by `flatten(2).transpose(1, 2)`, consuming source-layout `pixel_values` as `[B, C, H, W]`.
- `combine_image_text_embeddings` maps OCR box centers to patch indices:
  `floor((x0+x1)/2 * num_patches)` and `floor((y0+y1)/2 * num_patches) * num_patches`, clipped to `[0, num_patches-1]`.
- It gathers one patch embedding per text token and adds it into token embeddings, zeros gathered patches when bbox mean is `0.0` or `1.0`, removes used patches from the standalone patch-token list, pads remaining patch tokens to `image_embeddings.size(1)`, and concatenates `[text_tokens, remaining_patch_tokens]`.
- If `visual_bbox` is not supplied, `get_visual_bbox` creates normalized `[x0, y0, x1, y1]` patch boxes on a regular grid.

## Attention and relative bias

- `UdopAttention` uses bias-free `q/k/v/o` linear projections: `q,k,v: d_model -> num_heads*d_kv`, `o: num_heads*d_kv -> d_model`.
- Dense attention path is `scores = q @ k.T`, add position/mask bias, `softmax(scores.float(), dim=-1).type_as(scores)`, dropout, `attn @ v`, transpose/contiguous/reshape, output projection.
- Encoder position bias is not the T5 first-layer-only module bias alone; `UdopStack` computes aggregated 1D, horizontal, and vertical relative biases from sequence positions and bbox centers, then adds the mask once and shares the resulting `position_bias` through layers.
- Decoder self-attention resets `position_bias=None` initially, so the first decoder self-attention layer computes T5-style causal relative position bias. Decoder cross-attention has no learned relative bias, only encoder mask bias.

## Layout/OCR preprocessing

- `UdopProcessor` composes `LayoutLMv3ImageProcessor` and `UdopTokenizer`.
- `LayoutLMv3ImageProcessor` optionally runs Tesseract OCR, normalizes OCR boxes to a 0-1000 integer scale, resizes/rescales/normalizes images, and returns `pixel_values`, plus `words`/`boxes` when OCR is enabled.
- `UdopTokenizer` expects word-level boxes when OCR is not producing them. It expands each word box to all corresponding subword tokens; special separator token boxes are `[1000,1000,1000,1000]`; padding token boxes are `[0,0,0,0]`.
