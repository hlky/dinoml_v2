# CLIP Vision, Style Models, and Reference Image Features

## Coverage

- Diffusers: IP-Adapter and GLIGEN loaders use CLIP/SigLIP image encoders through `transformers`.
- Transformers: CLIP and SigLIP are covered under `src/transformers/models/clip` and `src/transformers/models/siglip`.
- Third-party/UI: UIs expose style/reference-image conditioning around these encoders.

## Runtime Contract

Image reference conditioning usually encodes a resized/normalized image into pooled embeddings or hidden-state sequences. IP-Adapter then projects those embeddings into adapter tokens or added K/V attention branches.

## Operators

- Vision transformer patch embedding, MHA, MLP, LayerNorm.
- Projection heads and optional hidden-state extraction.
- Image resize/crop/normalize.

## DinoML Notes

Reuse Transformers CLIP/SigLIP model coverage, but keep image-embed cache keys and adapter projection identity visible. This is a model component, not a raw preprocessor image map.

## Sources

- `X:/H/diffusers/src/diffusers/loaders/ip_adapter.py`
- `X:/H/diffusers/src/diffusers/models/attention_processor.py`
- `X:/H/transformers/src/transformers/models/clip`
- `X:/H/transformers/src/transformers/models/siglip`

