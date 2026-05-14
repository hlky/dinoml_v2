# Wav2Vec2 Encoders

## Coverage

- Diffusers: not covered.
- Transformers: covered under `src/transformers/models/wav2vec2` and related speech families.
- Third-party/UI: Comfy and Wan2GP use Wav2Vec2-style encoders, including patched MultiTalk variants.

## Runtime Contract

Comfy's local Wav2Vec2 averages channels to mono, optionally normalizes waveform, runs a 7-layer Conv1d feature extractor, projects features, adds convolutional positional embeddings, and runs a Transformer encoder. Defaults in the inspected file include embed dim 1024, 16 heads, 24 layers.

## Operators

- Conv1d stack with LayerNorm or GroupNorm.
- Weight-normalized positional Conv1d.
- MHA, GELU FFN, LayerNorm, residual add.
- Optional masking metadata.

## DinoML Notes

Use Transformers coverage for standard checkpoints, but retain patched UI wrappers separately. MultiTalk's subclass may change feature extraction, hidden-state choice, or projection shape.

## Sources

- `X:/H/transformers/src/transformers/models/wav2vec2`
- `H:/uis/Comfy-Org/ComfyUI/comfy/audio_encoders/wav2vec2.py`
- `H:/uis/deepbeepmeep/Wan2GP/models/wan/multitalk/wav2vec2.py`

