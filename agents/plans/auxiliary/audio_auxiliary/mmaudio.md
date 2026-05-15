# MMAudio

## Coverage

- Diffusers: not covered as a core Diffusers model in the inspected tree.
- Transformers: not covered as a named family.
- Third-party/UI: Wan2GP postprocessing stack.

## Runtime Contract

MMAudio is an audio generation/conditioning model. The inspected network projects audio latents, CLIP features, sync features, and text features into a shared hidden dimension, applies timestep embeddings, joint multimodal DiT blocks, fused single blocks, RoPE rotations for latent and CLIP sequences, and outputs latent audio deltas.

## Operators

- Channel-last Conv1d and ConvMLP.
- Linear/MLP, SiLU or SELU.
- Timestep embedding, RoPE.
- Joint attention and MMDiT-style blocks.
- Latent mean/std normalization.

## DinoML Notes

This is a separate audio/video generation lane. It needs explicit condition tensor contracts for `clip_f`, `sync_f`, `text_f`, negative/conditional features, latent sequence length, and audio codec/vocoder boundary.

## Sources

- `deepbeepmeep/Wan2GP/postprocessing/mmaudio/model/networks.py`
- `deepbeepmeep/Wan2GP/postprocessing/mmaudio/ext/bigvgan`

