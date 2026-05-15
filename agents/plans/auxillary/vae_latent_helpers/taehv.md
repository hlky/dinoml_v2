# TAEHV

## Coverage

- Diffusers: not covered as TAEHV in the inspected tree.
- Transformers: not covered.
- Third-party/UI: Comfy, SD.Next, and Wan2GP include TAEHV temporal/spatial preview decoders.

## Runtime Contract

TAEHV is a tiny temporal autoencoder for HunyuanVideo/Wan/LTX-style latents. Comfy's implementation accepts `[B, C, T, H, W]` or frame-major forms, uses `MemBlock` temporal memory, `TPool`/`TGrow` temporal down/up scaling, optional patch size for high-channel latents, and trims generated frames after decode.

## Operators

- Conv2d on folded time/frame batches.
- Temporal pooling/growing by reshape and 1x1 conv.
- Pixel shuffle/unshuffle for patch variants.
- Stateful or parallel memory block execution.

## DinoML Notes

Represent temporal chunking and memory explicitly. This is a good small target for testing video latent shape contracts without full video VAE cost.

## Sources

- `Comfy-Org/ComfyUI/comfy/taesd/taehv.py`
- `vladmandic/sdnext/modules/taesd/taehv.py`
- `deepbeepmeep/Wan2GP/postprocessing/flashvsr/tcdecoder.py`

