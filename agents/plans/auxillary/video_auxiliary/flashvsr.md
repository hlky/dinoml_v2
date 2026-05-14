# FlashVSR

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: Wan2GP postprocessing stack.

## Runtime Contract

FlashVSR is video super-resolution/upscaling. Wan2GP exposes modes `tiny`, `full`, and `tiny-long`, spatial ratios from 1.0 to 4.0, persistence options, sparse/dense attention backend selection, and separate files for transformer, low-quality projection, temporal decoder, positive prompt, and Wan VAE.

## Operators

- Wan-like video DiT transformer with attention backend dispatch.
- Low-quality projection and temporal decoder.
- Local/window sparse attention masks and optional block-sparse attention.
- Video chunking, persistence/offload, and upsampling ratio control.

## DinoML Notes

This is not a simple image upscaler. Treat it as a video postprocess pipeline with model residency, sparse attention backend, chunking, temporal shape, and VAE/decoder dependencies.

## Sources

- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/flashvsr/wgp_bridge.py`
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/flashvsr/wan_video_dit.py`
- `H:/uis/deepbeepmeep/Wan2GP/postprocessing/flashvsr/tcdecoder.py`

