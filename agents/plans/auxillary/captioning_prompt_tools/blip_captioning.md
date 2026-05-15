# BLIP Captioning

## Coverage

- Diffusers: only deprecated BLIP Diffusion helpers are present; normal UI captioning is not a Diffusers pipeline.
- Transformers: BLIP and BLIP-2 are covered under `src/transformers/models/blip` and `blip_2`.
- Third-party/UI: A1111 loads a cloned `models.blip.blip_decoder`; SD.Next can route through `clip_interrogator` or VLM captioning.

## Runtime Contract

A1111 resizes the image to 384x384, normalizes with CLIP-like mean/std, runs BLIP caption generation, then uses the result as prompt text. This is an image-to-text generation workflow, not a diffusion runtime branch.

## Operators

- Vision encoder, text decoder/generation, cross-attention.
- Tokenizer/generation controller.
- Image resize/normalize.

## DinoML Notes

Prefer reusing Transformers BLIP/BLIP-2 coverage for in-library models. The UI contract also needs model residency, beam settings, prompt assembly, and optional offload.

## Sources

- `AUTOMATIC1111/stable-diffusion-webui/modules/interrogate.py`
- `transformers/src/transformers/models/blip`
- `transformers/src/transformers/models/blip_2`

