# CLIP Interrogator and OpenCLIP Category Scoring

## Coverage

- Diffusers: not covered.
- Transformers: CLIP is covered, but OpenCLIP package models are separate unless mapped to Transformers configs.
- Third-party/UI: A1111 uses OpenAI CLIP package style scoring; SD.Next installs `clip_interrogator==0.6.0` and OpenCLIP.

## Runtime Contract

The UI encodes an image with CLIP/OpenCLIP, encodes category text lists, normalizes features, computes scaled similarity, applies softmax/top-k, and assembles prompt fragments. Category text files and top-N rules are part of the feature.

## Operators

- CLIP image/text encoders.
- L2 normalization, matrix multiply, softmax, top-k.
- Tokenization and prompt text assembly.

## DinoML Notes

The model graph is only half the task. DinoML needs a category database/cache contract, text-feature caching, ranking policy, and prompt formatter.

## Sources

- `AUTOMATIC1111/stable-diffusion-webui/modules/interrogate.py`
- `vladmandic/sdnext/modules/caption/openclip.py`

