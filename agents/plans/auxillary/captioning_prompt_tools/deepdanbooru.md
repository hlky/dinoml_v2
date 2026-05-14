# DeepDanbooru

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: A1111 includes `DeepDanbooruModel` from TorchDeepDanbooru-style code.

## Runtime Contract

DeepDanbooru is an anime tag classifier. A1111 loads model weights and tag names, runs image classification, filters probabilities by threshold, optionally removes rating tags, sorts by probability or alphabetically, formats underscores/spaces and optional scores, and returns a prompt string.

## Operators

- Large ResNet-like Conv2d stack with pooling and dense/classification head.
- Sigmoid/multi-label probabilities.
- CPU-side thresholding, sorting, tag filtering, and string formatting.

## DinoML Notes

The neural graph can be treated as image classifier inference. Product parity requires tag metadata, category filtering, threshold fields, sorting policy, and output formatting.

## Sources

- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru_model.py`
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru.py`

