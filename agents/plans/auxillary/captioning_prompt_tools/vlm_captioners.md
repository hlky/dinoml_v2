# VLM Captioners

## Coverage

- Diffusers: not covered.
- Transformers: many candidate VLMs are covered individually under Transformers model-family reports.
- Third-party/UI: SD.Next routes VLM captioning through `modules.caption.vqa` with model-specific handlers.

## Runtime Contract

VLM captioning accepts an image and a question/system prompt, builds model-specific conversation inputs, runs multimodal generation or VQA, strips reasoning tags when needed, and returns text. Some handlers use BLIP, JoyCaption, JoyTag, Moondream, DeepSeek-style VLMs, or remote-code models.

## Operators

- Model-specific vision encoder/projector/LLM generation.
- Placeholder token packing and image preprocessing.
- Generation controller and output cleanup.

## DinoML Notes

Do not treat VLM captioning as one model. Each selected VLM should compose its Transformers family report with a UI captioning contract for prompt templates, output cleanup, and residency.

## Sources

- `H:/uis/vladmandic/sdnext/modules/caption/vqa.py`
- `H:/uis/vladmandic/sdnext/modules/caption/models_def.py`

