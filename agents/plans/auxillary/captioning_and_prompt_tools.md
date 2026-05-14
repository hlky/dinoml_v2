# Captioning and Prompt Tools

## Why This Matters

Image generation UIs commonly include "interrogate" and tagging workflows that
convert an image into a caption, prompt hints, or tag lists. These are not
generation pipelines, but they matter for a complete optimized UI.

## Model Families

- BLIP captioning, sometimes from a cloned source tree rather than a
  transformers pipeline.
- OpenCLIP and CLIP-interrogator category scoring.
- DeepDanbooru custom torch tagger.
- WD taggers and VLM captioners in SD.Next-style caption stacks.

## Code Anchors

- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/launch_utils.py:352`
  BLIP repository install reference.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/interrogate.py:45`
  `InterrogateModels`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/interrogate.py:90`
  BLIP model load.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/interrogate.py:101`
  `models.blip.blip_decoder`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru.py:12`
  DeepDanbooru wrapper.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru.py:27`
  custom `DeepDanbooruModel`.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru_model.py:10`
  DeepDanbooru architecture.
- `H:/uis/AUTOMATIC1111/stable-diffusion-webui/modules/deepbooru_model.py:674`
  state dict load path.
- `H:/uis/vladmandic/sdnext/modules/caption/caption.py:16`
  OpenCLIP/BLIP caption stack.
- `H:/uis/vladmandic/sdnext/modules/caption/caption.py:26`
  tagger route.
- `H:/uis/vladmandic/sdnext/modules/caption/caption.py:43`
  VLM route.
- `H:/uis/vladmandic/sdnext/modules/api/caption.py:6`
  caption API names.
- `H:/uis/vladmandic/sdnext/installer.py:1228`
  `clip_interrogator==0.6.0` install.

## DinoML Gap

Moderate to high. BLIP/VLM pieces may overlap transformers, but the UI surface
also needs category scoring, tag thresholds, ranking, prompt assembly, model
residency, and image preprocessing. DeepDanbooru/WD taggers are a clearer
non-transformers auxiliary gap.

