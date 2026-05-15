# Panchovix/stable-diffusion-webui-reForge

## Source

- UI clone: `Panchovix/stable-diffusion-webui-reForge`

## Summary

reForge extends the Forge/WebUI surface with many first-party preprocessor
extensions and additional upscaler/restoration architecture support. It is a
good source for concrete preprocessor names and for memory/tiling behavior in
newer upscaler families.

## Model Support

- Main image/video models: continuation of Forge2 with Flux, Wan 2.2, Qwen
  Image, and broader new-model support.
- Formats/runtime: fp8, GGUF, Nunchaku, CUDA stream/offload, shared-memory
  offload, and low-VRAM SDXL/SD1.5 execution.
- Adapters: LoRA, ControlNet, T2I-Adapter/control model patchers, IP-Adapter,
  InstantID, PhotoMaker, reference/revision/tile preprocessors.
- Hunyuan-DiT support is referenced as an external extension path.
- Classic WebUI model support remains for SD1/SDXL checkpoints, VAEs,
  embeddings, and upscalers.

## Feature Surface

- Forge ControlNet extension and preprocessor registry.
- Explicit preprocessor extensions for tile, revision, reference, inpaint,
  recolor, NormalBae, Marigold depth, IP-Adapter, InstantID, and PhotoMaker.
- Advanced sampling modes for newer model families.
- Multidiffusion/tiled diffusion with ControlNet interaction.
- Additional upscaler model routes such as Compact and GRL.

## Auxiliary Model Families

- ControlNet preprocessors: Canny, tile, tile color-fix, reference, recolor,
  inpaint, LaMa inpaint, NormalBae, Marigold depth, IP-Adapter, InstantID,
  PhotoMaker, CLIP Vision revision.
- Legacy annotators: HED, PiDiNet, MLSD, pose, depth, normal, lineart and
  segmentation-style processors.
- Compact and GRL image restoration/upscalers.
- ControlNet, T2I-Adapter, Control LoRA patchers.

## Packages and Loaders

- Local Forge preprocessor extensions under `extensions-builtin`.
- Comfy/Forge patched control code under `modules_forge` and `ldm_patched`.
- Downloaded annotator weights from Hugging Face URLs in extension code.

## Code Anchors

- `Panchovix/stable-diffusion-webui-reForge/README.md:11`
  states Flux/fp8/GGUF/Wan 2.2/Qwen Image/Nunchaku scope.
- `Panchovix/stable-diffusion-webui-reForge/README.md:96`
  documents low-VRAM SDXL/SD1.5 targets.
- `Panchovix/stable-diffusion-webui-reForge/README.md:104`
  documents CUDA stream model movement.
- `Panchovix/stable-diffusion-webui-reForge/README.md:106`
  documents shared-memory offload.
- `Panchovix/stable-diffusion-webui-reForge/README.md:193`
  references Hunyuan-DiT extension support.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_controlnet/scripts/controlnet.py`
  integrates Forge ControlNet.
- `Panchovix/stable-diffusion-webui-reForge/modules_forge/supported_preprocessor.py:102`
  registers built-in none/canny preprocessors.
- `Panchovix/stable-diffusion-webui-reForge/modules_forge/supported_controlnet.py:37`
  defines `ControlNetPatcher`.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/forge_preprocessor_tile/scripts/preprocessor_tile.py:98`
  registers tile preprocessors.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/forge_preprocessor_revision/scripts/preprocessor_revision.py:91`
  registers CLIP vision revision preprocessors.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/forge_preprocessor_reference/scripts/forge_reference.py:214`
  registers reference preprocessors.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py:106`
  downloads `ControlNetLama.pth`.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/forge_preprocessor_inpaint/scripts/preprocessor_inpaint.py:217`
  registers LaMa inpaint.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/forge_preprocessor_normalbae/scripts/preprocessor_normalbae.py:16`
  defines NormalBae preprocessor.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/forge_preprocessor_marigold/scripts/preprocessor_marigold.py:20`
  tags Marigold as depth.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_ipadapter/scripts/forge_ipadapter.py:88`
  registers IP-Adapter CLIP Vision preprocessors.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_ipadapter/scripts/forge_ipadapter.py:106`
  registers InsightFace for InstantID.
- `Panchovix/stable-diffusion-webui-reForge/extensions-builtin/sd_forge_photomaker/scripts/forge_photomaker.py:23`
  registers PhotoMaker CLIP vision preprocessor.
- `Panchovix/stable-diffusion-webui-reForge/modules/compact_model.py:62`
  covers Compact upscaler route.
- `Panchovix/stable-diffusion-webui-reForge/modules/grl_model.py:62`
  covers GRL upscaler route.

## DinoML Gaps

- Broad new-model/runtime support: Flux, Qwen, Wan, fp8, GGUF, Nunchaku,
  offload policy, and WebUI-compatible components.
- Fine-grained ControlNet preprocessor registry with per-preprocessor params,
  weight download hints, and output artifact typing.
- IP/identity adapter preprocessing that combines CLIP Vision and face
  embedding/detection inputs.
- Tiled diffusion/control interaction contracts.

## Further Exploration Additions

- PixArt Sigma and Stable Cascade are listed as supported extension paths.
  Anchors: `Panchovix/stable-diffusion-webui-reForge/README.md:194`,
  `Panchovix/stable-diffusion-webui-reForge/README.md:195`.
- CFG++ sampler support should be tracked as a sampler/runtime feature.
  Anchors: `Panchovix/stable-diffusion-webui-reForge/README.md:18`,
  `Panchovix/stable-diffusion-webui-reForge/modules_forge/forge_alter_samplers.py:224`.
- Runtime dtype knobs include UNet fp8 e4m3/e5m2, VAE bf16/fp16/fp32, and CLIP
  fp8/fp16/fp32.
  Anchor: `Panchovix/stable-diffusion-webui-reForge/README.md:150`.
- Sage/Flash attention support is exposed as runtime acceleration.
  Anchors: `Panchovix/stable-diffusion-webui-reForge/README.md:113`,
  `Panchovix/stable-diffusion-webui-reForge/ldm_patched/ldm/modules/attention.py:688`.
