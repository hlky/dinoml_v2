# lllyasviel/stable-diffusion-webui-forge

## Source

- UI clone: `H:/uis/lllyasviel/stable-diffusion-webui-forge`

## Summary

Forge is the most important WebUI-family source for bundled ControlNet
preprocessors and low-memory/runtime variants. Its auxiliary value is less in
classic extras and more in the preprocessor registry, ControlNet patchers, and
backend/runtime additions around GGUF and memory policy.

## Model Support

- Main image models: Stable Diffusion WebUI-compatible checkpoints plus Flux.
- Formats/runtime: native Flux BNB NF4 and GGUF Q8_0/Q5_0/Q5_1/Q4_0/Q4_1 with
  GPU weight/offload controls.
- Adapters: LoRA, ControlNet, IP-Adapter, and Flux low-bit LoRA support.
- UI/API: txt2img/img2img endpoints remain WebUI-compatible, with Flux support
  called out as a separate path.
- Core emphasis: low-VRAM model residency, async swap/offload, and experimental
  model support.

## Feature Surface

- Built-in Forge ControlNet integration.
- Built-in legacy preprocessor suite for common ControlNet conditions.
- IP-Adapter, reference, tile, inpaint, recolor, normal/depth and similar
  preprocessor extensions.
- Control model patcher abstractions for ControlNet/T2I/Control LoRA-like
  models.
- Backend runtime and low-VRAM optimizations inherited from Comfy-style code.
- GGUF package code in third-party packages.

## Auxiliary Model Families

- Canny, HED, PiDiNet, MLSD, lineart, scribble, depth, normal, pose/openpose,
  segmentation and inpaint preprocessors through `forge_legacy_preprocessors`.
- LaMa inpaint preprocessor through Forge inpaint extension.
- CLIP Vision/IP-Adapter and InstantID-style preprocessors.
- Tile/recolor/reference conditioning preprocessors.
- ControlNet, T2I-Adapter, Control LoRA patchers.
- GGUF runtime/format support.

## Packages and Loaders

- Local bundled annotator/preprocessor code under `extensions-builtin`.
- Comfy-derived backend/control code under `backend`, `modules_forge`, and
  related patched modules.
- Bundled `gguf` package under `packages_3rdparty`.

## Code Anchors

- `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:15`
  links Flux BitsAndBytes/NF4/offload tutorial.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:17`
  links Flux separated full models and GGUF tutorial.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:21`
  discusses low-bit LoRA handling.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:35`
  states native Flux BNB NF4/GGUF support and LoRA support.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:75`
  lists LoRA status.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:77`
  lists ControlNet status.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:78`
  lists IP-Adapter status.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py`
  registers legacy preprocessor wrappers.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor_compiled.py`
  contains compiled/dispatch preprocessor definitions.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/annotator/`
  contains local annotator model implementations.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/modules_forge/supported_preprocessor.py`
  defines the preprocessor registration contract.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/modules_forge/supported_controlnet.py`
  defines ControlNet patching/loading behavior.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/modules_forge/shared.py`
  defines ControlNet and preprocessor model directories.
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/packages_3rdparty/gguf/constants.py`
  contains GGUF constants.

## DinoML Gaps

- Low-bit Flux/GGUF model format and residency policy support, plus
  WebUI-compatible adapter behavior.
- Preprocessor registry/admission model separate from ControlNet execution.
- Image-to-condition artifacts for each preprocessor family.
- Runtime support for low-memory control model patching and GGUF-style formats.

## Further Exploration Additions

- LayerDiffuse transparent image editing is called out as an integrated feature.
  Anchor: `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:29`.
- FreeU V2 is integrated as a runtime/generation modifier.
  Anchors: `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:98`,
  `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/sd_forge_freeu/scripts/forge_freeu.py:65`.
- Runtime formats include `bnb-fp4` and torch float8 `e4m3`/`e5m2`, in addition
  to NF4/GGUF.
  Anchor: `H:/uis/lllyasviel/stable-diffusion-webui-forge/modules_forge/main_entry.py:29`.
- Union ControlNets and OFT LoRAs should be documented with their explicit
  status caveats rather than assumed as fully supported.
  Anchors: `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:89`,
  `H:/uis/lllyasviel/stable-diffusion-webui-forge/README.md:92`.
