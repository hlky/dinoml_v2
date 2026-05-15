# LykosAI/StabilityMatrix

## Source

- UI clone: `LykosAI/StabilityMatrix`

## Summary

StabilityMatrix is not primarily a model runtime; it is a desktop manager and
workflow/package orchestrator. Its value for DinoML is the UI/product surface:
remote model catalogs, Comfy API node builders, downloadable upscaler catalogs,
package management, and workflow construction for common Comfy operations such
as FaceDetailer.

## Model Support

- Package/runtime support: A1111, Forge, reForge, SD.Next, Fooocus variants,
  ComfyUI, InvokeAI, SwarmUI, Wan2GP, FluxGym, CogVideo/CogStudio and related
  packages.
- Main model families exposed through package/catalog metadata: Stable
  Diffusion, SDXL variants, Flux, CogVideoX, Hunyuan, Hunyuan Video, Wan Video,
  and package-specific model surfaces.
- Shared model folders: StableDiffusion, VAE, ApproxVAE, Embeddings,
  ControlNet, T2IAdapter, IP-Adapter and package-specific mappings.
- Catalog/search: Civitai/Lykos base model types, remote model discovery,
  model index service, download/cache behavior, and generated Comfy workflows.
- Training/utility: FluxGym package for Flux LoRA training.

## Feature Surface

- Desktop package manager for multiple UIs and model folders.
- Remote model catalog helpers.
- Comfy API models and node builder surfaces.
- Upscaler catalog and Comfy upscaler API model.
- FaceDetailer workflow construction.
- Model/download management and UI integration.

## Auxiliary Model Families

- Comfy upscalers exposed through StabilityMatrix API models.
- FaceDetailer workflow components, including detector/detailer model nodes.
- Remote catalog entries for downloadable upscaler models.
- Indirect support for Comfy auxiliary surfaces through workflow builders.

## Packages and Loaders

- C# desktop application; it mostly manages external runtimes and model assets
  rather than directly running PyTorch models.
- Comfy API surfaces are represented as C# model/node builder types.

## Code Anchors

- `LykosAI/StabilityMatrix/README.md:54`
  lists WebUI package support.
- `LykosAI/StabilityMatrix/README.md:63`
  lists FluxGym package support.
- `LykosAI/StabilityMatrix/README.md:64`
  lists CogVideo via CogStudio.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/CivitBaseModelType.cs:14`
  lists CogVideoX base model type.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/CivitBaseModelType.cs:17`
  lists Flux base model types.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/CivitBaseModelType.cs:29`
  lists Hunyuan Video.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/CivitBaseModelType.cs:107`
  lists SDXL base model variants.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/CivitBaseModelType.cs:137`
  lists Wan Video.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/ComfyUI.cs:118`
  maps ControlNet/T2IAdapter shared folders for ComfyUI.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/Wan2GP.cs:16`
  describes Wan2GP package support.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/Wan2GP.cs:17`
  lists Wan/Qwen/Hunyuan Video/LTX Video/Flux support.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/FluxGym.cs:36`
  describes Flux LoRA training UI.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/ComfyUpscaler.cs:9`
  defines Comfy upscaler API model.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs:22`
  defines remote downloadable model catalog behavior.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/Nodes/ComfyNodeBuilder.cs:1160`
  builds Comfy FaceDetailer workflow nodes.
- `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/`
  contains package metadata/management types.
- `LykosAI/StabilityMatrix/StabilityMatrix.Avalonia/`
  contains desktop UI integration.

## DinoML Gaps

- Cross-UI model/package catalog support for all model categories and shared
  folder mappings.
- Distribution/catalog contract for auxiliary models and generated workflows.
- Desktop manager API expectations: install, update, locate, and wire models
  into backend runtimes.
- Workflow-level feature packaging, especially detailer/upscaler workflows that
  combine several auxiliary models.

## Further Exploration Additions

- Shared folder taxonomy is broader than first pass: Ultralytics, SAMs, and
  PromptExpansion, plus Invoke-specific IP-Adapter and CLIP Vision splits.
  Anchors: `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/SharedFolderType.cs:43`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/ComfyUI.cs:160`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/SharedFolders.cs:27`.
- Remote catalog includes ControlNet, prompt expansion, Ultralytics, SAM, CLIP,
  and CLIP Vision models, not only upscalers.
  Anchors: `LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs:173`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs:268`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs:392`.
- Prompt expansion is available through generated API types and Comfy node
  builder/node map.
  Anchors:
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Api/PromptGen/Generated/Refitter.g.cs:33`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/Nodes/ComfyNodeBuilder.cs:619`.
- Ultralytics/SAM integration appears through downloadable catalogs and Comfy
  node builder support.
  Anchors:
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Api/Comfy/Nodes/ComfyNodeBuilder.cs:698`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Helper/RemoteModels.cs:268`.
- Installable runtime/provider helpers include SageAttention and Nunchaku for
  Comfy packages.
  Anchors: `LykosAI/StabilityMatrix/StabilityMatrix.Core/Services/IPipWheelService.cs:28`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/PackageModification/InstallSageAttentionStep.cs:12`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/PackageModification/InstallNunchakuStep.cs:11`.
- Extension management is a package-level product surface, including Comfy
  extension manager and Git-backed install/update APIs.
  Anchors: `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/BasePackage.cs:278`,
  `LykosAI/StabilityMatrix/StabilityMatrix.Core/Models/Packages/Extensions/GitPackageExtensionManager.cs:14`,
  `LykosAI/StabilityMatrix/README.md:65`.
