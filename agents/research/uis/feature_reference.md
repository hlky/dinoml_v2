# Cross-UI Feature Reference

This reference compiles the feature/model support surfaced in the per-UI
reviews under `agents/research/uis/*/review.md`.

UI keys:

- `A1111`: `AUTOMATIC1111/stable-diffusion-webui`
- `Comfy`: `Comfy-Org/ComfyUI`
- `SDNext`: `vladmandic/sdnext`
- `Invoke`: `invoke-ai/InvokeAI`
- `Forge`: `lllyasviel/stable-diffusion-webui-forge`
- `reForge`: `Panchovix/stable-diffusion-webui-reForge`
- `Classic`: `Haoming02/sd-webui-forge-classic`
- `Wan2GP`: `deepbeepmeep/Wan2GP`
- `Swarm`: `mcmonkeyprojects/SwarmUI`
- `SM`: `LykosAI/StabilityMatrix`

## Main Image Models

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Stable Diffusion 1.x / 2.x | A1111, Comfy, SDNext, Forge, reForge, Classic, Swarm, SM | WebUI-compatible checkpoint baseline; Comfy/Swarm/SM expose via folders/catalogs. |
| SDXL / advanced SDXL | A1111, Comfy, SDNext, Invoke, Forge, reForge, Classic, Swarm, SM | Includes SDXL Turbo/Lightning/Hyper variants where cataloged. |
| SD3 / SD3.5 | Comfy, SDNext, Classic | Comfy README lists SD3/3.5; SDNext has SD3 pipeline wrapper; Classic README lists SD3. |
| Flux / Flux.1 | Comfy, SDNext, Invoke, Forge, reForge, Classic, Wan2GP, Swarm, SM | Forge/reForge emphasize low-bit Flux and offload; Invoke lists Flux.1 variants. |
| Flux Kontext / Fill / Redux / Krea | Comfy, Invoke, Classic | Invoke lists Kontext/Krea/Redux/Fill; Comfy lists Kontext; Classic lists Kontext. |
| Flux 2 / Flux.2 Klein | Comfy, Invoke, Classic | Comfy lists Flux 2; Invoke and Classic list Flux.2 Klein. |
| Qwen Image / Qwen Image Edit | Comfy, Invoke, reForge, Classic, Wan2GP, Swarm, SM | Also includes Qwen Image VAE handling in Invoke/Classic. |
| Z-Image | Invoke, Classic, Wan2GP, Swarm | Invoke lists Turbo/Base and Z-Image ControlNet; Classic/Wan2GP/Swarm list support. |
| AuraFlow | Comfy | Listed in Comfy model examples. |
| HunyuanDiT / Hunyuan Image | Comfy, reForge, SM | Comfy lists HunyuanDiT and Hunyuan Image 2.1; reForge references Hunyuan-DiT extension; SM catalogs Hunyuan. |
| Kandinsky | Wan2GP | Wan2GP README lists Kandinsky. |
| HiDream / HiDreamO1 | SDNext, Wan2GP | SDNext has HiDream-family pipeline wrapper; Wan2GP lists HiDreamO1. |
| Segmind SSD-1B | A1111 | A1111 README lists Segmind Stable Diffusion support. |
| Closed/API-hosted image models | Comfy | Comfy API nodes list closed-source/API model access. |

## Main Video Models

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Stable Video Diffusion | Comfy, Swarm, SM | Comfy lists SVD; Swarm inherits backend support and has SVD VAE setting. |
| Wan / Wan 2.1 / Wan 2.2 | Comfy, SDNext, Invoke, reForge, Classic, Wan2GP, Swarm, SM | Invoke is API-only; Wan2GP is deepest runtime/workflow source. |
| First/last-frame video | Classic | Classic README calls out FirstLastFrameToVideo for Wan 2.2. |
| Hunyuan Video | Comfy, SDNext, Wan2GP, Swarm, SM | SDNext has Hunyuan Video CFG-Zero wrapper. |
| Hunyuan Video 1.5 | Comfy | Comfy README lists Hunyuan Video 1.5. |
| LTX-Video / LTXV / LTX-2 | Comfy, Wan2GP, SM | Wan2GP includes LTX-2 workflows, audio/video, and Ic LoRAs. |
| Mochi | Comfy, Swarm | Comfy lists Mochi; Swarm has Mochi VAE defaults. |
| CogVideo / CogVideoX | SM | StabilityMatrix packages CogVideo via CogStudio and catalogs CogVideoX. |
| Cosmos/world models | Wan2GP | Wan2GP related projects include Cosmos/world generation. |
| Vista4D | Wan2GP | Wan2GP lists Vista4D video reshooting. |
| Video generation API/queueing | Wan2GP, Swarm, SM | Wan2GP exposes API queueing; Swarm/SM orchestrate backend/package execution. |

## Audio And TTS Models

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Stable Audio | Comfy | Comfy README lists Stable Audio. |
| ACE-Step | Swarm | Swarm docs list ACE-Step native audio model support. |
| MMAudio / video-to-audio | Wan2GP | Wan2GP integrates MMAudio and LTX video-to-audio. |
| Qwen3 TTS | Wan2GP | Listed in Wan2GP README. |
| Chatterbox | Wan2GP | Listed in Wan2GP README. |
| HearMula | Wan2GP | Listed in Wan2GP README. |
| Omnivoice | Wan2GP | TTS with voice cloning/dialogue mode. |
| ScenemeAI | Wan2GP | LTX-derived TTS/dialogue workflow. |
| IndexTTS2 | Wan2GP | Local pipeline and BigVGAN asset requirements. |
| BigVGAN / BigVGAN-v2 vocoder | Wan2GP | Used by MMAudio/IndexTTS paths. |
| Wav2Vec2-style semantic encoders | Wan2GP | Referenced in TTS codec/semantic features. |
| Whisper audio encoder | Comfy | Local Whisper encoder implementation. |
| Audio serving/settings | Swarm | Web audio route and audio settings exist around generated output. |

## Adapters, Conditioning, And Finetunes

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| LoRA | A1111, Comfy, SDNext, Invoke, Forge, reForge, Classic, Wan2GP, Swarm, SM | Nearly universal; SM catalogs/folder maps; Wan2GP includes system LoRAs. |
| Low-bit Flux LoRA | Forge, Classic | Forge discusses low-bit LoRA behavior; Classic notes Nunchaku support limits. |
| LTX Ic LoRA / system LoRA | Wan2GP | Used for LTX control video, outpaint/refocus/ungrade/uncompress, and system behaviors. |
| Textual inversion / embeddings | A1111, Comfy, Invoke, reForge, Classic, Swarm, SM | Explicit in A1111/Invoke/Swarm/SM; WebUI-family support inherited elsewhere. |
| Hypernetworks | A1111 | A1111 WebUI component support. |
| ControlNet | Comfy, SDNext, Invoke, Forge, reForge, Classic, Wan2GP, Swarm, SM | Forge/reForge richest bundled preprocessor/control registry. |
| T2I-Adapter | Comfy, SDNext, Invoke, Forge, reForge, Swarm, SM | Explicit in Comfy/Invoke/SM; Forge family patchers include adapter path. |
| Control LoRA | Invoke, Forge, reForge | Explicit in Invoke configs and Forge-family patchers. |
| ControlNet-XS | SDNext | SDNext control unit support. |
| LLLite | SDNext, Classic | SDNext control units; Classic README references LLLite/Union ControlNet. |
| IP-Adapter | Comfy, SDNext, Invoke, Forge, reForge, Swarm, SM | Forge/reForge include CLIP Vision preprocessing routes. |
| InstantID / InsightFace conditioning | reForge | reForge IP-Adapter extension registers InsightFace for InstantID. |
| PhotoMaker | reForge | reForge registers PhotoMaker CLIP vision preprocessor. |
| Reference/revision conditioning | reForge, Forge | reForge has explicit reference/revision preprocessors. |
| Style models | Comfy | Comfy folder/node support. |
| GLIGEN | Comfy | Comfy folder/node support. |
| VACE ControlNet | Wan2GP | Wan2GP links VACE docs and workflows. |

## Model Components And Encoders

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| VAE selection / standalone VAE | A1111, Comfy, SDNext, Invoke, reForge, Classic, Swarm, SM | Core component across UIs. |
| Approx VAE / VAE preview | Comfy, SM | Comfy TAESD/TAEHV; SM shared ApproxVAE folders. |
| TAESD / TAEHV | Comfy | Local latent preview implementation. |
| Flux.2 small decoder | Classic | README lists Flux.2 small decoder support. |
| Qwen2D VAE / Qwen Image VAE | Invoke, Classic | Invoke config identifies Qwen Image VAE; Classic lists Qwen2D VAE. |
| Flux.2 VAE | Invoke, Classic | Invoke config identifies Flux.2 VAE; Classic lists decoder/VAE support. |
| Wan VAE handling | Classic, Wan2GP | Classic notes Wan VAE memory behavior; Wan2GP manages video VAEs. |
| CLIP/text encoders | Comfy, SDNext, Invoke, Swarm, SM | Comfy/Invoke expose model component taxonomy; Swarm/SM catalog. |
| T5 encoder | Invoke | Dedicated model-manager config. |
| CLIP Vision | Comfy, Invoke, Forge, reForge, Swarm, SM | Used for IP/style/revision/vision conditioning. |
| SigLIP | Comfy, Invoke | Comfy CLIP vision configs; Invoke config. |
| Qwen VL encoder | Invoke, Wan2GP | Invoke has Qwen VL encoder config; Wan2GP uses Qwen3VL-style Deepy helpers. |
| Deepy agent/VLM helper | Wan2GP | Prompt/tool assistant for image/video/audio generation and inspection. |

## Image Upscaling And Restoration

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Extras upscaling/postprocess UI | A1111, SDNext, Classic | WebUI-family extras surface. |
| SD upscale / tiled upscale script | A1111, Classic | Scripted generation-time upscaling. |
| Spandrel upscalers | A1111, Comfy, SDNext, Invoke, Swarm | Invoke models Spandrel as typed model; Comfy/A1111/SDNext use loaders. |
| ESRGAN/RRDB | A1111, SDNext, Invoke, Classic | Legacy and Spandrel-backed paths. |
| RealESRGAN/SRVGG | A1111, SDNext, Invoke, Classic | Included in extras/upscale invocations. |
| SwinIR | A1111, SDNext | WebUI extension/core postprocess paths. |
| ScuNET | A1111, SDNext | WebUI extension/core postprocess paths. |
| DAT | A1111, Classic | A1111 DAT loader; Classic upscaler utilities. |
| HAT | A1111 | A1111 HAT loader. |
| ATD / DRCT | Classic | Classic upscaler utilities. |
| Compact / GRL | reForge | reForge model routes. |
| AuraSR | SDNext | SDNext postprocess route. |
| SeedVR / SeedVR2 | SDNext | SDNext postprocess route. |
| FlashVSR | Wan2GP | Video super-resolution/postprocess. |
| Remote upscaler catalog | SM | StabilityMatrix remote model catalog and Comfy upscaler API. |

## Face, Detection, Segmentation, And Masks

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| GFPGAN | A1111, Classic | Face restoration. |
| CodeFormer | A1111, Classic | Face restoration. |
| facexlib / RetinaFace helpers | A1111, Classic | Face detect/align/paste-back helpers. |
| FaceDetailer workflow | SM, SDNext | SM builds Comfy FaceDetailer workflow; SDNext has detailer API tests. |
| YOLO / YOLOv8 detailer/detection | SDNext, Swarm | SDNext detailer API; Swarm uses ultralytics. |
| SAM / SAM2 | Invoke | Segment Anything invocation. |
| SAM3 / SAM3.1 | Comfy, Wan2GP | Comfy SAM3 detection/blueprints; Wan2GP Magic Mask/video masks. |
| GroundingDINO | Invoke | Segment/bbox workflow evidence. |
| RT-DETR | Comfy | Extra node/detection surface noted. |
| rembg | Swarm | Optional package and background-removal support. |
| BiRefNet background removal | Comfy | Model and blueprint support. |
| Mask editor / mask utilities | Wan2GP, Swarm | Wan2GP mask editor; Swarm mask nodes. |
| Video masks / temporal masks | Comfy, Wan2GP | Comfy SAM3 video blueprint; Wan2GP temporally consistent masks. |

## Control Preprocessors And Image-To-Condition

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Canny | Invoke, Forge, reForge, Swarm | Built-in/preprocessor packages. |
| HED | SDNext, Invoke, Forge, reForge | Edge preprocessor. |
| PiDiNet | Invoke, Forge, reForge | Edge preprocessor. |
| MLSD | Invoke, Forge, reForge | Line segment preprocessor. |
| Lineart / scribble | Invoke, Forge, reForge | Control preprocessor family. |
| OpenPose / DWPose / pose | Invoke, Forge, reForge, Wan2GP | Invoke DWPose; Wan2GP pose extraction/alignment. |
| Depth / Depth Anything / DPT / GLPN / LeReS | SDNext, Invoke, Forge, reForge, Wan2GP | Depth extractors/preprocessors. |
| Marigold depth | reForge | Explicit reForge preprocessor. |
| NormalBae / normal maps | Invoke, Forge, reForge | Normal preprocessor family. |
| OneFormer / semantic segmentation | SDNext, Forge, reForge | SDNext processor list and Forge legacy suite. |
| LaMa inpaint preprocessor | Forge, reForge | Forge/reForge inpaint extension. |
| Tile / tile color-fix | Forge, reForge | Tile conditioning preprocessors. |
| Recolor | Forge, reForge | Recolor conditioning preprocessors. |
| Reference / revision | Forge, reForge | Reference/revision conditioning preprocessors. |
| PBR maps | Invoke | PBR map invocation. |
| Z-Image ControlNet modes | Invoke | Canny/HED/depth/pose/MLSD spatial controls. |
| Control video injection | Wan2GP | LTX/Wan-style video control workflow. |

## Captioning, Tagging, Prompting, And Agent Tools

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| BLIP interrogation/captioning | A1111, SDNext | A1111 interrogate; SDNext caption APIs. |
| CLIP interrogation / OpenCLIP | A1111, SDNext | Prompt/caption helpers. |
| DeepDanbooru | A1111 | Tagging model. |
| JoyTag/taggers/VLM captioning | SDNext | Caption module includes tagger/VLM-style support. |
| Prompt enhancer | Wan2GP | Integrated prompt enhancer. |
| Deepy agent | Wan2GP | Generates/inspects/transcribes image/video/audio workflows. |
| Model metadata usage hints/triggers | Swarm | Catalog-level model metadata. |

## Runtime, Formats, Providers, And Kernels

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| `.ckpt` / `.safetensors` | A1111, Comfy, Invoke, Classic, Swarm | Common checkpoint formats. |
| Diffusers format | Comfy, SDNext, Invoke | Explicit in Invoke and SDNext; Comfy component folders. |
| GGUF | Forge, reForge, Invoke, Wan2GP, Swarm, SM | Forge/Wan2GP runtime emphasis; Swarm installable feature; SM catalog support. |
| BNB NF4 | Forge | Flux low-bit support. |
| fp8 / scaled fp8 | reForge, Wan2GP | reForge README and Wan2GP qtype handlers. |
| NVFP4 | Comfy, Wan2GP | Comfy tensor layout; Wan2GP qtype handler. |
| Nunchaku / SVDQ | reForge, Classic, Wan2GP | New model/runtime support. |
| int8 / optimum-quanto | Wan2GP | qtype/kernel support. |
| TensorRT | Swarm | Installable TensorRT support and creation permission. |
| Triton kernels | Wan2GP | Kernel installation/runtime path. |
| SageAttention / SpargeAttention | Wan2GP | Sparse attention/runtime kernels. |
| CUDA stream/offload | reForge | Model movement/offload flags. |
| Shared GPU memory offload | reForge | Pin shared memory flag. |
| mmgp offload/profile/quant router | Wan2GP | Core low-VRAM runtime. |
| Model compile hooks | SDNext | Compile hooks and upscaler compile paths. |
| Backend orchestration/autoscaling | Swarm, SM | Swarm backend manager; SM package manager. |

## UI/Product Features Around Models

| Feature | UIs | Notes |
| --- | --- | --- |
| Model downloader/catalog | Swarm, SM, Invoke | Swarm downloader/metadata; SM remote catalog; Invoke starter models. |
| Multiple model roots/shared folders | Swarm, SM, Comfy | Swarm multiple roots; SM shared folder mappings; Comfy folder taxonomy. |
| Model metadata database | Swarm, SM | Model class, special format, preview, usage hints. |
| Package manager for UIs | SM | Installs/updates multiple UIs and maps model folders. |
| Comfy workflow generation/building | Swarm, SM | Swarm auto-workflow; SM Comfy node builder. |
| API routes for generation/models | A1111, Invoke, Wan2GP, Swarm | WebUI API, typed invocations, queue generation, model APIs. |
| Checkpoint merge | A1111 | A1111 extras/model utility. |
| Pickle-to-safetensors conversion | Swarm | Utility route/tool. |
| Training utility | SM | FluxGym Flux LoRA training package. |

## Further Exploration Addendum

The following additions came from read-only subagent follow-up passes. They
should be treated as part of the feature inventory even when they refine a
broader row above.

### Additional Main Image Families

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Stable Cascade | SDNext, reForge | SDNext APG/Stable Cascade pipeline; reForge extension path. |
| PixArt Sigma | reForge | Listed as supported extension path. |
| Lumina-Image-2.0 / Neta-Lumina / NetaYume-Lumina | Classic | Supported presets/families. |
| Cosmos Predict2 | Comfy, Invoke | Comfy supported models and Invoke taxonomy. |
| Anima | Comfy, Invoke | Invoke has distinct invocation suite; Comfy supported model family. |
| CogView4 | Invoke | Loader, text encoder, denoise, and image latent invocations. |
| Chroma / Radiance | Comfy, Swarm, Wan2GP | Comfy/Swarm model support; Wan2GP Flux Chroma/Radiance defaults. |
| OmniGen / OmniGen2 | Comfy, Swarm | Comfy supported model family and Swarm workflow generator. |
| Ernie Image | Comfy, Swarm | Supported model/workflow family. |
| LongCat Image | Comfy, Wan2GP | Comfy supported model; Wan2GP LongCat handlers. |
| Flux Chroma/Radiance/SRPO/UMO/USO/DreamOmni2 | Wan2GP | Wan2GP-specific Flux variants. |
| Kandinsky 5 Lite/Pro T2V/I2V | Comfy, Wan2GP | Refines the earlier generic Kandinsky row. |

### Additional Main Video Families

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| AnimateDiff / AnimateFace-style video | SDNext | Script-level video feature. |
| CogVideoX T2V/I2V/Inpaint | Comfy, SM | Comfy supported models; SM CogStudio package/catalog. |
| Cosmos T2V/I2V | Comfy, Swarm, Wan2GP | Comfy supported models; Swarm workflow generator; Wan2GP related projects. |
| Wan derived families | Wan2GP | Alpha/RGBA, Animate, Chrono Edit, Fun InP, Phantom, ReCamMaster, Stand-In, Lynx, Ditto, Multitalk/Infinitalk/Fantasy talking, Ovi, MoCha, SCAIL, SteadyDancer, WanMove, SVI2Pro, Lucy/Kiwi edit. |
| Wan Phantom | Swarm, Wan2GP | Swarm workflow generator and Wan2GP Wan-family defaults. |
| Hunyuan Video 1.5 variants | Comfy, Swarm, Wan2GP | Includes t2v/i2v 480p/720p, Lightx2v, SR/upsampler, avatar, custom-audio, custom-edit. |
| LTX 2.x synchronized audio/video | Comfy, Swarm, Wan2GP | Comfy blueprints; Swarm workflow generator; Wan2GP LTX-2 control/audio surface. |
| Magi Human | Wan2GP | Base/distill and SR 1080p talking-head pipeline. |
| LongCat video/avatar | Wan2GP | LongCat handler family. |

### Additional Audio And TTS Features

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| ACE-Step v1/v1.5/v1.5 XL | Wan2GP, Swarm, Comfy | Wan2GP defaults, Swarm controls, Comfy support. |
| KugelAudio | Wan2GP | TTS/audio handler. |
| YuE | Wan2GP | Music/audio handler. |
| HeartMuLa RL | Wan2GP | TTS/audio model default. |
| Latent audio and audio VAE encode/decode | Comfy, Swarm, Wan2GP | Comfy audio nodes; Swarm LTX-2 audio VAE catalog; Wan2GP LTX-2 audio VAE. |
| Audio save/load/preview/record/trim | Comfy | FLAC/MP3/Opus save plus audio utility nodes. |
| Ovi/MultiTalk/Infinitalk talking-video audio paths | Wan2GP | Refines the generic MMAudio/video-to-audio row. |
| ACE-Step generation controls | Swarm | Style, BPM, time signature, language, key scale. |

### Additional Adapters And Control

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| LyCORIS / OFT / BOFT LoRA variants | A1111 | Built-in LoRA extension has OFT/BOFT-style network support. |
| OFT LoRA caveat | Forge | Explicit status caveat in Forge README. |
| Union ControlNet | Classic; caveat in Forge | Classic supports/mentions Union ControlNet; Forge README marks Union ControlNet not implemented. |
| LayerDiffuse transparent image editing | Forge | Integrated transparent image editing feature. |
| FreeU V2 | Forge | Integrated runtime/generation modifier. |
| LTX-2 ID-LoRA / union control / HDR IC-LoRA | Wan2GP | Part of LTX-2 control/audio workflows. |
| MatAnyone | Wan2GP | Video mask creator/preprocess stack. |
| Magic Mask negative masks | Wan2GP | Magic Mask feature refinement. |
| Depth Anything v3 / NLFPose / SCAIL | Wan2GP | Preprocess/control extraction stack. |
| PhotoMaker | Comfy, reForge | Comfy folder taxonomy and reForge preprocessor. |
| Prompt expansion models | Invoke, SM | Invoke TextLLM/prompt expansion; SM prompt expansion catalog/node support. |

### Additional Components, Catalogs, And APIs

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Hypernetworks | A1111, Comfy, Classic | A1111/Classic WebUI support; Comfy folder taxonomy. |
| Classifiers | Comfy | Model folder taxonomy. |
| Model patches | Comfy, Swarm | Comfy folder taxonomy; Swarm model roots include patches/control folder. |
| Custom nodes / extensions | Comfy, Swarm, SM | Comfy taxonomy/API, Swarm installable features, SM extension manager. |
| Anima 3D latent/image-latent flow | Invoke | Anima invocation suite includes 3D latent shape and conversion. |
| Qwen2.5-VL / Qwen VL encoder | Invoke, Wan2GP | Invoke standalone/quantized encoder; Wan2GP Deepy Qwen VL usage. |
| Qwen3Encoder | Invoke | Component taxonomy. |
| TextLLM | Invoke | Model type, loader, invocation, route, starter models. |
| CLIP-L/G, T5-XXL, LLaVA, LLaMA, Qwen LLM, Mistral LLM text encoders | Swarm | T2I text encoder controls across modern families. |
| Media upload and job previews for image/video/audio/3D/text | Comfy | API/product surface. |
| First-class model discovery/object info/jobs/prompts API | Comfy | Server API endpoints. |
| Rich WebSocket/API surface | Swarm | Admin/backend/util/models/T2I/basic/grid/Comfy/image-batch APIs. |
| Remote ControlNet/SAM/Ultralytics/CLIP/CLIP Vision catalog | SM | Remote model catalog beyond upscalers. |
| Shared folders for Ultralytics/SAMs/PromptExpansion | SM | Broader shared-folder taxonomy. |
| Plugin catalog and manager | Wan2GP, SM | Wan2GP plugin ecosystem; SM Git-backed extension manager. |

### Additional Upscale, Restoration, And Video Postprocess

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Swin2SR | A1111 | Upstream-noted upscaler. |
| LDSR | A1111 | Diffusion upscaler extension. |
| Process Full Video plugin | Wan2GP | Chunked processing, resume metadata, outpaint, detailer, HDR conversion, refocus, ungrade, uncompress, FlashVSR. |
| Hunyuan 1.5 upsampler | Wan2GP | Dedicated upsampler variants. |
| Hunyuan Video 1.5 SR | Swarm, Wan2GP | Swarm workflow and Wan2GP defaults. |

### Additional Runtime, Formats, And Kernels

| Feature/model family | UIs | Notes |
| --- | --- | --- |
| Hypertile U-Net/VAE attention tiling | A1111 | Runtime/product feature. |
| CFG++ sampler family | reForge | Sampler/runtime feature. |
| bnb-fp4 | Forge | Low-bit runtime option. |
| torch float8 e4m3/e5m2 | Forge, reForge, Comfy | Runtime dtype support. |
| fp8 e8m0 / scaled fp8 / fp4 machinery | Comfy, Classic, Wan2GP | Comfy runtime formats; Classic/Wan2GP qtypes. |
| fp4mixed / fp8mixed / mxfp8 / nvfp4 / fp8_scaled | Classic | Runtime quant formats. |
| int8 Triton matmul | Classic | Runtime acceleration. |
| SageAttention / FlashAttention | reForge, Classic, Wan2GP, SM | Runtime/provider acceleration; SM install helpers. |
| Sage3 / radial attention | Wan2GP | Sparse/radial attention refinement. |
| Nunchaku attention/offload | SDNext, SM, Classic, Wan2GP, reForge | Runtime/provider support. |
| Nunchaku SVDQ/AWQ kernels | Wan2GP, Classic | Kernel/qtype support. |
| GGUF llama.cpp CUDA linear/embedding fast paths | Wan2GP | GGUF implementation detail. |
| GGUF qtypes Q2_K through Q8_0 | Wan2GP | More precise qtype range. |
| BNB int8 / BNB NF4 | Invoke, Forge | Invoke taxonomy; Forge Flux runtime. |
| ONNX model format/component | Invoke | Model taxonomy includes ONNX. |
| MPS early support | Wan2GP | Platform/runtime support. |
| torch compile hooks | Wan2GP, SDNext | Runtime optimization hooks. |
| TensorRT exact engine families | Swarm | SDXL 0.9/1.0, SD3 Medium, SDXL Turbo, SDXL Refiner, SVD. |

## Per-UI Missing Features And Compatibility Notes

This section is a second-pass compatibility layer over the feature lists above.
It distinguishes native support from support that is inherited through a
backend, extension, package manager, model catalog, or API-only path.

Compatibility modes:

- `native`: implemented directly in the UI/runtime.
- `extension`: present through built-in extensions, optional extensions, custom
  nodes, or external extension paths.
- `backend`: inherited from another execution backend, usually ComfyUI.
- `catalog`: visible through model/package catalogs or folder mapping, not
  necessarily executed by that app itself.
- `api-only`: exposed as an API path, not a full local UI/runtime feature.
- `partial`: supported with constraints, caveats, or narrower model coverage.

### A1111

| Area | Compatibility / missing notes |
| --- | --- |
| Main models | Native WebUI checkpoint flow covers SD 1.x/2.x, SDXL-class checkpoints, SSD-1B, ckpt/safetensors, VAE, textual inversion, hypernetworks, and LoRA. Modern families such as SD3/Flux/Qwen/Z-Image/Wan/Hunyuan/LTX are missing from core and would be extension-dependent. |
| Video/audio | No meaningful first-party video or audio model runtime in core. Treat video/audio support as extension territory. |
| Control/preprocessors | Core does not ship the full ControlNet preprocessor suite; ControlNet and related preprocessors are extension compatibility targets. |
| Upscaling/restoration | Strong native extras coverage. Missing from the first matrix but now tracked: Swin2SR, LDSR, Hypertile. |
| Runtime formats/providers | No native GGUF/Nunchaku/fp8/TensorRT-style provider story in core. WebUI compatibility is mostly PyTorch checkpoint-oriented. |
| API/product | WebUI API is broad for txt2img/img2img/extras, but not a typed model taxonomy like Invoke or graph model contract like Comfy. |

### Comfy

| Area | Compatibility / missing notes |
| --- | --- |
| Main models | Broadest native graph/runtime coverage among surveyed UIs, including SD, SDXL, SD3/3.5, Flux/Flux2, Qwen, Z-Image, Cosmos, Anima, Chroma/Radiance, OmniGen, Ernie, CogVideoX, Hunyuan, Wan, LTX, audio, 3D/API nodes, and more. |
| Workflow model | Compatibility is graph/node-based, not WebUI-style form/API compatibility. A DinoML UI using Comfy conventions needs node schemas, model folders, object info, prompt/job APIs, and workflow JSON compatibility. |
| Extensions/custom nodes | Some features are first-class, but many practical workflows depend on custom nodes, blueprints, or API nodes. Treat `custom_nodes`, `model_patches`, `photomaker`, classifiers, and extension APIs as compatibility surfaces. |
| Classic extras | WebUI-style extras, checkpoint merge, and face-restoration UI semantics are not the main native model. Those need separate compatibility if DinoML targets A1111-style clients. |
| Audio/media | Comfy has broad audio/media nodes, but compatibility includes audio/video/3D upload helpers, previews, and save/load formats, not just model execution. |
| Runtime formats/providers | Supports fp8/scaled fp8/NVFP4/fp4 machinery, but feature coverage varies by model/node path. |

### SDNext

| Area | Compatibility / missing notes |
| --- | --- |
| Main models | Native/diffusers-backed coverage includes SDXL/APG, SD3, Flux, Wan, Hunyuan Video, HiDream, Stable Cascade/APG, and AnimateDiff-style video surfaces. Coverage is broad but less exhaustive for the newest Wan/LTX/audio variants than Wan2GP or Comfy. |
| Video/audio | Has video pipeline wrappers and AnimateDiff-style surfaces; audio/TTS is not a first-class area. |
| Control/preprocessors | Strong native control processor/unit system. Needs image-to-condition artifact compatibility, not just ControlNet execution. |
| Captioning/tagging | Broader than a single caption row: OpenCLIP, WaifuDiffusion, DeepDanbooru, and VLM captioning are compatibility surfaces. |
| Runtime formats/providers | Includes compile hooks and Nunchaku attention/offload knobs. It is not a package manager and does not provide Swarm/SM-style cross-backend orchestration. |
| API/product | WebUI-like, not Comfy graph-compatible. DinoML would need separate compatibility if targeting SDNext scripts/API surfaces. |

### Invoke

| Area | Compatibility / missing notes |
| --- | --- |
| Main models | Strong typed taxonomy for SDXL, Flux/Flux2, Z-Image, Qwen Image, CogView4, Anima/Cosmos Predict2, VAEs, LoRAs, ControlNet, T2I-Adapter, Spandrel, TextLLM, Qwen VL, SigLIP, T5, and quantized formats. |
| Video | Wan is API-only; Invoke is not a deep local video-workbench like Wan2GP and does not cover the long-tail Wan/LTX/Hunyuan video variants in the same way. |
| Audio/TTS | TextLLM/prompt expansion is present, but broad TTS/music/video-audio support is not comparable to Wan2GP/Comfy. |
| Formats/providers | Supports ckpt, diffusers, some GGUF, BNB int8/NF4, ONNX and typed configs. It does not expose the broad low-level kernel/qtype matrix found in Wan2GP/Forge-family runtimes. |
| Workflow model | Invocation/schema based, not Comfy graph or WebUI scripts. Compatibility requires model-manager taxonomy, invocation input/output schemas, and artifact types. |
| Control/preprocessors | Strong typed preprocessors and SAM/SAM2/GroundingDINO-style workflows; Z-Image control has GGUF-aware denoise path. |

### Forge

| Area | Compatibility / missing notes |
| --- | --- |
| Main models | Native WebUI-compatible checkpoint flow plus Flux, especially low-bit Flux. Less broad than reForge/Classic for newer Qwen/Wan/Z-Image/Lumina-style families. |
| Explicit caveats | Union ControlNets are explicitly not implemented in the referenced README, and OFT LoRAs have a broken/pending status. Do not treat all ControlNet/LoRA variants as fully compatible. |
| Video/audio | No broad native video/audio/TTS runtime. |
| Control/preprocessors | Strong built-in Forge ControlNet/preprocessor suite; compatibility includes annotator code, preprocessor registry, and control patchers. |
| Runtime formats/providers | Strong low-VRAM and low-bit story: Flux BNB NF4, GGUF, bnb-fp4, torch float8 e4m3/e5m2, offload/GPU-weight behavior. |
| Product features | LayerDiffuse and FreeU V2 are compatibility surfaces in addition to pure model loading. |

### reForge

| Area | Compatibility / missing notes |
| --- | --- |
| Main models | Extends Forge2-style support with Flux, fp8, GGUF, Wan 2.2, Qwen Image, Nunchaku, and external extension paths for Hunyuan-DiT, PixArt Sigma, and Stable Cascade. Some newer model support is extension-path compatibility, not necessarily native core execution. |
| Video/audio | Has Wan 2.2/new-model paths, but not the full video/audio/TTS workbench surface of Wan2GP. |
| Control/preprocessors | Very rich preprocessor coverage: tile, revision, reference, inpaint, LaMa, recolor, NormalBae, Marigold, IP-Adapter, InstantID, PhotoMaker. |
| Runtime formats/providers | Strong experimental runtime surface: CUDA stream/offload, shared-memory offload, fp8/GGUF/Nunchaku, CFG++, Sage/Flash attention, UNet/VAE/CLIP dtype knobs. Compatibility may be hardware-sensitive. |
| Product/API | WebUI/Forge-style, not Comfy graph or SM/Swarm package orchestration. |

### Classic

| Area | Compatibility / missing notes |
| --- | --- |
| Main models | Despite the name, supports SD1, SDXL, SD3, Flux.2 Klein, Z-Image, Wan 2.2, Qwen Image/Edit, Flux Kontext, Lumina/Neta-Lumina, and special VAE/decoder support. |
| Video/audio | Wan 2.2 and FirstLastFrameToVideo are present, but broad LTX/Wan-derived talking-video/audio/TTS workflows are missing compared with Wan2GP. |
| Adapters/control | LoRA, ControlNet rewrite, LLLite, Union ControlNet, Hypernetworks, CLIP Interrogator. Nunchaku LoRA support has model-family limits called out in README. |
| Upscaling/restoration | Strong WebUI extras/API surface plus ATD/DAT/DRCT utilities. |
| Runtime formats/providers | Very broad qtype/runtime list: Nunchaku SVDQ, fp4mixed, fp8mixed, mxfp8, nvfp4, fp8_scaled, int8 Triton matmul, SageAttention, FlashAttention, fp16 accumulation, torch._scaled_mm. Compatibility is likely provider/hardware-specific. |
| Product/API | WebUI-compatible, not Comfy graph-compatible or SM package-manager compatible. |

### Wan2GP

| Area | Compatibility / missing notes |
| --- | --- |
| Main scope | Deepest video/audio/runtime source. Covers many Wan derivatives, Hunyuan 1.5 variants, LTX-2, Flux variants, Kandinsky 5, LongCat, Magi Human, TTS/music/audio, VLM/agent helpers, qtypes, and plugins. It is not a general A1111/Comfy-compatible image UI. |
| SD/WebUI compatibility | Classic SD checkpoint/WebUI extras compatibility is not the primary surface. DinoML compatibility here means matching Wan2GP-style handlers, presets/defaults, offload/qtypes, process plugins, and Gradio/API flows. |
| Video/audio | Very broad native surface; missing compatibility risk is in exact temporal/audio contracts: sliding windows, phase schedules, audio-window slicing, control-video injection, masks, VAE/audio sync, continuation metadata. |
| Control/preprocess | Includes pose/depth/flow extraction, MatAnyone, Magic Mask, SAM3/SAM3.1, negative masks, Depth Anything v3, NLFPose/SCAIL and pose alignment. |
| Runtime formats/providers | Broad qtype/kernel surface: GGUF qtypes Q2_K-Q8_0, llama.cpp CUDA fast paths, NVFP4 validation, Nunchaku SVDQ/AWQ, Sage3/radial attention, MPS early support, torch compile hooks, mmgp offload. Compatibility is highly runtime-policy dependent. |
| Product/plugins | Plugin ecosystem is a compatibility surface: LoRA manager/merger/organizer, multipliers UI, queue editor, gallery, motion designer, process full video, video mask creator, config/download/model/plugin manager. |

### Swarm

| Area | Compatibility / missing notes |
| --- | --- |
| Main scope | Orchestrator/product UI around Comfy backends. Feature compatibility is often inherited from Comfy or generated workflows, not native model execution. |
| Main models | Supports/targets SD, Flux, Qwen, Z-Image, Wan, Hunyuan Video, Cosmos, Chroma, Ernie, HiDreamO1, OmniGen, LTXv2, Qwen Image Edit Plus and more through workflow generation and backend support. |
| Audio | ACE-Step is a native UI/control surface; broader audio support depends on backend workflows. |
| Runtime/providers | TensorRT and GGUF are installable/backend features. TensorRT support is constrained to specific engine families and reduced flexibility. |
| Catalog/API | Strong API/model catalog surface: model download/edit/list, WebSocket generation, workflow extraction, Comfy feature install, LoRA extraction, TensorRT creation. Missing compatibility would be API and metadata behavior, not only model inference. |
| Dependency caveat | Since execution is backend-driven, DinoML would need either a Swarm-compatible backend adapter or Comfy-compatible API/workflow behavior. |

### StabilityMatrix

| Area | Compatibility / missing notes |
| --- | --- |
| Main scope | Desktop package/model manager, not primarily a model runtime. Feature rows attributed to `SM` are usually catalog/package/folder/workflow compatibility, not direct execution. |
| Model families | Catalogs/package mappings cover SD/SDXL/Flux/CogVideoX/Hunyuan/Hunyuan Video/Wan and many package-specific surfaces. Actual model compatibility depends on the installed target UI. |
| Shared folders/catalogs | Strong compatibility surface: StableDiffusion, VAE, ApproxVAE, Embeddings, ControlNet, T2IAdapter, IP-Adapter, CLIP Vision, Ultralytics, SAMs, PromptExpansion and remote catalogs. |
| Runtime/providers | Install helpers for SageAttention and Nunchaku exist for Comfy packages; SM itself does not implement those kernels. |
| Product/API | Extension management, package install/update, model index, remote catalog, and Comfy node/workflow builder behavior are the compatibility targets. |
| Missing direct runtime | No native image/video/audio inference engine should be assumed from SM feature rows. |
