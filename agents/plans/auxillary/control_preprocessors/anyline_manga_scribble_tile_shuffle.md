# Anyline, Manga Line, Scribble, Tile, Blur, Shuffle, and Color Maps

## Coverage

- Diffusers: not covered as model components.
- Transformers: not covered.
- Third-party/UI: SD.Next routes Anyline through `controlnet_aux`; tile/blur/shuffle/color-map preprocessors are mostly deterministic UI transforms.

## Runtime Contract

Anyline and manga-line helpers produce line/edge condition images. Scribble/sketch can be either a detector postprocess mode, such as HED scribble, or a deterministic threshold/cleanup operation. Tile, blur, shuffle, and color-map preprocessors are image transforms that generate ControlNet/T2I conditioning images without learned weights.

`controlnet_aux.AnylineDetector` is based on TheMistoAI Anyline code. It loads an MTEED/TED edge model, defaults to `detect_resolution=1280`, computes a TEED edge map with `safe_step(edge, 2)`, computes a second deterministic lineart map from Gaussian blur residuals (`guassian_sigma=2.0`, `intensity_threshold=3`), removes small objects, and combines the two maps into a final line condition.

`LineartStandardDetector` is deterministic rather than learned. It uses Gaussian blur residual intensity, median normalization above a threshold, and cubic resize. `ContentShuffleDetector` generates smooth random x/y remap fields and applies `cv2.remap`; related shuffle helpers in the same module include color shuffle, grayscale, downsample, and image-to-mask shuffle transforms.

## Operators

- Anyline/manga: TED/MTEED conv detector plus deterministic lineart residual cleanup.
- Scribble: NMS/blur/threshold/invert operations.
- Tile/blur/shuffle/color: resize, convolution/blur, remap fields, block shuffle, palette or color transform.

## DinoML Notes

Represent deterministic preprocessors as declarative image transforms with parameters. Anyline can share TEED graph coverage, but its final result depends on deterministic morphology and image-compositing postprocess, so it should remain a distinct preprocessor manifest entry.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/anyline/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/lineart_standard/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/shuffle/__init__.py`
- `H:/uis/vladmandic/sdnext/modules/control/proc/anyline/__init__.py:10`
- `H:/uis/lllyasviel/stable-diffusion-webui-forge/extensions-builtin/forge_legacy_preprocessors/legacy_preprocessors/preprocessor.py`
- `agents/plans/auxiliary/control_preprocessors.md`
