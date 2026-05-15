# Lineart

## Coverage

- Diffusers: not covered.
- Transformers: not covered.
- Third-party/UI: vendored from `controlnet_aux` style code in InvokeAI.

## Runtime Contract

InvokeAI loads `lllyasviel/Annotators/sk_model.pth` and `sk_model2.pth` into a small generator network. The model is an image-to-edge generator with reflection padding, Conv2d, InstanceNorm2d, residual blocks, ConvTranspose2d upsampling, and sigmoid output. The output is inverted and optionally thresholded for cleaner SDXL ControlNet conditioning.

`controlnet_aux.LineartDetector` has the same fine/coarse split and defaults to `sk_model.pth` plus `sk_model2.pth`. The processor registry exposes `lineart_realistic` and `lineart_coarse`.

`LineartAnimeDetector` is a separate U-Net generator, defaulting to `netG.pth`. It pads resized inputs to multiples of 256, scales RGB to -1..1, runs the U-Net, maps the result back to 0..255, resizes, and inverts the output. `LineartStandardDetector` is deterministic and uses Gaussian blur residual intensity rather than learned weights.

## Operators

- ReflectionPad2d, Conv2d, ConvTranspose2d.
- InstanceNorm2d, ReLU, Sigmoid.
- Residual add, resize, output inversion, threshold.
- Anime variant: U-Net skip connections, LeakyReLU/ReLU, InstanceNorm2d, Tanh.
- Standard variant: Gaussian blur, residual intensity, threshold/median normalization.

## DinoML Notes

Lineart is separate from Canny/HED because the model weights and output polarity matter. Track fine versus coarse checkpoint selection as part of the preprocessing schema.

## Sources

- `H:/controlnet_aux/src/controlnet_aux/lineart/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/lineart_anime/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/lineart_standard/__init__.py`
- `H:/controlnet_aux/src/controlnet_aux/processor.py`
- `invoke-ai/InvokeAI/invokeai/backend/image_util/lineart.py`
- `invoke-ai/InvokeAI/invokeai/backend/image_util/lineart_anime.py`
