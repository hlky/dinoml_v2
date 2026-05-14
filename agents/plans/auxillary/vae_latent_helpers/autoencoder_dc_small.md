# AutoencoderDC and AutoencoderSmall-Style Codecs

## Coverage

- Diffusers: `AutoencoderDC` is covered in the inspected Diffusers source.
- Transformers: not covered.
- Third-party/UI: SD.Next mentions AutoencoderSmall handling alongside TAESD/preview VAE paths.

## Runtime Contract

AutoencoderDC/Small-style codecs are lighter image latent codecs used by newer diffusion families or preview/upscale paths. They are not interchangeable with SD1/SDXL AutoencoderKL without matching latent channels, scaling, and architecture.

## Operators

- Conv/residual image autoencoder blocks.
- Downsample/upsample.
- Latent scaling/shift and output range conversion.

## DinoML Notes

Treat each codec class as its own artifact family. For first support, parse the component config and validate encode/decode boundary transforms before optimizing kernels.

## Sources

- `X:/H/diffusers/src/diffusers/models/autoencoders/autoencoder_dc.py`
- `H:/uis/vladmandic/sdnext/modules/vae/sd_vae_taesd.py:139`

