## activations
- `torch`: `sigmoid`
- `torch.Tensor`: `chunk`, `float`, `to`
- `torch.nn`: `Linear`, `SiLU`
- `torch.nn.functional`: `gelu`, `silu`

## adapter
- `torch`: `tensor`
- `torch.Tensor`: -
- `torch.nn`: `AvgPool2d`, `Conv2d`, `ModuleList`, `PixelUnshuffle`, `ReLU`, `Sequential`
- `torch.nn.functional`: -

## attention
- `torch`: `baddbmm`, `cat`, `empty`, `no_grad`, `ones_like`, `randn`, `tensor`, `where`, `zeros`, `zeros_like`
- `torch.Tensor`: `chunk`, `float`, `permute`, `repeat_interleave`, `reshape`, `size`, `softmax`, `split`, `squeeze`, `to`, `transpose`, `unsqueeze`, `values`
- `torch.nn`: `Dropout`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: `pad`, `silu`

## autoencoders/autoencoder_asym_kl
- `torch`: -
- `torch.Tensor`: `mode`
- `torch.nn`: `Conv2d`
- `torch.nn.functional`: -

## autoencoders/autoencoder_dc
- `torch`: `cat`
- `torch.Tensor`: `mean`, `movedim`, `repeat_interleave`, `size`, `split`, `unflatten`
- `torch.nn`: `Conv2d`, `Identity`, `ModuleList`, `Sequential`
- `torch.nn.functional`: `interpolate`, `pad`, `pixel_shuffle`, `pixel_unshuffle`

## autoencoders/autoencoder_kl
- `torch`: `cat`
- `torch.Tensor`: `mode`, `split`
- `torch.nn`: `Conv2d`
- `torch.nn.functional`: -

## autoencoders/autoencoder_kl_allegro
- `torch`: `arange`, `cat`, `is_grad_enabled`
- `torch.Tensor`: `contiguous`, `flatten`, `float`, `mode`, `new_zeros`, `permute`, `repeat_interleave`, `reshape`, `split`, `to`, `unflatten`, `unsqueeze`
- `torch.nn`: `Conv2d`, `Conv3d`, `Dropout`, `GroupNorm`, `ModuleList`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## autoencoders/autoencoder_kl_cogvideox
- `torch`: `cat`, `chunk`, `is_grad_enabled`
- `torch.Tensor`: `clone`, `mode`, `split`
- `torch.nn`: `Dropout`, `GroupNorm`, `Linear`, `ModuleList`, `SiLU`
- `torch.nn.functional`: `interpolate`, `pad`

## autoencoders/autoencoder_kl_cosmos
- `torch`: `arange`, `cat`, `chunk`, `is_grad_enabled`, `split`, `tensor`, `tril`
- `torch.Tensor`: `bool`, `clone`, `contiguous`, `flatten`, `flip`, `mode`, `new_ones`, `norm`, `permute`, `repeat`, `repeat_interleave`, `reshape`, `size`, `split`, `to`, `transpose`, `type_as`, `unflatten`, `unsqueeze`
- `torch.nn`: `Dropout`, `GroupNorm`, `Identity`, `ModuleList`
- `torch.nn.functional`: `avg_pool3d`, `conv3d`, `conv_transpose3d`, `pad`, `scaled_dot_product_attention`, `silu`

## autoencoders/autoencoder_kl_flux2
- `torch`: `cat`
- `torch.Tensor`: `mode`, `split`
- `torch.nn`: `BatchNorm2d`, `Conv2d`
- `torch.nn.functional`: -

## autoencoders/autoencoder_kl_hunyuan_video
- `torch`: `arange`, `cat`, `is_grad_enabled`, `meshgrid`, `where`
- `torch.Tensor`: `contiguous`, `expand`, `flatten`, `mode`, `permute`, `repeat_interleave`, `size`, `split`, `squeeze`, `to`, `unflatten`, `unsqueeze`
- `torch.nn`: `Conv3d`, `Dropout`, `GroupNorm`, `ModuleList`, `SiLU`
- `torch.nn.functional`: `interpolate`, `pad`

## autoencoders/autoencoder_kl_hunyuanimage
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: `contiguous`, `mean`, `mode`, `permute`, `repeat_interleave`, `reshape`, `split`, `view`
- `torch.nn`: `Conv2d`, `GroupNorm`, `ModuleList`
- `torch.nn.functional`: `scaled_dot_product_attention`

## autoencoders/autoencoder_kl_hunyuanimage_refiner
- `torch`: `cat`, `is_grad_enabled`, `ones`, `zeros`
- `torch.Tensor`: `contiguous`, `float`, `mean`, `mode`, `permute`, `repeat_interleave`, `reshape`, `split`, `squeeze`, `to`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: `normalize`, `pad`

## autoencoders/autoencoder_kl_hunyuanvideo15
- `torch`: `cat`, `full`, `is_grad_enabled`, `ones`, `zeros`
- `torch.Tensor`: `contiguous`, `expand`, `float`, `mean`, `mode`, `permute`, `repeat_interleave`, `reshape`, `split`, `squeeze`, `to`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: `normalize`, `pad`

## autoencoders/autoencoder_kl_kvae
- `torch`: `cat`, `is_grad_enabled`, `mean`
- `torch.Tensor`: `mode`, `repeat_interleave`, `split`, `view`
- `torch.nn`: `Conv2d`, `GroupNorm`, `Linear`, `Module`, `ModuleList`, `PixelShuffle`, `PixelUnshuffle`
- `torch.nn.functional`: `interpolate`

## autoencoders/autoencoder_kl_kvae_video
- `torch`: `cat`, `chunk`, `clone`, `empty`, `empty_like`, `is_grad_enabled`, `mean`, `split`, `zeros_like`
- `torch.Tensor`: `element_size`, `expand`, `item`, `mode`, `numel`, `permute`, `repeat_interleave`, `reshape`, `size`, `split`, `view`
- `torch.nn`: `AvgPool3d`, `Conv3d`, `GroupNorm`, `Linear`, `Module`, `ModuleList`, `PixelShuffle`, `PixelUnshuffle`
- `torch.nn.functional`: `avg_pool1d`, `interpolate`, `pad`, `silu`

## autoencoders/autoencoder_kl_ltx
- `torch`: `cat`, `concatenate`, `is_grad_enabled`, `ones`, `randn`, `tensor`, `zeros`
- `torch.Tensor`: `flatten`, `mean`, `mode`, `movedim`, `permute`, `repeat`, `reshape`, `size`, `split`, `unbind`, `unflatten`, `view`
- `torch.nn`: `Conv3d`, `Dropout`, `LayerNorm`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: -

## autoencoders/autoencoder_kl_ltx2
- `torch`: `cat`, `concatenate`, `is_grad_enabled`, `mean`, `ones`, `randn`, `sqrt`, `tensor`, `zeros`
- `torch.Tensor`: `flatten`, `mean`, `mode`, `movedim`, `permute`, `repeat`, `reshape`, `size`, `split`, `unbind`, `unflatten`, `view`
- `torch.nn`: `Conv3d`, `Dropout`, `LayerNorm`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: -

## autoencoders/autoencoder_kl_ltx2_audio
- `torch`: `bmm`, `cat`, `mean`, `ones`, `sqrt`, `tanh`, `zeros`
- `torch.Tensor`: `contiguous`, `mode`, `permute`, `reshape`, `split`, `view`
- `torch.nn`: `Conv2d`, `Dropout`, `GroupNorm`, `Identity`, `Linear`, `Module`, `ModuleList`, `SiLU`
- `torch.nn.functional`: `avg_pool2d`, `interpolate`, `pad`, `softmax`

## autoencoders/autoencoder_kl_magvit
- `torch`: `cat`, `concat`, `is_grad_enabled`
- `torch.Tensor`: `clone`, `flatten`, `mode`, `permute`, `size`, `split`, `to`, `unflatten`
- `torch.nn`: `Conv3d`, `Dropout`, `GroupNorm`, `Identity`, `ModuleList`
- `torch.nn.functional`: `interpolate`, `pad`

## autoencoders/autoencoder_kl_mochi
- `torch`: `arange`, `cat`, `cos`, `is_grad_enabled`, `pow`, `sin`
- `torch.Tensor`: `contiguous`, `flatten`, `mode`, `permute`, `repeat`, `repeat_interleave`, `size`, `split`, `to`, `unflatten`, `view`
- `torch.nn`: `Conv3d`, `GroupNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: -

## autoencoders/autoencoder_kl_qwenimage
- `torch`: `cat`, `clamp`, `ones`, `stack`, `zeros`, `zeros_like`
- `torch.Tensor`: `chunk`, `clone`, `contiguous`, `float`, `mode`, `permute`, `reshape`, `size`, `split`, `squeeze`, `to`, `type_as`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `Dropout`, `Identity`, `ModuleList`, `Parameter`, `Sequential`, `ZeroPad2d`
- `torch.nn.functional`: `normalize`, `pad`, `scaled_dot_product_attention`

## autoencoders/autoencoder_kl_temporal_decoder
- `torch`: `is_grad_enabled`, `zeros`
- `torch.Tensor`: `mode`, `permute`, `reshape`, `to`
- `torch.nn`: `Conv2d`, `Conv3d`, `GroupNorm`, `ModuleList`, `SiLU`
- `torch.nn.functional`: -

## autoencoders/autoencoder_kl_wan
- `torch`: `cat`, `clamp`, `ones`, `stack`, `zeros`, `zeros_like`
- `torch.Tensor`: `chunk`, `clone`, `contiguous`, `dim`, `float`, `mean`, `mode`, `permute`, `repeat_interleave`, `reshape`, `size`, `split`, `squeeze`, `to`, `type_as`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `Dropout`, `Identity`, `ModuleList`, `Parameter`, `Sequential`, `ZeroPad2d`
- `torch.nn.functional`: `normalize`, `pad`, `scaled_dot_product_attention`

## autoencoders/autoencoder_longcat_audio_dit
- `torch`: `exp`, `sin`, `zeros`
- `torch.Tensor`: `chunk`, `contiguous`, `float`, `mean`, `permute`, `pow`, `repeat_interleave`, `size`, `to`, `view`
- `torch.nn`: `Conv1d`, `ConvTranspose1d`, `ELU`, `Identity`, `Parameter`, `Sequential`, `Tanh`
- `torch.nn.functional`: `softplus`

## autoencoders/autoencoder_oobleck
- `torch`: `Tensor`, `cat`, `exp`, `log`, `pow`, `sin`, `zeros`
- `torch.Tensor`: `chunk`, `mean`, `mode`, `pow`, `reciprocal`, `reshape`, `split`, `sum`
- `torch.nn`: `Conv1d`, `ConvTranspose1d`, `ModuleList`, `Parameter`
- `torch.nn.functional`: -

## autoencoders/autoencoder_rae
- `torch`: `arange`, `cat`, `einsum`, `ones`, `rand`, `tensor`, `zeros`
- `torch.Tensor`: `clone`, `contiguous`, `cpu`, `detach`, `expand`, `float`, `permute`, `requires_grad_`, `reshape`, `size`, `split`, `to`, `tolist`, `transpose`, `unsqueeze`, `view`
- `torch.nn`: `Dropout`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: `interpolate`

## autoencoders/autoencoder_tiny
- `torch`: `arange`, `cat`, `meshgrid`, `ones_like`, `stack`, `zeros`
- `torch.Tensor`: `add`, `byte`, `clamp`, `copy_`, `div`, `mul`, `mul_`, `round_`, `split`, `sub`, `to`
- `torch.nn`: -
- `torch.nn.functional`: -

## autoencoders/autoencoder_vidtok
- `torch`: `Tensor`, `arange`, `cat`, `concat`, `concatenate`, `cumprod`, `is_grad_enabled`, `sigmoid`, `tensor`, `where`, `zeros`
- `torch.Tensor`: `atanh`, `clone`, `contiguous`, `detach`, `dim`, `float`, `item`, `mode`, `permute`, `repeat`, `reshape`, `round`, `split`, `squeeze`, `sum`, `tanh`, `to`, `type`, `unsqueeze`
- `torch.nn`: `AvgPool3d`, `Conv1d`, `Conv2d`, `Conv3d`, `Dropout`, `Identity`, `LayerNorm`, `Linear`, `Module`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: `interpolate`, `pad`, `scaled_dot_product_attention`

## autoencoders/consistency_decoder_vae
- `torch`: `cat`, `concat`, `tensor`
- `torch.Tensor`: `mode`, `split`
- `torch.nn`: `Conv2d`
- `torch.nn.functional`: `interpolate`

## autoencoders/vae
- `torch`: `Tensor`, `argmin`, `cdist`, `chunk`, `clamp`, `exp`, `gather`, `is_grad_enabled`, `mean`, `pow`, `randint`, `relu`, `sum`, `tanh`, `tensor`, `zeros_like`
- `torch.Tensor`: `add`, `argmax`, `contiguous`, `detach`, `div`, `long`, `mul`, `permute`, `reshape`, `sub`, `sum`, `to`, `view`
- `torch.nn`: `Conv2d`, `ConvTranspose2d`, `Embedding`, `GroupNorm`, `ModuleList`, `Sequential`, `SiLU`, `Upsample`
- `torch.nn.functional`: -

## autoencoders/vq_model
- `torch`: `zeros`
- `torch.Tensor`: `to`
- `torch.nn`: `Conv2d`
- `torch.nn.functional`: -

## controlnets/controlnet
- `torch`: `concat`, `flip`, `is_tensor`, `logspace`, `mean`, `tensor`
- `torch.Tensor`: `expand`, `flatten`, `reshape`, `to`, `unsqueeze`
- `torch.nn`: `Conv2d`, `Embedding`, `Identity`, `Linear`, `ModuleList`
- `torch.nn.functional`: `silu`

## controlnets/controlnet_cosmos
- `torch`: `cat`, `is_grad_enabled`, `zeros`, `zeros_like`
- `torch.Tensor`: `expand`, `flatten`, `repeat`, `resize`, `unsqueeze`, `view`
- `torch.nn`: `GELU`, `Linear`, `ModuleList`, `Sequential`
- `torch.nn.functional`: -

## controlnets/controlnet_flux
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: `permute`, `reshape`, `to`
- `torch.nn`: `Embedding`, `Linear`, `ModuleList`
- `torch.nn.functional`: -

## controlnets/controlnet_hunyuan
- `torch`: `cat`, `randn`, `where`
- `torch.Tensor`: `bool`, `unsqueeze`, `view`
- `torch.nn`: `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: -

## controlnets/controlnet_qwenimage
- `torch`: `cat`, `is_grad_enabled`, `ones`
- `torch.Tensor`: `to`
- `torch.nn`: `Linear`, `ModuleList`
- `torch.nn.functional`: -

## controlnets/controlnet_sana
- `torch`: `is_grad_enabled`
- `torch.Tensor`: `to`, `unsqueeze`, `view`
- `torch.nn`: `Linear`, `ModuleList`
- `torch.nn.functional`: -

## controlnets/controlnet_sd3
- `torch`: `is_grad_enabled`
- `torch.Tensor`: -
- `torch.nn`: `Linear`, `ModuleList`
- `torch.nn.functional`: -

## controlnets/controlnet_sparsectrl
- `torch`: `cat`, `flip`, `is_tensor`, `logspace`, `mean`, `tensor`, `zeros_like`
- `torch.Tensor`: `expand`, `permute`, `repeat_interleave`, `reshape`, `to`, `unsqueeze`
- `torch.nn`: `Conv2d`, `ModuleList`
- `torch.nn.functional`: `silu`

## controlnets/controlnet_union
- `torch`: `cat`, `concat`, `is_tensor`, `logspace`, `mean`, `randn`, `sigmoid`, `tensor`
- `torch.Tensor`: `expand`, `flatten`, `reshape`, `to`, `unsqueeze`
- `torch.nn`: `Conv2d`, `Embedding`, `Identity`, `LayerNorm`, `Linear`, `ModuleList`, `MultiheadAttention`, `Parameter`
- `torch.nn.functional`: -

## controlnets/controlnet_xs
- `torch`: `cat`, `concat`, `flip`, `is_grad_enabled`, `is_tensor`, `tensor`
- `torch.Tensor`: `expand`, `flatten`, `reshape`, `to`, `unsqueeze`
- `torch.nn`: `Conv2d`, `GroupNorm`, `ModuleList`, `SiLU`
- `torch.nn.functional`: -

## controlnets/controlnet_z_image
- `torch`: `arange`, `cat`, `cos`, `device`, `exp`, `is_grad_enabled`, `meshgrid`, `ones`, `ones_like`, `outer`, `polar`, `sin`, `stack`, `unbind`, `view_as_complex`, `view_as_real`, `where`, `zeros`, `zeros_like`
- `torch.Tensor`: `chunk`, `expand`, `flatten`, `float`, `permute`, `repeat`, `reshape`, `size`, `split`, `tanh`, `to`, `type_as`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Linear`, `ModuleDict`, `ModuleList`, `Sequential`, `SiLU`
- `torch.nn.functional`: `silu`

## controlnets/multicontrolnet
- `torch`: -
- `torch.Tensor`: -
- `torch.nn`: `ModuleList`
- `torch.nn.functional`: -

## controlnets/multicontrolnet_union
- `torch`: -
- `torch.Tensor`: -
- `torch.nn`: `ModuleList`
- `torch.nn.functional`: -

## downsampling
- `torch`: `arange`, `cat`, `outer`, `sum`, `tensor`
- `torch.Tensor`: `expand`, `new_zeros`, `permute`, `reshape`, `to`
- `torch.nn`: `AvgPool1d`, `AvgPool2d`, `Conv1d`, `Conv2d`, `LayerNorm`
- `torch.nn.functional`: `avg_pool1d`, `conv2d`, `pad`

## embeddings
- `torch`: `arange`, `cat`, `concat`, `cos`, `einsum`, `exp`, `from_numpy`, `linspace`, `log`, `meshgrid`, `ones`, `ones_like`, `outer`, `polar`, `rand`, `randn`, `sin`, `softmax`, `stack`, `tensor`, `view_as_complex`, `view_as_real`, `where`, `zeros`
- `torch.Tensor`: `bool`, `chunk`, `clamp`, `contiguous`, `copy_`, `cos`, `expand`, `flatten`, `float`, `mean`, `new_zeros`, `permute`, `repeat`, `repeat_interleave`, `reshape`, `sin`, `size`, `squeeze`, `sum`, `to`, `transpose`, `type`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `Embedding`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: `embedding`, `multi_head_attention_forward`, `pad`, `scaled_dot_product_attention`

## normalization
- `torch`: `Size`, `chunk`, `norm`, `ones`, `rsqrt`, `zeros`
- `torch.Tensor`: `chunk`, `float`, `mean`, `pow`, `to`
- `torch.nn`: `BatchNorm2d`, `Embedding`, `LayerNorm`, `Linear`, `Parameter`, `SiLU`
- `torch.nn.functional`: `group_norm`, `layer_norm`, `normalize`

## resnet
- `torch`: `Tensor`, `chunk`, `ones`, `sigmoid`, `where`
- `torch.Tensor`: `bool`, `contiguous`, `permute`, `reshape`, `to`
- `torch.nn`: `Conv1d`, `Conv2d`, `Conv3d`, `Dropout`, `GroupNorm`, `Identity`, `Linear`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## transformers/ace_step_transformer
- `torch`: `abs`, `all`, `arange`, `cat`, `finfo`, `full`, `is_grad_enabled`, `ones`, `randn`
- `torch.Tensor`: `any`, `chunk`, `flatten`, `item`, `masked_fill_`, `to`, `transpose`, `type_as`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv1d`, `ConvTranspose1d`, `Dropout`, `Linear`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: `pad`, `silu`

## transformers/auraflow_transformer_2d
- `torch`: `arange`, `cat`, `chunk`, `einsum`, `is_grad_enabled`, `meshgrid`, `randn`
- `torch.Tensor`: `flatten`, `permute`, `reshape`, `size`, `to`, `unsqueeze`, `view`
- `torch.nn`: `Linear`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: `silu`

## transformers/cogvideox_transformer_3d
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: `flatten`, `permute`, `reshape`, `size`, `to`
- `torch.nn`: `Dropout`, `LayerNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: -

## transformers/consisid_transformer_3d
- `torch`: `cat`, `is_grad_enabled`, `randn`, `softmax`
- `torch.Tensor`: `chunk`, `flatten`, `float`, `permute`, `reshape`, `size`, `to`, `transpose`, `type`
- `torch.nn`: `Dropout`, `GELU`, `LayerNorm`, `LeakyReLU`, `Linear`, `ModuleList`, `Parameter`, `Sequential`
- `torch.nn.functional`: -

## transformers/dit_transformer_2d
- `torch`: `einsum`, `is_grad_enabled`
- `torch.Tensor`: `chunk`, `reshape`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: `silu`

## transformers/dual_transformer_2d
- `torch`: -
- `torch.Tensor`: -
- `torch.nn`: `ModuleList`
- `torch.nn.functional`: -

## transformers/hunyuan_transformer_2d
- `torch`: `cat`, `einsum`, `randn`, `where`
- `torch.Tensor`: `bool`, `reshape`, `to`, `unsqueeze`, `view`
- `torch.nn`: `Linear`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: -

## transformers/latte_transformer_3d
- `torch`: `arange`, `einsum`, `is_grad_enabled`, `randn`
- `torch.Tensor`: `chunk`, `float`, `permute`, `repeat_interleave`, `reshape`, `unsqueeze`, `view`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: -

## transformers/lumina_nextdit2d
- `torch`: `empty`, `zeros`
- `torch.Tensor`: `bool`, `flatten`, `permute`, `size`, `tanh`, `to`, `unsqueeze`, `view`
- `torch.nn`: `Identity`, `ModuleList`, `Parameter`
- `torch.nn.functional`: -

## transformers/pixart_transformer_2d
- `torch`: `einsum`, `is_grad_enabled`, `randn`
- `torch.Tensor`: `chunk`, `reshape`, `squeeze`, `to`, `unsqueeze`, `view`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: -

## transformers/prior_transformer
- `torch`: `cat`, `full`, `is_tensor`, `ones`, `tensor`, `zeros`
- `torch.Tensor`: `expand`, `repeat_interleave`, `to`, `triu_`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: `pad`

## transformers/sana_transformer
- `torch`: `chunk`, `is_grad_enabled`, `randn`
- `torch.Tensor`: `chunk`, `flatten`, `movedim`, `permute`, `reshape`, `to`, `transpose`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: `scaled_dot_product_attention`

## transformers/stable_audio_transformer
- `torch`: `cat`, `cos`, `is_grad_enabled`, `log`, `ones`, `randn`, `sin`
- `torch.Tensor`: `to`, `transpose`, `unsqueeze`
- `torch.nn`: `Conv1d`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## transformers/t5_film_transformer
- `torch`: `arange`, `broadcast_to`, `cat`, `chunk`, `mul`, `ones`, `pow`, `rsqrt`, `tanh`, `where`
- `torch.Tensor`: `mean`, `pow`, `squeeze`, `to`, `unsqueeze`
- `torch.nn`: `Dropout`, `Embedding`, `Linear`, `ModuleList`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_2d
- `torch`: `einsum`, `is_grad_enabled`, `randn`
- `torch.Tensor`: `chunk`, `contiguous`, `double`, `float`, `permute`, `reshape`, `squeeze`, `to`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `GroupNorm`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: `log_softmax`, `silu`

## transformers/transformer_allegro
- `torch`: `is_grad_enabled`, `randn`
- `torch.Tensor`: `bool`, `chunk`, `flatten`, `numel`, `permute`, `reshape`, `squeeze`, `to`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: `max_pool3d`

## transformers/transformer_bria
- `torch`: `arange`, `cat`, `from_numpy`, `is_grad_enabled`, `ones_like`, `outer`, `polar`
- `torch.Tensor`: `chunk`, `clip`, `cos`, `flatten`, `float`, `repeat_interleave`, `sin`, `split_with_sizes`, `to`, `unflatten`, `unsqueeze`
- `torch.nn`: `Dropout`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`
- `torch.nn.functional`: -

## transformers/transformer_bria_fibo
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: `chunk`, `clip`, `contiguous`, `flatten`, `float`, `split_with_sizes`, `to`, `unflatten`, `unsqueeze`
- `torch.nn`: `Dropout`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`
- `torch.nn.functional`: -

## transformers/transformer_chroma
- `torch`: `arange`, `cat`, `chunk`, `is_grad_enabled`, `tensor`
- `torch.Tensor`: `chunk`, `clip`, `flatten`, `repeat`, `to`, `unsqueeze`
- `torch.nn`: `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`
- `torch.nn.functional`: -

## transformers/transformer_chronoedit
- `torch`: `cat`, `concat`, `device`, `empty_like`, `is_grad_enabled`, `no_grad`, `randn`, `zeros`
- `torch.Tensor`: `chunk`, `expand`, `flatten`, `float`, `permute`, `reshape`, `squeeze`, `to`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `Dropout`, `Identity`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_cogview3plus
- `torch`: `cat`, `einsum`, `is_grad_enabled`
- `torch.Tensor`: `clip`, `reshape`, `size`, `unsqueeze`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: -

## transformers/transformer_cogview4
- `torch`: `arange`, `cat`, `chunk`, `is_grad_enabled`, `max`, `ones`, `outer`, `split`, `sum`, `tensor`, `zeros`
- `torch.Tensor`: `chunk`, `cos`, `dim`, `expand`, `flatten`, `float`, `item`, `permute`, `reshape`, `sin`, `size`, `split`, `to`, `tolist`, `transpose`, `type_as`, `unflatten`, `unsqueeze`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: `scaled_dot_product_attention`, `silu`

## transformers/transformer_cosmos
- `torch`: `add`, `arange`, `cat`, `cos`, `is_grad_enabled`, `outer`, `sin`, `tensor`, `zeros`
- `torch.Tensor`: `chunk`, `expand`, `flatten`, `float`, `numel`, `permute`, `repeat`, `repeat_interleave`, `reshape`, `resize`, `size`, `transpose`, `type_as`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `GELU`, `Identity`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_easyanimate
- `torch`: `cat`, `concat`, `is_grad_enabled`
- `torch.Tensor`: `chunk`, `contiguous`, `flatten`, `permute`, `reshape`, `size`, `to`, `transpose`, `unflatten`, `unsqueeze`
- `torch.nn`: `Conv2d`, `LayerNorm`, `Linear`, `ModuleList`, `Sequential`, `SiLU`
- `torch.nn.functional`: `scaled_dot_product_attention`

## transformers/transformer_ernie_image
- `torch`: `arange`, `cat`, `cos`, `einsum`, `is_grad_enabled`, `meshgrid`, `ones`, `sin`, `stack`, `zeros`
- `torch.Tensor`: `chunk`, `contiguous`, `expand`, `flatten`, `float`, `numel`, `permute`, `reshape`, `to`, `transpose`, `type_as`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`, `Sequential`, `SiLU`
- `torch.nn.functional`: `gelu`

## transformers/transformer_flux
- `torch`: `cat`, `is_grad_enabled`, `zeros_like`
- `torch.Tensor`: `chunk`, `clip`, `contiguous`, `flatten`, `float`, `reshape`, `split_with_sizes`, `to`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Dropout`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`
- `torch.nn.functional`: -

## transformers/transformer_flux2
- `torch`: `cat`, `chunk`, `full_like`, `is_grad_enabled`, `split`
- `torch.Tensor`: `chunk`, `clip`, `clone`, `expand`, `flatten`, `float`, `split_with_sizes`, `to`, `unflatten`, `unsqueeze`
- `torch.nn`: `Dropout`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_glm_image
- `torch`: `arange`, `cat`, `chunk`, `is_grad_enabled`, `ones`, `outer`
- `torch.Tensor`: `chunk`, `cos`, `dim`, `expand`, `flatten`, `float`, `permute`, `reshape`, `sin`, `size`, `split`, `to`, `transpose`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Embedding`, `LayerNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: `silu`

## transformers/transformer_helios
- `torch`: `arange`, `cat`, `device`, `einsum`, `empty_like`, `is_grad_enabled`, `meshgrid`, `no_grad`, `ones`, `randn`, `sigmoid`, `split`, `zeros`
- `torch.Tensor`: `chunk`, `contiguous`, `cos`, `expand`, `flatten`, `float`, `permute`, `repeat_interleave`, `reshape`, `sin`, `squeeze`, `to`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `Dropout`, `Identity`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: `avg_pool3d`, `pad`

## transformers/transformer_hidream_image
- `torch`: `arange`, `cat`, `cos`, `einsum`, `empty_like`, `is_grad_enabled`, `no_grad`, `ones`, `randn`, `sin`, `split`, `stack`, `tensor`, `topk`, `zeros`, `zeros_like`
- `torch.Tensor`: `argsort`, `bincount`, `chunk`, `cpu`, `cumsum`, `div_`, `float`, `mean`, `mul_`, `numpy`, `permute`, `repeat`, `repeat_interleave`, `reshape`, `scatter_add_`, `scatter_reduce_`, `softmax`, `sum`, `to`, `transpose`, `type_as`, `unsqueeze`, `view`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `Sequential`, `SiLU`
- `torch.nn.functional`: `linear`, `one_hot`, `scaled_dot_product_attention`, `silu`

## transformers/transformer_hunyuan_video
- `torch`: `arange`, `cat`, `is_grad_enabled`, `meshgrid`, `ones`, `stack`, `zeros_like`
- `torch.Tensor`: `bool`, `chunk`, `flatten`, `float`, `masked_fill`, `mean`, `permute`, `repeat`, `reshape`, `sum`, `to`, `transpose`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_hunyuan_video15
- `torch`: `all`, `arange`, `cat`, `is_grad_enabled`, `meshgrid`, `ones`, `ones_like`, `stack`, `zeros`, `zeros_like`
- `torch.Tensor`: `bool`, `chunk`, `flatten`, `float`, `mean`, `permute`, `repeat`, `reshape`, `sum`, `to`, `transpose`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `Embedding`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `SiLU`
- `torch.nn.functional`: `pad`

## transformers/transformer_hunyuan_video_framepack
- `torch`: `arange`, `cat`, `is_grad_enabled`, `meshgrid`, `stack`, `zeros`
- `torch.Tensor`: `expand`, `flatten`, `new_ones`, `permute`, `reshape`, `squeeze`, `sum`, `to`, `transpose`, `unflatten`, `unsqueeze`
- `torch.nn`: `Conv3d`, `Linear`, `ModuleList`
- `torch.nn.functional`: `avg_pool3d`, `pad`, `silu`

## transformers/transformer_hunyuanimage
- `torch`: `arange`, `cat`, `is_grad_enabled`, `meshgrid`, `stack`
- `torch.Tensor`: `bool`, `chunk`, `flatten`, `float`, `mean`, `permute`, `repeat`, `reshape`, `sum`, `to`, `transpose`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `Conv3d`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `SiLU`
- `torch.nn.functional`: `pad`

## transformers/transformer_joyimage
- `torch`: `arange`, `cat`, `is_grad_enabled`, `linspace`, `meshgrid`, `outer`, `stack`, `zeros`
- `torch.Tensor`: `chunk`, `cos`, `flatten`, `float`, `item`, `max`, `permute`, `repeat_interleave`, `reshape`, `sin`, `squeeze`, `to`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_kandinsky
- `torch`: `arange`, `cat`, `chunk`, `cos`, `exp`, `is_grad_enabled`, `logical_or`, `outer`, `sin`, `softmax`, `stack`, `zeros_like`
- `torch.Tensor`: `argsort`, `contiguous`, `cumsum_`, `flatten`, `float`, `gather`, `int`, `mean`, `permute`, `repeat`, `reshape`, `sort`, `sum`, `to`, `transpose`, `type_as`, `unsqueeze`, `view`
- `torch.nn`: `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_longcat_audio_dit
- `torch`: `arange`, `cat`, `chunk`, `exp`, `norm`, `outer`, `randn`, `zeros`
- `torch.Tensor`: `bool`, `chunk`, `clamp`, `clone`, `contiguous`, `cos`, `flatten`, `float`, `logical_not`, `masked_fill`, `mean`, `repeat`, `sin`, `sum`, `to`, `transpose`, `type_as`, `unsqueeze`, `view`
- `torch.nn`: `Conv1d`, `Dropout`, `GELU`, `Identity`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: `layer_norm`

## transformers/transformer_longcat_image
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: `chunk`, `clip`, `flatten`, `float`, `split_with_sizes`, `to`, `unflatten`, `unsqueeze`
- `torch.nn`: `Dropout`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`
- `torch.nn.functional`: -

## transformers/transformer_ltx
- `torch`: `arange`, `cat`, `is_grad_enabled`, `linspace`, `meshgrid`, `ones_like`, `randn`, `stack`, `zeros_like`
- `torch.Tensor`: `cos`, `flatten`, `float`, `repeat`, `repeat_interleave`, `reshape`, `sin`, `size`, `to`, `transpose`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Dropout`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`
- `torch.nn.functional`: -

## transformers/transformer_ltx2
- `torch`: `all`, `arange`, `cat`, `concatenate`, `is_grad_enabled`, `lerp`, `linspace`, `meshgrid`, `ones_like`, `pow`, `randn`, `sigmoid`, `stack`, `swapaxes`, `tensor`, `zeros`, `zeros_like`
- `torch.Tensor`: `addcmul_`, `chunk`, `clamp`, `clip`, `cos`, `expand`, `flatten`, `float`, `repeat`, `repeat_interleave`, `reshape`, `sin`, `size`, `squeeze`, `swapaxes`, `to`, `transpose`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Dropout`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_lumina2
- `torch`: `arange`, `cat`, `gather`, `is_grad_enabled`, `stack`, `zeros`
- `torch.Tensor`: `bool`, `flatten`, `new_zeros`, `permute`, `repeat`, `reshape`, `sum`, `tanh`, `to`, `tolist`, `transpose`, `type_as`, `unsqueeze`, `view`
- `torch.nn`: `Linear`, `ModuleList`, `Sequential`
- `torch.nn.functional`: `scaled_dot_product_attention`

## transformers/transformer_mochi
- `torch`: `arange`, `autocast`, `cos`, `einsum`, `full`, `is_grad_enabled`, `linspace`, `meshgrid`, `sin`, `stack`, `tanh`
- `torch.Tensor`: `chunk`, `flatten`, `permute`, `reshape`, `to`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Linear`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_motif_video
- `torch`: `arange`, `cat`, `is_grad_enabled`, `meshgrid`, `ones`, `stack`
- `torch.Tensor`: `chunk`, `clone`, `flatten`, `permute`, `reshape`, `to`, `transpose`, `unflatten`, `unsqueeze`
- `torch.nn`: `Conv3d`, `Dropout`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: `pad`

## transformers/transformer_nucleusmoe_image
- `torch`: `arange`, `as_tensor`, `cat`, `cumsum`, `empty`, `finfo`, `full`, `is_grad_enabled`, `matmul`, `maximum`, `ones`, `ones_like`, `outer`, `polar`, `pow`, `sigmoid`, `split`, `stack`, `tensor`, `topk`, `view_as_complex`, `view_as_real`, `vstack`, `where`, `zeros`
- `torch.Tensor`: `any`, `chunk`, `clamp`, `clip`, `clone`, `contiguous`, `div`, `expand`, `flatten`, `flip`, `float`, `gather`, `max`, `new_zeros`, `repeat_interleave`, `reshape`, `scatter_add`, `scatter_add_`, `split`, `tanh`, `to`, `tolist`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Embedding`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: `grouped_mm`, `silu`, `softmax`

## transformers/transformer_omnigen
- `torch`: `arange`, `autocast`, `cat`, `finfo`, `is_grad_enabled`, `max`, `tensor`
- `torch.Tensor`: `chunk`, `cos`, `dim`, `expand`, `flatten`, `float`, `long`, `permute`, `reshape`, `sin`, `size`, `to`, `transpose`, `type_as`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `Embedding`, `Linear`, `ModuleList`, `SiLU`
- `torch.nn.functional`: `scaled_dot_product_attention`

## transformers/transformer_ovis_image
- `torch`: `cat`, `is_grad_enabled`, `split`
- `torch.Tensor`: `chunk`, `clip`, `flatten`, `float`, `split_with_sizes`, `to`, `unflatten`, `unsqueeze`
- `torch.nn`: `Dropout`, `LayerNorm`, `Linear`, `ModuleList`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_prx
- `torch`: `arange`, `cat`, `cos`, `einsum`, `is_grad_enabled`, `ones`, `sin`, `stack`, `zeros`
- `torch.Tensor`: `chunk`, `dim`, `expand`, `float`, `permute`, `repeat`, `reshape`, `to`, `transpose`, `type_as`, `unsqueeze`
- `torch.nn`: `Dropout`, `GELU`, `LayerNorm`, `Linear`, `ModuleList`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_qwenimage
- `torch`: `arange`, `as_tensor`, `cat`, `chunk`, `cos`, `exp`, `is_grad_enabled`, `ones`, `ones_like`, `outer`, `polar`, `pow`, `sin`, `stack`, `tensor`, `view_as_complex`, `view_as_real`, `where`
- `torch.Tensor`: `any`, `chunk`, `clip`, `clone`, `contiguous`, `div`, `expand`, `flatten`, `flip`, `float`, `max`, `new_zeros`, `reshape`, `size`, `split`, `to`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Embedding`, `LayerNorm`, `Linear`, `ModuleList`, `Sequential`, `SiLU`
- `torch.nn.functional`: `pad`

## transformers/transformer_sana_video
- `torch`: `cat`, `chunk`, `empty_like`, `is_grad_enabled`, `matmul`, `randn`
- `torch.Tensor`: `expand`, `flatten`, `float`, `movedim`, `permute`, `reshape`, `size`, `sum`, `to`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `Conv3d`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `SiLU`
- `torch.nn.functional`: `relu`

## transformers/transformer_sd3
- `torch`: `einsum`, `is_grad_enabled`
- `torch.Tensor`: `reshape`, `unsqueeze`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: -

## transformers/transformer_skyreels_v2
- `torch`: `arange`, `cat`, `concat`, `device`, `empty_like`, `is_grad_enabled`, `no_grad`, `randn`, `tensor`, `zeros`
- `torch.Tensor`: `chunk`, `contiguous`, `dim`, `expand`, `flatten`, `float`, `permute`, `repeat`, `repeat_interleave`, `reshape`, `squeeze`, `to`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `Dropout`, `Embedding`, `Identity`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_temporal
- `torch`: `arange`, `is_grad_enabled`
- `torch.Tensor`: `broadcast_to`, `contiguous`, `permute`, `repeat`, `reshape`, `to`
- `torch.nn`: `GroupNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: -

## transformers/transformer_wan
- `torch`: `cat`, `concat`, `device`, `empty_like`, `is_grad_enabled`, `no_grad`, `randn`, `zeros`
- `torch.Tensor`: `chunk`, `expand`, `flatten`, `float`, `permute`, `reshape`, `squeeze`, `to`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv3d`, `Dropout`, `Identity`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: -

## transformers/transformer_wan_animate
- `torch`: `cat`, `concat`, `device`, `diag_embed`, `empty_like`, `is_grad_enabled`, `matmul`, `no_grad`, `randn`, `split`, `sum`, `tensor`, `zeros`, `zeros_like`
- `torch.Tensor`: `chunk`, `contiguous`, `expand`, `flatten`, `float`, `permute`, `reshape`, `squeeze`, `sum`, `to`, `transpose`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `Conv1d`, `Conv3d`, `Dropout`, `Identity`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `RMSNorm`, `SiLU`
- `torch.nn.functional`: `conv2d`, `leaky_relu`, `linear`, `pad`

## transformers/transformer_wan_vace
- `torch`: `cat`, `concat`, `is_grad_enabled`, `randn`, `unbind`
- `torch.Tensor`: `chunk`, `flatten`, `float`, `new_ones`, `new_zeros`, `permute`, `reshape`, `size`, `to`, `transpose`, `type_as`, `unflatten`, `unsqueeze`
- `torch.nn`: `Conv3d`, `Identity`, `Linear`, `ModuleList`, `Parameter`
- `torch.nn.functional`: -

## transformers/transformer_z_image
- `torch`: `arange`, `cat`, `cos`, `device`, `empty`, `exp`, `is_grad_enabled`, `meshgrid`, `ones`, `ones_like`, `outer`, `polar`, `sin`, `stack`, `tensor`, `view_as_complex`, `view_as_real`, `where`, `zeros`, `zeros_like`
- `torch.Tensor`: `chunk`, `expand`, `flatten`, `float`, `permute`, `repeat`, `reshape`, `size`, `split`, `tanh`, `to`, `type_as`, `unbind`, `unflatten`, `unsqueeze`, `view`
- `torch.nn`: `LayerNorm`, `Linear`, `ModuleDict`, `ModuleList`, `Parameter`, `Sequential`, `SiLU`
- `torch.nn.functional`: `silu`

## unets/unet_1d
- `torch`: `is_tensor`, `tensor`
- `torch.Tensor`: `broadcast_to`, `repeat`, `to`
- `torch.nn`: `ModuleList`
- `torch.nn.functional`: -

## unets/unet_1d_blocks
- `torch`: `arange`, `cat`, `matmul`, `softmax`, `tensor`
- `torch.Tensor`: `contiguous`, `expand`, `new_zeros`, `permute`, `size`, `transpose`, `view`
- `torch.nn`: `Conv1d`, `Dropout`, `GELU`, `GroupNorm`, `Linear`, `ModuleList`
- `torch.nn.functional`: `conv1d`, `conv_transpose1d`, `pad`

## unets/unet_2d
- `torch`: `is_tensor`, `ones`, `tensor`
- `torch.Tensor`: `reshape`, `to`
- `torch.nn`: `Conv2d`, `Embedding`, `GroupNorm`, `Identity`, `ModuleList`, `SiLU`
- `torch.nn.functional`: -

## unets/unet_2d_blocks
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: `permute`, `reshape`
- `torch.nn`: `Conv2d`, `GroupNorm`, `Identity`, `ModuleList`, `ReLU`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## unets/unet_2d_condition
- `torch`: `cat`, `concat`, `is_tensor`, `tensor`
- `torch.Tensor`: `expand`, `flatten`, `reshape`, `to`, `unsqueeze`
- `torch.nn`: `Conv2d`, `Embedding`, `GroupNorm`, `Identity`, `Linear`, `ModuleList`
- `torch.nn.functional`: -

## unets/unet_3d_blocks
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: -
- `torch.nn`: `ModuleList`
- `torch.nn.functional`: -

## unets/unet_3d_condition
- `torch`: `is_tensor`, `tensor`
- `torch.Tensor`: `expand`, `permute`, `repeat_interleave`, `reshape`, `to`, `unsqueeze`
- `torch.nn`: `Conv2d`, `GroupNorm`, `ModuleList`
- `torch.nn.functional`: -

## unets/unet_i2vgen_xl
- `torch`: `cat`, `is_tensor`, `tensor`
- `torch.Tensor`: `expand`, `new_zeros`, `permute`, `repeat_interleave`, `reshape`, `squeeze`, `to`, `view`
- `torch.nn`: `AdaptiveAvgPool2d`, `Conv2d`, `GroupNorm`, `LayerNorm`, `Linear`, `ModuleList`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## unets/unet_kandinsky3
- `torch`: `cat`, `is_tensor`, `tensor`
- `torch.Tensor`: `chunk`, `expand`, `mean`, `permute`, `reshape`, `squeeze`, `to`, `unsqueeze`, `zero_`
- `torch.nn`: `Conv2d`, `ConvTranspose2d`, `GroupNorm`, `Identity`, `LayerNorm`, `Linear`, `ModuleList`, `Sequential`, `SiLU`
- `torch.nn.functional`: -

## unets/unet_motion_model
- `torch`: `cat`, `concat`, `is_grad_enabled`, `is_tensor`, `tensor`
- `torch.Tensor`: `contiguous`, `expand`, `flatten`, `permute`, `repeat_interleave`, `reshape`, `to`, `unsqueeze`, `values`
- `torch.nn`: `Conv2d`, `GroupNorm`, `Linear`, `ModuleList`, `SiLU`
- `torch.nn.functional`: -

## unets/unet_spatio_temporal_condition
- `torch`: `is_tensor`, `tensor`, `zeros`
- `torch.Tensor`: `expand`, `flatten`, `repeat_interleave`, `reshape`, `to`
- `torch.nn`: `Conv2d`, `GroupNorm`, `ModuleList`, `SiLU`
- `torch.nn.functional`: -

## unets/unet_stable_cascade
- `torch`: `arange`, `cat`, `is_grad_enabled`, `norm`, `zeros`, `zeros_like`
- `torch.Tensor`: `chunk`, `cos`, `exp`, `float`, `mean`, `mul`, `new_zeros`, `permute`, `sin`, `size`, `to`, `transpose`, `unsqueeze`, `view`
- `torch.nn`: `Conv2d`, `ConvTranspose2d`, `Dropout`, `GELU`, `Identity`, `LayerNorm`, `Linear`, `ModuleList`, `Parameter`, `PixelShuffle`, `PixelUnshuffle`, `Sequential`, `SiLU`, `Upsample`
- `torch.nn.functional`: `interpolate`

## unets/uvit_2d
- `torch`: `cat`, `is_grad_enabled`
- `torch.Tensor`: `chunk`, `flatten`, `permute`, `reshape`, `to`, `view`
- `torch.nn`: `Conv2d`, `Dropout`, `Embedding`, `GELU`, `Linear`, `ModuleList`
- `torch.nn.functional`: `silu`

## upsampling
- `torch`: `arange`, `cat`, `flip`, `outer`, `reshape`, `sum`, `tensor`
- `torch.Tensor`: `contiguous`, `expand`, `new_zeros`, `numel`, `permute`, `reshape`, `squeeze`, `to`, `view`
- `torch.nn`: `Conv1d`, `Conv2d`, `ConvTranspose1d`, `ConvTranspose2d`, `LayerNorm`
- `torch.nn.functional`: `conv2d`, `conv_transpose2d`, `interpolate`, `pad`

## vq_model
- `torch`: -
- `torch.Tensor`: -
- `torch.nn`: -
- `torch.nn.functional`: -
