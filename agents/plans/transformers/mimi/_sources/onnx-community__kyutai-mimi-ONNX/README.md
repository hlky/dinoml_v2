---
library_name: transformers.js
base_model:
- kyutai/mimi
license: cc-by-4.0
tags:
  - mimi
  - audio
---

ONNX-compatible weights for https://huggingface.co/kyutai/mimi

## Inference sample code
```py
import onnxruntime as ort

encoder_session = ort.InferenceSession("encoder_model.onnx")
decoder_session = ort.InferenceSession("decoder_model.onnx")

encoder_inputs = {encoder_session.get_inputs()[0].name: dummy_encoder_inputs.numpy()}
encoder_outputs = encoder_session.run(None, encoder_inputs)[0]

decoder_inputs = {decoder_session.get_inputs()[0].name: encoder_outputs}
decoder_outputs = decoder_session.run(None, decoder_inputs)[0]

# Print the results
print("Encoder Output Shape:", encoder_outputs.shape)
print("Decoder Output Shape:", decoder_outputs.shape)
```

## Conversion sample code
```py
import torch
import torch.nn as nn
from transformers import MimiModel

class MimiEncoder(nn.Module):
    def __init__(self, model):
        super(MimiEncoder, self).__init__()
        self.model = model

    def forward(self, input_values, padding_mask=None):
        return self.model.encode(input_values, padding_mask=padding_mask).audio_codes

class MimiDecoder(nn.Module):
    def __init__(self, model):
        super(MimiDecoder, self).__init__()
        self.model = model

    def forward(self, audio_codes, padding_mask=None):
        return self.model.decode(audio_codes, padding_mask=padding_mask).audio_values

model = MimiModel.from_pretrained("kyutai/mimi")
encoder = MimiEncoder(model)
decoder = MimiDecoder(model)

dummy_encoder_inputs = torch.randn((5, 1, 82500))
torch.onnx.export(
    encoder,
    dummy_encoder_inputs,
    "encoder_model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input_values'],
    output_names=['audio_codes'],
    dynamic_axes={
        'input_values': {0: 'batch_size', 1: 'num_channels', 2: 'sequence_length'},
        'audio_codes': {0: 'batch_size', 2: 'codes_length'},
    },
)

dummy_decoder_inputs = torch.randint(100, (4, model.config.num_quantizers, 91))
torch.onnx.export(
    decoder,
    dummy_decoder_inputs,
    "decoder_model.onnx",
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['audio_codes'],
    output_names=['audio_values'],
    dynamic_axes={
        'audio_codes': {0: 'batch_size', 2: 'codes_length'},
        'audio_values': {0: 'batch_size', 1: 'num_channels', 2: 'sequence_length'},
    },
)
```