---
library_name: transformers
tags: []
---

## Inference script :

```python

import torch, torchaudio
from datasets import load_dataset
from moshi.models import loaders

weight_path = loaders.hf_hub_download("Cnam-LMSSC/mimi_throat_microphone", "kyutai_implementation.safetensors")
model = loaders.get_mimi(weight_path).eval()
model.set_num_codebooks(model.total_codebooks)  # use all codebooks

test_dataset = load_dataset("Cnam-LMSSC/vibravox", "speech_clean", split="test", streaming=True)

audio_48kHz = torch.Tensor(next(iter(test_dataset))["audio.throat_microphone"]["array"])
audio_24kHz = torchaudio.functional.resample(audio_48kHz, orig_freq=48_000, new_freq=24_000)

enhanced_audio_24kHz = model.decode(model.encode(audio_24kHz[None, None, :]))

```

For streaming usage, please refer to this [script](https://github.com/kyutai-labs/moshi/blob/main/scripts/mimi_streaming_test.py)