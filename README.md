---
inference: false
---

![encodec image](https://github.com/facebookresearch/encodec/raw/2d29d9353c2ff0ab1aeadc6a3d439854ee77da3e/architecture.png)

# Model Card for EnCodec

This model card provides details and information about EnCodec 32kHz, a state-of-the-art real-time audio codec developed by Meta AI. 
This EnCodec checkpoint was trained specifically as part of the [MusicGen project](https://huggingface.co/docs/transformers/main/model_doc/musicgen),
and is intended to be used in conjuction with the MusicGen models.

## Model Details

### Model Description

EnCodec is a high-fidelity audio codec leveraging neural networks. It introduces a streaming encoder-decoder architecture with quantized latent space, trained in an end-to-end fashion. 
The model simplifies and speeds up training using a single multiscale spectrogram adversary that efficiently reduces artifacts and produces high-quality samples. 
It also includes a novel loss balancer mechanism that stabilizes training by decoupling the choice of hyperparameters from the typical scale of the loss. 
Additionally, lightweight Transformer models are used to further compress the obtained representation while maintaining real-time performance. This variant of EnCodec is 
trained on 20k of music data, consisting of an internal dataset of 10K high-quality music tracks, and on the ShutterStock and Pond5 music datasets.

- **Developed by:** Meta AI
- **Model type:** Audio Codec

### Model Sources

- **Repository:** [GitHub Repository](https://github.com/facebookresearch/audiocraft)
- **Paper:** [Simple and Controllable Music Generation](https://arxiv.org/abs/2306.05284)

## Uses
<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

EnCodec can be used directly as an audio codec for real-time compression and decompression of audio signals. 
It provides high-quality audio compression and efficient decoding. The model was trained on various bandwiths, which can be specified when encoding (compressing) and decoding (decompressing).
Two different setup exist for EnCodec: 

- Non-streamable: the input audio is split into chunks of 1 seconds, with an overlap of 10 ms, which are then encoded.
- Streamable: weight normalizationis used on the convolution layers, and the input is not split into chunks but rather padded on the left.

### Downstream Use

This variant of EnCodec is designed to be used in conjunction with the official [MusicGen checkpoints](https://huggingface.co/models?search=facebook/musicgen-).
However, it can also be used standalone to encode audio files.

## How to Get Started with the Model

Use the following code to get started with the EnCodec model using a dummy example from the LibriSpeech dataset (~9MB). First, install the required Python packages:

```
pip install --upgrade pip
pip install --upgrade transformers datasets[audio] 
```

Then load an audio sample, and run a forward pass of the model:

```python
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor


# load a demonstration datasets
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_48khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[0]["audio"]["array"]

# pre-process the inputs
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

# explicitly encode then decode the audio inputs
encoder_outputs = model.encode(inputs["input_values"], inputs["padding_mask"])
audio_values = model.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]

# or the equivalent with a forward pass
audio_values = model(inputs["input_values"], inputs["padding_mask"]).audio_values
```

## Evaluation

For evaluation results, refer to the [MusicGen evaluation scores](https://huggingface.co/facebook/musicgen-large#evaluation-results).

## Summary

EnCodec is a state-of-the-art real-time neural audio compression model that excels in producing high-fidelity audio samples at various sample rates and bandwidths. 
The model's performance was evaluated across different settings, ranging from 24kHz monophonic at 1.5 kbps to 48kHz stereophonic, showcasing both subjective and 
objective results. Notably, EnCodec incorporates a novel spectrogram-only adversarial loss, effectively reducing artifacts and enhancing sample quality. 
Training stability and interpretability were further enhanced through the introduction of a gradient balancer for the loss weights. 
Additionally, the study demonstrated that a compact Transformer model can be employed to achieve an additional bandwidth reduction of up to 40% without compromising 
quality, particularly in applications where low latency is not critical (e.g., music streaming).


## Citation

**BibTeX:**

```
@misc{copet2023simple,
      title={Simple and Controllable Music Generation}, 
      author={Jade Copet and Felix Kreuk and Itai Gat and Tal Remez and David Kant and Gabriel Synnaeve and Yossi Adi and Alexandre Défossez},
      year={2023},
      eprint={2306.05284},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```