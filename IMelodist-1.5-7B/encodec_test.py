import os
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import random
# load a demonstration datasets
librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained("facebook/encodec_32khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_32khz")

# cast the audio data to the correct sampling rate for the model
librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=processor.sampling_rate))
audio_sample = librispeech_dummy[random.choice(list(range(len(librispeech_dummy))))]["audio"]["array"]
audio_sample = audio_sample.reshape(2, -1)[0]
# pre-process the inputs
inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")
print(model.quantizer.num_quantizers)
# explicitly encode then decode the audio inputs
codes, scales = model.encode(inputs["input_values"], inputs["padding_mask"], return_dict=False)

# print(len(codes))
# print(len(codes[0]))
print(codes[0][0][0])
print(scales)
