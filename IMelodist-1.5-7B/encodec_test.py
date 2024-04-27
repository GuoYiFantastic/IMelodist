import os
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor
from utils import save_wav
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
home = os.environ['HOME']
encodec_path = f'{home}/models/encodec_32khz'
# load a demonstration datasets
# librispeech_dummy = load_dataset("hf-internal-testing/librispeech_asr_dummy",'clean',split='validation')

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained(encodec_path)
# processor = AutoProcessor.from_pretrained(encodec_path)

# # cast the audio data to the correct sampling rate for the model
# sr = processor.sampling_rate
# librispeech_dummy = librispeech_dummy.cast_column("audio", Audio(sampling_rate=sr))
# audio_sample = librispeech_dummy[1]["audio"]["array"]

# # save the original audio data as a wave file
# save_wav(audio_sample,'./ori.wav',sr=sr)
# # pre-process the inputs
# inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

# explicitly encode then decode the audio inputs
# codes, scales = model.encode(inputs["input_values"], return_dict=False)
# print(codes.shape)
codes = torch.randint(0,2048,(4,253),dtype=torch.int64)
codes = codes.unsqueeze(0).unsqueeze(0)
scales = [None]
# decode the quantized tokens to audio data and save
audio_values = model.decode(codes, scales)[0]
save_wav(audio_values[0][0].detach().numpy(),'./decode.wav')