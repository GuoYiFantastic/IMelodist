import os
from datasets import Dataset, Audio
from transformers import EncodecModel, AutoProcessor
import time
from utils import save_wav
home = os.environ['HOME']
encodec_path = f'{home}/models/encodec_32khz'
dataset_path = './any_instruct_sample'

# load the model + processor (for pre-processing the audio)
model = EncodecModel.from_pretrained(encodec_path)
processor = AutoProcessor.from_pretrained(encodec_path)

# cast the audio data to the correct sampling rate for the model
sr = processor.sampling_rate

audio_path = os.listdir(dataset_path)
audio_path = list(map(lambda s: f'{dataset_path}/{s}',audio_path))
dataset = Dataset.from_dict({'audio': audio_path}).cast_column('audio',Audio(sampling_rate=sr))

tot_enc_time = 0
tot_dec_time = 0
for path, d in zip(audio_path, dataset):
    audio_data = d['audio']['array']
    
    # encode
    start = time.time()
    inputs = processor(raw_audio=audio_data, sampling_rate=sr, return_tensors="pt")
    # explicitly encode then decode the audio inputs
    codes, scales = model.encode(inputs["input_values"],inputs['padding_mask'], return_dict=False)
    end = time.time()
    tot_enc_time += (end - start)
    
    # decode
    start = time.time()
    audio_values = model.decode(codes, scales)[0]
    save_wav(audio_values[0][0].detach().numpy(), f'{dataset_path}/decode_{path.split("/")[2]}', sr)
    end = time.time()
    tot_dec_time += (end - start)
print(f'Average encode time: {tot_enc_time / len(audio_path)}s')
print(f'Average decode time: {tot_dec_time / len(audio_path)}s')