import os
from datasets import Dataset, Audio
from transformers import EncodecModel, AutoProcessor
from utils import encodec_to_anygpt
import json
import random

# Settings
home = os.environ['HOME']
system = """
I am a music AI assistant named IMelodist based on InternLM2. I have plenty knowledge of basic music theories and am capable of composing beautiful 
music. I am also good at ABC notation which denotes music with pure ASCII characters. I can correct mistakes of an ABC notation score, generate ABC
notation score according to the music given by user and compose music with ABC notation score as well.
"""
encodec_path = f'{home}/models/encodec_32khz'
desc_path = f'{home}/IMelodist-1.5/IMelodist-1.5-7B/music_part1.json'
dataset_path = f'{home}/music-data/part1/music'

# Load model & processor
encodec_model = EncodecModel.from_pretrained(encodec_path)
encodec_processor = AutoProcessor.from_pretrained(encodec_path)
sr = encodec_processor.sampling_rate

# Load the description data
with open(desc_path, 'r',encoding='utf-8') as f:
    descs = json.load(f)

# Make dataset for audio
audio_path = os.listdir(dataset_path)
audio_path_full = list(map(lambda s: f'{dataset_path}/{s}',audio_path))
dataset = Dataset.from_dict({'audio': audio_path_full}).cast_column('audio',Audio(sampling_rate=sr))
file_names = list(map(lambda s: s.split('.')[0],audio_path))

# Make increment pretrain dataset
data_num = len(descs)
pre = data_num // 2
suf = data_num - pre
is_pres = [True] * pre + [False] * suf
random.shuffle(is_pres)
pt_dataset = []
printed = 0
i = 1
for file_name, data, is_pre in zip(file_names, dataset,is_pres):
    print(f'# {i}')
    audio_data = data['audio']['array']
    inputs = encodec_processor(raw_audio=audio_data, sampling_rate=sr, return_tensors="pt")
    codes, scales = encodec_model.encode(inputs["input_values"],inputs['padding_mask'], return_dict=False)
    audio_tokens = encodec_to_anygpt(codes)
    desc = descs[file_name]
    if is_pre:
        final_tokens = system + '\n' + desc + '\n' + audio_tokens
    else:
        final_tokens = system + '\n' + audio_tokens + '\n' + desc
    if printed < 10 and bool(random.getrandbits(1)):
        print(f'\n{final_tokens}')
        printed += 1
    pt_dataset.append({
        'conversation':[
            {
                'system' : "",
                'input': "",
                'output': final_tokens
            }
        ]
    })
    i += 1
with open('./datasets/dataset_anyinstruct_part1.json','w',encoding='utf-8') as f:
    json.dump(pt_dataset,ensure_ascii=False,indent=4)