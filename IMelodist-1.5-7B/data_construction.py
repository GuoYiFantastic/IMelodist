import os
from datasets import Dataset, Audio, load_dataset
from transformers import EncodecModel, AutoProcessor
from utils import encodec_to_anygpt_vec
import json
import random
from music21 import converter
from midi2audio import FluidSynth
import csv


# System and templates
system = """I am a music AI assistant named IMelodist based on InternLM2. I have plenty knowledge of basic music theories and am capable of composing beautiful 
music. I am also good at ABC notation which denotes music with pure ASCII characters. I can correct mistakes of an ABC notation score, generate ABC
notation score according to the music given by user and compose music with ABC notation score as well.
"""
abc_header_template = """Commonly used ABC notation header fields are:
X: reference number
L: unit note length
Q: tempo
M: meter
K: key
Below is a sample header:
{sample_header}
"""
irishman_template = """Here is a sample of music denoted in ABC notation, I should pay attention to these points and their appearance in the sample:
1. The format of a header;
2. The denotation of notes;
3. The denotation of bar lines(\'|\' or \'||\' or \'[|\' or \'|]\');
4. The denotation of repetitions (\'::\' or between \'|:\' and \':|\');
{abc_notation}
"""
abc_piece_template = """X:{num}
M:4/4
L:1/4
K:C
[|{music}|]
"""

# Settings
home = os.environ.get('HOME','./')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
encodec_path = f'{home}/models/encodec_32khz'
desc_path = f'{home}/IMelodist-1.5/IMelodist-1.5-7B/music_part2.json'
any_instruct_dataset_path = f'{home}/music-data/part2/music'
irishman_dataset_path = f'{home}/music-data/irishman'

"""
Any Instruct
"""

def generate_any_instruct():
    # Load model & processor
    encodec_model = EncodecModel.from_pretrained(encodec_path).cuda()
    encodec_processor = AutoProcessor.from_pretrained(encodec_path)
    sr = encodec_processor.sampling_rate

    # Load the description data
    with open(desc_path, 'r',encoding='utf-8') as f:
        descs = json.load(f)

    # Make dataset for audio
    audio_path = os.listdir(any_instruct_dataset_path)
    audio_path_full = list(map(lambda s: f'{any_instruct_dataset_path}/{s}',audio_path))
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
        codes, scales = encodec_model.encode(inputs["input_values"].cuda(),inputs['padding_mask'].cuda(), return_dict=False)
        audio_tokens = encodec_to_anygpt_vec(codes, to_cuda=True)
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
    with open('./datasets/dataset_anyinstruct_part2.json','w',encoding='utf-8') as f:
        json.dump(pt_dataset,f, ensure_ascii=False,indent=4)

"""
ABC header
"""

def generate_header(n: int = 30000):
    unit_note_lens = [f'1/{2 ** i}' for i in range(1, 5)]
    meters = ['2/2','4/4','3/4','6/8','3/8','5/4','9/8','12/8']
    keys = ['A','Am','Ab','Abm','B','Bm','Bb','Bbm','C','Cm','C#','C#m','D','Dm','E','Em','Eb','Ebm','F','Fm','F#','F#m','G','Gm']

    def sample_header_generate(n: int):
        data = []
        for _ in range(n):
            has_tempo = bool(random.getrandbits(1))
            ref = random.randint(1, 256)
            tempo = random.choice([f'1/4={random.randint(96, 200)}',f'1/8={random.randint(96, 200)}',f'3/8={random.randint(96, 200)}'])
            unl = random.choice(unit_note_lens)
            meter = random.choice(meters)
            key = random.choice(keys)
            header = f'X:{ref}\nL:{unl}\n'
            if has_tempo:
                header += f'Q:{tempo}\n'
            header += f'M:{meter}\nK:{key}'
            train_data = system + abc_header_template.replace('{sample_header}',header) 
            data.append({
                'conversation': [
                    {
                        'system':"",
                        'input': "",
                        'output': train_data
                    }
                ]
            })
        with open('./dataset_abc_header.json','w',encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    sample_header_generate(n)


"""
Irishman
"""
def generate_irishman():
    irishman_dataset = load_dataset('sander-wood/irishman',split='train',cache_dir=irishman_dataset_path)
    irishman_data = []
    for d in irishman_dataset:
        train_data = system + '\n' + irishman_template.replace('{abc_notation}',d['abc notation'])
        irishman_data.append({
            'conversation':[
                {
                    'system':'',
                    'input': '',
                    'output': train_data
                }
            ]
        })
    random.shuffle(irishman_data)
    with open('./datasets/dataset_irishman.json','w',encoding='utf-8') as f:
        json.dump(irishman_data, f, ensure_ascii=False, indent=4)

"""
ABC and audio corresponding
"""
def generate_abc_note():
    base_notes_big = ['C','^C','D','_E','E','F','^F','G','_A','A','_B','B']
    base_notes_small = ['c','^c','d','_e','e','f','^f','g','_a','a','_b','b']
    all_notes = []
    # generate A_2 to b^1
    low = ',,,,'
    for i in range(9, 60):
        idx = i % 12
        all_notes.append(base_notes_big[idx] + low)
        if idx == 11:
            low = low[:-1]
            print(i, low)
    # c^2 to c^5
    high = ''
    for i in range(37):
        idx = i % 12
        all_notes.append(base_notes_small[idx] + high)
        if idx == 11:
            high += "\'"
    
    return all_notes
all_notes = generate_abc_note()
print(all_notes)
metadata = open('./datasets/abc_audio/metadata.csv','w',encoding='utf-8',newline='')
writer = csv.writer(metadata)
writer.writerow(['wav_path','abc_notation'])

for i, note in enumerate(all_notes):
    abc_audio = abc_piece_template.replace('{num}',str(random.randint(1, 100))).replace('{music}',note)
    music_stream = converter.parse(abc_audio, format="abc")
    music_stream.write("midi", fp='./test.mid')
    # Please install a fluidsynth CLI before running this, doc: https://github.com/FluidSynth/fluidsynth/wiki/Download
    fs = FluidSynth('../assets/default_sound_font.sf2')
    fs.midi_to_audio('./test.mid', f'./datasets/abc_audio/wav/{i + 1}.wav')
    writer.writerow([f'wav/{i + 1}.wav',abc_audio])

metadata.close()