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
# the distance(in semitone) between each note and root of every common chord
# Reference: https://www.bilibili.com/video/BV1Xa41117Wy/?vd_source=96a5e3c8d3ee54a5b89d246a66d277e3 
# chord_dict = {
    
    
# }

# index 0 is the original chord, others are inversions
common_chord_dict = {
    'maj': [[4, 7], [-5, 4], [-8, -5]],
    'min': [[3, 7], [-5, 3], [-9, -5]],
    'maj7': [[4, 7, 11], [-1, 4, 7], [-5, -1, 4], [-8, -5, -1]],
    'min7': [[3, 7, 10], [-2, 3, 7], [-5, -2, 3], [-9, -5, -2]],
    '7':[[4, 7, 10], [-2, 4, 7], [-5, -2, 4], [-8, -5, -2]],
    '9': [[4, 7, 10, 14], [2, 4, 7, 10], [-2, 2, 4, 7], [-5, -2, 2, 4], [-8, -5, -2, 2]],
    'dim7': [[3, 6, 9], [-3, 3, 6], [-6, -3, 3], [-9, -6, -3]],
    '7/8': [[4, 7, 10, 12]]
}
uncommon_chord_dict = {
    'dim': [3, 6],
    'add6': [4, 7, 9],
    'madd6': [3, 7, 9],
    'aug': [4, 8],
    'maj9':[4, 7, 11, 14],
    'min9': [3, 7, 10, 14],
    # 'maj11': [4, 7, 11, 14, 17],
    # 'min11': [3, 7, 10, 14, 17],
    # '11': [4, 7, 10, 14, 17],
    # 'maj13': [4, 7, 11, 14, 17, 21],
    # 'min13': [3, 7, 10, 14, 17, 21],
    # '13': [4, 7, 10, 14, 17, 21],
    'sus2': [2, 7],
    'sus4': [5, 7],
    'dim7': [3, 6, 9],
    'augmaj7': [4, 8, 11],
    'add4': [4, 5, 7],
    'madd11': [3, 5, 7],
    '7add4': [4, 5, 7, 10],
    'add9': [2, 4, 7],
    'madd9': [2, 3, 7],
    'mM7': [3, 7, 11],
    'mM9': [3, 7, 11, 14],
    # 'mM11': [3, 7, 11, 14, 17],
    '7sus2': [2, 7, 10],
    '7sus4': [5, 7, 10],
    '9sus4': [5, 7, 10, 14],
    'dimadd4': [3, 5, 6],
    'augadd2': [2, 4, 8],
    'M7sus4': [5, 7, 11],
    'm7add4': [3, 5, 7, 10],
    '6/9': [4, 7, 9, 14],
    '7/6': [4, 7, 9, 11],
    # '9/6': [4, 7, 9, 11, 14],
    'M(b5)': [4, 6],
    'm(#5)': [3, 8],
    'M7(b5)': [4, 6, 11],
    'm7(b5)': [3, 6, 11],
    '7(b5)': [4, 6, 10],
    'm6/9': [3, 7, 9, 14],
    'm7(#5)': [3, 8, 11],
    '7(#5)': [4, 8, 10],
    'M7(b9)': [4, 7, 11, 13],
    'm7(b9)': [3, 7, 10, 13],
    '7(b9)': [4, 7, 10, 13],
    'M(#9)': [3, 4, 7],
    'M7(#9)': [4, 7, 11, 15],
    'm7(#9)': [3, 7, 10, 15],
    '7(#9)': [4, 7, 10, 15],
    # 'M7(#11)': [4, 7, 11, 14, 18],
    # 'm7(#11)': [3, 7, 10, 14, 18],
    # '7(#11)': [4, 7, 10, 14, 18],
    'M(b9)': [1, 4, 7],
    'M(#11)': [4, 6, 7],
    # 'm11(b5)': [3, 6, 10, 14, 17],
    '9(b5)': [4, 6, 10, 14],
    '9(#5)': [4, 8, 10, 14],
    'M9(#5)': [4, 8, 11, 14],
    # '11(b9)': [4, 7, 10, 13, 17],
    'aug7(#9)':[4, 8, 10, 15],
    'aug7': [4, 8, 10],
    # '13(b9)': [4, 7, 10, 13, 17, 21],
    # '13(#11)': [4, 7, 10, 14, 18, 21],
    'M(b9b5)': [4, 6, 11, 13],
    'M(b9#5)': [4, 8, 11, 13],
    # 'M(b9#11)': [4, 7, 11, 13, 18]
}


# Settings
home = os.environ.get('HOME','./')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
#os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
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
random.seed(42)
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
def generate_note_audio():
    """
    Generate all 88 notes in a piano
    """
    metadata = open('./datasets/abc_audio/metadata.csv','w',encoding='utf-8',newline='')
    writer = csv.writer(metadata)
    writer.writerow(['wav_path','abc_notation','chord_name'])

    for i, note in enumerate(all_notes):
        abc_audio = abc_piece_template.replace('{num}',str(random.randint(1, 100))).replace('{music}',note)
        music_stream = converter.parse(abc_audio, format="abc")
        music_stream.write("midi", fp='./test.mid')
        # Please install a fluidsynth CLI before running this, doc: https://github.com/FluidSynth/fluidsynth/wiki/Download
        fs = FluidSynth('../assets/default_sound_font.sf2')
        fs.midi_to_audio('./test.mid', f'./datasets/abc_audio/wav/notes/{i + 1}.wav')
        writer.writerow([f'wav/notes/{i + 1}.wav', abc_audio, '-'])

    metadata.close()

def generate_interval_audio(n: int = 1000):
    """
    generate intervals, steps:
    1. randomly choose a root 
    2. limit the range in 2 octaves: upper one and lower one, i.e. 24 semitones
    3. generate a random interval
    4. if retried for 200 times, get out of the loop
    """
    metadata = open('./datasets/abc_audio/metadata.csv','a',encoding='utf-8',newline='')
    writer = csv.writer(metadata)
    generated = []
    retry = 0
    while len(generated) < n:
        i = len(generated)
        if retry == 1000:
            break
        root_idx = random.randint(0, 87)
        note_range = all_notes[root_idx: min(87, root_idx + 12) + 1]
        other_note = random.choice(note_range)
        if other_note == all_notes[root_idx]:
            continue
        interval = f'[{all_notes[root_idx]}{other_note}]'
        if interval in generated:
            retry += 1
            continue
        generated.append(interval)
        abc_audio = abc_piece_template.replace('{num}',str(random.randint(1, n * 3))).replace('{music}',interval)
        music_stream = converter.parse(abc_audio, format="abc")
        music_stream.write("midi", fp='./test.mid')
        fs = FluidSynth('../assets/default_sound_font.sf2')
        fs.midi_to_audio('./test.mid', f'./datasets/abc_audio/wav/intervals/{i + 1}.wav')
        writer.writerow([f'wav/intervals/{i + 1}.wav', abc_audio, '-'])
    print(f"{len(generated)} of {n} generated")
    metadata.close()

def generate_common_chord_audio(n: int = 4000):
    """
    generate common chord audios, steps:
    1. randomly choose a root
    2. randomly choose a inversion
    3. choose a type of chord in the dict
    4. generate a random chord
    5. convert to wave form
    """
    metadata = open('./datasets/abc_audio/metadata.csv','a',encoding='utf-8',newline='')
    writer = csv.writer(metadata)
    chords = list(common_chord_dict.items())
    generated = []
    retry = 0
    while len(generated) < n:
        if retry == 1000:
            break
        i = len(generated)

        # step 1
        root_idx = random.randint(0, 87)
        notes = [all_notes[root_idx]]

        # step 2
        chord_class, distances = random.choice(chords)
        chord_name = all_notes[root_idx] + chord_class

        # step 3
        distance_idx = random.randint(0, len(distances) - 1)
        distance = distances[distance_idx]
        if distance_idx != 0:
            if root_idx + distance[0] >= 0 and root_idx + distance[0] <= 87:
                chord_name = chord_name + f'/{all_notes[root_idx + distance[0]]}'
        
        # step 4
        if chord_name in generated:
            retry += 1
            continue
        for d in distance:
            idx = root_idx + d
            if idx < 0 or idx > 87:
                break
            notes.append(all_notes[idx])
        
        if len(notes) != len(distance) + 1:
            continue

        generated.append(chord_name)
        chord = f"[{''.join(notes)}]"
        abc_audio = abc_piece_template.replace('{num}',str(random.randint(1, n * 3))).replace('{music}',chord)

        # step 5
        music_stream = converter.parse(abc_audio, format="abc")
        music_stream.write("midi", fp='./test.mid')
        fs = FluidSynth('../assets/default_sound_font.sf2')
        fs.midi_to_audio('./test.mid', f'./datasets/abc_audio/wav/common_chords/{i + 1}.wav')
        writer.writerow([f'wav/common_chords/{i + 1}.wav',abc_audio, chord_name])
    print(f"{len(generated)} of {n} generated")
    metadata.close()


def generate_uncommon_chord_audio(n: int = 2000):
    """
    generate uncommon chord audios, steps:
    1. randomly choose a root 
    2. choose a type of chord in the dict
    3. generate a random chord
    4. convert to wave form
    """
    metadata = open('./datasets/abc_audio/metadata.csv','a',encoding='utf-8',newline='')
    writer = csv.writer(metadata)
    chords = list(uncommon_chord_dict.items())
    generated = []
    retry = 0
    while len(generated) < n:
        i = len(generated)
        if retry == 1000:
            break
        # step 1
        root_idx = random.randint(0, 87)
        notes = [all_notes[root_idx]]

        # step 2
        chord_class, distances = random.choice(chords)
        chord_name = all_notes[root_idx] + chord_class

        # step 3
        if chord_name in generated:
            retry += 1
            continue
        for d in distances:
            idx = root_idx + d
            if idx > 87:
                break
            notes.append(all_notes[idx])
        
        if len(notes) != len(distances) + 1:
            continue

        generated.append(chord_name)
        chord = f"[{''.join(notes)}]"
        abc_audio = abc_piece_template.replace('{num}',str(random.randint(1, n * 3))).replace('{music}',chord)

        # step 4
        music_stream = converter.parse(abc_audio, format="abc")
        music_stream.write("midi", fp='./test.mid')
        fs = FluidSynth('../assets/default_sound_font.sf2')
        fs.midi_to_audio('./test.mid', f'./datasets/abc_audio/wav/uncommon_chords/{i + 1}.wav')
        writer.writerow([f'wav/uncommon_chords/{i + 1}.wav',abc_audio, chord_name])
    print(f"{len(generated)} of {n} generated")
    metadata.close()

# generate_note_audio()
# generate_interval_audio()
# generate_common_chord_audio()
generate_uncommon_chord_audio(n=500)

