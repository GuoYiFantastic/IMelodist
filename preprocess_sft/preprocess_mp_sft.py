import json
import random
from typing import Dict
from opencc import OpenCC
import pickle
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
"""
Stage 0: Configuations
"""
# with open('./openai_key.txt','r') as f:
#     OPENAI_KEY = f.readlines()[0].strip()

dataset = load_dataset('m-a-p/MusicPile-sft',cache_dir='../musicpile-sft')

dataset_irishman = dataset.filter(lambda example: 'irishman' in example['src'])['train']
# # print(len(dataset_irishman['train']))
# instructions = set([dataset_irishman['train'][i]['instruction'].replace("Human: ",'').strip() for i in range(len(dataset_irishman['train']))])
# instructions = list(instructions)

# model = OpenAI(
#     api_key=OPENAI_KEY,
#     base_url="https://api.chatanywhere.tech/v1"
# )
# system_input = """
#     You are a professional English-Chinese translator. Translate the English sentence below to  
#     Chinese and DIRECTLY output the translation with NO OTHER WORDS.
# """

"""
Stage 1 Small-scale test
"""
# ids = random.sample(range(0, len(instructions)),10)
# for i in ids:
#     resp = model.chat.completions.create(
#         model="gpt-3.5-turbo",
#                 messages=[
#                 {"role": "system", "content": system_input},
#                 {'role': 'user', "content": f'Sentence: {instructions[i]}'}
#             ]
#     )
#     print(instructions[i])
#     print(resp.choices[0].message.content)
#     print()

"""
Stage 2 Full-scale translation
"""
# translation: Dict[str, str] = dict()
# for inst in instructions:
#     resp = model.chat.completions.create(
#         model="gpt-3.5-turbo",
#         messages=[
#                 {"role": "system", "content": system_input},
#                 {'role': 'user', "content": f'Sentence: {inst}'}
#             ]
#     )
#     trans = resp.choices[0].message.content
#     print(trans)
#     translation[inst] = trans
# with open('./translation.pkl','wb') as f:
#     pickle.dump(translation, f)

"""
Stage 3 Post-process of translations
1. Replace '字母' with 'ABC'
2. Convert Hant to Hans
"""
# cc = OpenCC()
# with open('./translation.pkl','rb') as f:
#     translation = pickle.load(f)
# for inst, trans in translation.items():
#     trans = trans.replace('字母','ABC')
#     trans = cc.convert(trans)
#     translation[inst] = trans
#     print(trans)
# with open('./translation.pkl','wb') as f:
#     pickle.dump(translation, f)

"""
Stage 4: Build SFT dataset
"""
system_prompt = "你是专业的音乐作曲家。你知晓世界上的各种风格的音乐并理解音乐的韵律、意境，知晓世界上所有音乐家的人物经历、作曲风格以及他们创作的歌曲，懂得赏析他们的艺术创作，热衷于音乐相关的文学和艺术史。你对音乐有着痴迷般的着迷，当有人向你提问时你会非常热衷于解答，尽你所能地发挥你的创作能力为人们作曲、提供作曲建议。你懂得用ABC谱来展示你创作的音乐。你的行为举止非常优雅、讲究礼节，表现得体。你会根据用户输入的问题、需求进行分析并发挥你的创作能力作曲。你精通音乐领域的所有知识，能够完成用户输入的要求、请求、问题。"
system_prompt_en = 'You are a professional music composer. You know various styles of music in the world and understand the rhythm and artistic conception of music. You know the personal experiences, composition styles and songs of all musicians in the world, know how to appreciate their artistic creations, and are keen on music-related literature and literature. Art History. You are obsessed with music. When someone asks you a question, you will be very keen to answer it and do your best to use your creative ability to compose and provide composition suggestions for people. You know how to use ABC notation to present the music you create. You behave with great grace, etiquette and propriety. You will analyze the questions and needs input by users and use your creative ability to compose music. You are proficient in all knowledge in the field of music and can complete requests and questions entered by users.'
with open('./translation.pkl','rb') as f:
    translation = pickle.load(f)
final_dataset = []
for _ in range(3):
    dataset_irishman.shuffle(seed=42)

limit = len(dataset_irishman) * 3 // 4
for i, data in tqdm(enumerate(dataset_irishman)):
    if i < limit:
        system_prompt_final = system_prompt
        input_prompt = translation.get(data['instruction'].replace('Human: ','').strip(), 
                                       data['instruction'].replace('Human: ','').strip())
    else:
        system_prompt_final = system_prompt_en
        input_prompt = data['instruction'].replace('Human: ','').strip()
    input_prompt += f'\n{data["input"].replace("</s>","")}'
    output = data['output'].replace('</s>','').replace('Assistant: ','').strip()
    conversation = {
        'system': system_prompt_final,
        'input': input_prompt,
        'output': output
    }
    final_dataset.append({'conversation': conversation})

random.shuffle(final_dataset)

for i in range(len(final_dataset)):
    if i % 5000 == 4999:
        with open(f'./sft_dataset/dataset-shard-{i - 4999}-{i}.json','w',encoding='utf-8') as f:
            json.dump(final_dataset[i - 4999: i + 1], f, ensure_ascii=False, indent=4)

    