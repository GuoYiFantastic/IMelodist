import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
internlm2_path = os.environ['HOME'] + '/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b'
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch.nn as nn
import torch
'''
Stage 1: Tokenizer fusion
'''
# print("Start of tokenizer fusion")
# prefix = '🎶'
# internlm2_tokenizer = AutoTokenizer.from_pretrained(internlm2_path,trust_remote_code=True)
# music_tokens = []
# for i in range(8192):
#     music_tokens.append(f'<{prefix}{i}>')
# music_tokens.append('<somu>')
# music_tokens.append('<eomu>')
# internlm2_tokenizer.add_tokens(music_tokens)
# internlm2_tokenizer.save_pretrained('./imelodist-1_5')
# print("End of tokenizer fusion")

'''
Stage 2: Embedding layer fusion
'''
# print("Start of embedding layer fusion")
# # Load tokenizer and models
# internlm2 = AutoModel.from_pretrained(internlm2_path, trust_remote_code=True)
# anygpt = AutoModel.from_pretrained("fnlp/AnyGPT-base")
# tokenizer = AutoTokenizer.from_pretrained('./imelodist-1_5', trust_remote_code=True)
# vocab_size = tokenizer.vocab_size
# # new LM head
# new_output = nn.Linear(4096, vocab_size + 8194, bias=False)
# new_output.weight.data.normal_(mean=0.0,std=0.02)
# new_output = new_output.cpu()
# internlm2.model.output = new_output
# with torch.no_grad():
#     new_embed = nn.Embedding(vocab_size + 8194, 4096)

#     new_embed = new_embed.cpu()

#     # original internlm2-chat-7b embeddings
#     new_embed.weight[:vocab_size, ...] = internlm2.model.tok_embeddings.weight

#     # concat AnyGPT music embeddings
#     new_embed.weight[vocab_size:, ...] = anygpt.embed_tokens.weight[41224:49418] # 41224 = <🎶0> 49415 = <🎶8191> 49416 = <somu> 49417 = <eomu>
#     internlm2.model.tok_embeddings = new_embed

      # save the model
#     internlm2.save_pretrained('./imelodist-1_5')
# print("End of embedding layer fusion")
'''
Stage 3: Test
'''
# config = AutoConfig.from_pretrained('imelodist-1_5',trust_remote_code=True)
# imelodist = AutoModel.from_config(config, trust_remote_code=True)
# imelodist.eval()
# print(imelodist.output.weight.shape)