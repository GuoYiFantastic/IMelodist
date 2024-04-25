import numpy as np
from scipy.io.wavfile import write
import torch

prefix = '🎶'
start = '<somu>'
end = '<eomu>'
audio_codebook_size = 2048 # The quantization codebook size of encodec-32khz 

def save_wav(data, path, sr=32000):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    write(path,sr,scaled)



# derived from script open sourced by AnyGPT: https://github.com/OpenMOSS/AnyGPT/blob/main/anygpt/src/m_utils/anything2token.py
def encodec_to_anygpt(codes: torch.Tensor) -> str:
    music_tokens = []
    encodec_quantized_token = codes[0, 0]
    shape = encodec_quantized_token.shape
    # 第0层维持原值，第1层每个值+music_codebook_size，第2层每个值+music_codebook_size*2，第3层每个值+music_codebook_size*3    
    # 按层堆叠，依次加入第一帧的四层，第二帧的四层，依次类推
    for idx in range(shape[1]):
        for layer_idx in range(shape[0]):
                music_tokens.append(f"<{prefix}{encodec_quantized_token[layer_idx, idx].item() + audio_codebook_size * layer_idx}>")
    return start + ''.join(music_tokens) + end


def encodec_to_anygpt_vec(codes:torch.Tensor) -> str:
    """
    
    Args:
        codes(torch.Tensor) : The encode result of encodec-32khz
    Return:
        The AnyGPT-style music tokens to feed into IMelodist
    """
    music_tokens = []
    encodec_quantized_token = codes[0, 0]
    # shape = encodec_quantized_token.shape
    # 第0层维持原值，第1层每个值+music_codebook_size，第2层每个值+music_codebook_size*2，第3层每个值+music_codebook_size*3    
    # 按层堆叠，依次加入第一帧的四层，第二帧的四层，依次类推
    # for idx in range(shape[1]):
    #     for layer_idx in range(shape[0]):
    #             music_tokens.append(f"<{prefix}{encodec_quantized_token[layer_idx, idx].item() + audio_codebook_size * layer_idx}>")
    
    # a vectorized version
    offset = torch.LongTensor([[0],[2048],[4096],[6144]]).reshape(-1,1)
    encodec_quantized_token += offset
    encodec_quantized_token = encodec_quantized_token.T.reshape(-1)
    music_tokens = list(map(lambda elem: f'<{prefix}{elem.item()}>', encodec_quantized_token))
    return start + ''.join(music_tokens) + end

# test_data = torch.LongTensor([[[[1, 32, 23], [0, 32, 23], [0, 32, 23], [0, 32, 23]]]])
# assert encodec_to_anygpt(test_data) == encodec_to_anygpt_vec(test_data)