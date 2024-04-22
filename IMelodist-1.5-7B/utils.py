import numpy as np
from scipy.io.wavfile import write

def save_wav(data, path, sr=32000):
    scaled =  np.int16(data / np.max(np.abs(data)) * 32767)
    write(path,sr,scaled)
