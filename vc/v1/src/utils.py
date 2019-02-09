import numpy as np
from scipy.io.wavfile import read
from scipy.signal import resample
import torch

def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, out=torch.cuda.LongTensor(max_len))
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path):

    #data, sampling_rate = librosa.load(full_path, sr=22050)
    #data = data[:min(3 * sampling_rate, len(data))]
    sampling_rate, data = read(full_path)
    if sampling_rate != 22050:
        data = resample(data, 22050)
        sampling_rate = 22050
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

def to_gpu(x):
    x = x.contiguous()

    if torch.cuda.is_available():
        x = x.cuda(non_blocking=True)
    return torch.autograd.Variable(x)
