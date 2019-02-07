# encoding: utf-8
import numpy as np
import glob
import hp
import torch
from torch.utils.data.dataloader import default_collate
from utils import *
from scripts.extract_features import load_specs

def pad_zeros(xs):

    n_batch = len(xs)
    max_len = max(x.shape[1] for x in xs)
    pad = np.zeros((n_batch, xs[0].shape[0], max_len))

    for i in range(n_batch):
        pad[i, :, :xs[i].shape[1]] = xs[i]

    return pad

def pad_ref(xs):

    n_batch = len(xs)
    pad = np.zeros((n_batch, xs[0].shape[0], 300))

    for i in range(n_batch):
        l = min(300, xs[i].shape[1])
        pad[i, :, :l] = xs[i, :, :l]

    return pad

def collate_fn(batch):
    wavs = zip(*batch)

    ref, mag, mel, linear = load_specs(list(wavs))

    # Must be of Same Dim
    ref = pad_ref(ref)
    ref = default_collate(ref)

    mag_len = np.array([m.shape[1] for m in mag])
    mag = pad_zeros(mag)
    mag = default_collate(mag)
    mag_len = torch.LongTensor(mag_len)

    mel_len = [m.shape[1] + 1 for m in mel] # +1 for zero-frame
    stop = [np.array([0.] * (m - 1)) for m in mel_len]
    stop = prepare_stop_target(stop, hp.r)
    mel_len = torch.LongTensor(mel_len)

    linear = prepare_tensor(linear, hp.r)
    mel = prepare_tensor(mel, hp.r)
    mel = mel.transpose(0, 2, 1)
    mel = default_collate(mel)
    linear = linear.transpose(0, 2, 1)
    linear = default_collate(linear)

    return [ref, mag, mag_len, mel, mel_len, linear, stop]

class MyDataset():

    def __init__(self, phase, ds):

        data_files = []
        for sid in hp.speakers[phase]:
            data_files += glob.glob(ds[phase]['audio'].format(sid))
            
        self.list = {}
        self.len = 0

        for x in data_files:
            self.list[self.len] = {'wav'   : x}
            self.len += 1
        print('Load {} part'.format(phase))

    def __getitem__(self, idx):
        return self.list[idx]['wav']

    def __len__(self):
        return self.len

