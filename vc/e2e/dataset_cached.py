# encoding: utf-8
import numpy as np
import glob
import hp
import torch
from torch.utils.data.dataloader import default_collate
from utils import *

def pad_zeros(xs):

    n_batch = len(xs)
    max_len = max(x.shape[1] for x in xs)
    pad = np.zeros((n_batch, xs[0].shape[0], max_len))

    for i in range(n_batch):
        pad[i, :, :xs[i].shape[1]] = xs[i]

    return pad

def collate_fn(batch):
    ref, mag, mel, linear = zip(*batch)

    # Must be of Same Dim
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

        self.data_files = []
        for sid in hp.speakers[phase]:
            self.data_files += glob.glob(ds[phase]['audio'].format(sid))
            
        self.list = {}
        self.len = 0

        for x in self.data_files:
            self.list[self.len] = {
                'mel'   : x, 
                'ref'   : x.replace('mel', 'ref'), 
                'mag'   : x.replace('mel', 'mag'), 
                'linear': x.replace('mel', 'linear')
                }
            self.len += 1
        print('Load {} part'.format(phase))

    def __getitem__(self, idx):
        ref = np.load(self.list[idx]['ref'])
        mag = np.load(self.list[idx]['mag'])
        mel = np.load(self.list[idx]['mel'])
        linear = np.load(self.list[idx]['linear'])

        return ref, mag, mel, linear

    def __len__(self):
        return self.len

