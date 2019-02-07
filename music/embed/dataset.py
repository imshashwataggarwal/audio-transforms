# TODO:  1. Labels.txt file with: filepath label

# encoding: utf-8
import numpy as np
import hp
import torch
from torch.utils.data.dataloader import default_collate
from scripts.extract_features import load_specs

def pad_zeros(xs):

    n_batch = len(xs)
    pad = np.zeros((n_batch, xs[0].shape[0], 300))

    for i in range(n_batch):
        l = min(300, xs[i].shape[1])
        pad[i, :, :l] = xs[i][:, :l]
    return pad

def collate_fn(batch):
    wavs, labels = zip(*batch)

    spec = load_specs(wavs)
    spec = pad_zeros(spec)
    spec = default_collate(spec)

    labels = default_collate(labels)

    return [wavs, labels]

class MyDataset():

    def __init__(self, phase):
        self.list = {}
        self.len = 0
        with open('labels.txt', 'w') as f:
            for line in f:
                wav, label = line.strip().split(' ')
                self.list[self.len] = {'wav': wav, 'label': hp.m2i.get(label, -1)}
                self.len += 1
        print('Load {} part'.format(phase))

    def __getitem__(self, idx):
        return self.list[idx]['wav'], self.list[idx]['label']

    def __len__(self):
        return self.len

