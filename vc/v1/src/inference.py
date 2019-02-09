# TODO: Try Waveglow
import sys
import os
import numpy as np
import torch

from hparams import create_hparams
from model import Tacotron2
from train import load_model
from audio import AudioProcessor

save_dir = '../vis_v1/{}_'
ap = AudioProcessor()

hparams = create_hparams()
hparams.sampling_rate = 22050
checkpoint_path = "../model/checkpoint_new"

def get_input(fpath):
    # Later Create these instead of loading.
    mag_path = fpath.replace('.wav', '.npy').replace('wav', 'mag')
    ref_path = fpath.replace('.wav', '.npy').replace('wav', 'ref')

    mag = torch.from_numpy(np.load(mag_path)).float().unsqueeze(0)
    ref = torch.from_numpy(np.load(ref_path)).float().unsqueeze(0)

    return (ref.cuda(), mag.cuda())

def get_full_input(fpath):
    # Later Create these instead of loading.
    mag_path = fpath.replace('.wav', '.npy').replace('wav', 'mag')
    ref_path = fpath.replace('.wav', '.npy').replace('wav', 'ref')

    mel = torch.from_numpy(ap.melspectrogram(ap.load_wav(fpath, ap.sample_rate)[:ap.sample_rate * 3])).float().unsqueeze(0)
    mag = torch.from_numpy(np.load(mag_path)).float().unsqueeze(0)
    ref = torch.from_numpy(np.load(ref_path)).float().unsqueeze(0)

    mag_len = torch.LongTensor([mag.size(2)])
    mel_len = torch.LongTensor([mel.size(2)])

    return (ref.cuda(), mag.cuda(), mag_len.cuda(), mel.cuda(), mel_len.cuda())

def synthesis(fpath, full=False):
    
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model = model.cuda()
    model.eval()
    if full:
        inputs = get_full_input(fpath)
        mel_outputs, mel_outputs_postnet, _, alignments = model(inputs)
    else:
        inputs = get_input(fpath)
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(inputs)
    return mel_outputs, mel_outputs_postnet, alignments

if __name__ == "__main__":
    # prepare data
    fpath = '../data/RRBI/wav/arctic_a0005.wav'

    # Forward Pass
    mel_outputs, mel_outputs_postnet, alignments = synthesis(fpath, True)

    # Save output
    np.save(save_dir.format('mel') + 'mel_post.npy', mel_outputs_postnet.data.cpu().numpy()[0])
    np.save(save_dir.format('mel') + 'mel.npy', mel_outputs.data.cpu().numpy()[0])
    np.save(save_dir.format('att') + 'align.npy', alignments.data.cpu().numpy()[0])
    ap.save_wav(save_dir.format('wav') + 'post.wav', ap.inv_mel_spectrogram(mel_outputs_postnet.data.cpu().numpy()[0]))
    ap.save_wav(save_dir.format('wav') + 'mel.wav', ap.inv_mel_spectrogram(mel_outputs.data.cpu().numpy()[0]))