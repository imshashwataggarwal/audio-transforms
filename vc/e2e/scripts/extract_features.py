'''
Extract spectrograms and save them to file for training
'''
import os
import sys
import time
import librosa
import numpy as np
from audio import AudioProcessor
import math
import random
from pydub import AudioSegment

import warnings
warnings.filterwarnings("ignore")

mag_ap = AudioProcessor(sample_rate=16000, num_mels=80, min_level_db=-100, frame_shift_ms=10, frame_length_ms=20, ref_level_db=20, num_freq=161, power=1.5, preemphasis=0.97, signal_norm=True, symmetric_norm=None, max_norm=1, mel_fmin=50, mel_fmax=8000, clip_norm=True, griffin_lim_iters=60, do_trim_silence=True)
ap = AudioProcessor()

def vec2frames( vec, Nw, Ns ):
    # length of the input vector
    L = len( vec )
    # number of frames
    M = math.floor((L-Nw)/Ns+1)

    # compute index matrix in the direction ='rows'
    indf = Ns * np.array(list(range(0, M)))
    inds = np.array(list(range(0, Nw)))
    indexes = np.repeat(indf.reshape(-1,1), Nw ,1) + np.repeat(inds.reshape(1,-1), M ,0)

    # divide the input signal into frames using indexing
    frames = vec[indexes]
    window = np.hanning( Nw )
    frames = np.matmul(frames, np.diag( window ))
    return frames, indexes

def extract_specs(file_path):
    x = ap.load_wav(file_path, ap.sample_rate)

    mel = ap.melspectrogram(x.astype('float32')).astype('float32')
    lin = ap.spectrogram(x.astype('float32')).astype('float32')
    mag = mag_ap.spectrogram(x.astype('float32')).astype('float32')

    return mel, lin, mag

def extract_refs(fpath=None, fs=16000, Tw=25, Ts=10, alpha=0.97, nfft=512):
    # Loading sound file
    wav = AudioSegment.from_wav(fpath)
    if wav.duration_seconds < 3:
        silence = AudioSegment.silent(duration= (3.1 - wav.duration_seconds)* 1000)
        wav = wav + silence

    hi = wav.duration_seconds - 3
    lo = random.random() * hi
    wav = wav[lo * 1000: (lo + 3) * 1000]

    y = np.array(wav.get_array_of_samples()).astype('float32')

    # Trimming
    y = librosa.core.resample(y, 44100, 16000)
    y, _ = librosa.effects.trim(y)

    # Preemphasis
    y = np.append(y[0], y[1:] - alpha * y[:-1])

    Nw = int(1e-3 * Tw * fs)
    Ns = int(1e-3 * Ts * fs)

    frames, _ = vec2frames(y, Nw, Ns)

    mag = np.abs(np.fft.fft(frames, n=512))
    mag = 20 * np.log10(np.maximum(1e-5, mag))
    mag = np.clip((mag - 20 + 100) / 100, 1e-8, 1)
    mag = mag.T.astype(np.float32)

    rsize = mag.shape[1] - (mag.shape[1] % 100)
    rstart = int((mag.shape[1] - rsize) / 2)
    mag = mag[:, rstart:rstart+rsize] # (n_fft, T)
    return mag

def load_specs(wavs):
    ref = []
    mag = []
    mel = []
    linear = []

    for fpath in wavs:
        ref.append(extract_refs(fpath))
        x, y, z = extract_specs(fpath)
        mag.append(z)
        mel.append(x)
        linear.append(y)
    return ref, mag, mel, linear