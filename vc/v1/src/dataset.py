import random
import numpy as np
import torch
import torch.utils.data
import glob

import layers
from utils import load_wav_to_torch
from audio import AudioProcessor
import warnings
warnings.filterwarnings('ignore')

#speakers = ['ERMS', 'MBMPS', 'SVBI', 'RRBI', 'LXC', 'TXHC', 'ZHAA', 'YBAA', 'HJK', 'HKK', 'RMS', 'SLT']
speakers = ['RRBI']
data_path = '../data/{}/wav/*.wav'

class TextMelLoader(torch.utils.data.Dataset):
    """
        1) computes mel-spectrograms from audio files.
    """
    def __init__(self, hparams):
        self.data_files = []
        for sid in speakers:
            self.data_files += glob.glob(data_path.format(sid))

        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.ap = AudioProcessor()
        # self.stft = layers.TacotronSTFT(
        #     hparams.filter_length, hparams.hop_length, hparams.win_length,
        #     hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
        #     hparams.mel_fmax)
        random.seed(1234)
        random.shuffle(self.data_files)

    def get_mel_text_pair(self, fpath):
        # separate filename and text
        audiopath = fpath
        mag_path = fpath.replace('.wav', '.npy').replace('wav', 'mag')
        ref_path = fpath.replace('.wav', '.npy').replace('wav', 'ref')
        mag = torch.from_numpy(np.load(mag_path))
        ref = torch.from_numpy(np.load(ref_path))
        # mel = self.get_mel(audiopath)
        mel = torch.from_numpy(self.ap.melspectrogram(self.ap.load_wav(audiopath, self.ap.sample_rate)[:self.ap.sample_rate * 3]))
        return (ref, mag, mel)

    # def get_mel(self, filename):
    #     if not self.load_mel_from_disk:
    #         audio, sampling_rate = load_wav_to_torch(filename)
    #         if sampling_rate != self.stft.sampling_rate:
    #             raise ValueError("{} SR doesn't match target {} SR".format(
    #                 sampling_rate, self.stft.sampling_rate))
    #         audio_norm = audio / self.max_wav_value
    #         audio_norm = audio_norm.unsqueeze(0)
    #         audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
    #         melspec = self.stft.mel_spectrogram(audio_norm)
    #         melspec = torch.squeeze(melspec, 0)
    #     else:
    #         melspec = torch.from_numpy(np.load(filename))
    #         assert melspec.size(0) == self.stft.n_mel_channels, (
    #             'Mel dimension mismatch: given {}, expected {}'.format(
    #                 melspec.size(0), self.stft.n_mel_channels))

    #     return melspec

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.data_files[index])

    def __len__(self):
        return len(self.data_files)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        mag_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_input_len)
        ref_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), 200)
        
        mag_padded.zero_()
        ref_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mag = batch[ids_sorted_decreasing[i]][1]
            mag_padded[i, :, :mag.size(1)] = mag
            ref_padded[i, :, :] = batch[ids_sorted_decreasing[i]][0]

        # Right zero-pad mel-spec
        num_mels = batch[0][2].size(0)
        max_target_len = max([x[2].size(1) for x in batch])
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][2]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return ref_padded, mag_padded, input_lengths, mel_padded, gate_padded, output_lengths
