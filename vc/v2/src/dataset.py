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
data_path = 'drive/My Drive/data/{}/wav/*.wav'

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
        self.mag_ap = AudioProcessor(sample_rate=16000, num_mels=80, min_level_db=-100, frame_shift_ms=10, frame_length_ms=20, ref_level_db=20, num_freq=161, power=1.5, preemphasis=0.97, signal_norm=True, symmetric_norm=None, max_norm=1, mel_fmin=50, mel_fmax=8000, clip_norm=True, griffin_lim_iters=60, do_trim_silence=True)
        random.seed(1234)
        random.shuffle(self.data_files)

    def get_mel_text_pair(self, fpath):
        # separate filename and text
        audiopath = fpath
        #mag_path = fpath.replace('.wav', '.npy').replace('wav', 'mag')
        #mag = torch.from_numpy(np.load(mag_path))
        mag = torch.from_numpy(self.mag_ap.melspectrogram(self.mag_ap.load_wav(audiopath, self.mag_ap.sample_rate)[:self.mag_ap.sample_rate * 3]))
        mel = torch.from_numpy(self.ap.melspectrogram(self.ap.load_wav(audiopath, self.ap.sample_rate)[:self.ap.sample_rate * 3]))
        return (mag, mel)
        
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
            torch.LongTensor([x[0].size(1) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        mag_padded = torch.FloatTensor(len(batch), batch[0][0].size(0), max_input_len)
        
        mag_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            mag = batch[ids_sorted_decreasing[i]][0]
            mag_padded[i, :, :mag.size(1)] = mag

        # Right zero-pad mel-spec
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch])
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
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1)-1:] = 1
            output_lengths[i] = mel.size(1)

        return mag_padded, input_lengths, mel_padded, gate_padded, output_lengths
