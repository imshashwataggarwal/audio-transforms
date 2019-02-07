# coding: utf-8
import math
import numpy as np
import random
import hp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from models.embedding import Embedding
from models.encoder import DeepSpeech
from models.tacotron import Prenet, Decoder, PostCBHG

def reload_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        return model

def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = (sequence_length.unsqueeze(1)
                         .expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand

def expand_speaker_embed(inputs_btc, speaker_embed=None, tdim=1):
    if speaker_embed is None:
        return None
    # expand speaker embedding for all time steps
    # (B, N) -> (B, T, N)
    ss = speaker_embed.size()
    speaker_embed_btc = speaker_embed.unsqueeze(1).expand(
        ss[0], inputs_btc.size(tdim), ss[-1])
    return speaker_embed_btc


class E2E(nn.Module):
    def __init__(self,
                 embedding_dim=128,
                 idim=128,
                 n=8,
                 linear_dim=1025,
                 mel_dim=80,
                 r=5,
                 padding_idx=None):
                 
        super(E2E, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        
        # Initialize speaker embeddings
        self.embedding = nn.Sequential(
                reload_model(Embedding(n), hp.embed_path),
                nn.Linear(1024, embedding_dim)
        )
        
        # Initialize Encoder
        self.encoder = nn.Sequential(
                DeepSpeech.load_model(hp.encoder_path),
                nn.Linear(800, idim)
        )
        
        self.decoder = Decoder(256, mel_dim, r)
        self.postnet = PostCBHG(mel_dim + embedding_dim)
        self.last_linear = nn.Sequential(
            nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim),
            nn.Sigmoid())

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.encoder[0].conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.long()

    def forward(self, ref=None, mag=None, mag_len=None, mel=None):
        '''
        ref: Inputs for Speaker Embeddings, Shape: (B, F, T)
        mag: Inputs for Phone Encoder, Shape: (B, F, T)
        mel: Inputs for Decoder, Shape: (B, T, F)
        '''
        B = mag.size(0)
        mag_len = self.get_seq_lens(mag_len.cpu().long())
        mask = sequence_mask(mag_len.cuda())

        # (B, embedding_dim)
        embed = self.embedding(ref)
        
        # (B, Tx, idim)
        enc_outs  = self.encoder(mag)
        
        # (B, Tx, idim + embedding_dim)
        enc_outs = torch.cat([enc_outs, expand_speaker_embed(enc_outs, embed)], dim=-1)

        # (B, Ty, dim*r)
        mel_outputs, alignments, stop_tokens = self.decoder(enc_outs, mel, mask)
        
        # Reshape
        # (B, Ty, dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(torch.cat([mel_outputs, expand_speaker_embed(mel_outputs, embed)], dim=-1))
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens
