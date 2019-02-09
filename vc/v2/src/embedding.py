# coding: utf-8
import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init


class GST(nn.Module):

    def __init__(self, n_mels, E):

        super().__init__()
        self.encoder = Embedding(n_mels, E)
        self.stl = STL(E)

    def forward(self, inputs):
        enc_out = self.encoder(inputs)
        style_embed = self.stl(enc_out)

        return style_embed


class Embedding(nn.Module):
    
    def __init__(self, n_mels, E):
        super(Embedding, self).__init__()
        filters = [1, 32, 32, 64, 64, 128, 128]
        K = len(filters) - 1
        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=filters[i]) for i in range(1, K+1)])

        out_channels = self.calculate_channels(n_mels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=filters[-1] * out_channels,
                            hidden_size=E,
                            batch_first=True)

    def forward(self, x):
        N = x.size(0)
        T = x.size(2)

        x = x.transpose(1, 2)
        x = x.view(N, 1, T, -1)

        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)  # [N, 128, T//2^K, n_mels//2^K]

        x = x.transpose(1, 2)  # [N, T//2^K, 128, n_mels//2^K]
        T = x.size(1)
        N = x.size(0)
        x = x.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        _, x = self.gru(x)  # out --- [1, N, E]

        return x.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for i in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L

class STL(nn.Module):
    '''
    inputs --- [N, E]
    '''

    def __init__(self, E):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(10, E // 8))
        init.normal_(self.embed, mean=0, std=0.5)

        d_q = E 
        d_k = E // 8
        self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=E, num_heads=8)
       

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)  # [N, 1, E]
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed = self.attention(query, keys)

        return style_embed


class MultiHeadAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''

    def __init__(self, query_dim, key_dim, num_units, num_heads):

        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out
