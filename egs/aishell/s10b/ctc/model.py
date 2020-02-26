#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

from add_deltas_layer import AddDeltasLayer


# TODO(fangjun): remove proj_dim since we'll use TDNN-F.
def get_ctc_model(input_dim,
                  output_dim,
                  num_layers=4,
                  hidden_dim=512,
                  proj_dim=256,
                  frame_subsampling_factor=3):
    model = CtcModel(input_dim=input_dim,
                     output_dim=output_dim,
                     num_layers=num_layers,
                     hidden_dim=hidden_dim,
                     proj_dim=proj_dim,
                     frame_subsampling_factor=frame_subsampling_factor)

    return model


class CtcModel(nn.Module):

    def __init__(self, input_dim, output_dim, num_layers, hidden_dim, proj_dim,
                 frame_subsampling_factor):
        '''
        Args:
            input_dim: input dimension of the network

            output_dim: output dimension of the network

            num_layers: number of LSTM layers of the network

            hidden_dim: the dimension of the hidden state of LSTM layers

            proj_dim: dimension of the affine layer after every LSTM layer
        '''
        super().__init__()

        assert frame_subsampling_factor in [1, 3]
        self.sf = frame_subsampling_factor

        lstm_layer_list = []
        proj_layer_list = []

        # batchnorm requires input of shape [N, C, L] == [batch_size, dim, seq_len]
        self.input_batch_norm = nn.BatchNorm1d(num_features=input_dim * 3,
                                               affine=False)

        self.lstm = nn.LSTM(input_size=input_dim * 3,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            dropout=0.2,
                            batch_first=True)

        self.prefinal = nn.Linear(in_features=hidden_dim,
                                  out_features=output_dim)

        self.add_deltas_layer = AddDeltasLayer()

    def forward(self, feat, feat_len_list):
        '''
        Args:
            feat: a 3-D tensor of shape [batch_size, seq_len, feat_dim]
            feat_len_list: feat length of each utterance before padding

        Returns:
            a 3-D tensor of shape [batch_size, seq_len, output_dim]
            It is the output of `nn.Linear`. That is, **NO** log_softmax
            is applied to the output.
        '''
        x = feat

        # at his point, x is of shape [batch_size, seq_len, feat_dim]
        x = x.permute(0, 2, 1)

        # at his point, x is of shape [batch_size, feat_dim, seq_len] == [N, C, L]

        x = self.add_deltas_layer(x)

        if self.sf == 3:
            x = x[:, :, ::3]
            feat_len_list = (torch.tensor(feat_len_list).int() + 2) / 3
            # feat_len_list is still of type int32

        x = self.input_batch_norm(x)

        x = x.permute(0, 2, 1)

        # at his point, x is of shape [batch_size, seq_len, feat_dim] == [N, L, C]

        x = pack_padded_sequence(input=x,
                                 lengths=feat_len_list,
                                 batch_first=True,
                                 enforce_sorted=False)

        # TODO(fangjun): save intermediate LSTM state to support streaming inference
        x, _ = self.lstm(x)

        x, _ = pad_packed_sequence(x, batch_first=True)

        x = self.prefinal(x)

        return x, feat_len_list


def _test_ctc_model():
    input_dim = 5
    output_dim = 20
    model = CtcModel(input_dim=input_dim,
                     output_dim=output_dim,
                     num_layers=2,
                     hidden_dim=3,
                     proj_dim=4)

    feat1 = torch.randn((6, input_dim))
    feat2 = torch.randn((8, input_dim))

    from torch.nn.utils.rnn import pad_sequence
    feat = pad_sequence([feat1, feat2], batch_first=True)
    assert feat.shape == torch.Size([2, 8, input_dim])

    feat_len_list = [6, 8]
    x = model(feat, feat_len_list)

    assert x.shape == torch.Size([2, 8, output_dim])


if __name__ == '__main__':
    _test_ctc_model()
