#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
import torch.nn.functional as F


def compute_delta_feat(x, weight):
    '''
    Args:
        x: input feat of shape [batch_size, seq_len, feat_dim]

        weight: coefficients for computing delta features;
              it has a shape of [feat_dim, 1, kernel_size].

    Returns:
        a tensor fo shape [batch_size, seq_len, feat_dim]
    '''

    assert x.ndim == 3
    assert weight.ndim == 3
    assert weight.size(0) == x.size(2)
    assert weight.size(1) == 1
    assert weight.size(2) % 2 == 1

    feat_dim = x.size(2)

    pad_size = weight.size(2) // 2

    # F.pad requires a 4-D tensor in our case
    x = x.unsqueeze(0)

    # (0, 0, pad_size, pad_size) == (left, right, top, bottom)
    padded_x = F.pad(x, (0, 0, pad_size, pad_size), mode='replicate')

    # after padding, we have to convert it back to 3-D
    # since conv1d requires 3-D input
    padded_x = padded_x.squeeze(0)

    # conv1d requires a shape of [batch_size, feat_dim, seq_len]
    padded_x = padded_x.permute(0, 2, 1)

    # NOTE(fangjun): we perform a depthwise convolution here by
    # setting groups == number of channels
    y = F.conv1d(input=padded_x, weight=weight, groups=feat_dim)

    # now convert y back to be of shape [batch_size, seq_len, feat_dim]
    y = y.permute(0, 2, 1)

    return y


class AddDeltasTransform:
    '''
    This class implements `add-deltas` in kaldi with
    order == 2 and window == 2.

    It generates the identical output as kaldi's `add-deltas` with default
    parameters given the same input.
    '''

    def __init__(self):
        # yapf: disable
        self.first_order_coef = torch.tensor([-0.2, -0.1, 0, 0.1, 0.2])
        self.second_order_coef = torch.tensor([0.04, 0.04, 0.01, -0.04, -0.1, -0.04, 0.01, 0.04, 0.04])
        # yapf: enable

        # TODO(fangjun): change the coefficients to the following as suggested by Dan
        #  [-1, 0, 1]
        #  [1, 0, -2,  0, 1]

    def __call__(self, x):
        '''
        Args:
            x: a tensor of shape [batch_size, seq_len, feat_dim]

        Returns:
            a tensor of shape [batch_size, seq_len, feat_dim * 3]
        '''
        if self.first_order_coef.ndim != 3:
            num_duplicates = x.size(2)

            # yapf: disable
            self.first_order_coef = self.first_order_coef.reshape(1, 1, -1)
            self.first_order_coef = torch.cat([self.first_order_coef] * num_duplicates, dim=0)

            self.second_order_coef = self.second_order_coef.reshape(1, 1, -1)
            self.second_order_coef = torch.cat([self.second_order_coef] * num_duplicates, dim=0)
            # yapf: enable

            device = x.device
            self.first_order_coef = self.first_order_coef.to(device)
            self.second_order_coef = self.second_order_coef.to(device)

        first_order = compute_delta_feat(x, self.first_order_coef)
        second_order = compute_delta_feat(x, self.second_order_coef)

        y = torch.cat([x, first_order, second_order], dim=2)

        return y
