#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_delta_feat(x, weight, enable_padding):
    '''
    Args:
        x: input feat of shape [batch_size, seq_len, feat_dim]

        weight: coefficients for computing delta features;
              it has shape [feat_dim, 1, kernel_size].

        enable_padding: True to add padding.

    Returns:
        a tensor of shape [batch_size, seq_len, feat_dim]
    '''

    assert x.ndim == 3
    assert weight.ndim == 3
    assert weight.size(0) == x.size(2)
    assert weight.size(1) == 1
    assert weight.size(2) % 2 == 1

    feat_dim = x.size(2)

    if enable_padding:
        pad_size = weight.size(2) // 2

        # F.pad requires a 4-D tensor in our case
        x = x.unsqueeze(0)

        # (0, 0, pad_size, pad_size) == (left, right, top, bottom)
        x = F.pad(x, (0, 0, pad_size, pad_size), mode='replicate')

        # after padding, we have to convert it back to 3-D
        # since conv1d requires 3-D input
        x = x.squeeze(0)

    # conv1d requires a shape of [batch_size, feat_dim, seq_len]
    x = x.permute(0, 2, 1)

    # NOTE(fangjun): we perform a depthwise convolution here by
    # setting groups == number of channels
    y = F.conv1d(input=x, weight=weight, groups=feat_dim)

    # now convert y back to shape [batch_size, seq_len, feat_dim]
    y = y.permute(0, 2, 1)

    return y


class AddDeltasTransform(nn.Module):
    '''
    This class implements `add-deltas` in kaldi with
    order == 2 and window == 2.

    It can generate the identical output as kaldi's `add-deltas`.

    See transform_test.py
    '''

    def __init__(self,
                 first_order_coef=[-1, 0, 1],
                 second_order_coef=[1, 0, -2, 0, 1],
                 enable_padding=False):
        '''
        Note that this class has no trainable `nn.Parameters`.

        Args:
            first_order_coef: coefficient to compute the first order delta feature

            second_order_coef: coefficient to compute the second order delta feature
        '''
        super().__init__()

        self.first_order_coef = torch.tensor(first_order_coef)
        self.second_order_coef = torch.tensor(second_order_coef)
        self.enable_padding = enable_padding

    def forward(self, x):
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

        first_order = compute_delta_feat(x, self.first_order_coef,
                                         self.enable_padding)
        second_order = compute_delta_feat(x, self.second_order_coef,
                                          self.enable_padding)

        if self.enable_padding:
            y = torch.cat([x, first_order, second_order], dim=2)
        else:
            zeroth = (x.size(1) - second_order.size(1)) // 2
            first = (first_order.size(1) - second_order.size(1)) // 2

            y = torch.cat([
                x[:, zeroth:-zeroth, :],
                first_order[:, first:-first, :],
                second_order,
            ],
                          dim=2)

        return y
