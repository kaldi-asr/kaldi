# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_delta_feat(x, weight):
    '''
    Args:
        x: input feat of shape [batch_size, feat_dim, seq_len]

        weight: coefficients for computing delta features;
              it has shape [feat_dim, 1, kernel_size].

    Returns:
        a tensor of shape [batch_size, feat_dim, seq_len]
    '''

    assert x.ndim == 3

    assert weight.ndim == 3
    assert weight.size(0) == x.size(1)
    assert weight.size(1) == 1
    assert weight.size(2) % 2 == 1

    feat_dim = x.size(1)

    # NOTE(fangjun): we perform a depthwise convolution here by
    # setting groups == number of channels
    y = F.conv1d(input=x, weight=weight, groups=feat_dim)

    return y


class AddDeltasLayer(nn.Module):
    '''
    This class implements `add-deltas` with order == 2 and window == 2.

    Note that it has no trainable `nn.Parameter`s.
    '''

    def __init__(self,
                 first_order_coef=[-1, 0, 1],
                 second_order_coef=[1, 0, -2, 0, 1]):
        '''
        Args:
            first_order_coef: coefficient to compute the first order delta feature

            second_order_coef: coefficient to compute the second order delta feature
        '''
        super().__init__()

        self.first_order_coef = torch.tensor(first_order_coef).float()
        self.second_order_coef = torch.tensor(second_order_coef).float()

    def forward(self, x):
        '''
        Args:
            x: a tensor of shape [batch_size, feat_dim, seq_len]

        Returns:
            a tensor of shape [batch_size, feat_dim * 3, seq_len]
        '''
        if self.first_order_coef.ndim != 3:
            num_duplicates = x.size(1)

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

        # since we did not perform padding, we have to remove some frames
        # from the 0th and 1st order features
        zeroth_valid = (x.size(2) - second_order.size(2)) // 2
        first_valid = (first_order.size(2) - second_order.size(2)) // 2

        y = torch.cat([
            x[:, :, zeroth_valid:-zeroth_valid,],
            first_order[:, :, first_valid:-first_valid],
            second_order,
        ],
                      dim=1)

        return y
