#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch

import kaldi

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from ctc_loss import CTCLoss


def test_baidu_warp_ctc():
    device_id = 1
    kaldi.SelectGpuDevice(device_id=device_id)

    device = torch.device('cuda', index=device_id)

    ex1 = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float32)

    ex2 = torch.tensor(
        [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
        dtype=torch.float32)

    ex3 = torch.tensor([[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6],
                        [-15, -14, -13, -12, -11]],
                       dtype=torch.float32)

    activations = pad_sequence([ex1, ex2, ex3], batch_first=False)
    activations = activations.to(device)

    tmp_activations = activations.clone()

    activations.requires_grad_(True)
    tmp_activations.requires_grad_(True)

    targets = torch.tensor([1, 3, 3, 2, 3])
    target_lengths = torch.tensor([1, 2, 2])
    input_lengths = torch.tensor([1, 3, 3])

    loss_func = CTCLoss(use_warp_ctc=True, blank=0, reduction='mean')
    loss = loss_func(activations=activations,
                     targets=targets,
                     input_lengths=input_lengths,
                     target_lengths=target_lengths)

    print('warp ctc loss', loss)
    loss.backward()
    print('warp ctc activations grad', activations.grad)

    loss_func = CTCLoss(use_warp_ctc=False, blank=0, reduction='mean')
    loss = loss_func(activations=tmp_activations,
                     targets=targets,
                     input_lengths=input_lengths,
                     target_lengths=target_lengths)
    loss.backward()
    print('loss', loss)
    print('grad', tmp_activations.grad)
    print('grad x 6', tmp_activations.grad * 6)

    # It turns out that
    #   - the loss
    #   - and the gradients
    # computed by warp ctc and PyTorch's built-in CTCLoss are different.


def main():
    test_baidu_warp_ctc()


if __name__ == '__main__':
    torch.manual_seed(20200224)
    main()
