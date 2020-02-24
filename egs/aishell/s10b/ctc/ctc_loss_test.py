#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch

import kaldi
from kaldi import ctc

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.utils.dlpack import to_dlpack

from ctc_loss import CTCLoss


def test_baidu_warp_ctc():
    device_id = 1
    kaldi.SelectGpuDevice(device_id=device_id)

    device = torch.device('cuda', index=device_id)

    activations = torch.tensor([0.2] * 5).reshape(1, 1, -1).to(device)
    log_probs = torch.log(activations)
    log_probs.requires_grad_(True)

    tmp_log_probs = log_probs.clone()

    targets = torch.tensor([1])
    target_lengths = torch.tensor([1])
    input_lengths = torch.tensor([1])

    loss_func = CTCLoss(use_warp_ctc=True, blank=0, reduction='none')
    loss = loss_func(log_probs=log_probs,
                     targets=targets,
                     input_lengths=input_lengths,
                     target_lengths=target_lengths)

    print(loss)
    loss.backward()
    print(log_probs.grad)

    loss_func = CTCLoss(use_warp_ctc=False, blank=0, reduction='none')
    loss = loss_func(log_probs=tmp_log_probs,
                     targets=targets,
                     input_lengths=input_lengths,
                     target_lengths=target_lengths)
    loss.backward()
    print(log_probs.grad)
    print(loss)

    # It turns out that
    # (1) the cost values computed by warp-ctc and Pytorch's built-in ctc loss are identical
    # (2) But the gradient values differ!


def main():
    test_baidu_warp_ctc()


if __name__ == '__main__':
    torch.manual_seed(20200224)
    main()
