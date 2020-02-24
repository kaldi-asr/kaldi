#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
from torch.autograd import Function
from torch.utils.dlpack import to_dlpack
import torch.nn as nn
import torch.nn.functional as F

import kaldi
from kaldi import ctc


class WarpCtcLoss(Function):

    @staticmethod
    def forward(ctx, log_probs, targets, input_lengths, target_lengths, blank,
                reduction):
        '''
        Args:
            log_probs: `(seq_len, batch_size, C)`, where `C` is the number
                       of characters in aphabet including the blank symbol.

            targets: a tensor of [batch size] containing the concatenated labels.
                     Targets cannot be blank.

            input_lengths: a tensor of [batch_size] containing the number of input frames
                     for each utterance in the batch.

            target_lengths: a tensor of [batch_size] containing the label lengths

            blank: the index of the blank symbol.

            reduction: specifies the reduction to apply to
                       the output: `none` | `mean` | `sum`.

                       `none`: no reduction will be applied;

                       `mean`: the output losses will be divided
                       by the target lengths and then the mean
                       over the batch is taken.

                       `sum`: the output will be summed.
        '''
        device = log_probs.device
        assert device.type == 'cuda', 'we only support computing CTCLoss on GPU devices.'

        activations_tensor = log_probs.float().reshape(-1).contiguous()
        gradients_tensor = torch.empty_like(activations_tensor).contiguous()

        # NOTE(fangjun): foobar.cpu() is a no operation if foobar is already on CPU.
        flat_labels_tensor = targets.int().view(-1).cpu()
        label_lengths_tensor = target_lengths.int().view(-1).cpu()
        input_lengths_tensor = input_lengths.int().view(-1).cpu()

        alphabet_size = log_probs.size(2)
        minibatch = log_probs.size(1)

        costs_tensor = torch.empty(minibatch, dtype=torch.float32).contiguous()

        info = ctc.CtcOptions()
        info.loc = ctc.CtcComputeLocation.CTC_GPU
        info.blank_label = blank

        label_lengths = kaldi.IntSubVectorFromDLPack(
            to_dlpack(label_lengths_tensor))

        input_lengths = kaldi.IntSubVectorFromDLPack(
            to_dlpack(input_lengths_tensor))

        status, size_in_bytes = ctc.GetWorkspaceSize(
            label_lengths=label_lengths,
            input_lengths=input_lengths,
            alphabet_size=alphabet_size,
            minibatch=minibatch,
            info=info)

        assert status == ctc.CtcStatus.CTC_STATUS_SUCCESS

        num_floats = size_in_bytes // 4 + 1
        workspace_tensor = torch.empty(
            num_floats, dtype=torch.float32).contiguous().to(device)

        activations = kaldi.CuSubVectorFromDLPack(to_dlpack(activations_tensor))
        gradients = kaldi.CuSubVectorFromDLPack(to_dlpack(gradients_tensor))
        flat_labels = kaldi.IntSubVectorFromDLPack(
            to_dlpack(flat_labels_tensor))
        costs = kaldi.FloatSubVectorFromDLPack(to_dlpack(costs_tensor))
        workspace = kaldi.CuSubVectorFromDLPack(to_dlpack(workspace_tensor))

        stream = torch.cuda.default_stream(device)
        with torch.cuda.stream(stream):
            status = ctc.ComputeCtcLossGpu(activations=activations,
                                           gradients=gradients,
                                           flat_labels=flat_labels,
                                           label_lengths=label_lengths,
                                           input_lengths=input_lengths,
                                           alphabet_size=alphabet_size,
                                           minibatch=minibatch,
                                           costs=costs,
                                           workspace=workspace,
                                           options=info)

        gradients_tensor = gradients_tensor.reshape(*log_probs.shape)

        ctx.save_for_backward(gradients_tensor),

        if reduction == 'none':
            return costs_tensor

        total_loss = torch.sum(costs_tensor)

        if reduction == 'sum':
            return total_loss

        # else it is `mean`
        total_target_lengths = torch.sum(label_lengths_tensor)

        return total_loss / minibatch / total_target_lengths

    @staticmethod
    def backward(ctx, unused):
        '''
        The `forward` method has 6 inputs:
            `log_probs`, `targets`, `input_lengths`,
            `target_lengths`, `blank`, `reduction`

        We have to return 6 values.
        '''
        gradients, = ctx.saved_tensors
        return gradients, None, None, None, None, None


def warp_ctc_loss(log_probs, targets, input_lengths, target_lengths, blank,
                  reduction):
    '''
    A thin wrapper for WarpCtcLoss.

    We can use keyword arguments with this wrapper
    '''
    loss_func = WarpCtcLoss.apply
    return loss_func(log_probs, targets, input_lengths, target_lengths, blank,
                     reduction)


class CTCLoss(nn.Module):
    '''
    Note that PyTorch requires the probability to be log prob,
    while warp-ctc does not have this requirement.
    '''

    def __init__(self, use_warp_ctc=True, blank=0, reduction='mean'):
        '''
        Args:
            blank: the index of the blank label
            reduction: specifies the reduction to apply to
                       the output: `none` | `mean` | `sum`.

                       `none`: no reduction will be applied;

                       `mean`: the output losses will be divided
                       by the target lengths and then the mean
                       over the batch is taken.

                       `sum`: the output will be summed.
        '''
        super().__init__()
        assert reduction in ['none', 'mean', 'sum']
        if use_warp_ctc:
            self.loss_func = warp_ctc_loss
        else:
            self.loss_func = F.ctc_loss

        self.use_warp_ctc = use_warp_ctc

        self.blank = blank
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        '''
        Args:
            log_probs: `(seq_len, batch_size, C)`, where `C` is the number
                       of characters in aphabet including the blank symbol.

            targets: a tensor of [batch size] containing the concatenated labels.
                     Targets cannot be blank.

            input_lengths: a tensor of [batch_size] containing the number of input frames
                     for each utterance in the batch.

            target_lengths: a tensor of [batch_size] containing the label lengths
        '''
        if self.use_warp_ctc == False:
            # move all tensors to GPU
            device = log_probs.device
            targets = targets.to(device)
            input_lengths = input_lengths.to(device)
            target_lengths = target_lengths.to(device)

        return self.loss_func(log_probs=log_probs,
                              targets=targets,
                              input_lengths=input_lengths,
                              target_lengths=target_lengths,
                              blank=self.blank,
                              reduction=self.reduction)
