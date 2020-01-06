#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import torch
from torch.autograd import Function
from torch.utils.dlpack import to_dlpack

import kaldi
from kaldi import chain

g_nnet_output_deriv_tensor = None
g_xent_output_deriv_tensor = None


class KaldiChainObjfFunction(Function):

    @staticmethod
    def forward(ctx, opts, den_graph, supervision, nnet_output_tensor,
                xent_out_unused_tensor):
        '''
        opts:          Struct containing options

        den_graph:     The denominator graph, derived from denominator fst.

        supervision:   The supervision object, containing the supervision
                       paths and constraints on the alignment as an FST.

        nnet_output_tensor:   The output of the neural net; dimension must
                              equal ((supervision.num_sequences * supervision.frames_per_sequence)
                              by den_graph.NumPdfs()).  The rows are ordered
                              as: all sequences for frame 0; all sequences for
                              frame 1; etc.

        xent_out_unused_tensor: It MUST have the same structure as `nnet_output_tensor`;
                                it is passed but NOT used directly in this
                                function; it is still needed because we need
                                to do back propagation for it.
        '''
        global g_nnet_output_deriv_tensor
        global g_xent_output_deriv_tensor

        if g_nnet_output_deriv_tensor is None or g_nnet_output_deriv_tensor.shape != nnet_output_tensor.shape:
            # preallocate GPU memory in PyTorch and it will be shared with Kaldi
            g_nnet_output_deriv_tensor = torch.zeros_like(nnet_output_tensor)
            g_xent_output_deriv_tensor = torch.zeros_like(
                nnet_output_tensor).contiguous()
            # (fangjun): Note that g_xent_output_deriv needs to be contiguous,
            # i.e., stride == cols; otherwise, it will be re-allocated in Kaldi
        else:
            # (fangjun): we have to zero them out manually!
            g_nnet_output_deriv_tensor.zero_()
            g_xent_output_deriv_tensor.zero_()

        # it contains [objf, l2_term, weight] and will be returned to the caller
        objf_l2_term_weight_tensor = torch.zeros(3).float()

        nnet_output = kaldi.PytorchToCuSubMatrix(to_dlpack(nnet_output_tensor))

        nnet_output_deriv = kaldi.PytorchToCuSubMatrix(
            to_dlpack(g_nnet_output_deriv_tensor))

        xent_output_deriv = kaldi.PytorchToCuSubMatrix(
            to_dlpack(g_xent_output_deriv_tensor))

        objf_l2_term_weight = kaldi.PytorchToSubVector(
            to_dlpack(objf_l2_term_weight_tensor))

        chain.ComputeChainObjfAndDeriv(opts=opts,
                                       den_graph=den_graph,
                                       supervision=supervision,
                                       nnet_output=nnet_output,
                                       objf_l2_term_weight=objf_l2_term_weight,
                                       nnet_output_deriv=nnet_output_deriv,
                                       xent_output_deriv=xent_output_deriv)

        ctx.save_for_backward(g_nnet_output_deriv_tensor,
                              g_xent_output_deriv_tensor)

        objf_l2_term_weight_tensor = objf_l2_term_weight_tensor.to(
            nnet_output_tensor.device)
        return objf_l2_term_weight_tensor

    @staticmethod
    def backward(ctx, unused):
        nnet_output_deriv, xent_output_deriv = ctx.saved_tensors
        # Multiply by a negative number as we want to do
        # gradient **descent** and not **ascent**
        nnet_output_deriv *= -1
        xent_output_deriv *= -0.1  # TODO(fangjun): how to choose this value ??

        # return the derivative for the input parametes:
        # (opts, den_graph, supervision, nnet_output_tensor, xent_out_unused_tensor)
        return None, None, None, nnet_output_deriv, xent_output_deriv
