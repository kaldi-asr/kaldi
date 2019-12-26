#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import kaldi  # import kaldi before torch!
import kaldi_pybind.fst as fst
import kaldi_pybind.chain as chain

from dataset import read_nnet_chain_example

import torch
from torch.utils.dlpack import to_dlpack

device_id = 7

kaldi.SelectGpuDevice(device_id)
device = torch.device('cuda', device_id)

opts = chain.ChainTrainingOptions()
opts.l2_regularize = 5e-4
opts.leaky_hmm_coefficient = 0.1
opts.xent_regularize = 0

den_fst_filename = '/cache/fangjun/chain/aishell_kaldi_pybind/test/den.fst'
num_pdfs = 4336

den_fst = fst.StdVectorFst.Read(filename=den_fst_filename)

den_graph = chain.DenominatorGraph(fst=den_fst, num_pdfs=num_pdfs)

egs_scp_rxfilename = '/cache/fangjun/chain/aishell_kaldi_pybind/cegs.1.ark:13'
eg = read_nnet_chain_example(egs_scp_rxfilename)
supervision = eg.outputs[0].supervision
print('weight', supervision.weight)
print('num_sequences', supervision.num_sequences)
print('frames_per_sequence', supervision.frames_per_sequence)
print('label_dim', supervision.label_dim)

nnet_output_tensor = torch.rand(
    supervision.num_sequences * supervision.frames_per_sequence,
    num_pdfs).to(device)
nnet_output = kaldi.CuSubMatrixFromDLPack(to_dlpack(nnet_output_tensor))

objf_l2_term_weight_tensor = torch.zeros(3)
objf_l2_term_weight = kaldi.SubVectorFromDLPack(
    to_dlpack(objf_l2_term_weight_tensor))

nnet_output_deriv_tensor = torch.zeros_like(nnet_output_tensor)
nnet_output_deriv = kaldi.CuSubMatrixFromDLPack(
    to_dlpack(nnet_output_deriv_tensor))

xent_output_deriv_tensor = torch.zeros_like(nnet_output_tensor).contiguous()
assert xent_output_deriv_tensor.stride(0) == xent_output_deriv_tensor.shape[1]

xent_output_deriv = kaldi.CuSubMatrixFromDLPack(
    to_dlpack(xent_output_deriv_tensor))

chain.ComputeChainObjfAndDeriv(opts=opts,
                               den_graph=den_graph,
                               supervision=supervision,
                               nnet_output=nnet_output,
                               objf_l2_term_weight=objf_l2_term_weight,
                               nnet_output_deriv=nnet_output_deriv,
                               xent_output_deriv=xent_output_deriv)
print(objf_l2_term_weight_tensor)
print(nnet_output_deriv_tensor[0, 0])
print(xent_output_deriv_tensor[0, 0])
