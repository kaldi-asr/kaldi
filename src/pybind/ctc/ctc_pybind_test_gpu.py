#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import math
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi
from kaldi import ctc

try:
    import torch
except ImportError:
    print('This test needs PyTorch.')
    print('Please install PyTorch first.')
    print('PyTorch 1.3.0dev20191006 has been tested and is known to work.')
    sys.exit(0)

from torch.utils.dlpack import to_dlpack
from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available() == False:
    print('No GPU detected! Skip it')
    sys.exit(0)

if kaldi.CudaCompiled() == False:
    print('Kaldi is not compiled with CUDA! Skip it')
    sys.exit(0)

device_id = 0

kaldi.SelectGpuDevice(device_id=device_id)


class TestCtcGpu(unittest.TestCase):

    def test_case1(self):
        device = torch.device('cuda', device_id)

        # refer to https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
        # this is the simplest case
        # we have one sequence with probability: [0.2, 0.2, 0.2, 0.2, 0.2]
        label_lengths_tensor = torch.tensor([1], dtype=torch.int32)
        input_lengths_tensor = torch.tensor([1], dtype=torch.int32)
        alphabet_size = 5
        minibatch = 1
        info = ctc.CtcOptions()
        info.loc = ctc.CtcComputeLocation.CTC_GPU
        info.blank_label = 0

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
        self.assertEqual(status, ctc.CtcStatus.CTC_STATUS_SUCCESS)
        num_floats = size_in_bytes // 4 + 1
        workspace_tensor = torch.empty(
            num_floats, dtype=torch.float32).contiguous().to(device)

        activations_tensor = torch.tensor(
            [0.2, 0.2, 0.2, 0.2, 0.2],
            dtype=torch.float32).contiguous().to(device)
        gradients_tensor = torch.empty_like(activations_tensor)
        flat_labels_tensor = torch.tensor([1], dtype=torch.int32)
        costs_tensor = torch.empty(minibatch, dtype=torch.float32)

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

        # 1.6094379425049 is copied from
        # https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
        self.assertAlmostEqual(costs[0], 1.6094379425049)

    def test_case2(self):
        device = torch.device('cuda', device_id)
        # this is the second case
        # we have 3 sequences with probability:
        # [1, 2, 3, 4, 5]
        # [6, 7, 8, 9, 10]
        # [11, 12, 13, 14, 15]
        label_lengths_tensor = torch.tensor([2], dtype=torch.int32)
        input_lengths_tensor = torch.tensor([3], dtype=torch.int32)
        alphabet_size = 5
        minibatch = 1
        info = ctc.CtcOptions()
        info.loc = ctc.CtcComputeLocation.CTC_GPU
        info.blank_label = 0

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
        self.assertEqual(status, ctc.CtcStatus.CTC_STATUS_SUCCESS)
        num_floats = size_in_bytes // 4 + 1
        workspace_tensor = torch.empty(
            num_floats, dtype=torch.float32).contiguous().to(device)

        activations_tensor = torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            dtype=torch.float32).contiguous().view(-1).to(device)
        gradients_tensor = torch.empty_like(activations_tensor)

        # the target sequence is cc, which is 3 3
        flat_labels_tensor = torch.tensor([3, 3], dtype=torch.int32)
        costs_tensor = torch.empty(minibatch, dtype=torch.float32)

        activations = kaldi.CuSubVectorFromDLPack(to_dlpack(activations_tensor))
        gradients = kaldi.CuSubVectorFromDLPack(to_dlpack(gradients_tensor))
        flat_labels = kaldi.IntSubVectorFromDLPack(
            to_dlpack(flat_labels_tensor))
        costs = kaldi.FloatSubVectorFromDLPack(to_dlpack(costs_tensor))
        workspace = kaldi.CuSubVectorFromDLPack(to_dlpack(workspace_tensor))

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

        # 7.355742931366 is copied from
        # https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
        self.assertAlmostEqual(costs[0], 7.355742931366)

    def test_case3(self):
        device = torch.device('cuda', device_id)
        # this is the third case
        # we have 3 sequences with probability:
        # [-5, -4, -3, -2, -1]
        # [-10, -9, -8, -7, -6]
        # [-15, -14, -13, -12, -11]
        label_lengths_tensor = torch.tensor([2], dtype=torch.int32)
        input_lengths_tensor = torch.tensor([3], dtype=torch.int32)
        alphabet_size = 5
        minibatch = 1
        info = ctc.CtcOptions()
        info.loc = ctc.CtcComputeLocation.CTC_GPU
        info.blank_label = 0

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
        self.assertEqual(status, ctc.CtcStatus.CTC_STATUS_SUCCESS)
        num_floats = size_in_bytes // 4 + 1
        workspace_tensor = torch.empty(
            num_floats, dtype=torch.float32).contiguous().to(device)

        activations_tensor = torch.tensor(
            [[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6],
             [-15, -14, -13, -12, -11]],
            dtype=torch.float32).contiguous().view(-1).to(device)
        gradients_tensor = torch.empty_like(activations_tensor)
        # the target sequence is b c, whichis 2 3
        flat_labels_tensor = torch.tensor([2, 3], dtype=torch.int32)
        costs_tensor = torch.empty(minibatch, dtype=torch.float32)

        activations = kaldi.CuSubVectorFromDLPack(to_dlpack(activations_tensor))
        gradients = kaldi.CuSubVectorFromDLPack(to_dlpack(gradients_tensor))
        flat_labels = kaldi.IntSubVectorFromDLPack(
            to_dlpack(flat_labels_tensor))
        costs = kaldi.FloatSubVectorFromDLPack(to_dlpack(costs_tensor))
        workspace = kaldi.CuSubVectorFromDLPack(to_dlpack(workspace_tensor))

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

        # 4.938850402832 is copied from
        # https://github.com/baidu-research/warp-ctc/blob/master/torch_binding/TUTORIAL.md
        self.assertAlmostEqual(costs[0], 4.938850402832, places=6)

    def test_case4(self):
        device = torch.device('cuda', device_id)
        # combine case1 to case3 to a minibatch
        # the first example (a): input_length: 1, label_length: 1
        # the second example (c, c): input_length: 3, label_length: 2
        # the third example (b, c): input_length: 3, label_length: 2
        label_lengths_tensor = torch.tensor([1, 2, 2], dtype=torch.int32)
        input_lengths_tensor = torch.tensor([1, 3, 3], dtype=torch.int32)

        alphabet_size = 5
        minibatch = 3
        info = ctc.CtcOptions()
        info.loc = ctc.CtcComputeLocation.CTC_GPU
        info.blank_label = 0

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
        self.assertEqual(status, ctc.CtcStatus.CTC_STATUS_SUCCESS)
        num_floats = size_in_bytes // 4 + 1
        workspace_tensor = torch.empty(
            num_floats, dtype=torch.float32).contiguous().to(device)

        ex1 = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], dtype=torch.float32)

        ex2 = torch.tensor(
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]],
            dtype=torch.float32)

        ex3 = torch.tensor([[-5, -4, -3, -2, -1], [-10, -9, -8, -7, -6],
                            [-15, -14, -13, -12, -11]],
                           dtype=torch.float32)

        activations_tensor = pad_sequence([ex1, ex2, ex3], batch_first=False)

        activations_tensor = activations_tensor.contiguous().view(-1).to(device)
        gradients_tensor = torch.empty_like(activations_tensor)

        # labels are: (a), (c, c) (b, c)
        # which are:  (1), (3, 3), (2, 3)
        flat_labels_tensor = torch.tensor([1, 3, 3, 2, 3], dtype=torch.int32)
        costs_tensor = torch.empty(minibatch, dtype=torch.float32)

        activations = kaldi.CuSubVectorFromDLPack(to_dlpack(activations_tensor))
        gradients = kaldi.CuSubVectorFromDLPack(to_dlpack(gradients_tensor))
        flat_labels = kaldi.IntSubVectorFromDLPack(
            to_dlpack(flat_labels_tensor))
        costs = kaldi.FloatSubVectorFromDLPack(to_dlpack(costs_tensor))
        workspace = kaldi.CuSubVectorFromDLPack(to_dlpack(workspace_tensor))

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

        self.assertAlmostEqual(costs[0], 1.6094379425049)
        self.assertAlmostEqual(costs[1], 7.355742931366)
        self.assertAlmostEqual(costs[2], 4.938850402832, places=6)


if __name__ == '__main__':
    unittest.main()
