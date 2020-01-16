#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

try:
    import torch
except ImportError:
    print('This test needs PyTorch.')
    print('Please install PyTorch first.')
    print('PyTorch 1.3.0dev20191006 has been tested and is guaranteed to work.')
    sys.exit(0)

from torch.utils.dlpack import to_dlpack

import kaldi


class TestDLPackSubVector(unittest.TestCase):

    def test_dlpack_subvector(self):
        '''
        kaldi.Hello accepts two parameters:
            Param 1: reference to VectorBase<float>
            Param 2: pointer to VectorBase<float>

        This test shows that we can pass a DLPackSubVector
        to `Hello`.
        '''
        tensor1 = torch.tensor([1, 2]).float()
        v1 = kaldi.FloatSubVectorFromDLPack(to_dlpack(tensor1))

        tensor2 = torch.tensor([10, 20, 30]).float()
        v2 = kaldi.FloatSubVectorFromDLPack(to_dlpack(tensor2))

        kaldi.Hello(v1, v2)


if __name__ == '__main__':
    unittest.main()
