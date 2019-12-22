#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import math
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


class TestDLPack(unittest.TestCase):

    def test_pytorch_cpu_tensor_to_subvector(self):

        tensor = torch.arange(3).float()

        # v and tensor share the same memory
        # no data is copied
        v = kaldi.ToSubVector(to_dlpack(tensor))
        self.assertIsInstance(v, kaldi.FloatSubVector)

        v[0] = 100
        v[1] = 200
        v[2] = 300

        self.assertEqual(v[0], 100)
        self.assertEqual(v[1], 200)
        self.assertEqual(v[2], 300)

    def test_pytorch_cpu_tensor_to_submatrix(self):
        tensor = torch.arange(6).reshape(2, 3).float()

        m = kaldi.ToSubMatrix(to_dlpack(tensor))
        self.assertIsInstance(m, kaldi.FloatSubMatrix)

        m[0, 0] = 100  # also changes tensor, since memory is shared
        self.assertEqual(tensor[0, 0], 100)

    def test_pytorch_gpu_tensor_to_cu_subvector_and_cu_submatrix(self):
        if torch.cuda.is_available() == False:
            print('No GPU detected! Skip it')
            return

        device_id = 0

        # Kaldi and PyTorch will use the same GPU
        kaldi.SelectGpuDevice(device_id=device_id)

        device = torch.device('cuda', device_id)

        tensor = torch.arange(3).float()
        tensor = tensor.to(device)

        # make sure the tensor from PyTorch is indeed on GPU
        self.assertTrue(tensor.is_cuda)

        # GPU data is shared between kaldi::CuSubVector and PyTorch GPU tensor
        # no data is copied
        v = kaldi.ToCuSubVector(to_dlpack(tensor))
        self.assertIsInstance(v, kaldi.FloatCuSubVector)

        v.Add(value=10)
        self.assertEqual(tensor[0], 10)
        self.assertEqual(tensor[1], 11)
        self.assertEqual(tensor[2], 12)

        v.Scale(value=6)
        self.assertEqual(tensor[0], 60)
        self.assertEqual(tensor[1], 66)
        self.assertEqual(tensor[2], 72)

        v.SetZero()
        self.assertEqual(tensor[0], 0)
        self.assertEqual(tensor[1], 0)
        self.assertEqual(tensor[2], 0)

        # Now for CuSubMatrix
        tensor = torch.arange(3).reshape(1, 3).float()
        tensor = tensor.to(device)

        # make sure the tensor from PyTorch is indeed on GPU
        self.assertTrue(tensor.is_cuda)

        m = kaldi.ToCuSubMatrix(to_dlpack(tensor))
        m.ApplyExp()

        self.assertAlmostEqual(tensor[0, 0], math.exp(0), places=7)
        self.assertAlmostEqual(tensor[0, 1], math.exp(1), places=7)
        self.assertAlmostEqual(tensor[0, 2], math.exp(2), places=7)

        m.SetZero()
        self.assertEqual(tensor[0, 0], 0)
        self.assertEqual(tensor[0, 1], 0)
        self.assertEqual(tensor[0, 2], 0)


if __name__ == '__main__':
    unittest.main()
