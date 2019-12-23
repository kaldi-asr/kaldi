#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import gc
import math
import unittest

try:
    import torch
except ImportError:
    print('This test needs PyTorch.')
    print('Please install PyTorch first.')
    print('PyTorch 1.3.0dev20191006 has been tested and is guaranteed to work.')
    sys.exit(0)

from torch.utils.dlpack import from_dlpack
from torch.utils.dlpack import to_dlpack

import kaldi


class TestDLPack(unittest.TestCase):

    def test_pytorch_cpu_tensor_to_subvector(self):
        '''
        Note that we have two ways to convert a
        PyTorch CPU tensor to kaldi's FloatSubVector:

        Method 1:
            v = kaldi.ToSubVector(to_dlpack(tensor))

        Method 2:
            v = kaldi.FloatSubVector.from_dlpack(to_dlpack(tensor))
        '''
        tensor = torch.arange(3).float()

        # v and tensor share the same memory
        # no data is copied
        v = kaldi.ToSubVector(to_dlpack(tensor))
        self.assertIsInstance(v, kaldi.FloatSubVector)

        v[0] = 100
        v[1] = 200
        v[2] = 300

        self.assertEqual(tensor[0], 100)
        self.assertEqual(tensor[1], 200)
        self.assertEqual(tensor[2], 300)

        del v

        # memory is shared between `v` and `tensor`
        v = kaldi.FloatSubVector.from_dlpack(to_dlpack(tensor))
        self.assertEqual(v[0], 100)

        v[0] = 1  # also changes tensor
        self.assertEqual(tensor[0], 1)

    def test_pytorch_cpu_tensor_to_submatrix(self):
        '''
        Note that we have two ways to convert a
        PyTorch CPU tensor to kaldi's FloatSubMatrix:

        Method 1:
            v = kaldi.ToSubMatrix(to_dlpack(tensor))

        Method 2:
            v = kaldi.FloatSubMatrix.from_dlpack(to_dlpack(tensor))
        '''
        tensor = torch.arange(6).reshape(2, 3).float()

        m = kaldi.ToSubMatrix(to_dlpack(tensor))
        self.assertIsInstance(m, kaldi.FloatSubMatrix)

        m[0, 0] = 100  # also changes tensor, since memory is shared
        self.assertEqual(tensor[0, 0], 100)

        del m

        # memory is shared between `m` and `tensor`
        m = kaldi.FloatSubMatrix.from_dlpack(to_dlpack(tensor))
        m[0, 1] = 200
        self.assertEqual(tensor[0, 1], 200)

    def test_pytorch_and_kaldi_gpu_tensor_zero_copy(self):
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

        # now from Kaldi to PyTorch

        # (fangjun): we put all tests in this function to avoid
        # invoking SelectGpuDevice() twice

        dim = 2
        cpu_v = kaldi.FloatVector(size=dim)
        cpu_v[0] = 10
        cpu_v[1] = 20

        gpu_v = kaldi.FloatCuVector(cpu_v)
        self.assertEqual(gpu_v[0], 10)
        self.assertEqual(gpu_v[1], 20)

        # memory is shared between `gpu_v` and `tensor`
        tensor = from_dlpack(gpu_v.to_dlpack())

        self.assertTrue(tensor.is_cuda)
        self.assertEqual(tensor.device.index, device_id)

        self.assertTrue(tensor[0], 10)
        self.assertTrue(tensor[1], 20)

        tensor[0] = 1  # also changes `gpu_v`
        tensor[1] = 2

        self.assertEqual(gpu_v[0], 1)
        self.assertEqual(gpu_v[1], 2)

        gpu_v.Add(10)  # also changes `tensor`

        self.assertEqual(tensor[0], 11)
        self.assertEqual(tensor[1], 12)

        del tensor
        gc.collect()

        self.assertEqual(gpu_v[0], 11)  # gpu_v is still alive
        self.assertEqual(gpu_v[1], 12)

        # now for CuMatrix
        num_rows = 1
        num_cols = 2

        cpu_m = kaldi.FloatMatrix(row=num_rows, col=num_cols)
        cpu_m[0, 0] = 1
        cpu_m[0, 1] = 2

        gpu_m = kaldi.FloatCuMatrix(cpu_m)
        self.assertEqual(gpu_m[0, 0], 1)
        self.assertEqual(gpu_m[0, 1], 2)

        # memory is shared between `gpu_m` and `tensor`
        tensor = from_dlpack(gpu_m.to_dlpack())

        self.assertTrue(tensor.is_cuda)
        self.assertEqual(tensor.device.index, device_id)

        self.assertTrue(tensor[0, 0], 1)
        self.assertTrue(tensor[0, 1], 2)

        tensor[0, 0] = 6  # also changes `gpu_m`
        tensor[0, 1] = 8

        self.assertEqual(gpu_m[0, 0], 6)
        self.assertEqual(gpu_m[0, 1], 8)

        gpu_m.Add(2)  # also changes `tensor`
        self.assertTrue(tensor[0, 0], 8)
        self.assertTrue(tensor[0, 1], 10)

        del tensor
        gc.collect()

        self.assertEqual(gpu_m[0, 0], 8)  # `gpu_m` is still alive
        self.assertEqual(gpu_m[0, 1], 10)

        # now for CuVector from_dlpack
        tensor = torch.tensor([1, 2]).float()
        tensor = tensor.to(device)

        # memory is shared between `tensor` and `v`
        v = kaldi.FloatCuSubVector.from_dlpack(to_dlpack(tensor))
        self.assertEqual(v[0], 1)

        v.Add(1)  # also changes `tensor`
        self.assertEqual(tensor[0], 2)
        self.assertEqual(tensor[1], 3)

        del tensor
        # now for CuMatrix from_dlpack
        tensor = torch.tensor([1, 2]).reshape(1, 2).float()
        tensor = tensor.to(device)

        # memory is shared between `tensor` and `m`
        m = kaldi.FloatCuSubMatrix.from_dlpack(to_dlpack(tensor))
        self.assertEqual(m[0, 0], 1)

        m.Add(100)  # also changes `tensor`
        self.assertEqual(tensor[0, 0], 101)

    def test_vector_to_pytorch_cpu_tensor(self):
        dim = 2
        v = kaldi.FloatVector(size=dim)
        v[0] = 10
        v[1] = 20

        # memory is shared between kaldi::Vector and PyTorch Tensor
        tensor = from_dlpack(v.to_dlpack())
        self.assertEqual(tensor.is_cuda, False)

        self.assertEqual(tensor[0], 10)
        self.assertEqual(tensor[1], 20)

        tensor[0] = 100  # also changes `v`
        tensor[1] = 200

        self.assertEqual(v[0], 100)
        self.assertEqual(v[1], 200)

        v[0] = 9  # also changes `tensor`
        self.assertEqual(tensor[0], 9)

        del tensor
        gc.collect()

        # one more time
        self.assertEqual(v[0], 9)  # v is still alive
        self.assertEqual(v[1], 200)

        tensor = from_dlpack(v.to_dlpack())
        self.assertEqual(tensor.is_cuda, False)

        tensor[0] = 8
        tensor[1] = 10
        self.assertEqual(v[0], 8)
        self.assertEqual(v[1], 10)

    def test_matrix_to_pytorch_cpu_tensor(self):
        num_rows = 1
        num_cols = 2
        m = kaldi.FloatMatrix(row=num_rows, col=num_cols)
        m[0, 0] = 10
        m[0, 1] = 20

        # memory is shared between `tensor` and `m`
        tensor = from_dlpack(m.to_dlpack())
        self.assertEqual(tensor.is_cuda, False)
        self.assertEqual(tensor.ndim, 2)

        self.assertEqual(tensor[0, 0], 10)
        self.assertEqual(tensor[0, 1], 20)

        m[0, 0] = 100
        self.assertEqual(tensor[0, 0], 100)

        tensor[0, 0] = 1000
        self.assertEqual(m[0, 0], 1000)

        del tensor
        gc.collect()

        # one more time
        self.assertEqual(m[0, 0], 1000)  # m is still alive
        self.assertEqual(m[0, 1], 20)

        tensor = from_dlpack(m.to_dlpack())
        self.assertEqual(tensor.is_cuda, False)

        tensor[0, 0] = 8
        self.assertEqual(m[0, 0], 8)


if __name__ == '__main__':
    unittest.main()
