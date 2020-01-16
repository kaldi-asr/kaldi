#!/usr/bin/env python3

# Copyright 2019-2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
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
    print('PyTorch 1.3.0dev20191006 has been tested and is known to work.')
    sys.exit(0)

from torch.utils.dlpack import from_dlpack
from torch.utils.dlpack import to_dlpack

import kaldi


class TestDLPackGPU(unittest.TestCase):

    def test_pytorch_and_kaldi_gpu_tensor_zero_copy(self):
        # (fangjun): we put all tests in this function to avoid
        # invoking SelectGpuDevice() twice

        if torch.cuda.is_available() == False:
            print('No GPU detected! Skip it')
            return

        if kaldi.CudaCompiled() == False:
            print('Kaldi is not compiled with CUDA! Skip it')
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
        v = kaldi.CuSubVectorFromDLPack(to_dlpack(tensor))
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

        m = kaldi.CuSubMatrixFromDLPack(to_dlpack(tensor))
        m.ApplyExp()

        self.assertAlmostEqual(tensor[0, 0], math.exp(0), places=7)
        self.assertAlmostEqual(tensor[0, 1], math.exp(1), places=7)
        self.assertAlmostEqual(tensor[0, 2], math.exp(2), places=7)

        m.SetZero()
        self.assertEqual(tensor[0, 0], 0)
        self.assertEqual(tensor[0, 1], 0)
        self.assertEqual(tensor[0, 2], 0)

        # now from Kaldi to PyTorch

        dim = 2
        cpu_v = kaldi.FloatVector(size=dim)
        cpu_v[0] = 10
        cpu_v[1] = 20

        gpu_v = kaldi.FloatCuVector(cpu_v)
        self.assertEqual(gpu_v[0], 10)
        self.assertEqual(gpu_v[1], 20)

        gpu_v_reference_count = sys.getrefcount(gpu_v)

        # memory is shared between `gpu_v` and `tensor`
        tensor = from_dlpack(gpu_v.to_dlpack())

        # `gpu_v.to_dlpack()` increases the reference count of `gpu_v`
        self.assertEqual(gpu_v_reference_count + 1, sys.getrefcount(gpu_v))

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

        # now the reference count for gpu_v is decreased by one
        self.assertEqual(gpu_v_reference_count, sys.getrefcount(gpu_v))

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

        gpu_m_reference_count = sys.getrefcount(gpu_m)

        # memory is shared between `gpu_m` and `tensor`
        tensor = from_dlpack(gpu_m.to_dlpack())

        self.assertEqual(gpu_m_reference_count + 1, sys.getrefcount(gpu_m))

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

        self.assertEqual(gpu_m_reference_count, sys.getrefcount(gpu_m))

        self.assertEqual(gpu_m[0, 0], 8)  # `gpu_m` is still alive
        self.assertEqual(gpu_m[0, 1], 10)

        # now for CuVector from_dlpack
        tensor = torch.tensor([1, 2]).float()
        tensor = tensor.to(device)

        # memory is shared between `tensor` and `v`
        v = kaldi.DLPackFloatCuSubVector.from_dlpack(to_dlpack(tensor))
        self.assertEqual(v[0], 1)

        v.Add(1)  # also changes `tensor`
        self.assertEqual(tensor[0], 2)
        self.assertEqual(tensor[1], 3)

        del v
        del tensor

        # now for CuMatrix from_dlpack
        tensor = torch.tensor([1, 2]).reshape(1, 2).float()
        tensor = tensor.to(device)

        # memory is shared between `tensor` and `m`
        m = kaldi.DLPackFloatCuSubMatrix.from_dlpack(to_dlpack(tensor))
        self.assertEqual(m[0, 0], 1)

        m.Add(100)  # also changes `tensor`
        self.assertEqual(tensor[0, 0], 101)

        del m
        del tensor
        gc.collect()

        # now test the issue: https://github.com/pytorch/pytorch/issues/9261
        # it will not consume all GPU memory
        for i in range(100):
            b = torch.randn(1024 * 1024 * 1024 // 4, 1, device=device)  # 1G
            a = kaldi.CuSubMatrixFromDLPack(to_dlpack(b))
            gc.collect()
        torch.cuda.empty_cache()

        for i in range(100 * 4):
            b = kaldi.FloatCuMatrix(1024 * 1024, 64)  # 256 MB
            a = from_dlpack(b.to_dlpack())
            gc.collect()

    def test_vector_to_pytorch_cpu_tensor(self):
        dim = 2
        v = kaldi.FloatVector(size=dim)
        v[0] = 10
        v[1] = 20

        v_reference_count = sys.getrefcount(v)

        # memory is shared between kaldi::Vector and PyTorch Tensor
        tensor = from_dlpack(v.to_dlpack())

        self.assertEqual(v_reference_count + 1, sys.getrefcount(v))

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

        self.assertEqual(v_reference_count, sys.getrefcount(v))

        # one more time
        self.assertEqual(v[0], 9)  # v is still alive
        self.assertEqual(v[1], 200)

        tensor = from_dlpack(v.to_dlpack())

        self.assertEqual(v_reference_count + 1, sys.getrefcount(v))

        self.assertEqual(tensor.is_cuda, False)

        tensor[0] = 8
        tensor[1] = 10
        self.assertEqual(v[0], 8)
        self.assertEqual(v[1], 10)

        del tensor
        self.assertEqual(v_reference_count, sys.getrefcount(v))


if __name__ == '__main__':
    unittest.main()
