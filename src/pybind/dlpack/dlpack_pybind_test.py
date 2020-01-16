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
    print('PyTorch 1.3.0dev20191006 has been tested and is known to work.')
    sys.exit(0)

from torch.utils.dlpack import from_dlpack
from torch.utils.dlpack import to_dlpack

import kaldi


class TestDLPackCPU(unittest.TestCase):

    def test_pytorch_cpu_tensor_to_subvector(self):
        '''
        Note that we have two ways to convert a
        PyTorch CPU tensor to kaldi's FloatSubVector:

        Method 1:
            v = kaldi.FloatSubVectorFromDLPack(to_dlpack(tensor))

        Method 2:
            v = kaldi.DLPackFloatSubVector.from_dlpack(to_dlpack(tensor))
        '''
        tensor = torch.arange(3).float()

        # v and tensor share the same memory
        # no data is copied
        v = kaldi.FloatSubVectorFromDLPack(to_dlpack(tensor))
        # since DLPackFloatSubVector is a subclass of FloatSubVector
        # the following assertion is True
        self.assertIsInstance(v, kaldi.FloatSubVector)

        v[0] = 100
        v[1] = 200
        v[2] = 300

        self.assertEqual(tensor[0], 100)
        self.assertEqual(tensor[1], 200)
        self.assertEqual(tensor[2], 300)

        del v
        # the destructor of DLPackFloatSubVector is invoked
        # by the above `del v` statement, you should see the log message
        # here if you have put it in the destructor.

        # memory is shared between `v` and `tensor`
        v = kaldi.DLPackFloatSubVector.from_dlpack(to_dlpack(tensor))
        self.assertEqual(v[0], 100)

        v[0] = 1  # also changes tensor
        self.assertEqual(tensor[0], 1)

        # the destructor of DLPackFloatSubVector is also invoked here
        # after v is collected by the garbage collector.

    def test_pytorch_cpu_tensor_to_submatrix(self):
        '''
        Note that we have two ways to convert a
        PyTorch CPU tensor to kaldi's FloatSubMatrix:

        Method 1:
            v = kaldi.SubMatrixFromDLPack(to_dlpack(tensor))

        Method 2:
            v = kaldi.DLPackFloatSubMatrix.from_dlpack(to_dlpack(tensor))
        '''
        tensor = torch.arange(6).reshape(2, 3).float()

        m = kaldi.SubMatrixFromDLPack(to_dlpack(tensor))
        self.assertIsInstance(m, kaldi.FloatSubMatrix)

        m[0, 0] = 100  # also changes tensor, since memory is shared
        self.assertEqual(tensor[0, 0], 100)

        del m

        # memory is shared between `m` and `tensor`
        m = kaldi.DLPackFloatSubMatrix.from_dlpack(to_dlpack(tensor))
        m[0, 1] = 200
        self.assertEqual(tensor[0, 1], 200)

    def test_vector_to_pytorch_cpu_tensor(self):
        dim = 2
        v = kaldi.FloatVector(size=dim)
        v[0] = 10
        v[1] = 20

        v_reference_count = sys.getrefcount(v)

        # memory is shared between `tensor` and `v`
        tensor = from_dlpack(v.to_dlpack())

        self.assertEqual(v_reference_count + 1, sys.getrefcount(v))

        self.assertEqual(tensor.is_cuda, False)
        self.assertEqual(tensor.ndim, 1)

        self.assertEqual(tensor[0], 10)
        self.assertEqual(tensor[1], 20)

        v[0] = 100
        self.assertEqual(tensor[0], 100)

        tensor[0] = 1
        self.assertEqual(v[0], 1)

        del tensor
        gc.collect()

        self.assertEqual(v_reference_count, sys.getrefcount(v))

        # one more time
        self.assertEqual(v[0], 1)  # v is still alive
        self.assertEqual(v[1], 20)

        tensor = from_dlpack(v.to_dlpack())
        self.assertEqual(v_reference_count + 1, sys.getrefcount(v))

        self.assertEqual(tensor.is_cuda, False)

        tensor[0] = 8
        self.assertEqual(v[0], 8)

        del tensor

        self.assertEqual(v_reference_count, sys.getrefcount(v))

    def test_matrix_to_pytorch_cpu_tensor(self):
        num_rows = 1
        num_cols = 2
        m = kaldi.FloatMatrix(row=num_rows, col=num_cols)
        m[0, 0] = 10
        m[0, 1] = 20

        m_reference_count = sys.getrefcount(m)

        # memory is shared between `tensor` and `m`
        tensor = from_dlpack(m.to_dlpack())

        self.assertEqual(m_reference_count + 1, sys.getrefcount(m))

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

        self.assertEqual(m_reference_count, sys.getrefcount(m))

        # one more time
        self.assertEqual(m[0, 0], 1000)  # m is still alive
        self.assertEqual(m[0, 1], 20)

        tensor = from_dlpack(m.to_dlpack())
        self.assertEqual(m_reference_count + 1, sys.getrefcount(m))

        self.assertEqual(tensor.is_cuda, False)

        tensor[0, 0] = 8
        self.assertEqual(m[0, 0], 8)

        del tensor

        self.assertEqual(m_reference_count, sys.getrefcount(m))

    def test_cu_matrix_to_pytorch_cpu_tensor(self):
        if kaldi.CudaCompiled():
            print('This test is for constructing a CPU tensor from a CuMatrix')
            print('Kaldi is compiled with GPU, skip it')
            return

        num_rows = 1
        num_cols = 2

        cpu_m = kaldi.FloatMatrix(row=num_rows, col=num_cols)
        cpu_m[0, 0] = 1
        cpu_m[0, 1] = 2

        m = kaldi.FloatCuMatrix(cpu_m)
        self.assertEqual(m[0, 0], 1)
        self.assertEqual(m[0, 1], 2)

        m_reference_count = sys.getrefcount(m)

        # memory is shared between `m` and `tensor`
        tensor = from_dlpack(m.to_dlpack())

        self.assertEqual(m_reference_count + 1, sys.getrefcount(m))

        self.assertTrue(tensor.is_cuda == False)

        self.assertTrue(tensor[0, 0], 1)
        self.assertTrue(tensor[0, 1], 2)

        tensor[0, 0] = 6  # also changes `m`
        tensor[0, 1] = 8

        self.assertEqual(m[0, 0], 6)
        self.assertEqual(m[0, 1], 8)

        m.Add(2)  # also changes `tensor`
        self.assertTrue(tensor[-1, 0], 8)
        self.assertTrue(tensor[0, 1], 10)

        del tensor
        gc.collect()

        self.assertEqual(m_reference_count, sys.getrefcount(m))

        self.assertEqual(m[0, 0], 8)  # `m` is still alive
        self.assertEqual(m[0, 1], 10)

    def test_pytorch_cpu_tensor_to_cu_submatrix(self):
        if kaldi.CudaCompiled():
            print('This test is for constructing a CuSubMatrix from '
                  'a CPU tensor')
            print('Kaldi is compiled with GPU, skip it')
            return

        tensor = torch.arange(6).reshape(2, 3).float()

        m = kaldi.CuSubMatrixFromDLPack(to_dlpack(tensor))
        self.assertIsInstance(m, kaldi.FloatCuSubMatrix)

        m.SetZero()  # also changes tensor, since memory is shared
        self.assertEqual(tensor[0, 1], 0)
        m.Add(10)
        self.assertEqual(tensor[0, 1], 10)

        del m

        # memory is shared between `m` and `tensor`
        m = kaldi.DLPackFloatCuSubMatrix.from_dlpack(to_dlpack(tensor))

        m.Add(100)
        self.assertEqual(tensor[0, 1], 110)

    def test_cu_vector_to_pytorch_cpu_tensor(self):
        if kaldi.CudaCompiled():
            print('This test is for constructing a CPU tensor from a CuVector')
            print('Kaldi is compiled with GPU, skip it')
            return

        dim = 2
        cpu_v = kaldi.FloatVector(size=dim)
        cpu_v[0] = 10
        cpu_v[1] = 20

        v = kaldi.FloatCuVector(cpu_v)
        self.assertEqual(v[0], 10)
        self.assertEqual(v[1], 20)

        v_reference_count = sys.getrefcount(v)

        # memory is shared between `v` and `tensor`
        tensor = from_dlpack(v.to_dlpack())

        self.assertEqual(v_reference_count + 1, sys.getrefcount(v))

        self.assertTrue(tensor.is_cuda == False)

        self.assertTrue(tensor[0], 10)
        self.assertTrue(tensor[0], 20)

        tensor[0] = 6  # also changes `v`
        tensor[1] = 8

        self.assertEqual(v[0], 6)
        self.assertEqual(v[1], 8)

        v.Add(2)  # also changes `tensor`
        self.assertTrue(tensor[0], 8)
        self.assertTrue(tensor[1], 10)

        del tensor
        gc.collect()

        self.assertEqual(v_reference_count, sys.getrefcount(v))

        self.assertEqual(v[0], 8)  # `v` is still alive
        self.assertEqual(v[1], 10)

    def test_pytorch_cpu_tensor_to_cu_subvector(self):
        if kaldi.CudaCompiled():
            print('This test is for constructing a CuSubVector from '
                  'a CPU tensor')
            print('Kaldi is compiled with GPU, skip it')
            return

        tensor = torch.tensor([10, 20]).float()
        v = kaldi.CuSubVectorFromDLPack(to_dlpack(tensor))

        v.SetZero()  # also changes tensor, since memory is shared
        self.assertEqual(tensor[0], 0)
        self.assertEqual(tensor[1], 0)

        v.Add(8)
        self.assertEqual(tensor[0], 8)

        del v

        # memory is shared between `v` and `tensor`
        v = kaldi.DLPackFloatCuSubVector.from_dlpack(to_dlpack(tensor))

        v.Add(100)
        self.assertEqual(tensor[0], 108)

    def test_pytorch_cpu_int_tensor_to_subvector(self):
        tensor = torch.arange(3, dtype=torch.int32)
        v = kaldi.IntSubVectorFromDLPack(to_dlpack(tensor))

        #  tensor and v share the underlying memory
        v[0] = 100
        v[1] = 200
        v[2] = 300

        self.assertEqual(tensor[0], 100)
        self.assertEqual(tensor[1], 200)
        self.assertEqual(tensor[2], 300)

        tensor[0] = 10
        self.assertEqual(v[0], 10)


if __name__ == '__main__':
    unittest.main()
