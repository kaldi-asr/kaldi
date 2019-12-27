#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import numpy as np

import kaldi


class TestFloatSubVecotr(unittest.TestCase):

    def test_numpy(self):
        num_data = 10

        data = np.arange(num_data).astype(np.float32)

        # =============================================================
        # build a FloatSubVector() from a numpy array; memory is shared
        # -------------------------------------------------------------
        v = kaldi.FloatSubVector(data)
        self.assertEqual(v.Dim(), num_data)
        for i in range(num_data):
            self.assertEqual(i, v[i])

        # memory is shared between numpy array and FloatSubVector
        for i in range(num_data):
            v[i] += 10
            self.assertEqual(data[i], v[i])

        # =============================================================
        # Convert a FloatSubVector to a numpy array; memory is shared
        # -------------------------------------------------------------
        v_reference_count = sys.getrefcount(v)

        d = v.numpy()

        self.assertEqual(v_reference_count + 1, sys.getrefcount(v))

        self.assertIsInstance(d, np.ndarray)
        self.assertEqual(d.ndim, 1)
        self.assertEqual(d.dtype, np.float32)
        self.assertEqual(d.size, v.Dim())

        for i in range(num_data):
            d[i] += 10
            self.assertEqual(v[i], d[i])

        del d
        self.assertEqual(v_reference_count, sys.getrefcount(v))


class TestFloatVecotr(unittest.TestCase):

    def test_to_numpy(self):
        # first, build a kaldi vector
        dim = 8
        v = kaldi.FloatVector(size=dim)
        self.assertEqual(v.Dim(), dim)

        for i in range(dim):
            self.assertEqual(v[i], 0)

        # now to numpy; memory is shared
        d = v.numpy()

        d += 10
        for i in range(dim):
            self.assertEqual(d[i], v[i])


if __name__ == '__main__':
    unittest.main()
