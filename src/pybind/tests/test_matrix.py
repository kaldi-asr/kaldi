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

    def test_from_numpy(self):
        num_data = 10
        data = np.arange(num_data).astype(np.float32)
        v = kaldi.FloatSubVector(data)
        self.assertEqual(v.Dim(), num_data)
        for i in range(num_data):
            self.assertEqual(i, v[i])

        # memory is shared between numpy array and FloatSubVector
        for i in range(num_data):
            v[i] += 10
            self.assertEqual(data[i], i + 10)


class TestFloatSubMatrix(unittest.TestCase):

    def test_from_numpy(self):
        num_rows = 5
        num_cols = 6
        data = np.arange(num_rows * num_cols).reshape(
            num_rows, num_cols).astype(np.float32)
        m = kaldi.FloatSubMatrix(data)
        self.assertEqual(m.NumRows(), num_rows)
        self.assertEqual(m.NumCols(), num_cols)
        self.assertEqual(m.Stride(), data.strides[0] / 4)
        for r in range(num_rows):
            for c in range(num_cols):
                self.assertEqual(m[r, c], data[r, c])

        # memory is shared between numpy array and FloatSubMatrix
        for r in range(num_rows):
            for c in range(num_cols):
                m[r, c] += 10
                self.assertEqual(m[r, c], data[r, c])


if __name__ == '__main__':
    unittest.main()
