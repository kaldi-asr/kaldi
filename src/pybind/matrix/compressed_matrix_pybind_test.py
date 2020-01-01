#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi

class TestCompressedMatrix(unittest.TestCase):
    def test_from_float_matrix(self):
        num_rows = 2
        num_cols = 3
        m = kaldi.FloatMatrix(num_rows, num_cols)

        cm = kaldi.CompressedMatrix(m)

        self.assertEqual(cm.NumRows(), num_rows)
        self.assertEqual(cm.NumCols(), num_cols)

if __name__ == '__main__':
    unittest.main()
