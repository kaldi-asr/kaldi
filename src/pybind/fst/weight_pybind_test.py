#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import math  # for math.isnan
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

from kaldi import fst


class TestWeight(unittest.TestCase):

    def test_float_weight(self):
        w = fst.FloatWeight(100)
        self.assertEqual(w.Value(), 100)
        self.assertEqual(str(w), '100')

    def test_tropical_weight(self):
        w = fst.TropicalWeight(100)
        self.assertEqual(w.Value(), 100)
        self.assertEqual(str(w), '100')
        self.assertEqual(w.Type(), 'tropical')

        one = w.One()
        self.assertEqual(one.Value(), 0)

        zero = fst.TropicalWeight.Zero()
        self.assertEqual(zero.Value(), float('inf'))

        self.assertTrue(math.isnan(w.NoWeight().Value()))


if __name__ == '__main__':
    unittest.main()
