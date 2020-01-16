#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import math  # for math.isnan
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

from kaldi import fst


class TestLatticeWeight(unittest.TestCase):

    def test_lattice_weight(self):
        w = fst.LatticeWeight()
        self.assertEqual(w.Value1(), 0)  # lm cost
        self.assertEqual(w.Value2(), 0)  # acoustic cost

        w.SetValue1(1)
        w.SetValue2(2)
        self.assertEqual(w.Value1(), 1)
        self.assertEqual(w.Value2(), 2)

        w = fst.LatticeWeight(10, 20)
        self.assertEqual(w.Value1(), 10)
        self.assertEqual(w.Value2(), 20)

        w = fst.LatticeWeight.One()
        self.assertEqual(w.Value1(), 0)
        self.assertEqual(w.Value2(), 0)

        w = fst.LatticeWeight.Zero()
        self.assertEqual(w.Value1(), float('inf'))
        self.assertEqual(w.Value2(), float('inf'))

        self.assertEqual(w.Type(), 'lattice4')

        w = fst.LatticeWeight.NoWeight()

        self.assertTrue(math.isnan(w.Value1()))
        self.assertTrue(math.isnan(w.Value2()))

    def test_compact_lattice_weight(self):
        lat_w = fst.LatticeWeight(10, 20)
        s = [1, 2, 3, 4, 5]

        w = fst.CompactLatticeWeight(lat_w, s)
        self.assertEqual(w.Weight(), lat_w)
        self.assertEqual(w.String(), s)
        self.assertEqual(str(w), '10,20,1_2_3_4_5')

        # compactlattice44: the first 4 is for sizeof(float)
        # and the second is for sizeof(int)
        self.assertEqual(w.Type(), 'compactlattice44')


if __name__ == '__main__':
    unittest.main()
