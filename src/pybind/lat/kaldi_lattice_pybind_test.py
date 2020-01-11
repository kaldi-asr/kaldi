#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import math  # for math.isnan
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestKaldiLattice(unittest.TestCase):

    def test_lattice_arc(self):
        w = kaldi.LatticeWeight(10, 20)
        arc = kaldi.LatticeArc(ilabel=1, olabel=2, weight=w, nextstate=3)
        self.assertEqual(arc.ilabel, 1)
        self.assertEqual(arc.olabel, 2)
        self.assertEqual(arc.weight, w)
        self.assertEqual(arc.nextstate, 3)


if __name__ == '__main__':
    unittest.main()
