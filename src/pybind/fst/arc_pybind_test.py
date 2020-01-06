#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import math  # for math.isnan
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

from kaldi import fst


class TestArc(unittest.TestCase):

    def test_std_arc(self):
        arc = fst.StdArc()

        self.assertEqual(arc.Type(), 'standard')
        self.assertEqual(fst.StdArc.Type(), 'standard')

        ilabel = 0
        olabel = 1
        weight = fst.TropicalWeight.One()
        nextstate = 2

        arc = fst.StdArc(ilabel=ilabel,
                         olabel=olabel,
                         weight=weight,
                         nextstate=nextstate)
        self.assertEqual(arc.ilabel, ilabel)
        self.assertEqual(arc.olabel, olabel)
        self.assertEqual(arc.weight, weight)
        self.assertEqual(arc.nextstate, nextstate)
        self.assertEqual(str(arc),
                         '(ilabel: 0, olabel: 1, weight: 0, nextstate: 2)')


if __name__ == '__main__':
    unittest.main()
