#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

from kaldi import fst


class TestDeterminizeLatticePruned(unittest.TestCase):

    def test_determinize_lattice_pruned_options(self):
        opts = fst.DeterminizeLatticePrunedOptions()
        print('\ndefault value for DeterminizeLatticePrunedOptions:')
        print(opts)

    def test_determinize_lattice_phone_pruned_options(self):
        opts = fst.DeterminizeLatticePhonePrunedOptions()
        print('\ndefault value for DeterminizeLatticePhonePrunedOptions:')
        print(opts)


if __name__ == '__main__':
    unittest.main()
