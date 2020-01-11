#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestLatticeFasterDecoder(unittest.TestCase):

    def test_lattice_faster_decoder_config(self):
        opts = kaldi.LatticeFasterDecoderConfig()
        print('default value for LatticeFasterDecoderConfig:')
        print(opts)


if __name__ == '__main__':
    unittest.main()
