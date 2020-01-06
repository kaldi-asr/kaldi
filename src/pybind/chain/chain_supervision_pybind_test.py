#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import numpy as np

from kaldi import chain


class TestChainSupervision(unittest.TestCase):

    def test_chain_supervision(self):
        supervision = chain.Supervision()
        print(supervision)


if __name__ == '__main__':
    unittest.main()
