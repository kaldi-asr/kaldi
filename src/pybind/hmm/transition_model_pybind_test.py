#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestTransitionModel(unittest.TestCase):

    def test(self):
        rxfilename = './trans_0.txt'
        ki = kaldi.Input()
        is_opened, = ki.Open(rxfilename, read_header=False)
        self.assertTrue(is_opened)

        trans_model = kaldi.TransitionModel()
        trans_model.Read(ki.Stream(), binary=False)
        ki.Close()

        # you can print the model
        # print(trans_model)
        # to save space, we do NOT print it here


if __name__ == '__main__':
    unittest.main()
