#!/usr/bin/env python3

# Copyright 2019   Microsoft Corporation (author: Xingyu Na)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import numpy as np

import kaldi


class TestWaveData(unittest.TestCase):

    def test_duration(self):
        waveform = kaldi.FloatMatrix(1, 16000)
        wave_data = kaldi.WaveData(samp_freq=16000, data=waveform)
        self.assertEqual(1, wave_data.Duration())


if __name__ == '__main__':
    unittest.main()
