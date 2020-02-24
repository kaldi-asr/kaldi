#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import shutil
import tempfile
import unittest

import numpy as np

import torch
import torch.nn.functional as F

import kaldi

from transform import AddDeltasTransform


class TransformTest(unittest.TestCase):

    def test_add_deltas_transform(self):
        x = torch.tensor([
            [1, 3],
            [5, 10],
            [0, 1],
            [10, 20],
            [3, 1],
            [3, 2],
            [5, 1],
            [10, -2],
        ]).float()

        x = x.unsqueeze(0)

        transform = AddDeltasTransform()
        y = transform(x)

        # now use kaldi's add-deltas to compute the ground truth
        d = tempfile.mkdtemp()

        wspecifier = 'ark:{}/feats.ark'.format(d)

        writer = kaldi.MatrixWriter(wspecifier)
        writer.Write('utt1', x.squeeze(0).numpy())
        writer.Close()

        delta_feats_specifier = 'ark:{dir}/delta.ark'.format(dir=d)

        cmd = '''
        add-deltas --print-args=false --delta-order=2 --delta-window=2 {} {}
        '''.format(wspecifier, delta_feats_specifier)

        os.system(cmd)

        reader = kaldi.RandomAccessMatrixReader(delta_feats_specifier)

        expected = reader['utt1']

        y = y.squeeze(0)

        np.testing.assert_array_almost_equal(y.numpy(), expected.numpy())

        reader.Close()

        shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
