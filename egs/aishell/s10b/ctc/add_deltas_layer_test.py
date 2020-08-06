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

from add_deltas_layer import AddDeltasLayer


class AddDeltasLayerTest(unittest.TestCase):

    def test(self):
        x = torch.tensor([
            [1, 3],
            [5, 10],
            [0, 1],
            [10, 20],
            [3, 1],
            [3, 2],
            [5, 1],
            [10, -2],
            [10, 20],
            [100, 200],
        ]).float()

        x = x.unsqueeze(0)

        transform = AddDeltasLayer(first_order_coef=[-0.2, -0.1, 0, 0.1, 0.2],
                                   second_order_coef=[
                                       0.04, 0.04, 0.01, -0.04, -0.1, -0.04,
                                       0.01, 0.04, 0.04
                                   ])
        y = transform(x.permute(0, 2, 1)).permute(0, 2, 1)

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

        np.testing.assert_array_almost_equal(y.numpy(),
                                             expected.numpy()[4:-4, :],
                                             decimal=5)

        reader.Close()

        shutil.rmtree(d)


if __name__ == '__main__':
    unittest.main()
