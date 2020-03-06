#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi
from kaldi import NnetChainExampleWriter
from kaldi import RandomAccessNnetChainExampleReader
from kaldi import SequentialNnetChainExampleReader
from kaldi import chain
from kaldi import nnet3


class TestNnetChainExample(unittest.TestCase):

    def test_nnet_chain_example(self):

        # TODO(fangjun): find a place to store the test data
        egs_rspecifier = 'scp:./aishell_test.scp'
        reader = SequentialNnetChainExampleReader(egs_rspecifier)
        for key, value in reader:
            inputs = value.inputs
            self.assertEqual(len(inputs), 1)

            nnet_io = inputs[0]
            self.assertTrue(isinstance(nnet_io, nnet3.NnetIo))
            self.assertEqual(nnet_io.name, 'input')

            features = nnet_io.features
            m = kaldi.FloatMatrix()
            features.GetMatrix(m)
            m = m.numpy()
            print(m.shape)

            self.assertTrue(isinstance(key, str))
            self.assertTrue(isinstance(value, nnet3.NnetChainExample))
            outputs = value.outputs
            num_outputs = len(outputs)
            self.assertEqual(num_outputs, 1)

            nnet_chain_sup = outputs[0]
            self.assertTrue(
                isinstance(nnet_chain_sup, nnet3.NnetChainSupervision))
            self.assertEqual(nnet_chain_sup.name, 'output')

            sup = nnet_chain_sup.supervision
            self.assertTrue(isinstance(sup, chain.Supervision))
            weight = sup.weight
            self.assertEqual(sup.weight, 1)
            self.assertEqual(sup.num_sequences, 1)
            # we have to egs in the ark, with 30 and 50 frames per sequence respectively
            self.assertTrue(sup.frames_per_sequence == 30 or
                            sup.frames_per_sequence == 50)
            self.assertEqual(sup.label_dim, 4336)

            # now comes to the FST part !!!
            fst = sup.fst
            self.assertTrue(isinstance(sup.fst, kaldi.fst.StdVectorFst))
            # see pybind/fst/vector_fst_pybind_test.py for operations wrapped for fst::StdVectorFst
            # TODO(fangjun): finish the test


if __name__ == '__main__':
    unittest.main()
