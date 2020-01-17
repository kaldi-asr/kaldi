#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import unittest

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import shutil

from tempfile import mkdtemp

import numpy as np

import kaldi


class TestTableTypes(unittest.TestCase):

    def test_int_vector(self):
        tmp = mkdtemp()
        wspecifier = 'ark,scp:{dir}/ali.ark,{dir}/ali.scp'.format(dir=tmp)

        data = dict()

        key1 = 'key1'
        value1 = [0, 1, 3, 2]
        writer = kaldi.IntVectorWriter(wspecifier)
        writer.Write(key1, value1)

        key2 = 'key2'
        value2 = [0, 2]
        writer.Write(key2, value2)

        writer.Close()

        data[key1] = value1
        data[key2] = value2

        rspecifier = 'scp:{}/ali.scp'.format(tmp)
        reader = kaldi.SequentialIntVectorReader(rspecifier)
        for key, value in reader:
            self.assertTrue(key in data)
            np.testing.assert_array_equal(value, data[key])
        reader.Close()

        reader = kaldi.RandomAccessIntVectorReader(rspecifier)
        for key in data.keys():
            self.assertTrue(reader.HasKey(key))
            np.testing.assert_array_equal(reader.Value(key), data[key])
        reader.Close()

        shutil.rmtree(tmp)

    def test_float_vector(self):
        tmp = mkdtemp()
        wspecifier = 'ark,scp:{dir}/test.ark,{dir}/test.scp'.format(dir=tmp)

        data = dict()

        key1 = 'key1'
        value1 = np.random.rand(2).astype(np.float32)
        writer = kaldi.VectorWriter(wspecifier)
        writer.Write(key1, value1)

        key2 = 'key2'
        value2 = np.random.rand(10).astype(np.float32)
        writer.Write(key2, value2)

        writer.Close()

        data[key1] = value1
        data[key2] = value2

        rspecifier = 'scp:{}/test.scp'.format(tmp)
        reader = kaldi.SequentialVectorReader(rspecifier)
        for key, value in reader:
            self.assertTrue(key in data)
            np.testing.assert_array_almost_equal(value.numpy(), data[key])
        reader.Close()

        reader = kaldi.RandomAccessVectorReader(rspecifier)
        for key in data.keys():
            self.assertTrue(reader.HasKey(key))
            np.testing.assert_array_almost_equal(
                reader.Value(key).numpy(), data[key])
        reader.Close()

        shutil.rmtree(tmp)

    def test_float_matrix(self):
        tmp = mkdtemp()
        wspecifier = 'ark,scp:{dir}/test.ark,{dir}/test.scp'.format(dir=tmp)

        data = dict()

        key1 = 'key1'
        value1 = np.random.rand(6, 8).astype(np.float32)
        writer = kaldi.MatrixWriter(wspecifier)
        writer.Write(key1, value1)

        key2 = 'key2'
        value2 = np.random.rand(10, 20).astype(np.float32)
        writer.Write(key2, value2)

        writer.Close()

        data[key1] = value1
        data[key2] = value2

        rspecifier = 'scp:{}/test.scp'.format(tmp)
        reader = kaldi.SequentialMatrixReader(rspecifier)
        for key, value in reader:
            self.assertTrue(key in data)
            np.testing.assert_array_almost_equal(value.numpy(), data[key])
        reader.Close()

        reader = kaldi.RandomAccessMatrixReader(rspecifier)
        for key in data.keys():
            self.assertTrue(reader.HasKey(key))
            np.testing.assert_array_almost_equal(
                reader.Value(key).numpy(), data[key])
        reader.Close()
        
        # test RandomAccessReader with context manager        
        with kaldi.RandomAccessMatrixReader(rspecifier) as reader:
            for key in data.keys():
                self.assertTrue(reader.HasKey(key))
                np.testing.assert_array_almost_equal(
                    reader.Value(key).numpy(), data[key])

        shutil.rmtree(tmp)


if __name__ == '__main__':
    np.random.seed(20200110)
    unittest.main()
