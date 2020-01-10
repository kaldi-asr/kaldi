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


class TestIOUtil(unittest.TestCase):

    def test_read_vec_int(self):
        tmp = mkdtemp()
        for binary in [True, False]:
            if binary:
                wspecifier = 'ark,scp:{dir}/ali.ark,{dir}/ali.scp'.format(
                    dir=tmp)
            else:
                wspecifier = 'ark,scp,t:{dir}/ali.ark,{dir}/ali.scp'.format(
                    dir=tmp)

            data = dict()

            key1 = 'key1'
            value1 = [0, 1, 3, 2]
            writer = kaldi.IntVectorWriter(wspecifier)
            writer.Write(key1, value1)

            data[key1] = value1

            key2 = 'key2'
            value2 = [1, 2, 3, 4, 5, 6]
            writer.Write(key2, value2)

            data[key2] = value2

            writer.Close()

            filename = '{}/ali.scp'.format(tmp)
            with open(filename, 'r') as f:
                for line in f:
                    key, rxfilename = line.split()
                    value = kaldi.read_vec_int(rxfilename)
                    self.assertTrue(key in data)
                    self.assertEqual(value, data[key])

        shutil.rmtree(tmp)

    def test_read_vec_flt(self):
        tmp = mkdtemp()
        for binary in [True, False]:
            if binary:
                wspecifier = 'ark,scp:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)
            else:
                wspecifier = 'ark,scp,t:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)

            data = dict()

            key1 = 'key1'
            value1 = np.arange(3).astype(np.float32)
            writer = kaldi.VectorWriter(wspecifier)
            writer.Write(key1, value1)
            data[key1] = value1

            key2 = 'key2'
            value2 = value1 * 10
            writer.Write(key2, value2)
            data[key2] = value2
            writer.Close()

            filename = '{}/test.scp'.format(tmp)
            with open(filename, 'r') as f:
                for line in f:
                    key, rxfilename = line.split()
                    value = kaldi.read_vec_flt(rxfilename)
                    self.assertTrue(key in data)
                    np.testing.assert_array_equal(value, data[key])

        shutil.rmtree(tmp)

    def test_read_mat(self):
        tmp = mkdtemp()
        for binary in [True, False]:
            if binary:
                wspecifier = 'ark,scp:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)
            else:
                wspecifier = 'ark,scp,t:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)

            data = dict()

            key1 = 'key1'
            value1 = np.arange(6 * 8).reshape(6, 8).astype(np.float32)
            writer = kaldi.MatrixWriter(wspecifier)
            writer.Write(key1, value1)
            data[key1] = value1

            key2 = 'key2'
            value2 = value1 * 10
            writer.Write(key2, value2)
            data[key2] = value2
            writer.Close()

            filename = '{}/test.scp'.format(tmp)
            with open(filename, 'r') as f:
                for line in f:
                    key, rxfilename = line.split()
                    value = kaldi.read_mat(rxfilename)
                    self.assertTrue(key in data)
                    np.testing.assert_array_equal(value, data[key])

        shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
