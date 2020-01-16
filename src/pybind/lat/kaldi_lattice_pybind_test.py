#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import shutil
import unittest

from tempfile import mkdtemp

import kaldi


class TestKaldiLattice(unittest.TestCase):

    def test_lattice_arc(self):
        w = kaldi.LatticeWeight(10, 20)
        arc = kaldi.LatticeArc(ilabel=1, olabel=2, weight=w, nextstate=3)

        self.assertEqual(arc.ilabel, 1)
        self.assertEqual(arc.olabel, 2)
        self.assertEqual(arc.weight, w)
        self.assertEqual(arc.nextstate, 3)

    def test_compact_lattice_arc(self):
        lat_w = kaldi.LatticeWeight(10, 20)
        s = [1, 2, 3, 4, 5]
        w = kaldi.CompactLatticeWeight(lat_w, s)
        arc = kaldi.CompactLatticeArc(ilabel=1, olabel=2, weight=w, nextstate=3)

        self.assertEqual(arc.ilabel, 1)
        self.assertEqual(arc.olabel, 2)
        self.assertEqual(arc.weight, w)
        self.assertEqual(arc.nextstate, 3)

    def test_lattice_IO(self):
        rspecifier = 'scp:./test_lat.scp'

        expected_keys = ['BAC009S0764W0121', 'BAC009S0764W0122']

        data = dict()
        reader = kaldi.SequentialLatticeReader(rspecifier)
        for key, value in reader:
            data[key] = value.Copy()
        reader.Close()

        self.assertEqual(len(data), len(expected_keys))
        for key in expected_keys:
            self.assertTrue(key in data)

        # now for random access reader
        reader = kaldi.RandomAccessLatticeReader(rspecifier)
        for key in expected_keys:
            self.assertTrue(reader.HasKey(key))
            self.assertEqual(reader.Value(key).ToString(), data[key].ToString())
        reader.Close()

        # now for writer
        tmp = mkdtemp()
        for binary in [True, False]:
            if binary:
                wspecifier = 'ark,scp:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)
            else:
                wspecifier = 'ark,scp,t:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)

            writer = kaldi.LatticeWriter(wspecifier)
            for key in expected_keys:
                writer.Write(key, data[key])
            writer.Close()

            # read them back
            rspecifier = 'scp:{}/test.scp'.format(tmp)
            reader = kaldi.SequentialLatticeReader(rspecifier)
            for key, value in reader:
                self.assertTrue(key in data)
                self.assertEqual(value.ToString(), data[key].ToString())
            reader.Close()

        shutil.rmtree(tmp)

    def test_compact_lattice_IO(self):
        rspecifier = 'scp:./test_clat.scp'

        expected_keys = ['BAC009S0764W0121', 'BAC009S0764W0122']

        data = dict()
        reader = kaldi.SequentialCompactLatticeReader(rspecifier)
        for key, value in reader:
            data[key] = value.Copy()
        reader.Close()

        self.assertEqual(len(data), len(expected_keys))
        for key in expected_keys:
            self.assertTrue(key in data)

        # now for random access reader
        reader = kaldi.RandomAccessCompactLatticeReader(rspecifier)
        for key in expected_keys:
            self.assertTrue(reader.HasKey(key))
            self.assertEqual(reader.Value(key).ToString(), data[key].ToString())
        reader.Close()

        # now for writer
        tmp = mkdtemp()
        for binary in [True, False]:
            if binary:
                wspecifier = 'ark,scp:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)
            else:
                wspecifier = 'ark,scp,t:{dir}/test.ark,{dir}/test.scp'.format(
                    dir=tmp)

            writer = kaldi.CompactLatticeWriter(wspecifier)
            for key in expected_keys:
                writer.Write(key, data[key])
            writer.Close()

            # read them back
            rspecifier = 'scp:{}/test.scp'.format(tmp)
            reader = kaldi.SequentialCompactLatticeReader(rspecifier)
            for key, value in reader:
                self.assertTrue(key in data)
                self.assertEqual(value.ToString(), data[key].ToString())
            reader.Close()

        shutil.rmtree(tmp)


if __name__ == '__main__':
    unittest.main()
