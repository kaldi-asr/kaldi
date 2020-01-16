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

    def test_lattice_faster_decoder_config_parse_options(self):
        usage = 'testing'
        parse_options = kaldi.ParseOptions(usage)
        argv = [
            'a.out', '--print-args=false', '--beam=20', '--max-active=7000',
            'a.scp', 'b.scp'
        ]

        opts = kaldi.LatticeFasterDecoderConfig()
        opts.Register(parse_options)

        parse_options.Read(argv)
        self.assertEqual(parse_options.NumArgs(), 2)
        self.assertEqual(parse_options.GetArg(1), 'a.scp')
        self.assertEqual(parse_options.GetArg(2), 'b.scp')

        self.assertEqual(opts.beam, 20)
        self.assertEqual(opts.max_active, 7000)


if __name__ == '__main__':
    unittest.main()
