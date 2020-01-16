#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi


class TestParseOptions(unittest.TestCase):

    def test_args(self):
        b = kaldi.BoolArg()
        self.assertEqual(b.value, False)

        i = kaldi.Int32Arg()
        self.assertEqual(i.value, 0)

        u = kaldi.UInt32Arg()
        self.assertEqual(u.value, 0)

        f = kaldi.FloatArg()
        self.assertEqual(f.value, 0)

        d = kaldi.DoubleArg()
        self.assertEqual(d.value, 0)

        s = kaldi.StringArg()
        self.assertEqual(s.value, '')

    def test_parse_args(self):
        b = kaldi.BoolArg()
        i = kaldi.Int32Arg()
        u = kaldi.UInt32Arg()
        f = kaldi.FloatArg()
        d = kaldi.DoubleArg()
        s = kaldi.StringArg()

        parse_options = kaldi.ParseOptions(usage='test args')
        parse_options.Register(name='b', arg=b, doc='bool args')
        parse_options.Register('i', i, 'int32 args')
        parse_options.Register('u', u, 'uint32 args')
        parse_options.Register('f', f, 'float args')
        parse_options.Register('d', d, 'double args')
        parse_options.Register('s', s, 'string args')

        args = [
            'a.out', '--print-args=false', '--b', '--i=1', '--u=2', '--f=3',
            '--d=4', '--s=hello'
        ]
        parse_options.Read(args)
        self.assertEqual(b.value, True)
        self.assertEqual(i.value, 1)
        self.assertEqual(u.value, 2)
        self.assertEqual(f.value, 3)
        self.assertEqual(d.value, 4)
        self.assertEqual(s.value, 'hello')


if __name__ == '__main__':
    unittest.main()
