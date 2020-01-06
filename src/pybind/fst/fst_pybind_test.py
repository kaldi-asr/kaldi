#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

from kaldi import fst


class TestArc(unittest.TestCase):

    def test_FstWriteOptions(self):
        source = "source name"
        write_header = True
        write_isymbols = False
        write_osymbols = True
        align = False
        stream_write = True

        opt = fst.FstWriteOptions(source=source,
                                  write_header=write_header,
                                  write_isymbols=write_isymbols,
                                  write_osymbols=write_osymbols,
                                  align=align,
                                  stream_write=stream_write)

        self.assertEqual(opt.source, source)
        self.assertEqual(opt.write_header, write_header)
        self.assertEqual(opt.write_isymbols, write_isymbols)
        self.assertEqual(opt.write_osymbols, write_osymbols)
        self.assertEqual(opt.align, align)
        self.assertEqual(opt.stream_write, stream_write)


if __name__ == '__main__':
    unittest.main()
