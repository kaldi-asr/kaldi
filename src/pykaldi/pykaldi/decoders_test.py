#!/usr/bin/env python
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License. #
from __future__ import unicode_literals

import unittest
from pykaldi.decoders import PyGmmLatgenWrapper, DummyDecoder


# class TestHelperFunction(unittest.TestCase):
#     def test_version(self):
#         print 'pykaldi %s' % (str(pykaldi.__version__))
#         b, m, p = pykaldi.__version__
#         self.assertTrue(b + m + p > 0)
#
#     def test_git_revision(self):
#         print 'pykaldi last commit: %s' % (str(pykaldi.__git_revision__))
#         self.assertTrue(len(pykaldi.__git_revision__) == 40)


class TestPyGmmLatgenWrappeNotInit(unittest.TestCase):

    def setUp(self):
        self.d = PyGmmLatgenWrapper()

    def test_setup(self, args=['bad args']):
        self.assertFalse(self.d.setup(args))

    def test_decode(self, max_frames=10):
        self.assertEqual(self.d.decode(max_frames), 0)

    def test_frame_in(self):
        wav = b"ahoj"  # 16 bit audio -> 2 samples
        self.d.frame_in(wav)

    def test_frame_in_assert(self):
        wav = b"cau"  # 16 bit audio ->1.5 samples == bad
        with self.assertRaises(AssertionError):
            self.d.frame_in(wav)

    def get_best_path(self):
        self.assertEqual(self.d.get_best_path(), [])

    def get_Nbest(self):
        self.assertEqual(self.d.get_Nbest(), [])

    def get_lattice(self):
        self.assertEqual(self.d.get_lattice(), None)

    def get_raw_lattice(self):
        self.assertEqual(self.d.get_raw_lattice(), None)

    def prune_final(self):
        self.d.prune_final()

    def reset(self, keep_buffer_data=False):
        self.d.reset(keep_buffer_data)


class TestPyGmmLatgenWrappeRealInit(unittest.TestCase):
    # TODO how to initialize decoder without data
    # TODO create working dummy data
    pass


class TestDummyDecoder(unittest.TestCase):

    def setUp(self):
        self.d = DummyDecoder()

    def test_frame_in(self):
        self.d.frame_in("Ahoj")

    def test_decode(self):
        self.d.decode()

    def test_getNbest(self):
        self.assertEqual([(1.0, 'answer')], self.d.get_Nbest())

if __name__ == '__main__':
    unittest.main()
