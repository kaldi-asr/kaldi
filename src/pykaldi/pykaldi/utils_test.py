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
import copy
from utils import expand_prefix
from utils import fst_shortest_path_to_lists
import fst
import os
from subprocess import call


class TestExpandPref(unittest.TestCase):

    def setUp(self):
        self.test = {'x': '1',
                     'prefixdir': 'myownpath',
                     'mypath': {'prefix': 'prefixdir', 'value': 'origin'},
                     'y': {'prefix': 'mypath', 'value': 'file'},
                     'innerdic': {'a': {'prefix': 'mypath', 'value': ''}, 'b': ['arg1', 'arg2']},
                     'innerlist': [{'prefix': 'mypath', 'value': 'file2'}, 'arg3']
                     }
        self.gold = {'x': '1',
                     'prefixdir': 'myownpath',
                     'mypath': 'myownpath/origin',
                     'y': 'myownpath/origin/file',
                     'innerdic': {'a': 'myownpath/origin', 'b': ['arg1', 'arg2']},
                     'innerlist': ['myownpath/origin/file2', 'arg3']
                     }

    def test_expand_prefix(self):
        test, gold = copy.deepcopy(self.test), copy.deepcopy(self.gold)
        expand_prefix(test, gold)

        print '%r\ntest vs gold\n%r\n' % (test, gold)
        self.assertTrue(self.gold == gold, 'We modified gold!')

        self.assertFalse(set(test) ^ set(gold), 'symetric difference of keys should be empty')
        self.assertTrue(gold['y'] == test['y'], 'prefix expansion  fails')
        t, g = gold['innerdic']['a'], test['innerdic']['a']
        self.assertTrue(t == g, 'empty value for expansion fails: %r vs %r' % (t, g))
        self.assertTrue(gold['innerlist'][0] == test['innerlist'][0], 'list expansion fails')


@unittest.skipIf(call(['which', 'fstcompile']) != 0, 'We need to compile testing fst using fstcompile')
class TestLatticeToNbest(unittest.TestCase):

    def setUp(self):
        shortest_txt = os.path.join(os.path.dirname(__file__), 'test_shortest.txt')
        shortest_fst = os.path.join(os.path.dirname(__file__), 'test_shortest.fst')
        try:
            if not os.path.exists(shortest_fst):
                call(['fstcompile', shortest_txt, shortest_txt])
        except Exception as e:
            print 'Failed to generate testing fst'
            raise e
        self.s = fst.read_std(shortest_fst)
        self.s_result = [(110.40000001341105, [1, 3, 4]),
                         (110.6000000089407, [2, 3, 4]), (1000.2000000029802, [2])]

    def test_shortestPathToLists(self):
        shortest_fst = self.s.shortest_path(10)
        nbest_list = fst_shortest_path_to_lists(shortest_fst)
        self.assertSequenceEqual(nbest_list, self.s_result)


if __name__ == '__main__':
    unittest.main()
