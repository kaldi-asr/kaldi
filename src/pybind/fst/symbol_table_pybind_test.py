#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi
from kaldi import fst


class TestSymbolTable(unittest.TestCase):

    def test_symbol_table(self):
        self.assertEqual(fst.kNoSymbol, -1)

        # the name can be arbitrary string, or it can simply be omitted
        words = fst.SymbolTable(name='words.txt')
        self.assertEqual(words.Name(), 'words.txt')

        #        0     1      2     3    4     5    6
        text = '<eps> hello OpenFST in Python with Pybind11'.split()
        indices = [words.AddSymbol(w) for w in text]

        for i in range(len(text)):
            self.assertEqual(words.Find(key=i), text[i])
            self.assertEqual(words.Find(symbol=text[i]), i)
            self.assertTrue(words.Member(key=i))
            self.assertTrue(words.Member(symbol=text[i]))

        self.assertEqual(words.Find('Kaldi'), fst.kNoSymbol)
        self.assertEqual(words.AvailableKey(), len(text))
        self.assertEqual(words.NumSymbols(), len(text))

        self.assertEqual(words.GetNthKey(pos=5), 5)

        symbol_table_iterator = fst.SymbolTableIterator(words)
        i = 0
        while not symbol_table_iterator.Done():
            index = symbol_table_iterator.Value()
            symbol = symbol_table_iterator.Symbol()
            self.assertEqual(index, i)
            self.assertEqual(symbol, text[i])
            symbol_table_iterator.Next()
            i += 1

        # the following is more pythonic for iteration
        i = 0
        kaldi_symbol_iterator = kaldi.SymbolTableIterator(words)
        for index, symbol in kaldi_symbol_iterator:
            self.assertEqual(index, i)
            self.assertEqual(symbol, text[i])
            i += 1

        # to use the iterator again, we must reset it manually
        kaldi_symbol_iterator.Reset()
        i = 0
        for index, symbol in kaldi_symbol_iterator:
            self.assertEqual(index, i)
            self.assertEqual(symbol, text[i])
            i += 1

        # after removing the word 'with' whose index is 5

        words.RemoveSymbol(key=5)
        self.assertEqual(words.Find(key=5), '')
        self.assertEqual(words.Find(symbol='with'), fst.kNoSymbol)
        self.assertEqual(words.AvailableKey(), len(text))  # still 7
        self.assertEqual(words.NumSymbols(), len(text) - 1)  # now 6 = 7-1

        # at pos 5, we have the word `Pybind11` which has index 6
        self.assertEqual(words.GetNthKey(pos=5), 6)

        words.AddSymbol(symbol='with', key=5)
        self.assertEqual(words.Find(key=5), 'with')
        self.assertEqual(words.Find(symbol='with'), 5)
        self.assertEqual(words.AvailableKey(), len(text))  # still 7
        self.assertEqual(words.NumSymbols(), len(text))  # now 7

        self.assertEqual(words.GetNthKey(pos=5), 6)  # it's still 6 !

        # test I/O
        # to control the field separator, we can use
        # fst::SymbolTableTextOptions::fst_field_separator,
        # the default separator is controlled by FLAGS_fst_field_separator
        # whose default value is '\t ', e.g., a tab and a space
        filename = 'words.txt'
        words.WriteText(filename=filename)

        words_txt_read_back = fst.SymbolTable.ReadText(filename=filename)

        self.assertEqual(words.CheckSum(), words_txt_read_back.CheckSum())
        self.assertTrue(fst.CompatSymbols(words, words_txt_read_back))

        # now for binary
        filename = 'words.bin'
        words.Write(filename=filename)
        words_bin_read_back = fst.SymbolTable.Read(filename=filename)

        self.assertEqual(words.CheckSum(), words_bin_read_back.CheckSum())
        self.assertTrue(fst.CompatSymbols(words, words_bin_read_back))

        os.remove('words.bin')
        os.remove('words.txt')


if __name__ == '__main__':
    unittest.main()
