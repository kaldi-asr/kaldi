#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import unittest

import kaldi
from kaldi import fst


class TestStdVectorFst(unittest.TestCase):

    def test_std_vector_fst(self):
        vector_fst = fst.StdVectorFst()

        # create the same FST from
        # http://www.openfst.org/twiki/bin/view/FST/FstQuickTour#Creating%20FSTs%20Using%20Constructors
        # 1st state will be state 0 (returned by AddState)
        vector_fst.AddState()
        vector_fst.SetStart(0)
        vector_fst.AddArc(0, fst.StdArc(1, 1, fst.TropicalWeight(0.5), 1))
        vector_fst.AddArc(0, fst.StdArc(2, 2, fst.TropicalWeight(1.5), 1))

        vector_fst.AddState()
        vector_fst.AddArc(1, fst.StdArc(3, 3, fst.TropicalWeight(2.5), 2))

        vector_fst.AddState()
        vector_fst.SetFinal(2, fst.TropicalWeight(3.5))

        # fstprint with default options
        print(vector_fst)

        print('-' * 20)
        print('fstprint with customized options (default options)')
        print(
            vector_fst.ToString(is_acceptor=False,
                                show_weight_one=False,
                                fst_field_separator=" " * 6,
                                missing_symbol=""))
        # now build the symbol table
        input_words = '<eps> a b c'.split()
        output_words = '<eps> x y z'.split()

        isymbol_table = fst.SymbolTable()
        for w in input_words:
            isymbol_table.AddSymbol(w)

        osymbol_table = fst.SymbolTable()
        for w in output_words:
            osymbol_table.AddSymbol(w)

        vector_fst.SetInputSymbols(isyms=isymbol_table)
        vector_fst.SetOutputSymbols(osyms=osymbol_table)
        print(vector_fst)

        # now for I/O
        fst_filename = 'test.fst'
        vector_fst.Write(filename=fst_filename)

        read_back_fst = fst.StdVectorFst.Read(filename=fst_filename)
        print('fst after reading back is:')
        print(read_back_fst)

        # TODO(fangjun): check that the two fsts are the same: start/final/states/arcs/symbol tables
        # TODO(fangjun): add fstdraw support
        # TODO(fangjun): test fstcompile

        text_fst_str = read_back_fst.ToString()

        compiled_filename = "compiled.fst"
        fst.CompileFst(text_fst_str=text_fst_str,
                       out_binary_fst_filename=compiled_filename,
                       isymbols=isymbol_table,
                       osymbols=osymbol_table,
                       keep_isymbols=True,
                       keep_osymbols=True)

        read_back_compiled_fst = fst.StdVectorFst.Read(
            filename=compiled_filename)
        print('-' * 20)
        print('read back compiled fst is:')
        print(read_back_compiled_fst)

        os.remove(compiled_filename)
        os.remove(fst_filename)


if __name__ == '__main__':
    unittest.main()
