# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import kaldi_pybind
import kaldi_pybind.fst as fst


class SymbolTableIterator(fst.SymbolTableIterator):

    def __iter__(self):
        while not self.Done():
            index = self.Value()
            symbol = self.Symbol()
            yield index, symbol
            self.Next()
