'''
This file is modified from the PyKaldi project
https://github.com/pykaldi/pykaldi/blob/master/kaldi/util/table.py
'''
#
#
# Author:  Dogan Can
# Author:  Fanjun Kuang
#
#
'''
For detailed documentation of Kaldi tables, table readers/writers, table
read/write specifiers, see `Kaldi I/O mechanisms`_ and
`Kaldi I/O from a command-line perspective`_.

.. _Kaldi I/O mechanisms:
   http://kaldi-asr.org/doc/io.html
.. _Kaldi I/O from a command-line perspective:
   http://kaldi-asr.org/doc/io_tut.html
'''

# TODO(fangjun): set the PYTHONPATH environment variable outside this script
# to avoid set sys.path for every Python script

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np

import kaldi_pybind
from kaldi_pybind.nnet3 import _SequentialNnetChainExampleReader
from kaldi_pybind.nnet3 import _RandomAccessNnetChainExampleReader
from kaldi_pybind.nnet3 import _NnetChainExampleWriter

from kaldi_pybind.nnet3 import _SequentialNnetExampleReader
from kaldi_pybind.nnet3 import _RandomAccessNnetExampleReader

from kaldi_pybind.feat import _SequentialWaveReader
from kaldi_pybind.feat import _RandomAccessWaveReader
from kaldi_pybind.feat import _SequentialWaveInfoReader
from kaldi_pybind.feat import _RandomAccessWaveInfoReader

from kaldi_pybind import _SequentialBaseFloatMatrixReader
from kaldi_pybind import _RandomAccessBaseFloatMatrixReader
from kaldi_pybind import _BaseFloatMatrixWriter

from kaldi_pybind import _SequentialBaseFloatVectorReader
from kaldi_pybind import _RandomAccessBaseFloatVectorReader
from kaldi_pybind import _BaseFloatVectorWriter

from kaldi_pybind import _CompressedMatrixWriter

from kaldi_pybind import _SequentialInt32VectorReader
from kaldi_pybind import _RandomAccessInt32VectorReader
from kaldi_pybind import _Int32VectorWriter

from kaldi_pybind import _SequentialLatticeReader
from kaldi_pybind import _RandomAccessLatticeReader
from kaldi_pybind import _LatticeWriter

from kaldi_pybind import _SequentialCompactLatticeReader
from kaldi_pybind import _RandomAccessCompactLatticeReader
from kaldi_pybind import _CompactLatticeWriter

################################################################################
# Sequential Readers
################################################################################


class _SequentialReaderBase(object):
    '''Base class defining the Python API for sequential table readers.'''

    def __init__(self, rspecifier=''):
        '''
        This class is used for reading objects sequentially from an archive or
        script file. It implements the iterator protocol similar to how Python
        implements iteration over dictionaries. Each iteration returns a `(key,
        value)` pair from the table in sequential order.

        Args:
            rspecifier(str): Kaldi rspecifier for reading the table.
                If provided, the table is opened for reading.

        Raises:
            IOError: If opening the table for reading fails.
        '''
        super(_SequentialReaderBase, self).__init__()
        if rspecifier != '':
            if not self.Open(rspecifier):
                raise IOError('Error opening sequential table reader with '
                              'rspecifier: {}'.format(rspecifier))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.IsOpen():
            self.Close()

    def __iter__(self):
        while not self.Done():
            key = self.Key()
            value = self.Value()
            yield key, value
            self.Next()

    def Open(self, rspecifier):
        '''Opens the table for reading.

        Args:
            rspecifier(str): Kaldi rspecifier for reading the table.
                If provided, the table is opened for reading.

        Returns:
            True if table is opened successfully, False otherwise.

        Raises:
            IOError: If opening the table for reading fails.
        '''
        return super(_SequentialReaderBase, self).Open(rspecifier)

    def Done(self):
        '''Indicates whether the table reader is exhausted or not.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
          True if the table reader is exhausted, False otherwise.
        '''
        return super(_SequentialReaderBase, self).Done()

    def Key(self):
        '''Returns the current key.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
            str: The current key.
        '''
        return super(_SequentialReaderBase, self).Key()

    def FreeCurrent(self):
        '''Deallocates the current value.

        This method is provided as an optimization to save memory, for large
        objects.
        '''
        super(_SequentialReaderBase, self).FreeCurrent()

    def Value(self):
        '''Returns the current value.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
            The current value.
        '''
        return super(_SequentialReaderBase, self).Value()

    def Next(self):
        '''Advances the table reader.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.
        '''
        super(_SequentialReaderBase, self).Next()

    def IsOpen(self):
        '''Indicates whether the table reader is open or not.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
          True if the table reader is open, False otherwise.
        '''
        return super(_SequentialReaderBase, self).IsOpen()

    def Close(self):
        '''Closes the table.

        This method is provided for compatibility with the C++ API only;
        most users should use the Pythonic API.

        Returns:
            True if table is closed successfully, False otherwise.
        '''
        return super(_SequentialReaderBase, self).Close()


class SequentialNnetChainExampleReader(_SequentialReaderBase,
                                       _SequentialNnetChainExampleReader):
    '''Sequential table reader for nnet chain examples.'''
    pass

class SequentialNnetExampleReader(_SequentialReaderBase,
                                  _SequentialNnetExampleReader):
    '''Sequential table reader for nnet examples.'''
    pass


class SequentialWaveReader(_SequentialReaderBase, _SequentialWaveReader):
    '''Sequential table reader for wave files.'''
    pass


class SequentialWaveInfoReader(_SequentialReaderBase,
                               _SequentialWaveInfoReader):
    '''Sequential table reader for wave file headers.'''
    pass


class SequentialMatrixReader(_SequentialReaderBase,
                             _SequentialBaseFloatMatrixReader):
    '''Sequential table reader for single precision matrices.'''
    pass


class SequentialVectorReader(_SequentialReaderBase,
                             _SequentialBaseFloatVectorReader):
    '''Sequential table reader for single precision vectors.'''
    pass


class SequentialIntVectorReader(_SequentialReaderBase,
                                _SequentialInt32VectorReader):
    '''Sequential table reader for integer sequences.'''
    pass


class SequentialLatticeReader(_SequentialReaderBase, _SequentialLatticeReader):
    '''Sequential table reader for lattices.'''
    pass


class SequentialCompactLatticeReader(_SequentialReaderBase,
                                     _SequentialCompactLatticeReader):
    '''Sequential table reader for compact lattices.'''
    pass


################################################################################
# Random Access Readers
################################################################################


class _RandomAccessReaderBase(object):
    '''Base class defining the Python API for random access table readers.'''

    def __init__(self, rspecifier=''):
        '''
            This class is used for randomly accessing objects in an archive or
            script file. It implements `__contains__` and `__getitem__` methods to
            provide a dictionary-like interface for accessing table entries. e.g.
            `reader[key]` returns the `value` associated with the `key`.

            Args:
                rspecifier(str): Kaldi rspecifier for reading the table.
                    If provided, the table is opened for reading.

            Raises:
                IOError: If opening the table for reading fails.
            '''
        super(_RandomAccessReaderBase, self).__init__()
        if rspecifier != '':
            if not self.Open(rspecifier):
                raise IOError('Error opening random access table reader with '
                              'rspecifier: {}'.format(rspecifier))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.IsOpen():
            self.Close()

    def __contains__(self, key):
        return self.HasKey(key)

    def __getitem__(self, key):
        if self.HasKey(key):
            return self.Value(key)
        else:
            raise KeyError(key)

    def Open(self, rspecifier):
        '''Opens the table for reading.

            Args:
                rspecifier(str): Kaldi rspecifier for reading the table.
                    If provided, the table is opened for reading.

            Returns:
                True if table is opened successfully, False otherwise.

            Raises:
                IOError: If opening the table for reading fails.
            '''
        return super(_RandomAccessReaderBase, self).Open(rspecifier)

    def HasKey(self, key):
        '''Checks whether the table has the key.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Args:
                key (str): The key.

            Returns:
              True if the table has the key, False otherwise.
            '''
        return super(_RandomAccessReaderBase, self).HasKey(key)

    def Value(self, key):
        '''Returns the value associated with the key.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Args:
                key (str): The key.

            Returns:
                The value associated with the key.
            '''
        return super(_RandomAccessReaderBase, self).Value(key)

    def IsOpen(self):
        '''Indicates whether the table reader is open or not.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Returns:
              True if the table reader is open, False otherwise.
            '''
        return super(_RandomAccessReaderBase, self).IsOpen()

    def Close(self):
        '''Closes the table.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Returns:
                True if table is closed successfully, False otherwise.
            '''
        return super(_RandomAccessReaderBase, self).Close()


class RandomAccessNnetChainExampleReader(_RandomAccessReaderBase,
                                         _RandomAccessNnetChainExampleReader):
    '''Random access table reader for nnet chain examples.'''
    pass

class RandomAccessNnetExampleReader(_RandomAccessReaderBase,
                                    _RandomAccessNnetExampleReader):
    '''Random access table reader for nnet examples.'''
    pass

class RandomAccessWaveReader(_RandomAccessReaderBase, _RandomAccessWaveReader):
    '''Random access table reader for wave files.'''
    pass


class RandomAccessWaveInfoReader(_RandomAccessReaderBase,
                                 _RandomAccessWaveInfoReader):
    '''Random access table reader for wave file headers.'''
    pass


class RandomAccessMatrixReader(_RandomAccessReaderBase,
                               _RandomAccessBaseFloatMatrixReader):
    '''Random access table reader for single precision matrices.'''
    pass


class RandomAccessVectorReader(_RandomAccessReaderBase,
                               _RandomAccessBaseFloatVectorReader):
    '''Random access table reader for single precision vectors.'''
    pass


class RandomAccessIntVectorReader(_RandomAccessReaderBase,
                                  _RandomAccessInt32VectorReader):
    '''Random access table reader for integer sequences.'''
    pass


class RandomAccessLatticeReader(_RandomAccessReaderBase,
                                _RandomAccessLatticeReader):
    '''Random access table reader for lattices.'''
    pass


class RandomAccessCompactLatticeReader(_RandomAccessReaderBase,
                                       _RandomAccessCompactLatticeReader):
    '''Random access table reader for compact lattices.'''
    pass


################################################################################
# Writers
################################################################################


class _WriterBase(object):
    '''Base class defining the additional Python API for table writers.'''

    def __init__(self, wspecifier=''):
        '''

            This class is used for writing objects to an archive or script file. It
            implements the `__setitem__` method to provide a dictionary-like
            interface for writing table entries, e.g. `writer[key] = value` writes
            the pair `(key, value)` to the table.

            Args:
                wspecifier (str): Kaldi wspecifier for writing the table.
                    If provided, the table is opened for writing.

            Raises:
                IOError: If opening the table for writing fails.
            '''
        super(_WriterBase, self).__init__()
        if wspecifier != '':
            if not self.Open(wspecifier):
                raise IOError(
                    'Error opening table writer with wspecifier: {}'.format(
                        wspecifier))

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if self.IsOpen():
            self.Close()
    
    def __setitem__(self, key, value):
        self.Write(key, value)

    def Open(self, wspecifier):
        '''Opens the table for writing.

            Args:
                wspecifier(str): Kaldi wspecifier for writing the table.
                    If provided, the table is opened for writing.

            Returns:
                True if table is opened successfully, False otherwise.

            Raises:
                IOError: If opening the table for writing fails.
            '''
        return super(_WriterBase, self).Open(wspecifier)

    def Flush(self):
        '''Flushes the table contents to disk/pipe.'''
        super(_WriterBase, self).Flush()

    def Write(self, key, value):
        '''Writes the `(key, value)` pair to the table.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Args:
                key (str): The key.
                value: The value.
            '''
        super(_WriterBase, self).Write(key, value)

    def IsOpen(self):
        '''Indicates whether the table writer is open or not.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Returns:
              True if the table writer is open, False otherwise.
            '''
        return super(_WriterBase, self).IsOpen()

    def Close(self):
        '''Closes the table.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Returns:
                True if table is closed successfully, False otherwise.
            '''
        return super(_WriterBase, self).Close()


class NnetChainExampleWriter(_WriterBase, _NnetChainExampleWriter):
    '''Table writer for nnet chain examples.'''
    pass


class MatrixWriter(_WriterBase, _BaseFloatMatrixWriter):
    '''Table writer for single precision matrices.'''

    def Write(self, key, value):
        if isinstance(value, np.ndarray):
            m = kaldi_pybind.FloatSubMatrix(value)
            value = kaldi_pybind.FloatMatrix(m)
        super().Write(key, value)


class VectorWriter(_WriterBase, _BaseFloatVectorWriter):
    '''Table writer for single precision vectors.'''

    def Write(self, key, value):
        if isinstance(value, np.ndarray):
            v = kaldi_pybind.FloatSubVector(value)
            value = kaldi_pybind.FloatVector(v)
        super().Write(key, value)


class CompressedMatrixWriter(_WriterBase, _CompressedMatrixWriter):
    '''Table writer for single precision compressed matrices.'''
    pass


class IntVectorWriter(_WriterBase, _Int32VectorWriter):
    '''Table writer for integer sequences.'''
    pass


class LatticeWriter(_WriterBase, _LatticeWriter):
    '''Table writer for lattices.'''
    pass


class CompactLatticeWriter(_WriterBase, _CompactLatticeWriter):
    '''Table writer for compact lattices.'''
    pass


if False:
    # TODO(fangjun): enable the following once other wrappers are added

    class SequentialDoubleVectorReader(_SequentialReaderBase,
                                       _kaldi_table.SequentialDoubleVectorReader
                                      ):
        '''Sequential table reader for double precision vectors.'''
        pass

    class SequentialDoubleMatrixReader(_SequentialReaderBase,
                                       _kaldi_table.SequentialDoubleMatrixReader
                                      ):
        '''Sequential table reader for double precision matrices.'''
        pass

    class SequentialPosteriorReader(_SequentialReaderBase,
                                    _kaldi_table.SequentialPosteriorReader):
        '''Sequential table reader for frame posteriors.'''
        pass

    class SequentialGaussPostReader(_SequentialReaderBase,
                                    _kaldi_table.SequentialGaussPostReader):
        '''Sequential table reader for Gaussian-level frame posteriors.'''
        pass

    class SequentialFstReader(_SequentialReaderBase,
                              _kaldi_table_ext.SequentialFstReader):
        '''Sequential table reader for FSTs over the tropical semiring.'''
        pass

    class SequentialLogFstReader(_SequentialReaderBase,
                                 _kaldi_table_ext.SequentialLogFstReader):
        '''Sequential table reader for FSTs over the log semiring.'''
        pass

    class SequentialKwsIndexFstReader(
            _SequentialReaderBase,
            _kaldi_table_ext.SequentialKwsIndexFstReader):
        '''Sequential table reader for FSTs over the KWS index semiring.'''
        pass

    class SequentialRnnlmExampleReader(_SequentialReaderBase,
                                       _kaldi_table.SequentialRnnlmExampleReader
                                      ):
        '''Sequential table reader for RNNLM examples.'''
        pass

    class SequentialIntReader(_SequentialReaderBase,
                              _kaldi_table.SequentialIntReader):
        '''Sequential table reader for integers.'''
        pass

    class SequentialFloatReader(_SequentialReaderBase,
                                _kaldi_table.SequentialFloatReader):
        '''Sequential table reader for single precision floats.'''
        pass

    class SequentialDoubleReader(_SequentialReaderBase,
                                 _kaldi_table.SequentialDoubleReader):
        '''Sequential table reader for double precision floats.'''
        pass

    class SequentialBoolReader(_SequentialReaderBase,
                               _kaldi_table.SequentialBoolReader):
        '''Sequential table reader for Booleans.'''
        pass

    class SequentialIntVectorVectorReader(
            _SequentialReaderBase,
            _kaldi_table.SequentialIntVectorVectorReader):
        '''Sequential table reader for sequences of integer sequences.'''
        pass

    class SequentialIntPairVectorReader(
            _SequentialReaderBase, _kaldi_table.SequentialIntPairVectorReader):
        '''Sequential table reader for sequences of integer pairs.'''
        pass

    class SequentialFloatPairVectorReader(
            _SequentialReaderBase,
            _kaldi_table.SequentialFloatPairVectorReader):
        '''Sequential table reader for sequences of single precision float pairs.'''
        pass

    class RandomAccessDoubleVectorReader(
            _RandomAccessReaderBase,
            _kaldi_table.RandomAccessDoubleVectorReader):
        '''Random access table reader for double precision vectors.'''
        pass

    class RandomAccessDoubleMatrixReader(
            _RandomAccessReaderBase,
            _kaldi_table.RandomAccessDoubleMatrixReader):
        '''Random access table reader for double precision matrices.'''
        pass

    class RandomAccessPosteriorReader(_RandomAccessReaderBase,
                                      _kaldi_table.RandomAccessPosteriorReader):
        '''Random access table reader for frame posteriors.'''
        pass

    class RandomAccessGaussPostReader(_RandomAccessReaderBase,
                                      _kaldi_table.RandomAccessGaussPostReader):
        '''Random access table reader for Gaussian-level frame posteriors.'''
        pass

    class RandomAccessFstReader(_RandomAccessReaderBase,
                                _kaldi_table_ext.RandomAccessFstReader):
        '''Random access table reader for FSTs over the tropical semiring.'''
        pass

    class RandomAccessLogFstReader(_RandomAccessReaderBase,
                                   _kaldi_table_ext.RandomAccessLogFstReader):
        '''Random access table reader for FSTs over the log semiring.'''
        pass

    class RandomAccessKwsIndexFstReader(
            _RandomAccessReaderBase,
            _kaldi_table_ext.RandomAccessKwsIndexFstReader):
        '''Random access table reader for FSTs over the KWS index semiring.'''
        pass

    class RandomAccessIntReader(_RandomAccessReaderBase,
                                _kaldi_table.RandomAccessIntReader):
        '''Random access table reader for integers.'''
        pass

    class RandomAccessFloatReader(_RandomAccessReaderBase,
                                  _kaldi_table.RandomAccessFloatReader):
        '''Random access table reader for single precision floats.'''
        pass

    class RandomAccessDoubleReader(_RandomAccessReaderBase,
                                   _kaldi_table.RandomAccessDoubleReader):
        '''Random access table reader for double precision floats.'''
        pass

    class RandomAccessBoolReader(_RandomAccessReaderBase,
                                 _kaldi_table.RandomAccessBoolReader):
        '''Random access table reader for Booleans.'''
        pass

    class RandomAccessIntVectorVectorReader(
            _RandomAccessReaderBase,
            _kaldi_table.RandomAccessIntVectorVectorReader):
        '''Random access table reader for sequences of integer sequences.'''
        pass

    class RandomAccessIntPairVectorReader(
            _RandomAccessReaderBase,
            _kaldi_table.RandomAccessIntPairVectorReader):
        '''Random access table reader for sequences of integer pairs.'''
        pass

    class RandomAccessFloatPairVectorReader(
            _RandomAccessReaderBase,
            _kaldi_table.RandomAccessFloatPairVectorReader):
        '''
        Random access table reader for sequences of single precision float pairs.
        '''
        pass

################################################################################
# Mapped Random Access Readers
################################################################################

    class _RandomAccessReaderMappedBase(object):
        '''
        Base class defining the Python API for mapped random access table readers.
        '''

        def __init__(self, table_rspecifier='', map_rspecifier=''):
            '''
            This class is used for randomly accessing objects in an archive or
            script file. It implements `__contains__` and `__getitem__` methods to
            provide a dictionary-like interface for accessing table entries. If a
            **map_rspecifier** is provided, the map is used for converting the keys
            to the actual keys used to query the table, e.g. `reader[key]` returns
            the `value` associated with the key `map[key]`. Otherwise, it works like
            a random access table reader.

            Args:
                table_rspecifier(str): Kaldi rspecifier for reading the table.
                    If provided, the table is opened for reading.
                map_rspecifier (str): Kaldi rspecifier for reading the map.
                    If provided, the map is opened for reading.

            Raises:
                IOError: If opening the table or map for reading fails.
            '''
            super(_RandomAccessReaderMappedBase, self).__init__()
            if table_rspecifier != '' and map_rspecifier != '':
                if not self.open(table_rspecifier, map_rspecifier):
                    raise IOError(
                        'Error opening mapped random access table reader '
                        'with table_rspecifier: {}, map_rspecifier: {}'.format(
                            table_rspecifier, map_rspecifier))

        def __enter__(self):
            return self

        def __contains__(self, key):
            return self.has_key(key)

        def __getitem__(self, key):
            if self.has_key(key):
                return self.value(key)
            else:
                raise KeyError(key)

        def open(self, table_rspecifier, map_rspecifier):
            '''Opens the table for reading.

            Args:
                table_rspecifier(str): Kaldi rspecifier for reading the table.
                    If provided, the table is opened for reading.
                map_rspecifier (str): Kaldi rspecifier for reading the map.
                    If provided, the map is opened for reading.

            Returns:
                True if table is opened successfully, False otherwise.

            Raises:
                IOError: If opening the table or map for reading fails.
            '''
            return super(_RandomAccessReaderMappedBase,
                         self).open(table_rspecifier, map_rspecifier)

        def has_key(self, key):
            '''Checks whether the table has the key.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Args:
                key (str): The key.

            Returns:
              True if the table has the key, False otherwise.
            '''
            return super(_RandomAccessReaderMappedBase, self).has_key(key)

        def value(self, key):
            '''Returns the value associated with the key.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Args:
                key (str): The key.

            Returns:
                The value associated with the key.
            '''
            return super(_RandomAccessReaderMappedBase, self).value(key)

        def is_open(self):
            '''Indicates whether the table reader is open or not.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Returns:
              True if the table reader is open, False otherwise.
            '''
            return super(_RandomAccessReaderMappedBase, self).is_open()

        def close(self):
            '''Closes the table.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Returns:
                True if table is closed successfully, False otherwise.
            '''
            return super(_RandomAccessReaderMappedBase, self).close()

    class RandomAccessVectorReaderMapped(
            _RandomAccessReaderMappedBase,
            _kaldi_table.RandomAccessVectorReaderMapped):
        '''Mapped random access table reader for single precision vectors.'''
        pass

    class RandomAccessDoubleVectorReaderMapped(
            _RandomAccessReaderMappedBase,
            _kaldi_table.RandomAccessDoubleVectorReaderMapped):
        '''Mapped random access table reader for double precision vectors.'''
        pass

    class RandomAccessMatrixReaderMapped(
            _RandomAccessReaderMappedBase,
            _kaldi_table.RandomAccessMatrixReaderMapped):
        '''Mapped random access table reader for single precision matrices.'''
        pass

    class RandomAccessDoubleMatrixReaderMapped(
            _RandomAccessReaderMappedBase,
            _kaldi_table.RandomAccessDoubleMatrixReaderMapped):
        '''Mapped random access table reader for double precision matrices.'''
        pass

    class RandomAccessFloatReaderMapped(
            _RandomAccessReaderMappedBase,
            _kaldi_table.RandomAccessFloatReaderMapped):
        '''Mapped random access table reader for single precision floats.'''
        pass

    class DoubleVectorWriter(_WriterBase, _kaldi_table.DoubleVectorWriter):
        '''Table writer for double precision vectors.'''

        def write(self, key, value):
            '''Writes the `(key, value)` pair to the table.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Overrides write to accept both DoubleVector and DoubleSubVector.

            Args:
                key (str): The key.
                value: The value.
            '''
            super(DoubleVectorWriter, self).write(key,
                                                  _matrix.DoubleVector(value))

    class DoubleMatrixWriter(_WriterBase, _kaldi_table.DoubleMatrixWriter):
        '''Table writer for double precision matrices.'''

        def write(self, key, value):
            '''Writes the `(key, value)` pair to the table.

            This method is provided for compatibility with the C++ API only;
            most users should use the Pythonic API.

            Overrides write to accept both DoubleMatrix and DoubleSubMatrix.

            Args:
                key (str): The key.
                value: The value.
            '''
            super(DoubleMatrixWriter, self).write(key,
                                                  _matrix.DoubleMatrix(value))

    class WaveWriter(_WriterBase, _kaldi_table.WaveWriter):
        '''Table writer for wave files.'''
        pass

    class PosteriorWriter(_WriterBase, _kaldi_table.PosteriorWriter):
        '''Table writer for frame posteriors.'''
        pass

    class GaussPostWriter(_WriterBase, _kaldi_table.GaussPostWriter):
        '''Table writer for Gaussian-level frame posteriors.'''
        pass

    class FstWriter(_WriterBase, _kaldi_table_ext.FstWriter):
        '''Table writer for FSTs over the tropical semiring.'''
        pass

    class LogFstWriter(_WriterBase, _kaldi_table_ext.LogFstWriter):
        '''Table writer for FSTs over the log semiring.'''
        pass

    class KwsIndexFstWriter(_WriterBase, _kaldi_table_ext.KwsIndexFstWriter):
        '''Table writer for FSTs over the KWS index semiring.'''
        pass

    class NnetExampleWriter(_WriterBase, _kaldi_table.NnetExampleWriter):
        '''Table writer for nnet examples.'''
        pass

    class RnnlmExampleWriter(_WriterBase, _kaldi_table.RnnlmExampleWriter):
        '''Table writer for RNNLM examples.'''
        pass

    class IntWriter(_WriterBase, _kaldi_table.IntWriter):
        '''Table writer for integers.'''
        pass

    class FloatWriter(_WriterBase, _kaldi_table.FloatWriter):
        '''Table writer for single precision floats.'''
        pass

    class DoubleWriter(_WriterBase, _kaldi_table.DoubleWriter):
        '''Table writer for double precision floats.'''
        pass

    class BoolWriter(_WriterBase, _kaldi_table.BoolWriter):
        '''Table writer for Booleans.'''
        pass

    class IntVectorVectorWriter(_WriterBase,
                                _kaldi_table.IntVectorVectorWriter):
        '''Table writer for sequences of integer sequences.'''
        pass

    class IntPairVectorWriter(_WriterBase, _kaldi_table.IntPairVectorWriter):
        '''Table writer for sequences of integer pairs.'''
        pass

    class FloatPairVectorWriter(_WriterBase,
                                _kaldi_table.FloatPairVectorWriter):
        '''Table writer for sequences of single precision float pairs.'''
        pass


################################################################################
