# encoding: utf-8
# distutils: language = c++
from __future__ import unicode_literals

from cython cimport address
from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp cimport bool
cimport fst._fst
cimport fst.libfst
import fst
from pykaldi.utils import lattice_to_nbest


cdef extern from "dec-wrap/dec-wrap-latgen-wrapper.h" namespace "kaldi":
    cdef cppclass GmmLatgenWrapper:
        size_t Decode(size_t max_frames) except +
        void FrameIn(unsigned char *frame, size_t frame_len) except +
        bool GetBestPath(vector[int] v_out, float *prob) except +
        bool GetRawLattice(fst.libfst.StdVectorFst *fst_out) except +
        bool GetLattice(fst.libfst.LogVectorFst *fst_out, double *tot_prob) except +
        void PruneFinal() except +
        void Reset(bool keep_buffer_data) except +
        int Setup(int argc, char **argv) except +


cdef class PyGmmLatgenWrapper:
    """PyGmmLatgenWrapper"""
    cdef GmmLatgenWrapper * thisptr
    cdef long fs
    cdef int nchan, bits

    def __cinit__(self):
        self.thisptr = new GmmLatgenWrapper()

    def __init__(self, fs=16000, nchan=1, bits=16):
        """ __init__(self, fs=16000, nchan=1, bits=16)"""
        self.fs, self.nchan, self.bits = fs, nchan, bits
        assert(self.bits % 8 == 0)

    def __dealloc__(self):
        del self.thisptr

    def decode(self, max_frames=10):
        """decode(self, max_frames)"""
        return self.thisptr.Decode(max_frames)

    def frame_in(self, bytes frame_str):
        """frame_in(self, bytes frame_str, int num_samples)"""
        num_bytes = (self.bits / 8)
        num_samples = len(frame_str) / num_bytes
        assert(num_samples * num_bytes == len(frame_str)), "Not align audio to for %d bits" % self.bits
        self.thisptr.FrameIn(frame_str, num_samples)

    def get_best_path(self):
        """get_best_path(self)"""
        cdef vector[int] t
        cdef float prob
        self.thisptr.GetBestPath(t, address(prob))
        ids = [t[i] for i in xrange(t.size())]
        return (prob, ids)

    def get_nbest(self, n=1):
        """get_nbest(self, n=1)"""
        prob, lat = self.get_lattice()
        return lattice_to_nbest(lat, n)

    def get_lattice(self):
        cdef double prob
        r = fst.LogVectorFst()
        self.thisptr.GetLattice((<fst._fst.LogVectorFst?>r).fst, address(prob))
        return (prob, r)

    def get_raw_lattice(self):
        r = fst.StdVectorFst()
        self.thisptr.GetRawLattice((<fst._fst.StdVectorFst?>r).fst)
        return r

    def prune_final(self):
        """prune_final(self)"""
        self.thisptr.PruneFinal()

    def reset(self, keep_buffer_data=True):
        """reset(self, keep_buffer_data)"""
        self.thisptr.Reset(keep_buffer_data)

    def setup(self, args):
        """setup(self, args)"""
        args = ['PyGmmLatgenWrapper'] + args
        cdef char **string_buf = <char**>malloc(len(args) * sizeof(char*))
        if string_buf is NULL:
            raise MemoryError()
        try:
            for i, s in enumerate(args):
                string_buf[i] = s
            self.thisptr.Setup(len(args), string_buf)
        finally:
            free(string_buf)


class DummyDecoder(object):
    """For debugging purposes."""

    def __init__(self):
        print 'DummyDecoder initialized'

    def frame_in(self, frame):
        """rec_in(self, frame)"""
        print 'Dummy enqueing frame of length %d' % len(frame)

    def decode(self):
        """decode(self)"""
        pass

    def get_Nbest(self):
        return [(1.0, 'answer')]
