#!/usr/bin/env python3

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import kaldi_pybind


class FileNotOpenException(Exception):
    pass


# same name as
# https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py#L153
def read_vec_int(rxfilename):
    '''Read an int32 vector from an rxfilename.

    It can be used to read alignment information from `ali.scp`
    '''
    ki = kaldi_pybind.Input()
    is_opened, = ki.Open(rxfilename, read_header=False)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    holder = kaldi_pybind.IntVectorHolder()
    holder.Read(ki.Stream())
    v = holder.Value().copy()
    ki.Close()

    return v


# same name as
# https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py#L256
def read_vec_flt(rxfilename):
    '''Read a kaldi::Vector<float> from an rxfilename
    '''
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    v = kaldi_pybind.FloatVector()
    v.Read(ki.Stream(), is_binary)
    ki.Close()

    return v


# same name as
# https://github.com/vesis84/kaldi-io-for-python/blob/master/kaldi_io/kaldi_io.py#L376
def read_mat(rxfilename):
    '''Read a kaldi::Matrix<float> from an rxfilename
    '''
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    mat = kaldi_pybind.FloatMatrix()
    mat.Read(ki.Stream(), is_binary)
    ki.Close()

    return mat


def read_transition_model(rxfilename):
    '''Read binary transition model from an rxfilename.
    '''
    ki = kaldi_pybind.Input()
    is_opened, is_binary = ki.Open(rxfilename, read_header=True)
    if not is_opened:
        raise FileNotOpenException('Failed to open {}'.format(rxfilename))

    trans_model = kaldi_pybind.TransitionModel()
    trans_model.Read(ki.Stream(), is_binary)

    ki.Close()

    return trans_model
