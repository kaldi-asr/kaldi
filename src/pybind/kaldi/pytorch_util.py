#!/usr/bin/env python3

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir))

import kaldi_pybind


def PytorchToCuSubMatrix(dlpack_tensor):
    cu_sub_matrix = kaldi_pybind.CuSubMatrixFromDLPack(dlpack_tensor)
    return cu_sub_matrix


def PytorchToSubMatrix(dlpack_tensor):
    sub_matrix = kaldi_pybind.SubMatrixFromDLPack(dlpack_tensor)
    return sub_matrix


def PytorchToCuSubVector(dlpack_tensor):
    cu_sub_vector = kaldi_pybind.CuSubVectorFromDLPack(dlpack_tensor)
    return cu_sub_vector


def PytorchToSubVector(dlpack_tensor):
    sub_vector = kaldi_pybind.SubVectorFromDLPack(dlpack_tensor)
    return sub_vector
